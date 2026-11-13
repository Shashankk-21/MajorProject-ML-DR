"""
utils/checkpoint.py

Checkpoint save/load, per training-phase-decisions.md Section 5, the
"checkpoint-and-resume is a day-one requirement, not an add-on"
non-negotiable, and the discussion chat's patch spec (schema v1 -> v2).
One file holds everything needed to resume a run -- for FL runs, all 4
clients' state together (confirmed: not split per-client, since a round
can't proceed with only one client's state restored anyway).

SCHEMA v2 -- what changed from v1, and why:
- B1: model may now be either a plain peft model OR an Opacus
  GradSampleModule wrapping one. GradSampleModule stores the wrapped
  module under `_module` and does not forward arbitrary attribute
  access (like peft's own `peft_config`) to it -- calling
  get_peft_model_state_dict() directly on a GradSampleModule-wrapped
  model raises AttributeError. _unwrap_for_peft() below finds the real
  peft model by walking `_module` (duck-typed -- this file still never
  imports opacus), and every get/set_peft_model_state_dict call now
  goes through it. A checkpoint saved from a DP-wrapped model loads
  correctly into either a plain peft model or a fresh DP-wrapped one,
  since the stored state is always the unwrapped PEFT-only dict either
  way.
- B2: set_peft_model_state_dict() loads with strict=False semantics, so
  it doesn't raise on its own for a mismatched checkpoint (different
  LoRA rank, different target modules, a different head shape).
  load_checkpoint() now validates structurally in two independent
  passes: BEFORE loading, the saved key set and every tensor's shape
  are compared against what the current (unwrapped) model's own
  get_peft_model_state_dict() currently expects; AFTER loading, the
  returned missing/unexpected keys are inspected directly. Any
  unexpected_keys always raises. missing_keys only raises for adapter
  or head keys (name contains "lora_" or "modules_to_save") -- NOT for
  every missing key, since the frozen backbone is *always* absent from
  a PEFT-only checkpoint by design, and raising on that would fail
  every single resume, every time, unconditionally.
- M1: optimizer is now optional on both save and load. Locked project
  design: Experiment 1 (centralized) preserves optimizer state across
  resumes; in FL (Experiments 2-8), every client starts a FRESH
  optimizer every round, so the round-boundary checkpoint the server
  writes stores no optimizer at all -- that's correct, not a gap. One
  checkpoint file holds one model and AT MOST one optimizer; it never
  holds four clients' optimizers. If federated/client.py ever needs a
  mid-round per-client checkpoint (one file per client, with that
  client's own optimizer and accountant state), these same functions
  serve that too -- nothing extra needs building for it, the optional
  optimizer already covers it.
- M2: scheduler is now optional on both save and load, same
  presence-rule pattern as the optimizer, restored AFTER the optimizer.
  FL's round-level LR schedule needs no scheduler object saved here at
  all -- it's rebuilt from round_offset on every resume, not from a
  stored state_dict.
- M3: every saved RNG tensor is moved to CPU before
  torch.set_rng_state()/torch.cuda.set_rng_state_all() (those setters
  require CPU byte tensors; a CUDA map_location would otherwise hand
  them a CUDA tensor and break silently or loudly depending on the
  call). If the saved CUDA device count doesn't match what's available
  now, or CUDA isn't available at all, CUDA RNG restore is skipped with
  a warning rather than crashing -- exact CUDA RNG continuity isn't
  guaranteed in that case, but the rest of the resume still proceeds.
  Under DP, Opacus's Poisson sampler and noise generation draw from the
  global torch RNG, which this file already captures/restores -- no
  separate handling needed for that.
- A1+L2: optional run_config (a plain dict, stored verbatim) on save
  and expected_config on load. If expected_config is given, every key
  in it must exist in the checkpoint's saved run_config and match
  (exact equality for str/int/bool, math.isclose(rel_tol=1e-9) for
  floats) -- otherwise load_checkpoint raises ValueError listing every
  difference, rather than silently resuming into a different
  configuration than what actually produced this checkpoint (e.g. a
  different noise_multiplier or sample_rate would silently invalidate
  the privacy accounting). This file doesn't care what keys run_config
  contains -- suggested, caller's choice: experiment_id, model_name,
  lora_rank, lora_alpha, img_size, num_classes, accountant,
  noise_multiplier, sample_rate, max_grad_norm, delta,
  grad_sample_mode. The accountant string used for training must be the
  same one used for calibration -- verified separately, in
  privacy_calibration.py, not here. run_config alone is not a
  substitute for comparing actual accountant history; it's an
  additional, independent check.
- L1: durable write. f.flush() + os.fsync(f.fileno()) added before the
  final os.replace() -- the checkpoint destination is an NFS volume,
  where data can sit in client-side cache well after a write() call
  returns; without an explicit fsync, a pod kill shortly after a
  "completed" save could still lose data that was never actually
  pushed to the NFS server, even though the atomic-rename logic itself
  is otherwise correct.

What gets saved (full v2 field list): PEFT-only model weights (never
the frozen backbone -- it's reconstructed from timm's pretrained
weights every time create_coatnet_lora_model() runs), optional
optimizer state, optional scheduler state, RNG state (torch CPU + all
CUDA, numpy, python's `random`, plus caller-supplied extra_generators
-- whatever generator controls the caller's DataLoader shuffle, if any;
no particular sampler is assumed), round_offset, per-client Opacus
accountant state (opaque -- the caller supplies
{client_id: privacy_engine.accountant.state_dict()}, and on resume
calls .load_state_dict() on a fresh accountant of the SAME mechanism;
this file never imports opacus to interpret it), optional run_config,
the W&B run id, and an open-ended `extra` dict.

Writes are atomic: a full checkpoint is written to a temp file in the
SAME directory as the destination, flushed and fsync'd, then renamed
into place with os.replace() -- a same-filesystem rename is atomic on
POSIX, so a process killed mid-write (the pod's own hourly restart --
still happening; the sleep-infinity fix is centre-approved but not yet
applied) can never leave a half-written checkpoint at the real path.
Worst case after a kill: a stray, harmless .tmp file, and the previous
good checkpoint still fully intact.
"""

from __future__ import annotations

import logging
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from peft import get_peft_model_state_dict, set_peft_model_state_dict

logger = logging.getLogger(__name__)

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
]

# Bumped 1 -> 2: added optional scheduler state, optional run_config,
# and optional optimizer (v1 required one). A v1 checkpoint is rejected
# with a clear error below, not force-loaded against the new schema.
SCHEMA_VERSION = 2


def _unwrap_for_peft(model: nn.Module) -> nn.Module:
    """Find the underlying peft model inside a possible Opacus
    GradSampleModule wrapper, without importing opacus (duck-typed,
    consistent with this file never importing opacus anywhere else).

    GradSampleModule stores the wrapped module under `_module` and
    doesn't forward arbitrary attribute access -- peft's own
    `peft_config` attribute included -- to it. Walking `_module` a
    bounded number of levels (in case of more than one layer of
    wrapping) until something with `peft_config` is found handles both
    the wrapped and unwrapped case the same way, rather than requiring
    every caller to know which one it has.
    """
    current = model
    for _ in range(5):  # generous but bounded -- real wrapping depth is 0 or 1
        if hasattr(current, "peft_config"):
            return current
        inner = getattr(current, "_module", None)
        if inner is None:
            break
        current = inner
    raise TypeError(
        f"Could not find a peft model (an object with .peft_config) by "
        f"walking `_module` from {type(model).__name__}. Expected either "
        f"a plain peft model or an Opacus GradSampleModule wrapping one "
        f"-- got something else. Checkpointing needs the real peft model "
        f"to call get_peft_model_state_dict()/set_peft_model_state_dict() "
        f"on; refusing to guess."
    )


def _validate_peft_state_dict(
    unwrapped_model: nn.Module, saved_state_dict: Dict[str, torch.Tensor]
) -> None:
    """Pre-load structural check: compares the checkpoint's saved PEFT
    state dict against what the CURRENT model's own
    get_peft_model_state_dict() expects right now -- key set AND tensor
    shapes -- before attempting to load anything.

    set_peft_model_state_dict() loads with strict=False semantics and
    doesn't raise on its own for a mismatch, so this is what actually
    catches a checkpoint saved against a different LoRA config
    (different rank, different target modules, a different head shape)
    before any weights get loaded, rather than silently leaving some
    mix of fresh and loaded weights in place with no error at all.
    """
    expected = get_peft_model_state_dict(unwrapped_model)
    expected_keys = set(expected.keys())
    saved_keys = set(saved_state_dict.keys())

    missing = sorted(expected_keys - saved_keys)
    unexpected = sorted(saved_keys - expected_keys)
    shape_mismatched = sorted(
        key for key in (expected_keys & saved_keys)
        if expected[key].shape != saved_state_dict[key].shape
    )

    if missing or unexpected or shape_mismatched:
        raise ValueError(
            "Checkpoint's PEFT state dict doesn't match the current "
            "model's expected keys/shapes -- likely saved against a "
            "different LoRA config (rank, target modules, or head "
            "shape). Refusing to load a mismatched checkpoint silently.\n"
            f"Missing (expected by current model, absent in checkpoint): {missing}\n"
            f"Unexpected (in checkpoint, not expected by current model): {unexpected}\n"
            f"Shape-mismatched: {shape_mismatched}"
        )


def _apply_peft_state_dict(
    unwrapped_model: nn.Module, saved_state_dict: Dict[str, torch.Tensor]
) -> None:
    """Load the (already pre-validated) PEFT state dict, then inspect
    set_peft_model_state_dict()'s own returned result as a second,
    independent check.

    unexpected_keys always raises. missing_keys only raises for an
    adapter or head key (name contains "lora_" or "modules_to_save") --
    NOT for every missing key: the result lists every frozen backbone
    weight as missing, unconditionally, because the adapter-only state
    dict never contains them by design. Raising on all missing_keys
    would fail every single resume, every time.
    """
    result = set_peft_model_state_dict(unwrapped_model, saved_state_dict)
    if result is None:
        return

    unexpected_keys = list(getattr(result, "unexpected_keys", []) or [])
    missing_keys = list(getattr(result, "missing_keys", []) or [])
    relevant_missing = [
        key for key in missing_keys if "lora_" in key or "modules_to_save" in key
    ]

    if unexpected_keys:
        raise ValueError(
            f"set_peft_model_state_dict reported unexpected key(s) that "
            f"shouldn't exist in this model at all: {unexpected_keys}. "
            f"A mismatched checkpoint must never silently leave fresh "
            f"LoRA/head weights in place -- investigate before "
            f"proceeding."
        )
    if relevant_missing:
        raise ValueError(
            f"set_peft_model_state_dict reported missing adapter/head "
            f"key(s) that should have been restored from the checkpoint: "
            f"{relevant_missing}. A mismatched checkpoint must never "
            f"silently leave fresh LoRA/head weights in place -- "
            f"investigate before proceeding."
        )


def _config_values_match(expected_value: Any, saved_value: Any) -> bool:
    if isinstance(expected_value, float) or isinstance(saved_value, float):
        try:
            return math.isclose(float(expected_value), float(saved_value), rel_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return expected_value == saved_value


def _validate_run_config(
    saved_run_config: Optional[Dict[str, Any]], expected_config: Dict[str, Any]
) -> None:
    saved_run_config = saved_run_config or {}
    mismatches = []
    for key, expected_value in expected_config.items():
        if key not in saved_run_config:
            mismatches.append(
                f"{key}: expected {expected_value!r}, not present in checkpoint's run_config"
            )
            continue
        saved_value = saved_run_config[key]
        if not _config_values_match(expected_value, saved_value):
            mismatches.append(f"{key}: expected {expected_value!r}, checkpoint has {saved_value!r}")

    if mismatches:
        raise ValueError(
            "load_checkpoint: expected_config mismatch against this "
            "checkpoint's saved run_config. Resuming into a different "
            "configuration than what produced this checkpoint could "
            "silently corrupt the run (e.g. a different noise_multiplier "
            "or sample_rate would invalidate the privacy accounting "
            "entirely, with no other symptom). Differences:\n  "
            + "\n  ".join(mismatches)
        )


def _capture_rng_state(
    extra_generators: Optional[Dict[str, torch.Generator]],
) -> Dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "extra_generators": {
            name: gen.get_state() for name, gen in (extra_generators or {}).items()
        },
    }


def _restore_rng_state(
    rng_state: Dict[str, Any],
    extra_generators: Optional[Dict[str, torch.Generator]],
) -> None:
    # Every saved tensor is moved to CPU before use: torch.set_rng_state
    # and torch.cuda.set_rng_state_all both require CPU byte tensors,
    # and a CUDA map_location on the torch.load() call that produced
    # this dict would otherwise hand them CUDA tensors here, breaking
    # silently or loudly depending on the call.
    torch.set_rng_state(rng_state["torch_cpu"].cpu())

    saved_cuda = rng_state.get("torch_cuda")
    if saved_cuda is not None:
        if not torch.cuda.is_available():
            logger.warning(
                "load_checkpoint: checkpoint has saved CUDA RNG state, "
                "but CUDA is not available in this process -- skipping "
                "CUDA RNG restore. Exact CUDA RNG continuity across this "
                "resume is NOT guaranteed."
            )
        elif len(saved_cuda) != torch.cuda.device_count():
            logger.warning(
                "load_checkpoint: checkpoint was saved with %d CUDA "
                "device(s)' RNG state, but %d are available now -- "
                "skipping CUDA RNG restore rather than guessing a "
                "mapping between them. Exact CUDA RNG continuity across "
                "this resume is NOT guaranteed.",
                len(saved_cuda), torch.cuda.device_count(),
            )
        else:
            torch.cuda.set_rng_state_all([t.cpu() for t in saved_cuda])

    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["python"])

    saved_generators = rng_state.get("extra_generators", {})
    for name, gen in (extra_generators or {}).items():
        if name not in saved_generators:
            logger.warning(
                "load_checkpoint: no saved RNG state found for generator "
                "'%s' -- it will keep running from its current state "
                "rather than the checkpointed one. If this generator "
                "controls the caller's DataLoader shuffle order, resumed "
                "data order will diverge from the original run from this "
                "point on.",
                name,
            )
            continue
        gen.set_state(saved_generators[name].cpu())


def save_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    round_offset: int = 0,
    opacus_accountant_states: Optional[Dict[str, Any]] = None,
    extra_generators: Optional[Dict[str, torch.Generator]] = None,
    wandb_run_id: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one complete, atomic checkpoint to `path`.

    Args:
        path: destination file path. The temp file used for the atomic
            write is created in the same directory, so `path`'s parent
            directory must exist (or be creatable) and be writable.
        model: the peft-wrapped model, OR an Opacus GradSampleModule
            wrapping one (unwrapped internally -- see _unwrap_for_peft).
            Only adapter/head weights are saved
            (get_peft_model_state_dict()), never the frozen backbone.
        optimizer: optional. Saved via the standard .state_dict() when
            given. Locked design: Experiment 1 (centralized) passes one
            and preserves its state across resumes; FL experiments
            (2-8) start a fresh optimizer every round and pass None at
            the round boundary -- that's correct, not a gap. This
            function also serves an optional mid-round per-client
            checkpoint (one file per client, with that client's own
            optimizer and accountant state) if federated/client.py ever
            needs one -- nothing extra needs building for that case,
            the optional optimizer already covers it. One checkpoint
            file holds one model and AT MOST one optimizer, never four
            clients' optimizers at once.
        scheduler: optional, same treatment as optimizer. FL's
            round-level LR schedule needs no scheduler object here at
            all -- it's rebuilt from round_offset on resume, never from
            a saved state_dict.
        round_offset: 0 for centralized training (Experiment 1, not FL);
            for federated experiments, the true round number at save
            time, so a resumed run's LR schedule continues correctly
            instead of restarting from round 0 (Section 5).
        opacus_accountant_states: opaque, caller-prepared object holding
            all 4 clients' Opacus accountant state together (confirmed:
            one file, not per-client) -- typically
            {client_id: privacy_engine.accountant.state_dict()}. None
            for the non-DP experiments (1-3). Not interpreted or
            validated here; this file never imports opacus.
        extra_generators: additional torch.Generator objects whose state
            must survive a restart for exact resume -- whatever
            generator controls the caller's DataLoader shuffle order, if
            any (no particular sampler is assumed). Keyed by a
            caller-chosen name; restored under the same name in
            load_checkpoint().
        wandb_run_id: the active W&B run id, so a resumed run reattaches
            instead of starting a new one.
        run_config: optional plain dict, stored verbatim -- e.g.
            experiment_id, model_name, lora_rank, lora_alpha, img_size,
            num_classes, accountant, noise_multiplier, sample_rate,
            max_grad_norm, delta, grad_sample_mode (caller's choice;
            this file doesn't interpret any of it). Paired with
            expected_config in load_checkpoint() to catch a resume into
            a different configuration than what produced this
            checkpoint.
        extra: anything else the caller wants preserved verbatim (e.g.
            current epoch number, best-selection-metric-so-far for early
            stopping) -- stored and returned as-is, uninterpreted.
    """
    unwrapped_model = _unwrap_for_peft(model)

    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": get_peft_model_state_dict(unwrapped_model),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": _capture_rng_state(extra_generators),
        "round_offset": round_offset,
        "opacus_accountant_states": opacus_accountant_states,
        "wandb_run_id": wandb_run_id,
        "run_config": run_config,
        "extra": extra or {},
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Temp file in the SAME directory as the destination -- required
    # for the final os.replace() to be a same-filesystem rename, which
    # is what actually makes it atomic (a cross-filesystem rename would
    # raise, not silently degrade -- but same-directory sidesteps the
    # question regardless).
    fd, tmp_path = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(checkpoint, f)
            # Destination is an NFS volume -- data can sit in client-side
            # cache well after write() returns. Without an explicit
            # flush+fsync, a pod kill shortly after a "completed" save
            # could still lose data that was never actually pushed to
            # the NFS server, even though the rename logic below is
            # otherwise correct.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(destination))
    except BaseException:
        # Covers a mid-pod-restart kill as much as a real exception --
        # best-effort cleanup of the temp file; a hard kill can still
        # beat this, which is fine, since the real `path` was never
        # touched either way.
        Path(tmp_path).unlink(missing_ok=True)
        raise

    logger.info(
        "save_checkpoint: wrote %s (round_offset=%d, has_optimizer=%s, "
        "has_scheduler=%s, dp=%s, wandb_run_id=%s)",
        destination,
        round_offset,
        optimizer is not None,
        scheduler is not None,
        opacus_accountant_states is not None,
        wandb_run_id,
    )


def load_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    extra_generators: Optional[Dict[str, torch.Generator]] = None,
    expected_config: Optional[Dict[str, Any]] = None,
    map_location=None,
) -> Dict[str, Any]:
    """Load a checkpoint written by save_checkpoint(), restoring `model`
    (PEFT-only; works whether `model` is a plain peft model or an
    Opacus GradSampleModule wrapping one), `optimizer` and `scheduler`
    (if both given and both present in the checkpoint), and RNG state
    globally plus into any `extra_generators` supplied.

    Returns the checkpoint's non-restored fields for the caller to use:
    {"round_offset": int, "opacus_accountant_states": Any | None,
     "wandb_run_id": str | None, "run_config": dict | None,
     "extra": dict}.

    Raises:
        ValueError: schema_version mismatch (a v1 checkpoint is
            rejected, not force-loaded against the v2 schema); the
            saved PEFT state dict's keys/shapes don't match what the
            current model expects; `expected_config` was given and
            doesn't match the checkpoint's saved run_config; `optimizer`
            was passed but the checkpoint has no saved optimizer state
            (a silent fresh optimizer on a resume is a caller bug, not
            something to paper over); same for `scheduler`; or
            set_peft_model_state_dict() itself reports an unexpected or
            a missing adapter/head key once actually applied (see the
            ordering note below for why this one case is different).
        TypeError: `model` (after unwrapping) still doesn't look like a
            peft model.

    Ordering guarantee: every check above that CAN be done without
    touching model/optimizer/scheduler/RNG state -- schema version, the
    PEFT structural pre-check, expected_config, and the optimizer/
    scheduler PRESENCE rules -- runs first, before anything is mutated.
    A load rejected by any of those leaves model, optimizer, scheduler,
    and the global RNG completely untouched, so the caller's objects
    are exactly as they were before the call. Only once every one of
    those has passed does this function actually mutate anything, in
    this order: model, then optimizer, then scheduler, then RNG last.
    One exception to the "rejected means untouched" guarantee: the
    post-hoc check inside _apply_peft_state_dict (unexpected/missing
    keys reported by set_peft_model_state_dict() itself) can only be
    evaluated by actually calling it, so a failure there has, by
    definition, already applied some state to the model -- this is the
    one mutating step whose own correctness can't be fully confirmed
    without attempting it. Everything after it (optimizer, scheduler,
    RNG) is still guaranteed untouched in that case, since they run
    strictly later.
    """
    # weights_only=False (PyTorch's newer default is True) is required
    # here: this checkpoint deliberately carries non-tensor objects
    # (numpy/python RNG state tuples, plain dicts, the opaque Opacus
    # accountant blob) that a weights_only load would refuse. Safe in
    # this context specifically because these checkpoints are always
    # self-generated by save_checkpoint() above, never loaded from an
    # untrusted source. No mutation of model/optimizer/scheduler/RNG
    # happens from this call itself -- it only reads a file.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # --- Validation phase: nothing below this point, down to the
    # "Apply phase" comment, mutates model, optimizer, scheduler, or
    # RNG state. Every check that can run before touching any of them
    # does so here, so a raise anywhere in this phase leaves all four
    # exactly as the caller handed them in. ---

    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint at {path} has schema_version="
            f"{checkpoint.get('schema_version')!r}, but this code expects "
            f"{SCHEMA_VERSION}. A v1 checkpoint (optimizer required, no "
            f"scheduler/run_config fields) is NOT compatible with this "
            f"v2 loader -- do not force-load it; regenerate the "
            f"checkpoint with the current code, or load with the "
            f"matching code version."
        )

    unwrapped_model = _unwrap_for_peft(model)
    _validate_peft_state_dict(unwrapped_model, checkpoint["model_state_dict"])

    if expected_config is not None:
        _validate_run_config(checkpoint.get("run_config"), expected_config)

    has_saved_optimizer = checkpoint.get("optimizer_state_dict") is not None
    if optimizer is not None and not has_saved_optimizer:
        raise ValueError(
            "load_checkpoint: an optimizer was passed to restore into, "
            "but this checkpoint has no saved optimizer state. Loading "
            "it would silently leave a completely fresh optimizer in "
            "place (wrong Adam moments, wrong step count) while "
            "everything else resumes correctly -- almost certainly a "
            "caller bug (e.g. loading a round-boundary FL checkpoint, "
            "which never saves an optimizer by design, as if it were a "
            "centralized checkpoint). Investigate before proceeding."
        )
    if optimizer is None and has_saved_optimizer:
        logger.warning(
            "load_checkpoint: this checkpoint has saved optimizer state, "
            "but no optimizer was passed to restore it into -- ignoring "
            "the saved state. If the resumed run is supposed to continue "
            "this optimizer's momentum/state, this is a caller bug."
        )

    has_saved_scheduler = checkpoint.get("scheduler_state_dict") is not None
    if scheduler is not None and not has_saved_scheduler:
        raise ValueError(
            "load_checkpoint: a scheduler was passed to restore into, "
            "but this checkpoint has no saved scheduler state. Investigate "
            "before proceeding -- a silently fresh scheduler (restarting "
            "its warmup/cosine schedule from step 0) is a caller bug, not "
            "something to paper over."
        )
    if scheduler is None and has_saved_scheduler:
        logger.warning(
            "load_checkpoint: this checkpoint has saved scheduler state, "
            "but no scheduler was passed to restore it into -- ignoring "
            "the saved state. If the resumed run is supposed to continue "
            "this schedule, this is a caller bug."
        )

    # --- Apply phase: every check above has passed. Only now does
    # anything actually get mutated -- model first, then optimizer,
    # then scheduler, then RNG last, so that a failure partway through
    # this phase still leaves whatever comes AFTER the failure point
    # untouched (RNG in particular, being last, is never touched by a
    # load that fails anywhere in this phase). ---

    _apply_peft_state_dict(unwrapped_model, checkpoint["model_state_dict"])

    if optimizer is not None and has_saved_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and has_saved_scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    _restore_rng_state(checkpoint["rng_state"], extra_generators)

    logger.info(
        "load_checkpoint: restored %s (round_offset=%d, has_optimizer=%s, "
        "has_scheduler=%s, dp=%s, wandb_run_id=%s)",
        path,
        checkpoint["round_offset"],
        optimizer is not None and has_saved_optimizer,
        scheduler is not None and has_saved_scheduler,
        checkpoint["opacus_accountant_states"] is not None,
        checkpoint["wandb_run_id"],
    )

    return {
        "round_offset": checkpoint["round_offset"],
        "opacus_accountant_states": checkpoint["opacus_accountant_states"],
        "wandb_run_id": checkpoint["wandb_run_id"],
        "run_config": checkpoint.get("run_config"),
        "extra": checkpoint["extra"],
    }


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pt") -> Optional[str]:
    """Find the most recently modified checkpoint in `checkpoint_dir`
    matching `pattern`, or None if there isn't one.

    Exists for the restart-recovery path: on every pod restart (still
    roughly hourly -- Section 8.4's fix is approved but not applied
    yet), a training script needs to automatically discover whether a
    checkpoint to resume from already exists, without a human deciding
    which file that is each time.

    Sorts by filesystem modification time rather than parsing filenames
    for a round/epoch number -- simpler, and correct regardless of
    whatever naming convention a caller chooses, as long as checkpoints
    are written through save_checkpoint() above (and therefore always
    finalized via os.replace() -- a `.tmp` file mid-write never matches
    `pattern` and is never considered a candidate). Doesn't decide which
    round is "best" -- that's the caller's job; this just finds the
    newest file.
    """
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        return None

    candidates = list(directory.glob(pattern))
    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)