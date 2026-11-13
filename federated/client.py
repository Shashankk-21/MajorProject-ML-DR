"""
federated/client.py

The federated client for Experiments 2-8. Two layers:

  (A) PURE functions with no Flower types -- client_name_for, derive_seed,
      train_client_round, evaluate_client_round. They run on CPU without
      Flower or Ray, which is how they are unit-tested.
  (B) Thin Flower wrappers -- a ClientApp whose train/evaluate handlers
      only convert Message records <-> plain dicts and call (A).

Importing this module does not import ray. The Flower imports are guarded:
without flwr installed, the module still imports and (A) still works;
`app` is simply None.

Flower API names used (checked against the Flower reference docs; the
docs available while writing were 1.39, the pod runs 1.37.0 -- re-run the
Flower-side tests on the pod):
  from flwr.app import ArrayRecord, ConfigRecord, Context, Message,
                       MetricRecord, RecordDict
  from flwr.clientapp import ClientApp
  @app.train() / @app.evaluate() handlers taking (msg: Message, context:
  Context); msg.content["arrays"] (ArrayRecord, .to_torch_state_dict()) and
  msg.content["config"] (ConfigRecord) are the default keys the built-in
  strategies use; context.node_config["partition-id"]; replies are
  Message(content=RecordDict(...), reply_to=msg).
  Record value types: ConfigRecord takes int/float/str/bytes/bool and lists
  of them; MetricRecord takes ONLY int/float and lists of int/float (no
  str, no bool). So: the accountant state travels as a JSON string in a
  ConfigRecord, and the 25-integer confusion matrix travels as a list of
  ints in the MetricRecord.

HARD RULES this file implements (each was a bug found in validation):
  1. Every Opacus make_private uses cfg grad-sample-mode, and config
     validation REJECTS anything but "functorch" on the DP path (the
     default "hooks" mode gives silently wrong qkv-LoRA gradients). The
     standing per-sample-gradient gate (utils.dp_checks) still runs once
     per process before the first DP training and verifies that mode.
  2. PrivacyEngine(accountant=<calibration["accountant"]>) -- never
     defaulted -- and engine.accountant.mechanism() must equal it.
  3. The DP loader is built at the LOGICAL batch (cfg dp-logical-batch-size,
     which must equal effective-batch-size AND the locked
     LOGICAL_BATCH_SIZE = 64 defined in this file); check_loader_matches_calibration runs on the
     real wrapped loader every round.
  4. Order per round: factory model -> strict global-state load ->
     freeze_bn_eval -> optimizer AFTER peft -> make_private LAST.
  5. The optimizer is built fresh every round, never persisted. Two AdamW
     groups (LoRA lr-lora, head lr-head) sent by the server each round;
     no scheduler (FL's LR is round-level, set by the server).
  6. No Mixup, no WeightedRandomSampler. Class weights come from THIS
     client's own train labels only.
  7. Weights move as get/set_peft_model_state_dict on the original
     (unwrapped) peft model, with the key-set + shape comparison before
     loading and the missing/unexpected-key inspection after (same rules as
     checkpoint.py).
  8. The FedProx snapshot is taken after the global state is loaded and
     before any local step. fedprox without a positive proximal-mu, or
     fedavg WITH a proximal-mu, raises -- a silent FedProx -> FedAvg
     downgrade is impossible.
  9. Per-round determinism and diversity: python/numpy/torch are seeded
     from derive_seed(run_seed, true_round, partition_id), and the loader
     seed is that same value.
 10. GPU memory is released after every call (the 4 simulated clients run
     sequentially in one process on a 16 GiB slice).
 11. A client only ever reads its OWN manifest
     ({manifest-root}/{client_name}_manifest.csv); partition 0 APTOS,
     1 IDRiD, 2 Messidor-2, 3 EyePACS, anything else raises.

CONFIG CONTRACT (the ConfigRecord the client receives, as a dict)
  Static, always: experiment-id (str), algorithm ("fedavg"|"fedprox"),
    dp (bool), run-seed (int), num-rounds (int), lora-rank, lora-alpha,
    img-size (224), num-classes (5), pretrained (bool, optional, default
    True), local-epochs, physical-batch-size, accumulation-steps,
    effective-batch-size, label-smoothing, weight-decay, num-workers,
    manifest-root.
  Static, DP only: calibration-path, max-grad-norm, delta,
    grad-sample-mode (must be "functorch"), dp-logical-batch-size (must
    equal effective-batch-size and the locked LOGICAL_BATCH_SIZE = 64
    defined in this file), AND accountant (str) and
    target-epsilon (float). The last two are not in the original contract
    list, but the DP checklist requires validating the calibration file's
    accountant and target epsilon against "the experiment's" -- they have
    to come from somewhere, and a calibration file for the wrong epsilon
    would otherwise be used silently. server.py reads both from the
    experiment YAML.
  Per round: round (TRUE round, 1 <= round <= num-rounds), lr-lora, lr-head, proximal-mu (fedprox
    only), accountant-states (DP only; JSON string {"0": state, ...};
    absent/empty on the first round).
  Evaluate needs only: lora-rank, lora-alpha, img-size, num-classes,
    manifest-root, num-workers, run-seed (+ optional pretrained, round).

TRAIN REPLY: ArrayRecord "arrays" (peft state dict, CPU float32);
  MetricRecord "metrics" with num-examples, partition-id, train-loss,
  local-epochs (+ DP: epsilon, noise-multiplier, sample-rate,
  steps-this-round, steps-total); DP also ConfigRecord "accountant" holding
  {"accountant": <JSON string>, "partition-id": int}.
EVALUATE REPLY: MetricRecord "metrics" with num-examples, partition-id,
  val-loss, val-macro-f1, recall-class-0..4, confusion-matrix (25 ints,
  row-major, rows = true class, columns = predicted class).

EARLY DP BUDGET CHECK (before the gradient gate, the model, the loader or
any training): steps_before is the sum of the client's stored accountant
history steps (0 only when no state exists); planned = local-epochs *
num_batches. PrivacyBudgetError if steps_before + planned exceeds the
calibrated steps, and PrivacyBudgetError unless steps_before is exactly
(round - 1) * local-epochs * num_batches (catches a replayed round, a lost
state or a stale state that the overflow check alone would let through);
and every prior history entry must equal this client's calibrated noise
multiplier and sample rate (math.isclose, rel_tol 1e-9), which catches a
swapped client state or a changed calibration file whose step total still
looks right. A malformed state raises ValueError and is never read as zero
steps. On TRUE round 1 ANY supplied accountant-state mapping (non-empty,
whether or not it holds this client's key, whether or not its history is
empty) raises PrivacyBudgetError after the structure check: only None or an
empty mapping is a fresh first round. The post-training budget checks remain
as a backstop.

NOT HERE: checkpoints (server.py), wandb, calibration computation,
optimizer persistence, schedulers, server logic.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import numbers
import random
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from sklearn.metrics import confusion_matrix
from torch.optim import AdamW

from models.coatnet_lora import (
    NUM_DR_GRADES,
    create_coatnet_lora_model,
    freeze_bn_eval,
    split_trainable_parameters,
)
from preprocessing.dataset import compute_class_weights, get_dataloaders

# 64 is locked by the training-phase decisions (Section 4). This local constant
# is a deliberate drift guard, not a second source of truth to edit: the
# experiment YAML key dp_logical_batch_size (client config key
# dp-logical-batch-size) must match it, and _validate_cfg rejects any other
# value. Changing it requires a design decision, not a config edit.
LOGICAL_BATCH_SIZE = 64
from training.local_trainer import (
    compute_selection_metric,
    run_training_epoch,
    run_validation_epoch,
)
from utils.dp_checks import ensure_per_sample_gradients_verified
from utils.privacy_calibration import check_loader_matches_calibration, load_calibration

try:  # guarded so (A) works without Flower; ray is never imported here
    from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp
except ImportError:  # pragma: no cover - exercised only where flwr is absent
    ArrayRecord = ConfigRecord = Context = Message = MetricRecord = RecordDict = None
    ClientApp = None

logger = logging.getLogger(__name__)

__all__ = [
    "PrivacyBudgetError",
    "PARTITION_CLIENTS",
    "client_name_for",
    "derive_seed",
    "train_client_round",
    "evaluate_client_round",
    "app",
]

PARTITION_CLIENTS: Dict[int, str] = {0: "APTOS", 1: "IDRiD", 2: "Messidor-2", 3: "EyePACS"}
_ALGORITHMS = ("fedavg", "fedprox")
_IMG_SIZE = 224
_EVAL_BATCH_SIZE = 64


class PrivacyBudgetError(RuntimeError):
    """A client's cumulative accountant steps exceed the steps its
    noise multiplier was calibrated for -- the reported epsilon would be
    larger than the experiment's target."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def client_name_for(partition_id: Any) -> str:
    """0 APTOS, 1 IDRiD, 2 Messidor-2, 3 EyePACS; anything else raises."""
    if not _is_int(partition_id) or int(partition_id) not in PARTITION_CLIENTS:
        raise ValueError(
            f"partition-id must be one of {sorted(PARTITION_CLIENTS)} "
            f"(0 APTOS, 1 IDRiD, 2 Messidor-2, 3 EyePACS), got {partition_id!r}"
        )
    return PARTITION_CLIENTS[int(partition_id)]


def derive_seed(run_seed: int, true_round: int, partition_id: int) -> int:
    """A deterministic seed in [0, 2**32) for one (run, round, client).

    sha256 of the three integers, not Python's hash() (which is salted per
    process) and not run_seed + round arithmetic (which collides across
    clients/rounds). Different rounds therefore shuffle differently, and a
    resumed or re-run round reproduces itself exactly.
    """
    for name, value in (("run_seed", run_seed), ("true_round", true_round), ("partition_id", partition_id)):
        if not _is_int(value):
            raise ValueError(f"{name} must be an int, got {value!r}")
    digest = hashlib.sha256(
        f"flwr-dr-client:{int(run_seed)}:{int(true_round)}:{int(partition_id)}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _manifest_path(cfg: Mapping[str, Any], client: str) -> Path:
    path = Path(cfg["manifest-root"]) / f"{client}_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{client}: manifest not found at {path}")
    return path


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

_COMMON_KEYS = {
    "lora-rank": "pos_int", "lora-alpha": "pos_int", "img-size": "int",
    "num-classes": "int", "manifest-root": "str", "num-workers": "nonneg_int",
    "run-seed": "int",
}
_TRAIN_KEYS = {
    "experiment-id": "str", "algorithm": "str", "dp": "bool",
    "num-rounds": "pos_int", "local-epochs": "pos_int",
    "physical-batch-size": "pos_int", "accumulation-steps": "pos_int",
    "effective-batch-size": "pos_int", "label-smoothing": "nonneg_num",
    "weight-decay": "nonneg_num", "round": "pos_int",
    "lr-lora": "pos_num", "lr-head": "pos_num",
}
_DP_KEYS = {
    "calibration-path": "str", "max-grad-norm": "pos_num", "delta": "prob",
    "grad-sample-mode": "str", "dp-logical-batch-size": "pos_int",
    "accountant": "str", "target-epsilon": "pos_num",
}


def _kind_ok(kind: str, value: Any) -> bool:
    if kind == "str":
        return isinstance(value, str) and value != ""
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "int":
        return _is_int(value)
    if kind == "pos_int":
        return _is_int(value) and value > 0
    if kind == "nonneg_int":
        return _is_int(value) and value >= 0
    if kind == "pos_num":
        return _is_num(value) and value > 0
    if kind == "nonneg_num":
        return _is_num(value) and value >= 0
    if kind == "prob":
        return _is_num(value) and 0 < value < 1
    raise AssertionError(f"unknown kind {kind}")  # programming error, not input


def _validate_cfg(cfg: Any, *, for_train: bool) -> None:
    """Raise ValueError listing every problem with the config."""
    if not isinstance(cfg, Mapping):
        raise ValueError(f"cfg must be a mapping, got {type(cfg).__name__}")

    problems: List[str] = []

    def check(keys: Mapping[str, str]) -> None:
        for key, kind in keys.items():
            if key not in cfg:
                problems.append(f"missing key {key!r} (expected {kind})")
            elif not _kind_ok(kind, cfg[key]):
                problems.append(f"{key!r}: expected {kind}, got {cfg[key]!r}")

    check(_COMMON_KEYS)
    if cfg.get("img-size") != _IMG_SIZE or not _is_int(cfg.get("img-size")):
        problems.append(
            f"'img-size' must be {_IMG_SIZE} (the offline cache holds 224x224 images), "
            f"got {cfg.get('img-size')!r}"
        )
    if cfg.get("num-classes") != NUM_DR_GRADES or not _is_int(cfg.get("num-classes")):
        problems.append(f"'num-classes' must be {NUM_DR_GRADES}, got {cfg.get('num-classes')!r}")
    if "pretrained" in cfg and not isinstance(cfg["pretrained"], bool):
        problems.append(f"'pretrained': expected bool, got {cfg['pretrained']!r}")

    def check_round_bounds() -> None:
        rnd, num_rounds = cfg.get("round"), cfg.get("num-rounds")
        if _is_int(rnd) and _is_int(num_rounds) and num_rounds > 0 and rnd >= 1 and rnd > num_rounds:
            problems.append(f"'round' ({rnd}) must be between 1 and 'num-rounds' ({num_rounds})")

    if not for_train:
        if "num-rounds" in cfg:  # optional for evaluation, but never invalid when present
            check({"num-rounds": "pos_int"})
        if "round" in cfg:
            if not (_is_int(cfg["round"]) and cfg["round"] >= 1):
                problems.append(f"'round': expected an int >= 1 (the TRUE round), got {cfg['round']!r}")
            else:
                check_round_bounds()  # upper bound only when num-rounds is present
    else:
        check(_TRAIN_KEYS)
        check_round_bounds()
        if _is_num(cfg.get("label-smoothing")) and not cfg["label-smoothing"] < 1:
            problems.append(f"'label-smoothing' must be < 1, got {cfg['label-smoothing']!r}")

        algorithm = cfg.get("algorithm")
        if "algorithm" in cfg and algorithm not in _ALGORITHMS:
            problems.append(f"'algorithm' must be one of {_ALGORITHMS}, got {algorithm!r}")
        if algorithm == "fedprox":
            mu = cfg.get("proximal-mu")
            if not _is_num(mu) or not mu > 0:
                problems.append(
                    f"algorithm 'fedprox' requires a positive 'proximal-mu', got {mu!r} "
                    f"(a missing mu would silently downgrade FedProx to FedAvg)"
                )
        elif algorithm == "fedavg" and "proximal-mu" in cfg:
            problems.append(
                f"algorithm 'fedavg' must NOT receive 'proximal-mu' (got {cfg['proximal-mu']!r}) "
                f"-- that would be a FedProx config mislabelled as FedAvg"
            )

        if cfg.get("dp") is True:
            check(_DP_KEYS)
            # grad-sample-mode is enforced HERE, not left to the per-process
            # gradient gate: Opacus's default "hooks" mode returns silently
            # wrong per-sample gradients for the qkv LoRA layers, and no
            # other mode has been verified for this architecture.
            mode = cfg.get("grad-sample-mode")
            if isinstance(mode, str) and mode != "functorch":
                problems.append(
                    f"DP path: 'grad-sample-mode' must be 'functorch', got {mode!r}"
                )
            effective = cfg.get("effective-batch-size")
            logical = cfg.get("dp-logical-batch-size")
            if _is_int(effective) and _is_int(logical) and effective != logical:
                problems.append(
                    f"DP path: 'effective-batch-size' ({effective}) must equal "
                    f"'dp-logical-batch-size' ({logical})"
                )
            if _is_int(logical) and logical != LOGICAL_BATCH_SIZE:
                problems.append(
                    f"DP path: 'dp-logical-batch-size' ({logical}) must equal the "
                    f"locked LOGICAL_BATCH_SIZE ({LOGICAL_BATCH_SIZE}) defined in client.py"
                )
        elif cfg.get("dp") is False:
            phys, acc, eff = (cfg.get(k) for k in ("physical-batch-size", "accumulation-steps", "effective-batch-size"))
            if all(_is_int(v) and v > 0 for v in (phys, acc, eff)) and phys * acc != eff:
                problems.append(
                    f"non-DP path: effective-batch-size ({eff}) must equal "
                    f"physical-batch-size * accumulation-steps ({phys} * {acc} = {phys * acc})"
                )

    if problems:
        raise ValueError("invalid client config:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# Model, weights, metrics, accountant helpers
# ---------------------------------------------------------------------------

def _build_model(cfg: Mapping[str, Any]) -> torch.nn.Module:
    return create_coatnet_lora_model(
        lora_rank=cfg["lora-rank"],
        lora_alpha=cfg["lora-alpha"],
        num_classes=cfg["num-classes"],
        img_size=cfg["img-size"],
        pretrained=cfg.get("pretrained", True),
    )


def _load_global_state(model: torch.nn.Module, global_state: Mapping[str, Any]) -> None:
    """Strict load of the received peft state dict into the ORIGINAL
    (unwrapped) peft model: compare key set and tensor shapes first, raise
    on any unexpected key and on any missing lora_/modules_to_save key
    afterwards (every other "missing" key is the frozen backbone, which a
    PEFT-only state dict never contains -- expected, not an error).
    """
    state = {k: torch.as_tensor(v) for k, v in global_state.items()}
    expected = get_peft_model_state_dict(model)

    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatched = sorted(
        k for k in set(expected) & set(state)
        if tuple(expected[k].shape) != tuple(state[k].shape)
    )
    if missing or unexpected or shape_mismatched:
        raise ValueError(
            "global state does not match the model's PEFT state dict "
            "(different LoRA rank/targets/head?).\n"
            f"Missing: {missing}\nUnexpected: {unexpected}\nShape-mismatched: {shape_mismatched}"
        )

    result = set_peft_model_state_dict(model, state)
    if result is not None:
        bad_unexpected = list(getattr(result, "unexpected_keys", []) or [])
        bad_missing = [
            k for k in (getattr(result, "missing_keys", []) or [])
            if "lora_" in k or "modules_to_save" in k
        ]
        if bad_unexpected or bad_missing:
            raise ValueError(
                f"set_peft_model_state_dict reported unexpected keys {bad_unexpected} "
                f"and missing adapter/head keys {bad_missing}"
            )


def _extract_state(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """The peft state dict of the ORIGINAL model as detached CPU float32
    clones (they must outlive the model, which is freed after the call)."""
    state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in get_peft_model_state_dict(model).items():
        state[key] = value.detach().cpu().to(torch.float32).clone()
    non_finite = [k for k, v in state.items() if not bool(torch.isfinite(v).all())]
    if non_finite:
        raise RuntimeError(
            f"trained weights contain NaN/inf in {non_finite[:5]} -- refusing to "
            f"send them to the server (they would poison the global model)"
        )
    return state


def _snapshot_trainable(module: torch.nn.Module) -> List[torch.Tensor]:
    """FedProx's round-start snapshot, taken from the module that is handed
    to run_training_epoch, in its parameter order."""
    return [p.detach().clone() for p in module.parameters() if p.requires_grad]


def _clean_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Plain Python int/float (or lists of them) only -- what MetricRecord
    accepts. bool, NaN and inf are rejected."""
    clean: Dict[str, Any] = {}

    def scalar(key: str, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (numbers.Integral, numbers.Real)):
            raise ValueError(f"metric {key!r} must be an int or float, got {value!r}")
        if _is_int(value):
            return int(value)
        if not math.isfinite(float(value)):
            raise ValueError(f"metric {key!r} is not finite: {value!r}")
        return float(value)

    for key, value in metrics.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            clean[key] = [scalar(key, v) for v in value]
        else:
            clean[key] = scalar(key, value)
    return clean


def _parse_accountant_states(raw: Any) -> Optional[Dict[str, Any]]:
    """The per-round JSON string {"0": state, ...} -> dict; absent or empty
    -> None."""
    if raw is None or raw == "":
        return None
    parsed = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    if not isinstance(parsed, dict):
        raise ValueError(f"accountant-states must be a JSON object, got {type(parsed).__name__}")
    return parsed or None


def _own_accountant_state(accountant_states: Optional[Mapping[Any, Any]], partition_id: int) -> Optional[dict]:
    if not accountant_states:
        return None
    for key in (str(partition_id), partition_id):
        if key in accountant_states:
            return accountant_states[key]
    return None


def _state_for_load(state: Mapping[str, Any]) -> Dict[str, Any]:
    """JSON turns the history tuples into lists; turn them back into
    (noise_multiplier, sample_rate, num_steps) tuples before loading."""
    loaded = dict(state)
    loaded["history"] = [
        (float(h[0]), float(h[1]), int(h[2])) for h in state["history"]
    ]
    return loaded


def _json_safe(obj: Any) -> Any:
    """Round-trip through JSON so the state holds plain types only."""
    return json.loads(json.dumps(obj, default=float))


def _history_steps(history: Any) -> int:
    return int(sum(int(entry[2]) for entry in history))


def _recorded_steps(state: Any, client: str, true_round: int) -> int:
    """Total steps recorded in a client's stored accountant state, with the
    structure checked first. A malformed state raises ValueError -- it is
    never read as "zero steps", which would let a corrupted or foreign
    state pass for a fresh one.
    """
    problems: List[str] = []
    history = state.get("history") if isinstance(state, Mapping) else None
    if not isinstance(state, Mapping):
        problems.append(f"state is a {type(state).__name__}, expected a mapping")
    elif not isinstance(history, (list, tuple)):
        problems.append(f"state has no 'history' list (got {type(history).__name__})")
    else:
        for i, entry in enumerate(history):
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                problems.append(f"history[{i}] must be (noise_multiplier, sample_rate, steps), got {entry!r}")
                continue
            noise_multiplier, sample_rate, steps = entry
            if not _is_num(noise_multiplier) or not _is_num(sample_rate):
                problems.append(f"history[{i}]: noise_multiplier/sample_rate must be finite numbers, got {entry!r}")
            if not _is_int(steps) or steps < 0:
                problems.append(f"history[{i}]: steps must be an int >= 0, got {steps!r}")
    if problems:
        raise ValueError(
            f"{client}: malformed accountant state for round {true_round}:\n  " + "\n  ".join(problems)
        )
    return int(sum(int(entry[2]) for entry in history))


def _check_dp_round_progress(
    *,
    client: str,
    true_round: int,
    cfg: Mapping[str, Any],
    calibration: Mapping[str, Any],
    entry: Mapping[str, Any],
    own_state: Any,
    accountant_states: Optional[Mapping[Any, Any]] = None,
) -> int:
    """Early DP budget and round-progress validation. Runs BEFORE the
    gradient gate, the model, the DataLoader or any training work.

    Order: (0) the structure of this client's own state, if one was supplied
    (malformed -> ValueError); (0b) on TRUE round 1, ANY supplied accountant
    state (own_state present, or a non-empty accountant_states mapping with or
    without this client's key) -> PrivacyBudgetError, since only None or an
    empty collection is a fresh first round; then:

    steps_before is the sum of the stored history's step counts (0 only
    when NO state exists). Two checks, in this order:
      1. budget: steps_before + planned must not exceed the calibrated
         steps (planned = local-epochs * num_batches);
      2. exact progress: every client takes part in every round with a
         fixed number of local epochs and batches, so steps_before must be
         exactly (round - 1) * local-epochs * num_batches. The budget check
         alone cannot catch a replayed round whose steps still fit inside
         the total, nor a lost or stale state.
    Both raise PrivacyBudgetError naming the client, round, actual prior
    steps and expected prior steps. A malformed state raises ValueError.
    Returns steps_before.
    """
    steps_before = 0 if own_state is None else _recorded_steps(own_state, client, true_round)
    if true_round == 1 and (own_state is not None or accountant_states):
        keys = (
            sorted(str(k) for k in accountant_states)
            if isinstance(accountant_states, Mapping)
            else f"<{type(accountant_states).__name__}>"
        )
        raise PrivacyBudgetError(
            f"{client} round {true_round}: a first round must start with NO accountant "
            f"state, but one was supplied (keys: {keys}; this client's own state "
            f"{'present' if own_state is not None else 'absent'}). Only None or an empty "
            f"mapping is accepted for round 1; a non-empty mapping means the server is "
            f"mis-wired or the round is being replayed."
        )
    planned = cfg["local-epochs"] * entry["num_batches"]
    expected_before = (true_round - 1) * cfg["local-epochs"] * entry["num_batches"]

    if steps_before + planned > entry["steps"]:
        raise PrivacyBudgetError(
            f"{client} round {true_round}: budget overspent -- {steps_before} prior "
            f"steps + {planned} planned this round = {steps_before + planned} would "
            f"exceed the {entry['steps']} steps the noise multiplier was calibrated "
            f"for (expected prior steps for this round: {expected_before})"
        )
    if steps_before != expected_before:
        raise PrivacyBudgetError(
            f"{client} round {true_round}: accountant progress is inconsistent -- the "
            f"stored state records {steps_before} prior steps but exactly "
            f"{expected_before} are expected (= (round - 1) * local-epochs * "
            f"num_batches = {true_round - 1} * {cfg['local-epochs']} * "
            f"{entry['num_batches']}). A replayed round, a lost state or a stale "
            f"state would make the reported epsilon wrong."
        )
    if own_state is not None and own_state.get("mechanism") != calibration["accountant"]:
        raise ValueError(
            f"{client} round {true_round}: stored accountant state is for mechanism "
            f"{own_state.get('mechanism')!r} but the calibration uses "
            f"{calibration['accountant']!r}"
        )
    if own_state is not None:
        # The step total can look right while the history belongs to a
        # different client (swapped state) or to a different calibration (a
        # changed calibration file): every prior entry must have been
        # recorded at THIS client's calibrated noise multiplier and sample rate.
        mismatched = [
            (i, h[0], h[1]) for i, h in enumerate(own_state["history"])
            if not math.isclose(h[0], entry["noise_multiplier"], rel_tol=1e-9)
            or not math.isclose(h[1], entry["sample_rate"], rel_tol=1e-9)
        ]
        if mismatched:
            raise PrivacyBudgetError(
                f"{client} round {true_round}: the stored accountant history does not match "
                f"this client's calibration (noise_multiplier {entry['noise_multiplier']!r}, "
                f"sample_rate {entry['sample_rate']!r}); mismatched entries (index, "
                f"noise_multiplier, sample_rate): {mismatched}. A swapped client state or a "
                f"changed calibration file would make the reported epsilon wrong even when "
                f"the step total looks right."
            )
    return steps_before


def _train_loader(client: str, manifest: Path, batch_size: int, num_workers: int, seed: int):
    """Own-client train loader. get_dataloaders returns the (train, val,
    test) loaders; only the first is used here."""
    return get_dataloaders(
        client, manifest_path=manifest, batch_size=batch_size,
        num_workers=num_workers, seed=seed,
    )[0]


def _own_class_weights(train_loader: Any, device: torch.device) -> torch.Tensor:
    """Class weights from THIS client's own train labels only -- counts are
    never pooled across clients (strict isolation)."""
    if len(train_loader.dataset) == 0:
        raise ValueError("this client's train split is empty")
    labels = train_loader.dataset.data["label"].to_numpy()
    weights = np.asarray(compute_class_weights(labels), dtype=np.float64)
    return torch.as_tensor(weights, dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# (A) Pure functions
# ---------------------------------------------------------------------------

def train_client_round(
    *,
    partition_id: int,
    cfg: Mapping[str, Any],
    global_state: Mapping[str, torch.Tensor],
    accountant_states: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """One client's local training for one round.

    Returns {"state": peft state dict (CPU float32), "num_examples": int,
    "metrics": dict of floats/ints, "accountant_state": JSON-serialisable
    dict, or None when dp is off}.

    Raises ValueError (bad config / mismatched weights / calibration
    disagreeing with the config), PrivacyBudgetError (cumulative steps
    exceed the calibrated steps), DPGradientCheckError (the per-sample
    gradient gate failed), FileNotFoundError (own manifest missing).
    """
    client = client_name_for(partition_id)
    _validate_cfg(cfg, for_train=True)
    try:
        return _train_impl(client, int(partition_id), cfg, global_state, accountant_states)
    finally:
        _release_memory()


def _validate_calibration(cal: Mapping[str, Any], cfg: Mapping[str, Any], client: str) -> Dict[str, Any]:
    problems: List[str] = []
    if cal["accountant"] != cfg["accountant"]:
        problems.append(f"accountant: calibration {cal['accountant']!r} vs config {cfg['accountant']!r}")
    if not math.isclose(cal["target_delta"], cfg["delta"], rel_tol=1e-9):
        problems.append(f"delta: calibration {cal['target_delta']!r} vs config {cfg['delta']!r}")
    if cal["logical_batch_size"] != cfg["dp-logical-batch-size"]:
        problems.append(
            f"logical batch size: calibration {cal['logical_batch_size']} vs "
            f"config dp-logical-batch-size {cfg['dp-logical-batch-size']}"
        )
    expected_epochs = cfg["num-rounds"] * cfg["local-epochs"]
    if cal["epochs"] != expected_epochs:
        problems.append(
            f"epochs: calibration {cal['epochs']} vs num-rounds * local-epochs = {expected_epochs}"
        )
    if not math.isclose(cal["target_epsilon"], cfg["target-epsilon"], rel_tol=1e-9):
        problems.append(
            f"target epsilon: calibration {cal['target_epsilon']!r} vs config {cfg['target-epsilon']!r}"
        )
    if client not in cal["clients"]:
        problems.append(f"no calibration entry for client {client!r} (has {sorted(cal['clients'])})")
    if problems:
        raise ValueError(
            "calibration file does not match the experiment config:\n  " + "\n  ".join(problems)
        )
    return cal["clients"][client]


def _train_impl(client, partition_id, cfg, global_state, accountant_states) -> Dict[str, Any]:
    dp = cfg["dp"]
    algorithm = cfg["algorithm"]
    fedprox = algorithm == "fedprox"
    mu = float(cfg["proximal-mu"]) if fedprox else 0.0
    true_round = cfg["round"]

    seed = derive_seed(cfg["run-seed"], true_round, partition_id)
    _seed_everything(seed)
    device = _select_device()
    calibration = entry = None
    if dp:
        calibration = load_calibration(cfg["calibration-path"])
        entry = _validate_calibration(calibration, cfg, client)
        own_state = _own_accountant_state(accountant_states, partition_id)
        # Budget and exact round progress, before ANYTHING expensive (it also
        # covers a missing state at round > 1, which is lost accounting, and
        # any supplied state at round 1).
        _check_dp_round_progress(
            client=client, true_round=true_round, cfg=cfg,
            calibration=calibration, entry=entry, own_state=own_state,
            accountant_states=accountant_states,
        )

    manifest = _manifest_path(cfg, client)

    if dp:
        # The standing gate: once per process, before the first DP training.
        ensure_per_sample_gradients_verified(
            lambda: _build_model(cfg), device=device, grad_sample_mode=cfg["grad-sample-mode"],
        )

    # Wrapping order (rule 4): factory -> strict global load -> BN eval ->
    # optimizer after peft -> make_private last.
    model = _build_model(cfg)
    _load_global_state(model, global_state)
    freeze_bn_eval(model)
    model.to(device)

    batch_size = cfg["dp-logical-batch-size"] if dp else cfg["physical-batch-size"]
    train_loader = _train_loader(client, manifest, batch_size, cfg["num-workers"], seed)
    num_examples = len(train_loader.dataset)
    class_weights = _own_class_weights(train_loader, device)

    lora_params, head_params = split_trainable_parameters(model)
    optimizer = AdamW(
        [
            {"params": lora_params, "lr": cfg["lr-lora"]},
            {"params": head_params, "lr": cfg["lr-head"]},
        ],
        weight_decay=cfg["weight-decay"],
    )

    metrics: Dict[str, Any] = {}
    accountant_state = None
    epoch_losses: List[float] = []
    common = dict(class_weights=class_weights, label_smoothing=cfg["label-smoothing"], proximal_mu=mu)

    if not dp:
        snapshot = _snapshot_trainable(model) if fedprox else None
        for _ in range(cfg["local-epochs"]):
            epoch_losses.append(run_training_epoch(
                model, train_loader, optimizer, device,
                accumulation_steps=cfg["accumulation-steps"], global_params=snapshot, **common,
            ))
    else:
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Secure RNG turned off")
            engine = PrivacyEngine(accountant=calibration["accountant"])
        if engine.accountant.mechanism() != calibration["accountant"]:
            raise RuntimeError(
                f"PrivacyEngine accountant mechanism {engine.accountant.mechanism()!r} != "
                f"calibration accountant {calibration['accountant']!r}"
            )
        if own_state is not None:
            engine.accountant.load_state_dict(_state_for_load(own_state))
        steps_before = _history_steps(engine.accountant.history)

        dp_model = None
        try:
            dp_model, dp_optimizer, dp_loader = engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=entry["noise_multiplier"],
                max_grad_norm=cfg["max-grad-norm"],
                poisson_sampling=True,
                grad_sample_mode=cfg["grad-sample-mode"],
            )
            check_loader_matches_calibration(dp_loader, entry)
            snapshot = _snapshot_trainable(dp_model) if fedprox else None
            with BatchMemoryManager(
                data_loader=dp_loader,
                max_physical_batch_size=cfg["physical-batch-size"],
                optimizer=dp_optimizer,
            ) as memory_safe_loader:
                for _ in range(cfg["local-epochs"]):
                    epoch_losses.append(run_training_epoch(
                        dp_model, memory_safe_loader, dp_optimizer, device,
                        accumulation_steps=1, global_params=snapshot, **common,
                    ))
        finally:
            remove_hooks = getattr(dp_model, "remove_hooks", None)
            if callable(remove_hooks):
                try:
                    remove_hooks()
                except Exception:  # cleanup only
                    logger.debug("remove_hooks() failed", exc_info=True)

        history = engine.accountant.history
        steps_total = _history_steps(history)
        steps_this_round = steps_total - steps_before
        if steps_this_round <= 0:
            raise RuntimeError(
                "the accountant recorded no steps this round -- training would be "
                "unaccounted; refusing to report an epsilon"
            )
        if steps_total > entry["steps"]:
            raise PrivacyBudgetError(
                f"{client}: cumulative accountant steps {steps_total} exceed the "
                f"{entry['steps']} steps the noise multiplier was calibrated for "
                f"(target epsilon {cfg['target-epsilon']})"
            )
        expected_this_round = cfg["local-epochs"] * entry["num_batches"]
        if steps_this_round != expected_this_round:
            logger.warning(
                "%s round %d: accountant recorded %d steps, expected local-epochs * "
                "num_batches = %d", client, true_round, steps_this_round, expected_this_round,
            )
        if true_round == cfg["num-rounds"] and steps_total != entry["steps"]:
            logger.warning(
                "%s: final round ends with %d accountant steps but calibration assumed %d",
                client, steps_total, entry["steps"],
            )

        epsilon = float(engine.accountant.get_epsilon(delta=cfg["delta"]))
        accountant_state = _json_safe(engine.accountant.state_dict())
        metrics.update({
            "epsilon": epsilon,
            "noise-multiplier": entry["noise_multiplier"],
            "sample-rate": entry["sample_rate"],
            "steps-this-round": steps_this_round,
            "steps-total": steps_total,
        })

    train_loss = float(sum(epoch_losses) / len(epoch_losses))
    if not math.isfinite(train_loss):
        raise RuntimeError(f"{client}: non-finite training loss ({train_loss}) in round {true_round}")
    metrics.update({"train-loss": train_loss, "local-epochs": cfg["local-epochs"]})

    # State is read from the ORIGINAL peft model: the wrapped module shares
    # its parameters, so the optimizer's updates are visible here.
    state = _extract_state(model)
    return {
        "state": state,
        "num_examples": int(num_examples),
        "metrics": _clean_metrics(metrics),
        "accountant_state": accountant_state,
    }


def evaluate_client_round(
    *,
    partition_id: int,
    cfg: Mapping[str, Any],
    global_state: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """Evaluate the received GLOBAL weights on this client's own val split:
    eval mode, no DP wrapper, batch 64, no shuffle. Returns
    {"num_examples": int, "metrics": {...}} (keys in the module docstring).
    """
    client = client_name_for(partition_id)
    _validate_cfg(cfg, for_train=False)
    try:
        return _evaluate_impl(client, int(partition_id), cfg, global_state)
    finally:
        _release_memory()


def _evaluate_impl(client, partition_id, cfg, global_state) -> Dict[str, Any]:
    seed = derive_seed(cfg["run-seed"], cfg.get("round", 0), partition_id)
    _seed_everything(seed)
    device = _select_device()
    manifest = _manifest_path(cfg, client)

    model = _build_model(cfg)
    _load_global_state(model, global_state)
    freeze_bn_eval(model)
    model.to(device)

    val_loader = get_dataloaders(
        client, manifest_path=manifest, batch_size=_EVAL_BATCH_SIZE,
        num_workers=cfg["num-workers"], seed=seed,
    )[1]
    num_examples = len(val_loader.dataset)
    if num_examples == 0:
        raise ValueError(f"{client}: this client's val split is empty")

    val_loss, logits, targets = run_validation_epoch(model, val_loader, device)
    predictions = logits.argmax(axis=1)
    labels = np.arange(NUM_DR_GRADES)
    cm = confusion_matrix(targets, predictions, labels=labels)
    if int(cm.sum()) != num_examples:
        raise RuntimeError(
            f"{client}: confusion matrix covers {int(cm.sum())} samples but the val "
            f"split has {num_examples}"
        )

    support = cm.sum(axis=1)
    metrics: Dict[str, Any] = {
        "partition-id": partition_id,
        "val-loss": float(val_loss),
        "val-macro-f1": float(compute_selection_metric(logits, targets)),
        # recall of a class with no val examples is reported as 0.0; the
        # server computes pooled recall from the confusion matrices instead.
        **{
            f"recall-class-{c}": (float(cm[c, c]) / float(support[c]) if support[c] > 0 else 0.0)
            for c in range(NUM_DR_GRADES)
        },
        "confusion-matrix": [int(v) for v in cm.reshape(-1)],
    }
    return {"num_examples": int(num_examples), "metrics": _clean_metrics(metrics)}


# ---------------------------------------------------------------------------
# (B) Flower wrappers
# ---------------------------------------------------------------------------

def _partition_id_from_context(context: Any) -> int:
    try:
        partition_id = context.node_config["partition-id"]
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError("context.node_config has no 'partition-id'") from exc
    client_name_for(partition_id)  # raises on anything outside 0-3
    return int(partition_id)


def _train_handler(msg: "Message", context: "Context") -> "Message":
    partition_id = _partition_id_from_context(context)
    cfg = dict(msg.content["config"])
    accountant_states = _parse_accountant_states(cfg.pop("accountant-states", None))
    global_state = msg.content["arrays"].to_torch_state_dict()

    result = train_client_round(
        partition_id=partition_id, cfg=cfg, global_state=global_state,
        accountant_states=accountant_states,
    )

    metrics = _clean_metrics({
        "num-examples": result["num_examples"],
        "partition-id": partition_id,
        **result["metrics"],
    })
    content = RecordDict({
        "arrays": ArrayRecord(result["state"]),
        "metrics": MetricRecord(metrics),
    })
    if result["accountant_state"] is not None:
        content["accountant"] = ConfigRecord({
            "accountant": json.dumps(result["accountant_state"], sort_keys=True),
            "partition-id": partition_id,
        })
    return Message(content=content, reply_to=msg)


def _evaluate_handler(msg: "Message", context: "Context") -> "Message":
    partition_id = _partition_id_from_context(context)
    cfg = dict(msg.content["config"])
    global_state = msg.content["arrays"].to_torch_state_dict()

    result = evaluate_client_round(partition_id=partition_id, cfg=cfg, global_state=global_state)

    metrics = _clean_metrics({"num-examples": result["num_examples"], **result["metrics"]})
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)


app = ClientApp() if ClientApp is not None else None

if app is not None:

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        return _train_handler(msg, context)

    @app.evaluate()
    def evaluate(msg: Message, context: Context) -> Message:
        return _evaluate_handler(msg, context)