"""
federated/server.py

The federated server for Experiments 2-8: the round-level LR schedule, a
FedAvg / FedProx strategy subclass, strict reply checking, pooled evaluation
metrics, per-round checkpointing with resume, DP calibration and the
ServerApp entry point.

Experiments: 2 = FedAvg, 3 = FedProx (mu 0.01), 4-8 = FedProx + DP at
epsilon 50 / 10 / 5 / 1 / 0.5. This file takes a PLAIN CONFIG DICT (no YAML
parsing, no wandb); configs/ and utils/experiment_tracker.py plug in around it.

LAUNCH MECHANISM (the decisions doc says `flwr run` and a ServerApp, but not
how the config arrives -- assumption): `flwr run` starts `app` (a ServerApp)
whose main reads the flat `context.run_config` mapping and calls
run_experiment(); the federation must have exactly 4 supernodes (client.py
maps partition-id 0..3 from the node config). For scripts and tests,
run_experiment(cfg, grid) and ExperimentRunner are callable directly.
set_round_end_callback() lets experiment_tracker.py hook the ServerApp path.

RUN HOOK (experiment tracking under `flwr run`): the ServerApp runs in its own
process and the experiment config exists only inside main(), so a tracker
cannot be started from outside. run_experiment_tracked(cfg, grid) works in
this order: (a) validate_server_config(cfg) first; an invalid config raises
ValueError before any hook runs; (b) resolve the hook: the explicit hook
argument, else set_run_hook(), else the "run-hook" config key, else the
environment variable DR_RUN_HOOK (all "package.module:function"; an empty
"run-hook" means unset); (c) call hook(validated cfg copy with defaults filled)
-> (on_round_end, wandb_run_id, finish); (d) run_experiment(cfg, ...). Hook
START failures are fatal (RunHookError before any grid call or checkpoint
write): a silently untracked run would write checkpoints without a wandb id and
break resume-safety. on_round_end failures and finish failures are logged and
swallowed. finish(summary, error) is called after the run (error is None on
success, summary is None on failure). If the hook supplies no wandb_run_id the
id stored in latest.pt is kept. The "run-hook" key never enters build_run_config.

FLOWER NAMES. Module-level Flower imports are limited to names present in
BOTH flwr 1.30.0 and 1.37.0: flwr.app (ArrayRecord, ConfigRecord, Context,
Message, MetricRecord, RecordDict), flwr.serverapp (Grid, ServerApp),
flwr.serverapp.strategy (FedAvg, FedProx). They are import-guarded so the
pure functions (round_lr, pooled_eval_metrics, parse/check reply functions,
CheckpointStore, calibration helpers) work without Flower. Facts used, from
the Flower reference and strategy source: Strategy.start(grid,
initial_arrays, num_rounds, timeout=3600, train_config, evaluate_config,
evaluate_fn); evaluate_fn is called as evaluate_fn(0, initial_arrays) before
round 1 and evaluate_fn(round, arrays) after every round; FedProx.
configure_train sets config["proximal-mu"] on the config it receives and
then calls FedAvg.configure_train, so per-round keys go into the config
BEFORE super().configure_train; FedAvg's constructor takes fraction_train,
fraction_evaluate, min_train_nodes, min_evaluate_nodes, min_available_nodes.

WHAT THE SERVER GUARANTEES THE CLIENTS
 - train/evaluate configs use exactly the keys in client.py's docstring.
 - "round" is the TRUE round (round_offset + server_round); "num-rounds" is
   always the TOTAL number of rounds, never the remaining count on a resume.
 - accountant-states: none on true round 1; from round 2 on, all four
   partitions' states from the immediately preceding round (on a resume,
   exactly the checkpoint's states).
 - the config is checked with client.py's own validator at startup, so a
   configuration the clients would reject fails here, once, not four times.
 - a fedavg experiment never sends "proximal-mu"; a fedprox experiment's
   outgoing configs are verified to contain it (the silent FedProx -> FedAvg
   downgrade the decisions doc warns about cannot happen).

WHAT THE SERVER REFUSES
 - any round where the four partitions did not ALL reply, without error,
   with distinct partition-ids (Flower's built-in strategies tolerate failed
   replies silently; averaging three clients is never acceptable here);
 - non-finite aggregated weights;
 - the default evaluate averaging, which would average the confusion-matrix
   lists: aggregate_evaluate pools the four 5x5 matrices itself.

CHECKPOINT LAYOUT (/data/checkpoints/<experiment-id>/): round_XXX.pt for
every round (the best round is picked retrospectively), best.pt when the
selection metric (pooled macro-F1) improves, latest.pt written LAST, and
done.json when all rounds are complete. Each file holds the PEFT state of a
CPU template model (built once from the factory), round_offset = the true
round just completed, the four clients' accountant states (DP only), a
run_config that a resume must match exactly, and the round's metrics. The
optimizer is never stored: every client builds a fresh one each round.

FINAL EVALUATION SOURCE OF TRUTH: final evaluation must load
round_<best-round>.pt, where best-round is the "best-round" recorded in
latest.pt's extra / done.json. best.pt is only a convenience artifact: a crash
between its write and latest.pt's, followed by a redo, can leave it holding an
attempt that was never recorded. prepare() therefore reconciles it on every
resume (rebuilt from round_<best-round>.pt when it is missing or its
round_offset differs), and the resumed run's weights and RNG state always come
from latest.pt, never from the rebuilt best checkpoint.

RESUME: if latest.pt exists it is loaded with expected_config=run_config
(any changed setting raises); the template becomes the initial arrays and
start() runs total - round_offset rounds. A restart in the middle of a round
redoes that round from the previous checkpoint. Calibration is never redone
on resume: an existing file is loaded and verified, including that each
client's calibrated dataset_size equals its manifest's current train count.

CONFIG KEYS (plain dict)
 Required: experiment-id, algorithm ("fedavg"|"fedprox"), dp (bool), run-seed,
   num-rounds, local-epochs, lora-rank, lora-alpha, img-size (224),
   num-classes (5), physical-batch-size, accumulation-steps,
   effective-batch-size, label-smoothing, weight-decay, num-workers,
   manifest-root, lr-lora-peak, lr-head-peak.
 fedprox only (required there, forbidden for fedavg): proximal-mu.
 DP only (required): calibration-path, accountant (no default), target-epsilon,
   delta, max-grad-norm (no default: the decisions doc leaves it open),
   grad-sample-mode ("functorch"), dp-logical-batch-size (64).
 Optional, defaults NOT pinned by the decisions doc (exposed as keys):
   pretrained (True), warmup-rounds (1), warmup-start-factor (0.5: round 1
   uses half the peak LR, the average of a one-round linear ramp),
   lr-min-factor (0.1: "about 10% of peak"), timeout (604800.0 s, overriding
   Flower's 3600 s default, which coincides with the pod's hourly restart),
   checkpoint-root ("/data/checkpoints"), epsilon-tolerance (0.01).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import numbers
import os
import re
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from federated import client as _client
from models.coatnet_lora import NUM_DR_GRADES, create_coatnet_lora_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.privacy_calibration import (
    calibrate_client_noise_multipliers,
    load_calibration,
    save_calibration,
)

try:  # guarded: the pure functions below work without Flower
    from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
except ImportError:  # pragma: no cover
    ArrayRecord = ConfigRecord = Context = Message = MetricRecord = RecordDict = None
try:
    from flwr.serverapp import Grid, ServerApp
except ImportError:  # pragma: no cover
    Grid = ServerApp = None
try:
    from flwr.serverapp.strategy import FedAvg, FedProx
except ImportError:  # pragma: no cover
    FedAvg = FedProx = None

logger = logging.getLogger(__name__)

__all__ = [
    "FederationError", "NUM_CLIENTS", "CLIENT_NAMES", "SELECTION_METRIC",
    "round_lr", "macro_f1_from_confusion", "recall_from_confusion", "pooled_eval_metrics",
    "validate_server_config", "static_train_config", "static_evaluate_config", "build_run_config",
    "parse_train_reply", "check_train_replies", "check_evaluate_replies",
    "load_peft_state_strict", "CheckpointStore", "ExperimentRunner", "build_strategy",
    "run_experiment", "read_resume_info", "set_round_end_callback", "app",
    "set_run_hook", "run_experiment_tracked", "RUN_HOOK_ENV", "RunHookError",
]

NUM_CLIENTS = 4
CLIENT_NAMES = tuple(_client.PARTITION_CLIENTS[i] for i in range(NUM_CLIENTS))
SELECTION_METRIC = "macro-f1-pooled"
_EXPERIMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class RunHookError(RuntimeError):
    """The experiment-tracking hook could not be loaded, raised, or returned
    something malformed. Fatal before the run starts."""


class FederationError(RuntimeError):
    """The federation misbehaved (missing, errored, duplicate or malformed
    client replies, non-finite weights, inconsistent state). Never recovered
    from silently: it aborts the experiment."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_int(v: Any) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Round-level learning-rate schedule
# ---------------------------------------------------------------------------

def round_lr(
    true_round: int,
    num_rounds: int,
    peak: float,
    *,
    warmup_rounds: int = 1,
    warmup_start_factor: float = 0.5,
    min_factor: float = 0.1,
) -> float:
    """The LR the server broadcasts for one TRUE round (Section 2).

    Rounds 1..warmup_rounds ramp linearly from peak * warmup_start_factor
    (round 1) towards peak. After the warmup, cosine decay: the first
    post-warmup round runs at the peak and the LAST round (num_rounds) at
    peak * min_factor (about 10% of peak). One value per round, the same for
    every client -- client sizes differ >200x, so a step-level schedule
    would decay inconsistently across them.
    """
    for name, v in (("true_round", true_round), ("num_rounds", num_rounds), ("warmup_rounds", warmup_rounds)):
        if not _is_int(v):
            raise ValueError(f"{name} must be an int, got {v!r}")
    if not 1 <= true_round <= num_rounds:
        raise ValueError(f"true_round must satisfy 1 <= round <= num_rounds ({num_rounds}), got {true_round}")
    if warmup_rounds < 0:
        raise ValueError(f"warmup_rounds must be >= 0, got {warmup_rounds}")
    if not _is_num(peak) or peak <= 0:
        raise ValueError(f"peak must be a finite number > 0, got {peak!r}")
    if not _is_num(warmup_start_factor) or not 0 < warmup_start_factor <= 1:
        raise ValueError(f"warmup_start_factor must be in (0, 1], got {warmup_start_factor!r}")
    if not _is_num(min_factor) or not 0 <= min_factor <= 1:
        raise ValueError(f"min_factor must be in [0, 1], got {min_factor!r}")

    if true_round <= warmup_rounds:
        factor = warmup_start_factor + (1.0 - warmup_start_factor) * (true_round - 1) / warmup_rounds
        return float(peak * factor)
    span = max(num_rounds - warmup_rounds - 1, 1)
    progress = min(max((true_round - warmup_rounds - 1) / span, 0.0), 1.0)
    return float(peak * (min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))))


# ---------------------------------------------------------------------------
# Pooled evaluation metrics from confusion matrices
# ---------------------------------------------------------------------------

def _as_confusion(values: Any, label: str) -> np.ndarray:
    flat = list(values)
    if len(flat) != NUM_DR_GRADES * NUM_DR_GRADES or not all(_is_int(v) and v >= 0 for v in flat):
        raise ValueError(
            f"{label}: confusion matrix must be {NUM_DR_GRADES * NUM_DR_GRADES} non-negative "
            f"ints (row-major, rows = true class), got {flat[:6]}..."
        )
    return np.asarray(flat, dtype=np.int64).reshape(NUM_DR_GRADES, NUM_DR_GRADES)


def recall_from_confusion(cm: np.ndarray) -> List[float]:
    """Per-class recall; a class with no true examples gets 0.0."""
    support = cm.sum(axis=1)
    return [float(cm[c, c]) / float(support[c]) if support[c] > 0 else 0.0 for c in range(NUM_DR_GRADES)]


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Macro-F1 over all 5 grades with zero_division=0 -- the same definition
    as local_trainer.compute_selection_metric (labels=0..4, zero_division=0),
    computed from a confusion matrix: F1_c = 2TP / (2TP + FP + FN)."""
    f1s = []
    for c in range(NUM_DR_GRADES):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum()) - tp
        fn = float(cm[c, :].sum()) - tp
        denom = 2.0 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(sum(f1s) / NUM_DR_GRADES)


def pooled_eval_metrics(per_client: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    """Pool the four clients' validation results.

    per_client[partition_id] = {"confusion-matrix": 25 ints, "num-examples":
    int, "val-loss": float}. Returns a flat dict of scalars/lists:
      macro-f1-pooled           (the round-selection metric; EyePACS dominates it)
      macro-f1-client-mean      (unweighted mean of the per-client macro-F1s)
      recall-pooled-class-0..4  (Mild = 1 and Moderate = 2 are the priority)
      macro-f1-client-<pid>, recall-client-<pid>-class-<c>,
      val-loss-client-<pid>, num-examples-client-<pid>
      val-loss-pooled (weighted by examples), num-examples,
      confusion-matrix-pooled (25 ints), confusion-matrix-client-<pid> (25 ints
      each, row-major, rows = true class).
    """
    if set(per_client) != set(range(NUM_CLIENTS)):
        raise ValueError(f"need results for partitions {list(range(NUM_CLIENTS))}, got {sorted(per_client)}")
    cms: Dict[int, np.ndarray] = {}
    for pid in range(NUM_CLIENTS):
        entry = per_client[pid]
        cm = _as_confusion(entry["confusion-matrix"], f"partition {pid}")
        n = entry["num-examples"]
        if not _is_int(n) or n <= 0 or int(cm.sum()) != n:
            raise ValueError(f"partition {pid}: confusion matrix sums to {int(cm.sum())} but num-examples is {n!r}")
        if not _is_num(entry["val-loss"]):
            raise ValueError(f"partition {pid}: val-loss must be finite, got {entry['val-loss']!r}")
        cms[pid] = cm

    pooled = sum(cms.values())
    total = int(pooled.sum())
    out: Dict[str, Any] = {
        "num-examples": total,
        "val-loss-pooled": float(sum(per_client[p]["num-examples"] * per_client[p]["val-loss"] for p in cms) / total),
        "macro-f1-pooled": macro_f1_from_confusion(pooled),
        "confusion-matrix-pooled": [int(v) for v in pooled.reshape(-1)],
    }
    for c, r in enumerate(recall_from_confusion(pooled)):
        out[f"recall-pooled-class-{c}"] = r
    client_f1 = []
    for pid, cm in cms.items():
        f1 = macro_f1_from_confusion(cm)
        client_f1.append(f1)
        out[f"macro-f1-client-{pid}"] = f1
        out[f"confusion-matrix-client-{pid}"] = [int(v) for v in cm.reshape(-1)]
        out[f"val-loss-client-{pid}"] = float(per_client[pid]["val-loss"])
        out[f"num-examples-client-{pid}"] = int(per_client[pid]["num-examples"])
        for c, r in enumerate(recall_from_confusion(cm)):
            out[f"recall-client-{pid}-class-{c}"] = r
    out["macro-f1-client-mean"] = float(sum(client_f1) / len(client_f1))
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REQUIRED = {
    "experiment-id": "id", "algorithm": "str", "dp": "bool", "run-seed": "int",
    "num-rounds": "pos_int", "local-epochs": "pos_int", "lora-rank": "pos_int",
    "lora-alpha": "pos_int", "img-size": "int", "num-classes": "int",
    "physical-batch-size": "pos_int", "accumulation-steps": "pos_int",
    "effective-batch-size": "pos_int", "label-smoothing": "nonneg_num",
    "weight-decay": "nonneg_num", "num-workers": "nonneg_int", "manifest-root": "str",
    "lr-lora-peak": "pos_num", "lr-head-peak": "pos_num",
}
_DEFAULTS = {
    "pretrained": True, "warmup-rounds": 1, "warmup-start-factor": 0.5,
    "lr-min-factor": 0.1, "timeout": 604800.0, "checkpoint-root": "/data/checkpoints",
    "epsilon-tolerance": 0.01,
}
_DP_REQUIRED = {
    "calibration-path": "str", "accountant": "str", "target-epsilon": "pos_num",
    "delta": "prob", "max-grad-norm": "pos_num", "grad-sample-mode": "str",
    "dp-logical-batch-size": "pos_int",
}


def _kind_ok(kind: str, v: Any) -> bool:
    if kind == "id":
        return isinstance(v, str) and _EXPERIMENT_ID_RE.fullmatch(v) is not None
    if kind == "str":
        return isinstance(v, str) and v != ""
    if kind == "bool":
        return isinstance(v, bool)
    if kind == "int":
        return _is_int(v)
    if kind == "pos_int":
        return _is_int(v) and v > 0
    if kind == "nonneg_int":
        return _is_int(v) and v >= 0
    if kind == "pos_num":
        return _is_num(v) and v > 0
    if kind == "nonneg_num":
        return _is_num(v) and v >= 0
    if kind == "prob":
        return _is_num(v) and 0 < v < 1
    raise AssertionError(kind)


def static_train_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """The static part of the train ConfigRecord (the per-round part --
    round, lr-lora, lr-head, proximal-mu, accountant-states -- is added in
    configure_train). Plain Python types only."""
    out: Dict[str, Any] = {
        "experiment-id": str(cfg["experiment-id"]), "algorithm": str(cfg["algorithm"]),
        "dp": bool(cfg["dp"]), "run-seed": int(cfg["run-seed"]), "num-rounds": int(cfg["num-rounds"]),
        "lora-rank": int(cfg["lora-rank"]), "lora-alpha": int(cfg["lora-alpha"]),
        "img-size": int(cfg["img-size"]), "num-classes": int(cfg["num-classes"]),
        "pretrained": bool(cfg.get("pretrained", True)), "local-epochs": int(cfg["local-epochs"]),
        "physical-batch-size": int(cfg["physical-batch-size"]),
        "accumulation-steps": int(cfg["accumulation-steps"]),
        "effective-batch-size": int(cfg["effective-batch-size"]),
        "label-smoothing": float(cfg["label-smoothing"]), "weight-decay": float(cfg["weight-decay"]),
        "num-workers": int(cfg["num-workers"]), "manifest-root": str(cfg["manifest-root"]),
    }
    if cfg["dp"]:
        out.update({
            "calibration-path": str(cfg["calibration-path"]), "max-grad-norm": float(cfg["max-grad-norm"]),
            "delta": float(cfg["delta"]), "grad-sample-mode": str(cfg["grad-sample-mode"]),
            "dp-logical-batch-size": int(cfg["dp-logical-batch-size"]),
            "accountant": str(cfg["accountant"]), "target-epsilon": float(cfg["target-epsilon"]),
        })
    return out


def static_evaluate_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """The evaluate ConfigRecord: only what evaluate_client_round needs."""
    return {
        "lora-rank": int(cfg["lora-rank"]), "lora-alpha": int(cfg["lora-alpha"]),
        "img-size": int(cfg["img-size"]), "num-classes": int(cfg["num-classes"]),
        "pretrained": bool(cfg.get("pretrained", True)), "manifest-root": str(cfg["manifest-root"]),
        "num-workers": int(cfg["num-workers"]), "run-seed": int(cfg["run-seed"]),
        "num-rounds": int(cfg["num-rounds"]),
    }


def validate_server_config(cfg: Any) -> Dict[str, Any]:
    """Validate the experiment config and return a copy with defaults filled.
    Raises ValueError listing every problem. After the server's own checks the
    config is also run through client.py's validator (on a representative
    round-1 config), so anything the clients would reject fails here."""
    if not isinstance(cfg, Mapping):
        raise ValueError(f"cfg must be a mapping, got {type(cfg).__name__}")
    out: Dict[str, Any] = dict(cfg)
    for key, value in _DEFAULTS.items():
        out.setdefault(key, value)

    problems: List[str] = []

    def check(keys: Mapping[str, str]) -> None:
        for key, kind in keys.items():
            if key not in out:
                problems.append(f"missing key {key!r} (expected {kind})")
            elif not _kind_ok(kind, out[key]):
                problems.append(f"{key!r}: expected {kind}, got {out[key]!r}")

    check(_REQUIRED)
    if not isinstance(out["pretrained"], bool):
        problems.append(f"'pretrained': expected bool, got {out['pretrained']!r}")
    if not _kind_ok("nonneg_int", out["warmup-rounds"]):
        problems.append(f"'warmup-rounds': expected int >= 0, got {out['warmup-rounds']!r}")
    if not (_is_num(out["warmup-start-factor"]) and 0 < out["warmup-start-factor"] <= 1):
        problems.append(f"'warmup-start-factor': expected a number in (0, 1], got {out['warmup-start-factor']!r}")
    if not (_is_num(out["lr-min-factor"]) and 0 <= out["lr-min-factor"] <= 1):
        problems.append(f"'lr-min-factor': expected a number in [0, 1], got {out['lr-min-factor']!r}")
    for key in ("timeout", "epsilon-tolerance"):
        if not _kind_ok("pos_num", out[key]):
            problems.append(f"{key!r}: expected a number > 0, got {out[key]!r}")
    if not _kind_ok("str", out["checkpoint-root"]):
        problems.append(f"'checkpoint-root': expected a non-empty str, got {out['checkpoint-root']!r}")

    if "run-hook" in out and not isinstance(out["run-hook"], str):
        problems.append(f"'run-hook': expected a str ('package.module:function', empty = unset), got {out['run-hook']!r}")

    algorithm = out.get("algorithm")
    if "algorithm" in out and algorithm not in ("fedavg", "fedprox"):
        problems.append(f"'algorithm' must be 'fedavg' or 'fedprox', got {algorithm!r}")
    if algorithm == "fedprox" and not _kind_ok("pos_num", out.get("proximal-mu")):
        problems.append(f"fedprox requires a positive 'proximal-mu', got {out.get('proximal-mu')!r}")
    if algorithm == "fedavg" and "proximal-mu" in out and not (_is_num(out["proximal-mu"]) and out["proximal-mu"] == 0):
        problems.append("fedavg takes no 'proximal-mu' (0.0 is tolerated: flwr merges the pyproject default into every run-config)")
    if out.get("dp") is True:
        check(_DP_REQUIRED)

    if problems:
        raise ValueError("invalid server config:\n  " + "\n  ".join(problems))

    probe = static_train_config(out)
    probe.update({"round": 1, "lr-lora": float(out["lr-lora-peak"]), "lr-head": float(out["lr-head-peak"])})
    if algorithm == "fedprox":
        probe["proximal-mu"] = float(out["proximal-mu"])
    try:
        _client._validate_cfg(probe, for_train=True)
        _client._validate_cfg(static_evaluate_config(out), for_train=False)
    except ValueError as exc:
        raise ValueError(f"config would be rejected by the clients:\n{exc}") from exc
    return out


def build_run_config(
    cfg: Mapping[str, Any],
    manifest_sha256: Mapping[str, str],
    noise_multipliers: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """The run_config stored in every checkpoint; a resume must match it
    exactly (checkpoint.load_checkpoint's expected_config)."""
    rc: Dict[str, Any] = {
        "experiment-id": cfg["experiment-id"], "algorithm": cfg["algorithm"], "dp": bool(cfg["dp"]),
        "run-seed": int(cfg["run-seed"]), "num-rounds": int(cfg["num-rounds"]),
        "local-epochs": int(cfg["local-epochs"]), "lora-rank": int(cfg["lora-rank"]),
        "lora-alpha": int(cfg["lora-alpha"]), "img-size": int(cfg["img-size"]),
        "num-classes": int(cfg["num-classes"]), "pretrained": bool(cfg.get("pretrained", True)),
        "physical-batch-size": int(cfg["physical-batch-size"]),
        "accumulation-steps": int(cfg["accumulation-steps"]),
        "effective-batch-size": int(cfg["effective-batch-size"]),
        "lr-lora-peak": float(cfg["lr-lora-peak"]), "lr-head-peak": float(cfg["lr-head-peak"]),
        "warmup-rounds": int(cfg.get("warmup-rounds", 1)),
        "warmup-start-factor": float(cfg.get("warmup-start-factor", 0.5)),
        "lr-min-factor": float(cfg.get("lr-min-factor", 0.1)),
        "weight-decay": float(cfg["weight-decay"]), "label-smoothing": float(cfg["label-smoothing"]),
        "manifest-sha256": dict(manifest_sha256),
    }
    if cfg["algorithm"] == "fedprox":
        rc["proximal-mu"] = float(cfg["proximal-mu"])
    if cfg["dp"]:
        if noise_multipliers is None:
            raise ValueError("DP run_config needs the per-client noise multipliers")
        rc.update({
            "accountant": str(cfg["accountant"]), "target-epsilon": float(cfg["target-epsilon"]),
            "delta": float(cfg["delta"]), "max-grad-norm": float(cfg["max-grad-norm"]),
            "grad-sample-mode": str(cfg["grad-sample-mode"]),
            "dp-logical-batch-size": int(cfg["dp-logical-batch-size"]),
            "noise-multipliers": {k: float(v) for k, v in noise_multipliers.items()},
        })
    return rc


# ---------------------------------------------------------------------------
# Client replies
# ---------------------------------------------------------------------------

def _metric(metrics: Mapping[str, Any], key: str, kind: str, *, where: str) -> Any:
    if key not in metrics:
        raise FederationError(f"{where}: reply metrics lack {key!r}")
    v = metrics[key]
    ok = {
        "int": _is_int(v), "pos_int": _is_int(v) and v > 0, "nonneg_int": _is_int(v) and v >= 0,
        "float": _is_num(v), "nonneg_float": _is_num(v) and v >= 0, "pos_float": _is_num(v) and v > 0,
    }[kind]
    if not ok:
        raise FederationError(f"{where}: reply metric {key!r} must be {kind}, got {v!r}")
    return int(v) if kind.endswith("int") else float(v)


def parse_train_reply(content: Mapping[str, Any], *, dp: bool) -> Dict[str, Any]:
    """Validate one train reply's content (a RecordDict: "arrays", "metrics",
    and for DP "accountant") and return a plain dict: partition-id,
    num-examples, train-loss, local-epochs, and for DP epsilon,
    noise-multiplier, sample-rate, steps-this-round, steps-total and the
    parsed accountant-state."""
    if "arrays" not in content or "metrics" not in content:
        raise FederationError(f"train reply lacks 'arrays'/'metrics' (has {sorted(content)})")
    metrics = dict(content["metrics"])
    pid = _metric(metrics, "partition-id", "nonneg_int", where="train reply")
    if pid >= NUM_CLIENTS:
        raise FederationError(f"train reply: partition-id {pid} outside 0..{NUM_CLIENTS - 1}")
    where = f"train reply of partition {pid}"
    info: Dict[str, Any] = {
        "partition-id": pid,
        "num-examples": _metric(metrics, "num-examples", "pos_int", where=where),
        "train-loss": _metric(metrics, "train-loss", "float", where=where),
        "local-epochs": _metric(metrics, "local-epochs", "pos_int", where=where),
    }
    if dp:
        for key in ("epsilon",):
            info[key] = _metric(metrics, key, "nonneg_float", where=where)
        for key in ("noise-multiplier", "sample-rate"):
            info[key] = _metric(metrics, key, "pos_float", where=where)
        for key in ("steps-this-round", "steps-total"):
            info[key] = _metric(metrics, key, "nonneg_int", where=where)
        if "accountant" not in content:
            raise FederationError(f"{where}: DP reply lacks the 'accountant' record")
        record = dict(content["accountant"])
        raw = record.get("accountant")
        if not isinstance(raw, str) or record.get("partition-id") != pid:
            raise FederationError(
                f"{where}: accountant record must hold a JSON string and partition-id {pid}, "
                f"got partition-id {record.get('partition-id')!r}"
            )
        try:
            state = json.loads(raw)
        except ValueError as exc:
            raise FederationError(f"{where}: accountant state is not valid JSON: {exc}") from exc
        if not isinstance(state, dict) or not isinstance(state.get("history"), list) \
                or not isinstance(state.get("mechanism"), str):
            raise FederationError(f"{where}: accountant state needs 'history' (list) and 'mechanism' (str)")
        info["accountant-state"] = state
    return info


def _error_text(msg: Any) -> str:
    try:
        return str(msg.error.reason)
    except Exception:  # noqa: BLE001
        return "unknown error"


def check_train_replies(replies: Any, *, dp: bool) -> Dict[int, Dict[str, Any]]:
    """Raise FederationError unless ALL four partitions replied without
    error, with distinct partition-ids and well-formed contents. Returns
    {partition_id: parsed reply}. Never silently averages fewer clients."""
    replies = list(replies)
    if len(replies) != NUM_CLIENTS:
        raise FederationError(f"expected {NUM_CLIENTS} train replies, got {len(replies)}")
    infos: Dict[int, Dict[str, Any]] = {}
    for msg in replies:
        if msg.has_error():
            raise FederationError(f"a client returned an error: {_error_text(msg)}")
        info = parse_train_reply(msg.content, dp=dp)
        if info["partition-id"] in infos:
            raise FederationError(f"duplicate train reply for partition {info['partition-id']}")
        infos[info["partition-id"]] = info
    if set(infos) != set(range(NUM_CLIENTS)):
        raise FederationError(f"train replies cover partitions {sorted(infos)}, need {list(range(NUM_CLIENTS))}")
    return infos


def check_evaluate_replies(replies: Any) -> Dict[int, Dict[str, Any]]:
    """Same all-four rule for evaluate replies. Returns {partition_id:
    {"num-examples", "val-loss", "confusion-matrix"}}."""
    replies = list(replies)
    if len(replies) != NUM_CLIENTS:
        raise FederationError(f"expected {NUM_CLIENTS} evaluate replies, got {len(replies)}")
    infos: Dict[int, Dict[str, Any]] = {}
    for msg in replies:
        if msg.has_error():
            raise FederationError(f"a client returned an error: {_error_text(msg)}")
        if "metrics" not in msg.content:
            raise FederationError(f"evaluate reply lacks 'metrics' (has {sorted(msg.content)})")
        metrics = dict(msg.content["metrics"])
        pid = _metric(metrics, "partition-id", "nonneg_int", where="evaluate reply")
        if pid >= NUM_CLIENTS or pid in infos:
            raise FederationError(f"evaluate reply: bad or duplicate partition-id {pid}")
        where = f"evaluate reply of partition {pid}"
        cm = metrics.get("confusion-matrix")
        if not isinstance(cm, (list, tuple)):
            raise FederationError(f"{where}: 'confusion-matrix' must be a list of 25 ints, got {type(cm).__name__}")
        infos[pid] = {
            "num-examples": _metric(metrics, "num-examples", "pos_int", where=where),
            "val-loss": _metric(metrics, "val-loss", "float", where=where),
            "confusion-matrix": [int(v) if _is_int(v) else v for v in cm],
        }
    if set(infos) != set(range(NUM_CLIENTS)):
        raise FederationError(f"evaluate replies cover partitions {sorted(infos)}, need {list(range(NUM_CLIENTS))}")
    return infos


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class _DRStrategyMixin:
    """The overrides shared by the FedAvg and FedProx variants. Cooperative:
    super() resolves to FedAvg / FedProx through the MRO."""

    def _init_dr(self, cfg: Mapping[str, Any], round_offset: int,
                 accountant_states: Optional[Mapping[str, Any]]) -> None:
        self.dr_cfg = dict(cfg)
        self.round_offset = int(round_offset)
        self.dp = bool(cfg["dp"])
        self.algorithm = cfg["algorithm"]
        self.accountant_states: Dict[str, Any] = dict(accountant_states or {})
        self.train_report: Dict[int, Dict[str, Any]] = {}
        self.eval_report: Dict[int, Dict[str, Any]] = {}
        self._sent_lrs: Dict[int, Any] = {}

    def true_round(self, server_round: int) -> int:
        return self.round_offset + int(server_round)

    # -- configure_train ---------------------------------------------------
    def configure_train(self, server_round, arrays, config, grid):
        cfg = self.dr_cfg
        true_round = self.true_round(server_round)
        # Copy first: FedProx mutates the config it receives.
        out = ConfigRecord({k: v for k, v in config.items()})
        if self.algorithm == "fedavg" and "proximal-mu" in out:
            raise ValueError("a fedavg experiment must never send 'proximal-mu'")

        out["round"] = int(true_round)
        out["lr-lora"] = round_lr(
            true_round, cfg["num-rounds"], cfg["lr-lora-peak"], warmup_rounds=cfg["warmup-rounds"],
            warmup_start_factor=cfg["warmup-start-factor"], min_factor=cfg["lr-min-factor"])
        out["lr-head"] = round_lr(
            true_round, cfg["num-rounds"], cfg["lr-head-peak"], warmup_rounds=cfg["warmup-rounds"],
            warmup_start_factor=cfg["warmup-start-factor"], min_factor=cfg["lr-min-factor"])

        if self.dp and true_round >= 2:
            missing = [str(i) for i in range(NUM_CLIENTS) if str(i) not in self.accountant_states]
            if missing:
                raise FederationError(
                    f"round {true_round} needs the accountant states of all four partitions; "
                    f"missing {missing}")
            out["accountant-states"] = json.dumps(
                {k: self.accountant_states[k] for k in sorted(self.accountant_states)}, sort_keys=True)
        else:
            if "accountant-states" in out:
                del out["accountant-states"]  # true round 1 sends none

        # Per-round keys are in; NOW the parent (FedProx adds "proximal-mu").
        messages = list(super().configure_train(server_round, arrays, out, grid))
        self._verify_outgoing(messages, true_round)
        self._sent_lrs[true_round] = (float(out["lr-lora"]), float(out["lr-head"]))
        return messages

    def _verify_outgoing(self, messages: List[Any], true_round: int) -> None:
        if len(messages) != NUM_CLIENTS:
            raise FederationError(f"configure_train produced {len(messages)} messages, need {NUM_CLIENTS}")
        for msg in messages:
            sent = msg.content[self.configrecord_key]
            if sent.get("round") != true_round:
                raise FederationError(f"outgoing config has round {sent.get('round')!r}, expected {true_round}")
            if self.algorithm == "fedprox":
                mu = sent.get("proximal-mu")
                if mu is None or not math.isclose(float(mu), float(self.dr_cfg["proximal-mu"]), rel_tol=1e-12):
                    raise FederationError(
                        "a fedprox round would be sent WITHOUT its proximal-mu -- the silent "
                        "FedProx -> FedAvg downgrade; refusing to run the round")
            elif "proximal-mu" in sent:
                raise FederationError("a fedavg round would be sent WITH proximal-mu")

    # -- aggregate_train ---------------------------------------------------
    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        true_round = self.true_round(server_round)
        infos = check_train_replies(replies, dp=self.dp)
        if true_round not in self._sent_lrs:
            raise FederationError(f"round {true_round}: no learning rates were recorded for this round's broadcast")
        replies.sort(key=lambda msg: int(dict(msg.content["metrics"])["partition-id"]))

        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is None:
            raise FederationError(f"round {true_round}: aggregation returned no arrays")
        bad = [i for i, a in enumerate(arrays.to_numpy_ndarrays()) if not np.isfinite(a).all()]
        if bad:
            raise FederationError(f"round {true_round}: aggregated arrays {bad[:5]} contain NaN/inf")

        per_client = {pid: {k: v for k, v in info.items() if k != "accountant-state"} for pid, info in infos.items()}
        if self.dp:
            # Commit the states only now that every check has passed.
            self.accountant_states = {str(pid): infos[pid]["accountant-state"] for pid in range(NUM_CLIENTS)}
        self.train_report[true_round] = {
            "aggregated": dict(metrics) if metrics is not None else {},
            "per-client": per_client,
            "lr-lora": self._sent_lrs[true_round][0], "lr-head": self._sent_lrs[true_round][1],
        }
        return arrays, metrics

    # -- aggregate_evaluate ------------------------------------------------
    def aggregate_evaluate(self, server_round, replies):
        infos = check_evaluate_replies(replies)
        flat = pooled_eval_metrics(infos)  # NOT the default averaging
        self.eval_report[self.true_round(server_round)] = flat
        return MetricRecord(flat)

    def privacy_report(self, true_round: int) -> Optional[Dict[str, Any]]:
        if not self.dp:
            return None
        per_client = self.train_report[true_round]["per-client"]
        keys = ("epsilon", "noise-multiplier", "sample-rate", "steps-this-round", "steps-total")
        report: Dict[str, Any] = {k: {pid: per_client[pid][k] for pid in sorted(per_client)} for k in keys}
        report["max-epsilon"] = max(per_client[p]["epsilon"] for p in per_client)
        report["delta"] = float(self.dr_cfg["delta"])
        report["accountant"] = str(self.dr_cfg["accountant"])
        return report


if FedAvg is not None:
    class DRFedAvg(_DRStrategyMixin, FedAvg):
        """FedAvg with the server logic of this project (Experiment 2)."""

        def __init__(self, cfg, *, round_offset=0, accountant_states=None):
            FedAvg.__init__(
                self, fraction_train=1.0, fraction_evaluate=1.0, min_train_nodes=NUM_CLIENTS,
                min_evaluate_nodes=NUM_CLIENTS, min_available_nodes=NUM_CLIENTS)
            self._init_dr(cfg, round_offset, accountant_states)

if FedProx is not None:
    class DRFedProx(_DRStrategyMixin, FedProx):
        """FedProx (mu from the config) with the server logic of this project."""

        def __init__(self, cfg, *, round_offset=0, accountant_states=None):
            FedProx.__init__(
                self, fraction_train=1.0, fraction_evaluate=1.0, min_train_nodes=NUM_CLIENTS,
                min_evaluate_nodes=NUM_CLIENTS, min_available_nodes=NUM_CLIENTS,
                proximal_mu=float(cfg["proximal-mu"]))
            self._init_dr(cfg, round_offset, accountant_states)


def build_strategy(cfg: Mapping[str, Any], *, round_offset: int = 0,
                   accountant_states: Optional[Mapping[str, Any]] = None):
    """DRFedProx for algorithm "fedprox", DRFedAvg otherwise."""
    if cfg["algorithm"] == "fedprox":
        if FedProx is None:
            raise ImportError("flwr.serverapp.strategy.FedProx is not importable")
        return DRFedProx(cfg, round_offset=round_offset, accountant_states=accountant_states)
    if FedAvg is None:
        raise ImportError("flwr.serverapp.strategy.FedAvg is not importable")
    return DRFedAvg(cfg, round_offset=round_offset, accountant_states=accountant_states)


# ---------------------------------------------------------------------------
# Template model and checkpoints
# ---------------------------------------------------------------------------

def build_template_model(cfg: Mapping[str, Any]) -> torch.nn.Module:
    """The CPU template peft model, built once from the factory. Seeded from
    run-seed (RNG state restored afterwards) so every experiment starts from
    the same initial adapters and head."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(cfg["run-seed"]) % (2 ** 32))
        return create_coatnet_lora_model(
            lora_rank=cfg["lora-rank"], lora_alpha=cfg["lora-alpha"],
            num_classes=cfg["num-classes"], img_size=cfg["img-size"],
            pretrained=cfg.get("pretrained", True))


def model_state(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """The PEFT state dict as detached CPU float32 clones."""
    state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in get_peft_model_state_dict(model).items():
        state[key] = value.detach().cpu().to(torch.float32).clone()
    return state


def load_peft_state_strict(model: torch.nn.Module, state: Mapping[str, Any]) -> None:
    """Strict load into the peft model: key set and tensor shapes compared
    first; afterwards any unexpected key, and any missing lora_/
    modules_to_save key, raises (every other "missing" key is the frozen
    backbone, which a PEFT-only state never contains). Same rules as
    checkpoint.py."""
    tensors = {k: torch.as_tensor(v) for k, v in state.items()}
    expected = get_peft_model_state_dict(model)
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    shape_mismatched = sorted(
        k for k in set(expected) & set(tensors) if tuple(expected[k].shape) != tuple(tensors[k].shape))
    if missing or unexpected or shape_mismatched:
        raise ValueError(
            "state does not match the template model's PEFT state dict.\n"
            f"Missing: {missing}\nUnexpected: {unexpected}\nShape-mismatched: {shape_mismatched}")
    result = set_peft_model_state_dict(model, tensors)
    if result is not None:
        bad_unexpected = list(getattr(result, "unexpected_keys", []) or [])
        bad_missing = [k for k in (getattr(result, "missing_keys", []) or [])
                       if "lora_" in k or "modules_to_save" in k]
        if bad_unexpected or bad_missing:
            raise ValueError(
                f"set_peft_model_state_dict reported unexpected keys {bad_unexpected} and "
                f"missing adapter/head keys {bad_missing}")


class CheckpointStore:
    """/data/checkpoints/<experiment-id>/: round_XXX.pt (every round),
    best.pt, latest.pt (written LAST), done.json. Thin wrapper over
    utils.checkpoint; the server never stores an optimizer."""

    def __init__(self, directory: Any) -> None:
        self.directory = Path(directory)

    def round_path(self, true_round: int) -> Path:
        return self.directory / f"round_{int(true_round):03d}.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.pt"

    @property
    def done_path(self) -> Path:
        return self.directory / "done.json"

    def save_round(self, *, model, true_round: int, improved: bool, run_config: Mapping[str, Any],
                   accountant_states: Optional[Mapping[str, Any]], extra: Mapping[str, Any],
                   wandb_run_id: Optional[str] = None) -> None:
        common = dict(
            model=model, optimizer=None, round_offset=int(true_round),
            opacus_accountant_states=dict(accountant_states) if accountant_states is not None else None,
            run_config=dict(run_config), wandb_run_id=wandb_run_id, extra=dict(extra))
        save_checkpoint(str(self.round_path(true_round)), **common)
        if improved:
            save_checkpoint(str(self.best_path), **common)
        save_checkpoint(str(self.latest_path), **common)  # LAST

    def load_latest(self, *, model, run_config: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.latest_path.is_file():
            return None
        return load_checkpoint(
            str(self.latest_path), model=model, optimizer=None, expected_config=dict(run_config))

    def write_done(self, summary: Mapping[str, Any]) -> None:
        _atomic_write_json(self.done_path, summary)

    def read_done(self) -> Optional[Dict[str, Any]]:
        if not self.done_path.is_file():
            return None
        with open(self.done_path, "r") as f:
            return json.load(f)


def read_resume_info(cfg: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """For experiment_tracker.py: what latest.pt says (round_offset,
    wandb_run_id, extra) WITHOUT building a model, so wandb can be resumed
    (wandb.init(id=..., resume="must")) before the experiment starts."""
    path = CheckpointStore(Path(cfg.get("checkpoint-root", _DEFAULTS["checkpoint-root"])) / cfg["experiment-id"]).latest_path
    if not path.is_file():
        return None
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    return {"round_offset": ckpt["round_offset"], "wandb_run_id": ckpt.get("wandb_run_id"),
            "extra": ckpt.get("extra", {})}


# ---------------------------------------------------------------------------
# Manifests and calibration
# ---------------------------------------------------------------------------

def manifest_path(cfg: Mapping[str, Any], client_name: str) -> Path:
    """A client's manifest: {manifest-root}/{client_name}_manifest.csv."""
    return Path(cfg["manifest-root"]) / f"{client_name}_manifest.csv"


def count_train_rows(path: Path) -> int:
    return int((pd.read_csv(path, usecols=["split"])["split"] == "train").sum())


def verify_calibration(payload: Mapping[str, Any], cfg: Mapping[str, Any], train_sizes: Mapping[str, int]) -> None:
    """The loaded calibration must match the experiment: accountant, target
    epsilon, delta, logical batch size, epochs, and each client's
    dataset_size equal to the manifest's CURRENT train count."""
    problems: List[str] = []
    epochs = cfg["num-rounds"] * cfg["local-epochs"]
    if payload["accountant"] != cfg["accountant"]:
        problems.append(f"accountant: file {payload['accountant']!r} vs config {cfg['accountant']!r}")
    if not math.isclose(payload["target_epsilon"], cfg["target-epsilon"], rel_tol=1e-9):
        problems.append(f"target epsilon: file {payload['target_epsilon']!r} vs config {cfg['target-epsilon']!r}")
    if not math.isclose(payload["target_delta"], cfg["delta"], rel_tol=1e-9):
        problems.append(f"delta: file {payload['target_delta']!r} vs config {cfg['delta']!r}")
    if payload["logical_batch_size"] != cfg["dp-logical-batch-size"]:
        problems.append(f"logical batch size: file {payload['logical_batch_size']} vs config {cfg['dp-logical-batch-size']}")
    if payload["epochs"] != epochs:
        problems.append(f"epochs: file {payload['epochs']} vs num-rounds * local-epochs = {epochs}")
    for name in CLIENT_NAMES:
        entry = payload["clients"].get(name)
        if entry is None:
            problems.append(f"no calibration entry for {name}")
        elif entry["dataset_size"] != train_sizes[name]:
            problems.append(
                f"{name}: calibrated dataset_size {entry['dataset_size']} vs the manifest's current "
                f"train count {train_sizes[name]}")
    if problems:
        raise ValueError("calibration file does not match the experiment:\n  " + "\n  ".join(problems))


def ensure_calibration(cfg: Mapping[str, Any], train_sizes: Mapping[str, int]) -> Dict[str, Any]:
    """Compute the calibration ONCE if its file does not exist (from the
    clients' TRAIN-split sizes); in every case load it and verify it. An
    existing file is never recomputed -- a resume must reuse the noise
    multipliers its accountants have been spending against."""
    path = Path(cfg["calibration-path"])
    epochs = cfg["num-rounds"] * cfg["local-epochs"]
    if not path.is_file():
        logger.info("calibrating %s (target epsilon %s) -> %s", cfg["experiment-id"], cfg["target-epsilon"], path)
        clients = calibrate_client_noise_multipliers(
            target_epsilon=cfg["target-epsilon"], client_train_sizes=dict(train_sizes),
            accountant=cfg["accountant"], logical_batch_size=cfg["dp-logical-batch-size"],
            epochs=epochs, target_delta=cfg["delta"], epsilon_tolerance=cfg["epsilon-tolerance"])
        save_calibration(
            str(path), target_epsilon=cfg["target-epsilon"], clients=clients, accountant=cfg["accountant"],
            logical_batch_size=cfg["dp-logical-batch-size"], epochs=epochs, target_delta=cfg["delta"],
            epsilon_tolerance=cfg["epsilon-tolerance"],
            metadata={"experiment-id": cfg["experiment-id"], "train-sizes": dict(train_sizes)})
    payload = load_calibration(str(path))
    verify_calibration(payload, cfg, train_sizes)
    return payload


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """One experiment: prepare() validates, calibrates, builds the template
    and resumes; run(grid) drives strategy.start(); evaluate_fn is the
    end-of-round checkpoint hook. On resume prepare() also reconciles best.pt
    (see _reconcile_best_checkpoint); final evaluation must use
    round_<best-round>.pt as its source of truth, not best.pt."""

    def __init__(self, cfg: Mapping[str, Any], *,
                 on_round_end: Optional[Callable[[int, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]], None]] = None,
                 wandb_run_id: Optional[str] = None) -> None:
        self.raw_cfg = cfg
        self.on_round_end = on_round_end
        self.wandb_run_id = wandb_run_id
        self.cfg: Dict[str, Any] = {}
        self.store: Optional[CheckpointStore] = None
        self.template = None
        self.strategy = None
        self.run_config: Dict[str, Any] = {}
        self.round_offset = 0
        self.best_selection_metric = float("-inf")
        self.best_round = 0
        self.last_completed_round = 0

    # -- preparation -------------------------------------------------------
    def prepare(self) -> Dict[str, Any]:
        cfg = self.cfg = validate_server_config(self.raw_cfg)
        self.store = CheckpointStore(Path(cfg["checkpoint-root"]) / cfg["experiment-id"])
        self.store.directory.mkdir(parents=True, exist_ok=True)

        sha256, train_sizes = {}, {}
        for name in CLIENT_NAMES:
            path = manifest_path(cfg, name)
            if not path.is_file():
                raise FileNotFoundError(f"{name}: manifest not found at {path}")
            sha256[name] = _sha256_of_file(path)
            if cfg["dp"]:
                train_sizes[name] = count_train_rows(path)

        noise = None
        if cfg["dp"]:
            payload = ensure_calibration(cfg, train_sizes)
            noise = {n: payload["clients"][n]["noise_multiplier"] for n in CLIENT_NAMES}
        self.run_config = build_run_config(cfg, sha256, noise)
        self.template = build_template_model(cfg)

        accountant_states = None
        latest = self.store.load_latest(model=self.template, run_config=self.run_config)
        if latest is not None:
            self.round_offset = int(latest["round_offset"])
            if not 0 < self.round_offset <= cfg["num-rounds"]:
                raise ValueError(f"checkpoint round_offset {self.round_offset} outside 1..{cfg['num-rounds']}")
            extra = latest["extra"]
            self.best_selection_metric = float(extra.get("best-selection-metric", float("-inf")))
            self.best_round = int(extra.get("best-round", 0))
            self.last_completed_round = self.round_offset
            if self.wandb_run_id is None:
                self.wandb_run_id = latest.get("wandb_run_id")
            if cfg["dp"]:
                accountant_states = latest["opacus_accountant_states"]
                if not isinstance(accountant_states, dict) or set(accountant_states) != {str(i) for i in range(NUM_CLIENTS)}:
                    raise ValueError("checkpoint has no accountant state for all four partitions")
            self._reconcile_best_checkpoint()
        elif self.store.read_done() is not None:
            raise RuntimeError(f"{self.store.done_path} exists but latest.pt does not")

        remaining = cfg["num-rounds"] - self.round_offset
        if remaining > 0:
            self.strategy = build_strategy(cfg, round_offset=self.round_offset, accountant_states=accountant_states)
        logger.info("experiment %s: round_offset=%d, %d round(s) remaining", cfg["experiment-id"], self.round_offset, remaining)
        return {"done": remaining == 0, "round_offset": self.round_offset, "remaining": remaining}

    def _reconcile_best_checkpoint(self) -> None:
        """Make best.pt agree with latest.pt's recorded best round (resume only).

        Called by prepare() after latest.pt was loaded into the template.
        1. latest.pt's best-round must satisfy 1 <= best-round <= round_offset
           (a completed round always has a best round), else FederationError.
        2. If best.pt exists and its round_offset equals best-round, nothing
           is changed. best.pt is inspected with a plain torch.load, which
           touches neither the template nor any RNG state.
        3. Otherwise (missing, or a different round_offset) it is rebuilt from
           round_<best-round>.pt, loaded and validated through load_checkpoint
           with the current run_config, and written through save_checkpoint
           (atomic). A missing source file, or a source whose recorded
           round_offset differs from best-round, raises FederationError.
        4. Rebuilding loads the historical weights (and RNG state) into the
           template, so latest.pt is loaded again afterwards. The template and
           RNG state from which the next round resumes are then exactly those
           of latest.pt, and this is verified before returning.
        """
        best_round, offset = self.best_round, self.round_offset
        if not 1 <= best_round <= offset:
            raise FederationError(
                f"{self.cfg['experiment-id']}: {self.store.latest_path} (round_offset={offset}) records "
                f"best-round={best_round}; a completed round needs a best round in 1..{offset}")

        best_path = self.store.best_path
        if best_path.is_file():
            try:
                existing = torch.load(str(best_path), map_location="cpu", weights_only=False)
            except Exception as exc:
                raise FederationError(
                    f"{best_path} is unreadable ({type(exc).__name__}: {exc}); delete it to have it "
                    f"rebuilt from {self.store.round_path(best_round).name} on the next resume") from exc
            recorded = existing.get("round_offset") if isinstance(existing, Mapping) else None
            if _is_int(recorded) and recorded == best_round:
                return
            reason = f"holds round_offset={recorded!r}, latest.pt records best-round={best_round}"
        else:
            reason = "is missing"

        source_path = self.store.round_path(best_round)
        if not source_path.is_file():
            raise FederationError(
                f"{best_path} {reason}, and the source {source_path} needed to rebuild it does not exist")

        # Snapshot the resume state (latest.pt, already loaded) to verify it is restored below.
        weights_before = model_state(self.template)
        rng_before = torch.get_rng_state()

        source = load_checkpoint(
            str(source_path), model=self.template, optimizer=None, expected_config=dict(self.run_config))
        if source["round_offset"] != best_round:
            raise FederationError(
                f"{source_path} records round_offset={source['round_offset']!r}, expected {best_round}")
        # The template now holds the best round's weights (load_checkpoint validated them).
        save_checkpoint(
            str(best_path), model=self.template, optimizer=None, round_offset=best_round,
            opacus_accountant_states=source.get("opacus_accountant_states"),
            run_config=dict(self.run_config), wandb_run_id=source.get("wandb_run_id"),
            extra=dict(source["extra"]))
        logger.warning("%s %s: rebuilt it from %s", best_path, reason, source_path.name)

        # CRITICAL: resume from latest.pt, not from the historical best round.
        restored = self.store.load_latest(model=self.template, run_config=self.run_config)
        weights_after = model_state(self.template)
        if (restored is None or int(restored["round_offset"]) != offset
                or set(weights_after) != set(weights_before)
                or any(not torch.equal(weights_after[k], weights_before[k]) for k in weights_before)
                or not torch.equal(rng_before, torch.get_rng_state())):
            raise FederationError(
                f"{self.cfg['experiment-id']}: restoring latest.pt after rebuilding best.pt did not "
                f"reproduce the resume weights/RNG state; refusing to continue")

    # -- checkpoint hook (Strategy.start's evaluate_fn) --------------------
    def evaluate_fn(self, server_round: int, arrays: Any) -> None:
        if server_round == 0:
            return None  # called once with the initial arrays; never checkpoint it
        true_round = self.round_offset + int(server_round)
        eval_metrics = self.strategy.eval_report.get(true_round)
        train_metrics = self.strategy.train_report.get(true_round)
        if eval_metrics is None or train_metrics is None:
            raise FederationError(f"round {true_round}: aggregate_train/aggregate_evaluate results are missing")

        load_peft_state_strict(self.template, arrays.to_torch_state_dict())
        selection = float(eval_metrics[SELECTION_METRIC])
        improved = selection > self.best_selection_metric
        if improved:
            self.best_selection_metric, self.best_round = selection, true_round
        privacy = self.strategy.privacy_report(true_round)
        extra = {
            "round": true_round, "train": train_metrics, "eval": eval_metrics, "privacy": privacy,
            "selection-metric-name": SELECTION_METRIC, "selection-metric": selection,
            "best-selection-metric": self.best_selection_metric, "best-round": self.best_round,
        }
        self.store.save_round(
            model=self.template, true_round=true_round, improved=improved, run_config=self.run_config,
            accountant_states=self.strategy.accountant_states if self.cfg["dp"] else None,
            extra=extra, wandb_run_id=self.wandb_run_id)
        self.last_completed_round = true_round
        if self.on_round_end is not None:
            try:
                self.on_round_end(true_round, train_metrics, eval_metrics, privacy)
            except Exception:  # a tracker failure must not kill an expensive run
                logger.exception("on_round_end callback failed at round %d; continuing", true_round)
        return None

    # -- run ---------------------------------------------------------------
    def _summary(self, rounds_run: int) -> Dict[str, Any]:
        return {
            "experiment-id": self.cfg["experiment-id"], "done": True, "num-rounds": self.cfg["num-rounds"],
            "rounds-run": rounds_run, "final-round": self.last_completed_round,
            "best-round": self.best_round, "best-selection-metric": self.best_selection_metric,
            "selection-metric-name": SELECTION_METRIC, "wandb-run-id": self.wandb_run_id,
        }

    def run(self, grid: Any) -> Dict[str, Any]:
        info = self.prepare()
        if info["done"]:
            summary = self._summary(0)
            if self.store.read_done() is None:
                self.store.write_done({**summary, "completed-at": time.time()})
            logger.info("experiment %s is already complete", self.cfg["experiment-id"])
            return summary

        cfg = self.cfg
        initial_arrays = ArrayRecord(model_state(self.template))
        self.strategy.start(
            grid=grid, initial_arrays=initial_arrays, num_rounds=info["remaining"],
            timeout=float(cfg["timeout"]),  # overrides Flower's 3600 s default
            train_config=ConfigRecord(static_train_config(cfg)),
            evaluate_config=ConfigRecord(static_evaluate_config(cfg)),
            evaluate_fn=self.evaluate_fn)
        if self.last_completed_round != cfg["num-rounds"]:
            raise FederationError(
                f"start() returned after round {self.last_completed_round} of {cfg['num-rounds']}")
        summary = self._summary(info["remaining"])
        self.store.write_done({**summary, "completed-at": time.time()})
        return summary


def run_experiment(cfg: Mapping[str, Any], grid: Any, *, on_round_end: Optional[Callable] = None,
                   wandb_run_id: Optional[str] = None) -> Dict[str, Any]:
    """Run (or resume) one experiment on `grid`; returns a summary dict.
    on_round_end(true_round, train_metrics, eval_metrics, privacy) is called
    after each round's checkpoint is written; privacy is None for non-DP."""
    return ExperimentRunner(cfg, on_round_end=on_round_end, wandb_run_id=wandb_run_id).run(grid)


# ---------------------------------------------------------------------------
# ServerApp
# ---------------------------------------------------------------------------

_ROUND_END_CALLBACK: Optional[Callable] = None
_RUN_HOOK: Optional[Callable[[Dict[str, Any]], Any]] = None
RUN_HOOK_ENV = "DR_RUN_HOOK"


def set_round_end_callback(callback: Optional[Callable]) -> None:
    """experiment_tracker.py registers its per-round logging here for the
    `flwr run` path."""
    global _ROUND_END_CALLBACK
    _ROUND_END_CALLBACK = callback


def set_run_hook(hook: Optional[Callable[[Dict[str, Any]], Any]]) -> None:
    """Register hook(cfg) -> (on_round_end, wandb_run_id, finish); see the
    module docstring (RUN HOOK)."""
    global _RUN_HOOK
    _RUN_HOOK = hook


def _load_hook_spec(spec: str, origin: str) -> Callable[[Dict[str, Any]], Any]:
    try:
        module_name, sep, func_name = spec.partition(":")
        if not sep or not module_name or not func_name:
            raise ValueError("expected 'package.module:function'")
        hook = getattr(importlib.import_module(module_name), func_name)
        if not callable(hook):
            raise TypeError("not callable")
    except Exception as exc:
        raise RunHookError(f"run hook {spec!r} (from {origin}) could not be loaded: {exc!r}") from exc
    return hook


def _resolve_run_hook(cfg_v: Mapping[str, Any],
                      explicit: Optional[Callable[[Dict[str, Any]], Any]]) -> Optional[Callable[[Dict[str, Any]], Any]]:
    """explicit argument, else set_run_hook(), else cfg 'run-hook', else
    $DR_RUN_HOOK, else None. An unloadable spec is fatal (RunHookError)."""
    if explicit is not None:
        return explicit
    if _RUN_HOOK is not None:
        return _RUN_HOOK
    spec = cfg_v.get("run-hook")
    if isinstance(spec, str) and spec.strip():
        return _load_hook_spec(spec.strip(), "the 'run-hook' config key")
    spec = os.environ.get(RUN_HOOK_ENV, "").strip()
    if spec:
        return _load_hook_spec(spec, f"${RUN_HOOK_ENV}")
    return None


def _call_hook(hook: Callable[[Dict[str, Any]], Any], cfg_v: Mapping[str, Any]) -> Any:
    """Call the hook and check its result; any problem is a RunHookError."""
    try:
        result = hook(dict(cfg_v))
    except Exception as exc:
        raise RunHookError(f"run hook raised: {exc!r}") from exc
    if not isinstance(result, tuple) or len(result) != 3:
        raise RunHookError("run hook must return a 3-tuple (on_round_end, wandb_run_id, finish)")
    on_round_end, wandb_run_id, finish = result
    if on_round_end is not None and not callable(on_round_end):
        raise RunHookError("run hook on_round_end must be callable or None")
    if wandb_run_id is not None and not isinstance(wandb_run_id, str):
        raise RunHookError(f"run hook wandb_run_id must be None or str, got {type(wandb_run_id).__name__}")
    if finish is not None and not callable(finish):
        raise RunHookError("run hook finish must be callable or None")
    return result


def _call_finish(finish: Optional[Callable], summary: Optional[Mapping[str, Any]],
                 error: Optional[BaseException]) -> None:
    if finish is None:
        return
    try:
        finish(summary, error)
    except Exception:
        logger.exception("run hook finish callback failed; ignoring")


def run_experiment_tracked(cfg: Mapping[str, Any], grid: Any, *,
                           hook: Optional[Callable[[Dict[str, Any]], Any]] = None) -> Dict[str, Any]:
    """run_experiment with the run hook around it (module docstring, RUN HOOK).
    Without a hook this is run_experiment(cfg, grid, on_round_end=the callback
    from set_round_end_callback)."""
    cfg_v = validate_server_config(cfg)  # invalid config: ValueError before any hook
    hook = _resolve_run_hook(cfg_v, hook)
    on_round_end, wandb_run_id, finish = _ROUND_END_CALLBACK, None, None
    if hook is not None:
        hook_round_end, wandb_run_id, finish = _call_hook(hook, cfg_v)
        if hook_round_end is not None:
            on_round_end = hook_round_end
    try:
        summary = run_experiment(cfg, grid, on_round_end=on_round_end, wandb_run_id=wandb_run_id)
    except BaseException as exc:
        _call_finish(finish, None, exc)
        raise
    _call_finish(finish, summary, None)
    return summary


app = ServerApp() if ServerApp is not None else None

if app is not None:

    @app.main()
    def main(grid: Any, context: Any) -> None:
        """`flwr run` entry point: the experiment config arrives as the flat
        run-config mapping (same keys as the config dict above)."""
        run_experiment_tracked(dict(context.run_config), grid)