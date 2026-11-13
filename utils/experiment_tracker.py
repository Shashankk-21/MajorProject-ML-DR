"""
utils/experiment_tracker.py: Weights & Biases logging for the experiments.

Federated runs (Experiments 2-8): the server calls make_hook(cfg) through its
run hook ("utils.experiment_tracker:make_hook" in the 'run-hook' config key).
make_hook returns (on_round_end, wandb_run_id, finish) exactly as
federated.server.run_experiment_tracked expects. Experiment 1 (centralized):

    tracker, run_id = start_centralized(cfg, checkpoint_dir)
    run_centralized_training(..., wandb_run_id=run_id,
                             on_epoch_end=make_epoch_callback(tracker))
    tracker.finish(result, None)        # or tracker.finish(None, exc) on failure

Logging never kills training: wandb is imported lazily; without it (or with
WANDB_MODE=disabled) the tracker is a logged no-op that still returns a run id.
Start policy: online init, else a loud warning and an offline retry, else
TrackerStartError (the server treats a hook start failure as fatal). Every call
after start is wrapped; failures are logged and swallowed.

The API key comes from WANDB_API_KEY, else the file /data/.wandb_api_key (the
flwr SuperLink freezes the environment, so no per-run variable is needed). It
is never logged or stored. WANDB_DIR / WANDB_CACHE_DIR are left to the
environment. Nothing per-sample is ever logged.

Does not import federated.server at module level (circular); it is imported
lazily for read_resume_info and the client names.
"""
from __future__ import annotations

import logging
import math
import numbers
import os
import re
import secrets
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "PROJECT", "DEFAULT_ENTITY", "DEFAULT_KEY_FILE", "NUM_CLASSES", "DEFAULT_CM_EVERY", "LATEST_CENTRALIZED",
    "TrackerStartError", "f1_per_class", "recall_per_class", "quadratic_weighted_kappa",
    "resolve_api_key", "sanitize_config", "run_name", "round_metrics", "epoch_metrics",
    "ExperimentTracker", "make_round_callback", "make_epoch_callback", "make_hook",
    "peek_centralized_run_id", "start_centralized",
]

PROJECT = "fl-dr-detection"
DEFAULT_ENTITY = "shashankmpatil77-gogte-institute-of-technology"
DEFAULT_KEY_FILE = "/data/.wandb_api_key"
NUM_CLASSES = 5
DEFAULT_CM_EVERY = 5
LATEST_CENTRALIZED = "latest_centralized.pt"
_SECRET_RE = re.compile(r"key|token|secret|password|credential", re.IGNORECASE)
_EVAL_PID_RE = re.compile(r"^(?:confusion-matrix|macro-f1|val-loss)-client-(\d+)$")
_SUCCESS = object()  # finish() with no arguments is a clean finish


class TrackerStartError(RuntimeError):
    """wandb could not be started online or offline."""


# ---------------------------------------------------------------------------
# Pure metric functions (flat 25-int row-major confusion matrix, rows = true)
# ---------------------------------------------------------------------------

def _matrix(cm: Any) -> np.ndarray:
    arr = np.asarray(cm)
    if arr.ndim != 1 or arr.size != NUM_CLASSES * NUM_CLASSES:
        raise ValueError(f"confusion matrix must be {NUM_CLASSES * NUM_CLASSES} flat values, got shape {arr.shape}")
    if arr.dtype == bool or not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise ValueError(f"confusion matrix must be numeric, got {arr.dtype}")
    if np.issubdtype(arr.dtype, np.floating) and not (np.isfinite(arr).all() and (np.mod(arr, 1) == 0).all()):
        raise ValueError("confusion matrix floats must be finite whole numbers")
    arr = arr.astype(np.int64)
    if (arr < 0).any():
        raise ValueError("confusion matrix has negative counts")
    return arr.reshape(NUM_CLASSES, NUM_CLASSES)


def f1_per_class(cm: Sequence[int]) -> List[float]:
    """Per-class F1 = 2tp / (2tp + fp + fn); 0.0 when that is 0 (zero_division=0)."""
    m = _matrix(cm)
    tp = np.diag(m).astype(np.float64)
    fp = m.sum(axis=0) - np.diag(m)
    fn = m.sum(axis=1) - np.diag(m)
    den = 2 * tp + fp + fn
    return [float(2 * t / d) if d > 0 else 0.0 for t, d in zip(tp, den)]


def recall_per_class(cm: Sequence[int]) -> List[float]:
    """Per-class recall = tp / true count; 0.0 for a class with no true samples."""
    m = _matrix(cm)
    rows = m.sum(axis=1)
    return [float(m[i, i] / rows[i]) if rows[i] > 0 else 0.0 for i in range(NUM_CLASSES)]


def quadratic_weighted_kappa(cm: Sequence[int]) -> float:
    """1 - sum(w*O) / sum(w*E), w_ij = (i-j)^2, E = outer(row, col) / N.
    Defined as 0.0 when the denominator is zero (empty matrix, or all true and
    predicted mass on one class), where sklearn would return nan."""
    m = _matrix(cm).astype(np.float64)
    n = m.sum()
    if n <= 0:
        return 0.0
    idx = np.arange(NUM_CLASSES)
    w = (idx[:, None] - idx[None, :]) ** 2
    expected = np.outer(m.sum(axis=1), m.sum(axis=0)) / n
    den = float((w * expected).sum())
    if den <= 0.0:
        return 0.0
    return float(1.0 - (w * m).sum() / den)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def resolve_api_key(env: Optional[Mapping[str, str]] = None, key_file: Any = DEFAULT_KEY_FILE) -> Tuple[Optional[str], str]:
    """(key, source) with source in {"environment", "file", "none"}."""
    env = os.environ if env is None else env
    key = str(env.get("WANDB_API_KEY", "") or "").strip()
    if key:
        return key, "environment"
    try:
        text = Path(key_file).read_text(encoding="utf-8").strip()
    except OSError:
        return None, "none"
    return (text, "file") if text else (None, "none")


def sanitize_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Full hyperparameter dump for wandb.config: secret-looking keys dropped,
    non-scalar values stringified."""
    out: Dict[str, Any] = {}
    for key, value in dict(cfg).items():
        if _SECRET_RE.search(str(key)):
            continue
        out[str(key)] = value if (value is None or isinstance(value, (bool, int, float, str))) else str(value)
    return out


def run_name(cfg: Mapping[str, Any]) -> str:
    """exp{N}-{algorithm}[-eps{value}] is the experiment id by definition."""
    name = cfg.get("experiment-id")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("cfg needs a non-empty 'experiment-id' (exp{N}-{algorithm}[-eps{value}])")
    return name.strip()


def _tags(cfg: Mapping[str, Any]) -> List[str]:
    tags: List[str] = []
    if isinstance(cfg.get("algorithm"), str) and cfg["algorithm"]:
        tags.append(cfg["algorithm"])
    if cfg.get("dp") is True:
        tags.append("dp")
    if cfg.get("diagnostic") is True:
        tags.append("diagnostic")
    return tags


def _new_run_id() -> str:
    return secrets.token_hex(4)


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


def _client_names() -> Optional[Tuple[str, ...]]:
    try:
        from federated.server import CLIENT_NAMES  # lazy: circular at module level
        return tuple(CLIENT_NAMES)
    except Exception:
        logger.warning("client names unavailable; per-client metrics use client<pid>")
        return None


def _cname(names: Optional[Sequence[str]], pid: int) -> str:
    return str(names[pid]) if names is not None and 0 <= pid < len(names) else f"client{pid}"


def _add_cm_metrics(out: Dict[str, Any], cm: Any, f1_base: str, recall_base: str, qwk_key: str) -> None:
    f1, rc = f1_per_class(cm), recall_per_class(cm)
    for c in range(NUM_CLASSES):
        out[f"{f1_base}/class-{c}"] = f1[c]
        out[f"{recall_base}/class-{c}"] = rc[c]
    out[qwk_key] = quadratic_weighted_kappa(cm)


# ---------------------------------------------------------------------------
# Metric dictionaries (pure; the exact keys that reach wandb)
# ---------------------------------------------------------------------------

def round_metrics(train_metrics: Optional[Mapping[str, Any]], eval_metrics: Optional[Mapping[str, Any]],
                  privacy: Optional[Mapping[str, Any]], client_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Scalars for one federated round, from the server's on_round_end payload."""
    out: Dict[str, Any] = {}
    ev = dict(eval_metrics or {})
    tr = dict(train_metrics or {})

    def put(key: str, value: Any) -> None:
        v = _num(value)
        if v is not None:
            out[key] = v

    put("val/macro-f1-pooled", ev.get("macro-f1-pooled"))
    put("val/macro-f1-client-mean", ev.get("macro-f1-client-mean"))
    put("val/loss-pooled", ev.get("val-loss-pooled"))

    pooled = ev.get("confusion-matrix-pooled")
    try:
        _add_cm_metrics(out, pooled, "val/f1-pooled", "val/recall-pooled", "val/qwk-pooled")
    except (ValueError, TypeError):
        for c in range(NUM_CLASSES):  # no usable matrix: only the server's recalls
            put(f"val/recall-pooled/class-{c}", ev.get(f"recall-pooled-class-{c}"))

    pids = sorted({int(m.group(1)) for k in ev for m in [_EVAL_PID_RE.match(str(k))] if m})
    for pid in pids:
        name = _cname(client_names, pid)
        put(f"val/macro-f1/{name}", ev.get(f"macro-f1-client-{pid}"))
        put(f"val/loss/{name}", ev.get(f"val-loss-client-{pid}"))
        try:
            _add_cm_metrics(out, ev.get(f"confusion-matrix-client-{pid}"),
                            f"val/f1/{name}", f"val/recall/{name}", f"val/qwk/{name}")
        except (ValueError, TypeError):
            for c in range(NUM_CLASSES):
                put(f"val/recall/{name}/class-{c}", ev.get(f"recall-client-{pid}-class-{c}"))

    per_client: Dict[int, Mapping[str, Any]] = {}
    for pid, info in dict(tr.get("per-client") or {}).items():
        try:
            per_client[int(pid)] = info
        except (TypeError, ValueError):
            continue
    agg = dict(tr.get("aggregated") or {})
    loss = _num(agg.get("train-loss"))
    if loss is None and per_client:  # fallback: example-weighted mean of the clients
        pairs = [(_num(i.get("train-loss")), _num(i.get("num-examples"))) for i in per_client.values()]
        pairs = [(l, n) for l, n in pairs if l is not None and n is not None and n > 0]
        if pairs:
            loss = sum(l * n for l, n in pairs) / sum(n for _l, n in pairs)
    if loss is not None:
        out["train/loss"] = loss
    put("train/lr-lora", tr.get("lr-lora"))
    put("train/lr-head", tr.get("lr-head"))
    for pid in sorted(per_client):
        put(f"train/loss/{_cname(client_names, pid)}", per_client[pid].get("train-loss"))

    if privacy:
        for key in ("epsilon", "noise-multiplier", "sample-rate", "steps-this-round", "steps-total"):
            for pid, value in dict(privacy.get(key) or {}).items():
                try:
                    put(f"privacy/{key}/{_cname(client_names, int(pid))}", value)
                except (TypeError, ValueError):
                    continue
        put("privacy/max-epsilon", privacy.get("max-epsilon"))
    return out


def epoch_metrics(metrics: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Scalars for one Experiment 1 epoch (train_centralized's on_epoch_end).
    The validation set is pooled, so the pooled key names are reused."""
    m = dict(metrics or {})
    out: Dict[str, Any] = {}
    for src, dst in (("train_loss", "train/loss"), ("val_loss", "val/loss-pooled"),
                     ("val_macro_f1", "val/macro-f1-pooled"), ("lr_lora", "train/lr-lora"),
                     ("lr_head", "train/lr-head")):
        v = _num(m.get(src))
        if v is not None:
            out[dst] = v
    if m.get("val_confusion_matrix") is not None:  # absent: skip quietly
        try:
            _add_cm_metrics(out, m["val_confusion_matrix"], "val/f1-pooled", "val/recall-pooled", "val/qwk-pooled")
        except (ValueError, TypeError):
            logger.debug("val_confusion_matrix unusable; per-class metrics skipped")
    return out


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """One wandb run per experiment. start() once, then log_round / log_epoch,
    then finish()."""

    def __init__(self, *, wandb_module: Any = None, project: str = PROJECT, entity: Optional[str] = None,
                 cm_every: int = DEFAULT_CM_EVERY, api_key_file: Any = DEFAULT_KEY_FILE,
                 client_names: Optional[Sequence[str]] = None) -> None:
        self._injected = wandb_module
        self.project = project
        self.entity = entity or os.environ.get("WANDB_ENTITY") or DEFAULT_ENTITY
        self.cm_every = cm_every if isinstance(cm_every, int) and not isinstance(cm_every, bool) else DEFAULT_CM_EVERY
        self.api_key_file = api_key_file
        self._client_names = tuple(client_names) if client_names is not None else None
        self._names_loaded = client_names is not None
        self._wandb: Any = None
        self.run: Any = None
        self.run_id: Optional[str] = None
        self.mode = "unstarted"  # online | offline | noop
        self.num_rounds: Optional[int] = None
        self._finished = False
        self._table_warned = False

    # -- start ---------------------------------------------------------------
    def _load_wandb(self) -> Any:
        if self._injected is not None:
            return self._injected
        if os.environ.get("WANDB_MODE", "").strip().lower() == "disabled":
            logger.warning("WANDB_MODE=disabled: experiment tracking is a no-op")
            return None
        try:
            import wandb  # lazy
            return wandb
        except ImportError:
            logger.warning("wandb is not importable: experiment tracking is a no-op")
            return None

    def _ensure_api_key(self) -> None:
        key, source = resolve_api_key(key_file=self.api_key_file)
        if key and source == "file":
            os.environ["WANDB_API_KEY"] = key  # in memory only; the file is the store
            logger.info("wandb API key read from %s", self.api_key_file)
        elif not key:
            logger.warning("no wandb API key in WANDB_API_KEY or %s; online init will probably fail", self.api_key_file)

    def start(self, cfg: Mapping[str, Any], resume_run_id: Optional[str] = None,
              tags: Optional[Sequence[str]] = None) -> str:
        """Start (or resume) the run and return its id."""
        if self.run_id is not None:
            logger.warning("tracker already started (run %s); start() ignored", self.run_id)
            return self.run_id
        cfg = dict(cfg)
        name = run_name(cfg)
        nr = cfg.get("num-rounds")
        self.num_rounds = nr if isinstance(nr, int) and not isinstance(nr, bool) else None
        resume = resume_run_id if isinstance(resume_run_id, str) and resume_run_id else None
        run_id = resume or _new_run_id()
        module = self._load_wandb()
        if module is None:
            self.run_id, self.mode = run_id, "noop"
            return run_id
        self._wandb = module
        self._ensure_api_key()
        kwargs: Dict[str, Any] = {
            "project": self.project, "entity": self.entity, "name": name, "id": run_id,
            "config": sanitize_config(cfg), "tags": list(tags if tags is not None else _tags(cfg)),
        }
        if resume:
            kwargs["resume"] = "must"
        try:
            run = module.init(**kwargs)
            self.mode = "online"
        except Exception as exc:
            logger.warning("WANDB ONLINE INIT FAILED (%r): retrying in OFFLINE mode; sync the run later with "
                           "`wandb sync`", exc)
            retry = dict(kwargs, mode="offline")
            if resume:
                retry["resume"] = "allow"  # "must" needs the server
            try:
                run = module.init(**retry)
                self.mode = "offline"
            except Exception as exc2:
                raise TrackerStartError(f"wandb init failed online ({exc!r}) and offline ({exc2!r})") from exc2
        self.run = run
        rid = getattr(run, "id", None)
        if run is None:
            logger.warning("wandb.init returned no run: experiment tracking is a no-op")
            self.mode = "noop"
        self.run_id = rid if isinstance(rid, str) and rid else run_id
        return self.run_id

    # -- logging (never raises) ----------------------------------------------
    def _names(self) -> Optional[Tuple[str, ...]]:
        if not self._names_loaded:
            self._client_names, self._names_loaded = _client_names(), True
        return self._client_names

    def _log(self, data: Dict[str, Any], step: int) -> None:
        if self.run is None or not data:
            return
        self.run.log(data, step=step)

    def _tables(self, true_round: int, ev: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        labels = [str(c) for c in range(NUM_CLASSES)]

        def table(flat: Any) -> Any:
            m = _matrix(flat)
            rows = [[labels[i], labels[j], int(m[i, j])] for i in range(NUM_CLASSES) for j in range(NUM_CLASSES)]
            return self._wandb.Table(columns=["true", "pred", "count"], data=rows)

        out["val/confusion-matrix-pooled"] = table(ev.get("confusion-matrix-pooled"))
        for pid in sorted({int(m.group(1)) for k in ev for m in [_EVAL_PID_RE.match(str(k))] if m}):
            flat = ev.get(f"confusion-matrix-client-{pid}")
            if flat is not None:
                out[f"val/confusion-matrix/{_cname(self._names(), pid)}"] = table(flat)
        return out

    def _tables_due(self, true_round: int) -> bool:
        if self.cm_every <= 0:
            return False
        return true_round % self.cm_every == 0 or (self.num_rounds is not None and true_round == self.num_rounds)

    def log_round(self, true_round: int, train_metrics: Any, eval_metrics: Any, privacy: Any) -> None:
        """Matches the server's on_round_end(true_round, train_metrics, eval_metrics, privacy)."""
        if self.run is None:
            return
        try:
            data = round_metrics(train_metrics, eval_metrics, privacy, self._names())
            if self._tables_due(true_round) and self._wandb is not None:
                try:
                    data.update(self._tables(true_round, dict(eval_metrics or {})))
                except Exception:
                    if not self._table_warned:
                        self._table_warned = True
                        logger.exception("confusion-matrix tables could not be built; scalars still logged")
            self._log(data, int(true_round))
        except Exception:
            logger.exception("wandb log_round failed at round %s; continuing", true_round)

    def log_epoch(self, epoch: int, metrics: Any) -> None:
        """Matches train_centralized's on_epoch_end(epoch, metrics)."""
        if self.run is None:
            return
        try:
            self._log(epoch_metrics(metrics), int(epoch))
        except Exception:
            logger.exception("wandb log_epoch failed at epoch %s; continuing", epoch)

    # -- finish --------------------------------------------------------------
    def finish(self, summary: Any = _SUCCESS, error: Optional[BaseException] = None) -> None:
        """finish(summary, error) as the server hook calls it: summary None
        means failure. finish() with no arguments is a clean finish. Never raises."""
        if self._finished:
            return
        self._finished = True
        run = self.run
        if run is None:
            return
        try:
            failed = error is not None or summary is None
            if failed:
                text = f"{type(error).__name__}: {error}" if error is not None else "run failed (no error given)"
                self._set_summary(run, "status", "failed")
                self._set_summary(run, "error", text[:2000])
            else:
                self._set_summary(run, "status", "finished")
                if isinstance(summary, Mapping):
                    for k, v in summary.items():
                        if k not in ("wandb-run-id", "experiment-id") and (
                                v is None or isinstance(v, (bool, int, float, str))):
                            self._set_summary(run, f"result/{k}", v)
            try:
                run.finish(exit_code=1 if failed else 0)
            except TypeError:
                run.finish()
        except Exception:
            logger.exception("wandb finish failed; ignoring")

    @staticmethod
    def _set_summary(run: Any, key: str, value: Any) -> None:
        try:
            run.summary[key] = value
        except Exception:
            logger.debug("could not set run.summary[%r]", key)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def make_round_callback(tracker: ExperimentTracker) -> Callable[[int, Any, Any, Any], None]:
    """The function to pass as the server's on_round_end."""

    def on_round_end(true_round: int, train_metrics: Any, eval_metrics: Any, privacy: Any) -> None:
        tracker.log_round(true_round, train_metrics, eval_metrics, privacy)

    return on_round_end


def make_epoch_callback(tracker: ExperimentTracker) -> Callable[[int, Any], None]:
    """The function to pass as train_centralized's on_epoch_end."""

    def on_epoch_end(epoch: int, metrics: Any) -> None:
        tracker.log_epoch(epoch, metrics)

    return on_epoch_end


def _cm_every(cfg: Mapping[str, Any]) -> int:
    v = cfg.get("wandb-cm-every", DEFAULT_CM_EVERY)
    return v if isinstance(v, int) and not isinstance(v, bool) else DEFAULT_CM_EVERY


def _federated_resume_id(cfg: Mapping[str, Any]) -> Optional[str]:
    try:
        from federated.server import read_resume_info  # lazy: circular at module level
        info = read_resume_info(cfg)
    except Exception:
        logger.exception("could not read the resume info; starting a new wandb run (the server validates the checkpoint itself)")
        return None
    rid = info.get("wandb_run_id") if info else None
    return rid if isinstance(rid, str) and rid else None


def make_hook(cfg: Mapping[str, Any], *, wandb_module: Any = None) -> Tuple[Callable, str, Callable]:
    """Server run hook: hook(validated cfg) -> (on_round_end, wandb_run_id, finish).
    Optional cfg keys: wandb-cm-every (int, default 5; <= 0 disables the
    confusion-matrix tables), diagnostic (bool, adds the 'diagnostic' tag)."""
    tracker = ExperimentTracker(wandb_module=wandb_module, cm_every=_cm_every(cfg))
    run_id = tracker.start(cfg, resume_run_id=_federated_resume_id(cfg))
    return make_round_callback(tracker), run_id, tracker.finish


def peek_centralized_run_id(checkpoint_dir: Any) -> Optional[str]:
    """wandb_run_id stored in <checkpoint_dir>/latest_centralized.pt, or None
    (absent file, unreadable file, or no id)."""
    path = Path(checkpoint_dir) / LATEST_CENTRALIZED
    if not path.is_file():
        return None
    try:
        import torch  # lazy
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        rid = ckpt.get("wandb_run_id")
    except Exception:
        logger.exception("could not read %s; starting a new wandb run", path)
        return None
    return rid if isinstance(rid, str) and rid else None


def start_centralized(cfg: Mapping[str, Any], checkpoint_dir: Any, *, wandb_module: Any = None,
                      tracker: Optional[ExperimentTracker] = None) -> Tuple[ExperimentTracker, str]:
    """Experiment 1: peek the stored run id, start or resume the run; returns
    (tracker, run_id). Pass run_id as wandb_run_id and make_epoch_callback(tracker)
    as on_epoch_end to run_centralized_training."""
    tracker = tracker or ExperimentTracker(wandb_module=wandb_module, cm_every=_cm_every(cfg))
    run_id = tracker.start(cfg, resume_run_id=peek_centralized_run_id(checkpoint_dir))
    return tracker, run_id