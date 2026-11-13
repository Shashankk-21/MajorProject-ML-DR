"""
notebooks/stepzero/stepzero_lib.py: pure logic for the step-zero diagnostics.

numpy and the stdlib only; no torch import at module level (torch tensors are
handled by duck typing). Used by rank_sweep.py, grad_norm_probe.py and
timing_probe.py.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

_EPS = 1e-12  # float slack so threshold boundaries behave as written


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) \
        and math.isfinite(float(v))


def _is_pos_int(v: Any) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and int(v) > 0


# ---------------------------------------------------------------------------
# Rank selection
# ---------------------------------------------------------------------------

def select_rank(results: Mapping[int, Mapping[str, float]], tie: float = 0.01, qwk_flag: float = 0.02) -> Dict[str, Any]:
    """results = {rank: {"val_macro_f1": f, "qwk": q}}. Tie set = ranks whose
    val_macro_f1 >= best_f1 - tie; chosen = the LOWEST rank in it;
    manual_review when the chosen rank's QWK is below best_qwk - qwk_flag
    (best_qwk is over all ranks; the chosen rank is still returned)."""
    if not isinstance(results, Mapping) or not results:
        raise ValueError("results must be a non-empty mapping of rank -> metrics")
    if not _is_num(tie) or tie < 0 or not _is_num(qwk_flag) or qwk_flag < 0:
        raise ValueError(f"tie and qwk_flag must be finite numbers >= 0, got {tie!r}, {qwk_flag!r}")
    clean: Dict[int, Dict[str, float]] = {}
    for rank, m in results.items():
        if not _is_pos_int(rank):
            raise ValueError(f"rank must be a positive int, got {rank!r}")
        if not isinstance(m, Mapping) or "val_macro_f1" not in m or "qwk" not in m:
            raise ValueError(f"rank {rank}: needs 'val_macro_f1' and 'qwk'")
        f1, qwk = m["val_macro_f1"], m["qwk"]
        if not _is_num(f1) or not _is_num(qwk):
            raise ValueError(f"rank {rank}: non-finite or non-numeric metrics ({f1!r}, {qwk!r})")
        clean[int(rank)] = {"val_macro_f1": float(f1), "qwk": float(qwk)}
    best_f1 = max(m["val_macro_f1"] for m in clean.values())
    best_qwk = max(m["qwk"] for m in clean.values())
    tie_set = sorted(r for r, m in clean.items() if m["val_macro_f1"] >= best_f1 - tie - _EPS)
    chosen = tie_set[0]
    manual_review = clean[chosen]["qwk"] < best_qwk - qwk_flag - _EPS
    reason = (f"ranks within {tie:g} of the best val macro-F1 {best_f1:.4f}: {tie_set}; "
              f"chose the lowest, rank {chosen} (val macro-F1 {clean[chosen]['val_macro_f1']:.4f}, "
              f"QWK {clean[chosen]['qwk']:.4f}, best QWK {best_qwk:.4f})")
    if manual_review:
        reason += (f"; MANUAL REVIEW: the chosen rank's QWK is more than {qwk_flag:g} below the best QWK "
                   f"({best_qwk - clean[chosen]['qwk']:.4f} below)")
    return {"chosen": chosen, "tie_set": tie_set, "best_f1": best_f1, "best_qwk": best_qwk,
            "manual_review": bool(manual_review), "reason": reason}


# ---------------------------------------------------------------------------
# Per-sample gradient norms
# ---------------------------------------------------------------------------

def _finite_array(values: Any, what: str) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if a.size == 0:
        raise ValueError(f"{what}: empty")
    if not np.isfinite(a).all():
        raise ValueError(f"{what}: contains non-finite values")
    return a


def percentile_summary(norms: Any) -> Dict[str, Any]:
    """{"n","p25","p50","p75","p95","max"} (numpy linear percentiles)."""
    a = _finite_array(norms, "norms")
    p25, p50, p75, p95 = (float(np.percentile(a, q)) for q in (25, 50, 75, 95))
    return {"n": int(a.size), "p25": p25, "p50": p50, "p75": p75, "p95": p95, "max": float(a.max())}


def fraction_clipped(norms: Any, c: float) -> float:
    """Share of norms strictly greater than the clip value c."""
    if not _is_num(c) or c <= 0:
        raise ValueError(f"c must be a finite number > 0, got {c!r}")
    return float(np.mean(_finite_array(norms, "norms") > float(c)))


def geometric_mean(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if a.size == 0 or not np.isfinite(a).all() or not (a > 0).all():
        raise ValueError("geometric_mean needs a non-empty set of finite values > 0")
    return float(np.exp(np.mean(np.log(a))))


def recommend_max_grad_norm(per_client_medians: Mapping[str, float], flag_ratio: float = 3.0) -> Dict[str, Any]:
    """One global value = geometric mean of the per-client medians; flag when
    the largest and smallest median differ by more than flag_ratio."""
    if not isinstance(per_client_medians, Mapping) or not per_client_medians:
        raise ValueError("per_client_medians must be a non-empty mapping")
    if not _is_num(flag_ratio) or flag_ratio <= 0:
        raise ValueError(f"flag_ratio must be a finite number > 0, got {flag_ratio!r}")
    vals = list(per_client_medians.values())
    value = geometric_mean(vals)  # validates finite and > 0
    ratio = float(max(vals)) / float(min(vals))
    return {"value": value, "ratio": ratio, "flag": bool(ratio > flag_ratio)}


def phase_ratio(a: float, b: float, flag_ratio: float = 3.0) -> Dict[str, Any]:
    """max/min of two positive values, flagged when above flag_ratio."""
    if not (_is_num(a) and _is_num(b) and a > 0 and b > 0):
        raise ValueError(f"phase_ratio needs two finite values > 0, got {a!r}, {b!r}")
    ratio = float(max(a, b)) / float(min(a, b))
    return {"ratio": ratio, "flag": bool(ratio > flag_ratio)}


def _per_sample_squares(g: Any) -> np.ndarray:
    if hasattr(g, "detach"):  # torch tensor, any device
        t = g.detach()
        if t.ndim < 1:
            raise ValueError("per-sample gradient must have a leading batch dimension")
        flat = t.reshape(int(t.shape[0]), -1).double()
        return (flat * flat).sum(dim=1).cpu().numpy().astype(np.float64)
    a = np.asarray(g, dtype=np.float64)
    if a.ndim < 1:
        raise ValueError("per-sample gradient must have a leading batch dimension")
    flat = a.reshape(a.shape[0], -1)
    return (flat * flat).sum(axis=1)


def flat_per_sample_norms(per_param_grad_samples: Sequence[Any]) -> np.ndarray:
    """Flat L2 norm per sample over ALL trainable parameters, as Opacus clips:
    sqrt(sum over parameters of each sample's squared L2). Input: one array or
    tensor shaped (B, ...) per trainable parameter. Returns float64 (B,); may
    contain inf/nan if the gradients do (callers check)."""
    grads = list(per_param_grad_samples)
    if not grads:
        raise ValueError("need at least one per-parameter grad_sample")
    total: Optional[np.ndarray] = None
    for k, g in enumerate(grads):
        sq = _per_sample_squares(g)
        if total is None:
            total = sq
        elif sq.shape != total.shape:
            raise ValueError(f"parameter {k}: batch size {sq.shape[0]} differs from {total.shape[0]}")
        else:
            total = total + sq
    return np.sqrt(total)


def round_sig(x: float, sig: int = 3) -> float:
    """x rounded to `sig` significant digits (for the line the user pastes)."""
    if not _is_num(x) or x <= 0:
        raise ValueError(f"round_sig needs a finite value > 0, got {x!r}")
    return float(f"{float(x):.{sig}g}")


# ---------------------------------------------------------------------------
# Timing extrapolation
# ---------------------------------------------------------------------------

def epoch_seconds_nondp(n: int, physical_batch: int, seconds_per_step: float) -> float:
    """Seconds per non-DP epoch: ceil(n / physical_batch) micro-batches."""
    if not _is_pos_int(n) or not _is_pos_int(physical_batch) or not _is_num(seconds_per_step) or seconds_per_step < 0:
        raise ValueError("need n, physical_batch positive ints and seconds_per_step >= 0")
    return math.ceil(n / physical_batch) * float(seconds_per_step)


def expected_micro_batches_dp(n: int, logical_batch: int, physical_batch: int) -> float:
    """Expected micro-batches per DP epoch. Poisson sampling with q = 1/ceil(n /
    logical_batch) gives ceil(n / logical_batch) logical batches of size
    Binomial(n, q); each splits into ceil(size / physical_batch) chunks."""
    if not (_is_pos_int(n) and _is_pos_int(logical_batch) and _is_pos_int(physical_batch)):
        raise ValueError("n, logical_batch and physical_batch must be positive ints")
    n, logical_batch, physical_batch = int(n), int(logical_batch), int(physical_batch)
    n_logical = math.ceil(n / logical_batch)
    if n_logical == 1:
        return float(math.ceil(n / physical_batch))
    q = 1.0 / n_logical
    mean, sd = n * q, math.sqrt(n * q * (1.0 - q))
    lo, hi = max(0, int(mean - 12 * sd) - 1), min(n, int(mean + 12 * sd) + 1)
    log_q, log_1q, lg_n = math.log(q), math.log1p(-q), math.lgamma(n + 1)
    mass = expect = 0.0
    for k in range(lo, hi + 1):
        pmf = math.exp(lg_n - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * log_q + (n - k) * log_1q)
        mass += pmf
        expect += pmf * math.ceil(k / physical_batch)
    return n_logical * expect / mass


def epoch_seconds_dp(n: int, logical_batch: int, physical_batch: int, seconds_per_step: float) -> float:
    if not _is_num(seconds_per_step) or seconds_per_step < 0:
        raise ValueError("seconds_per_step must be >= 0")
    return expected_micro_batches_dp(n, logical_batch, physical_batch) * float(seconds_per_step)


def largest_fitting(status_by_batch: Mapping[int, str]) -> Optional[int]:
    """Largest physical batch whose status is 'ok' (None if none fit)."""
    ok = [int(b) for b, s in status_by_batch.items() if s == "ok"]
    return max(ok) if ok else None


# ---------------------------------------------------------------------------
# Small I/O and formatting helpers
# ---------------------------------------------------------------------------

def fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cells = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def read_json(path: Any) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def atomic_write_json(path: Any, obj: Any) -> None:
    """Write via a temp file in the same folder, so a crash never leaves a
    half-written results file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
