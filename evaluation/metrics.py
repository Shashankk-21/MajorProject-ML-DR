"""
evaluation/metrics.py: pure metric functions for the final test-set evaluation.

numpy only (no torch, no pandas). Labels are integers 0..K-1 (K = 5 DR grades).

NaN policy (differs from training-time selection, which used zero_division=0 and
therefore counted an absent class as 0):
  * support == 0 (class absent from y_true): recall and F1 are NaN. Precision is NaN
    only when the class is never predicted (sklearn zero_division=nan); an absent
    class that IS predicted has precision 0.0, but F1 stays NaN (recall undefined).
  * AUC one-vs-rest is NaN for a class with no positives or no negatives.
  * QWK is NaN when its denominator is 0 (nothing to agree or disagree about).
  * Macro averages are taken over the DEFINED (non-NaN) classes only; all-NaN gives NaN.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

NUM_CLASSES = 5


def _labels(y: Any, num_classes: int, name: str) -> np.ndarray:
    a = np.asarray(y)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.dtype == bool or not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)):
        raise ValueError(f"{name} must be integer class ids, got {a.dtype}")
    if np.issubdtype(a.dtype, np.floating):
        if not np.isfinite(a).all() or not (np.mod(a, 1) == 0).all():
            raise ValueError(f"{name} must hold whole-number class ids")
    a = a.astype(np.int64)
    if a.size and (a.min() < 0 or a.max() >= num_classes):
        raise ValueError(f"{name} must be in [0, {num_classes - 1}], got [{a.min()}, {a.max()}]")
    return a


def _pair(y_true: Any, y_pred: Any, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    t, p = _labels(y_true, num_classes, "y_true"), _labels(y_pred, num_classes, "y_pred")
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_pred differ in length: {t.shape[0]} vs {p.shape[0]}")
    return t, p


def confusion(y_true: Any, y_pred: Any, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """(K, K) int64 matrix, rows = true class, columns = predicted class."""
    t, p = _pair(y_true, y_pred, num_classes)
    return np.bincount(t * num_classes + p, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def accuracy(y_true: Any, y_pred: Any) -> float:
    t, p = _pair(y_true, y_pred, 10 ** 9)
    return float("nan") if t.size == 0 else float(np.mean(t == p))


def per_class_report(y_true: Any, y_pred: Any, num_classes: int = NUM_CLASSES) -> Dict[str, np.ndarray]:
    """{"precision","recall","f1","support"}, each of length K; NaN where undefined (module docstring)."""
    cm = confusion(y_true, y_pred, num_classes).astype(np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        recall = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(support > 0, 2.0 * tp / (support + predicted), np.nan)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support.astype(np.int64)}


def macro_over_defined(values: Any) -> float:
    """Mean over the non-NaN entries; NaN if there are none."""
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


def macro_f1(y_true: Any, y_pred: Any, num_classes: int = NUM_CLASSES) -> float:
    return macro_over_defined(per_class_report(y_true, y_pred, num_classes)["f1"])


def qwk(y_true: Any, y_pred: Any, num_classes: int = NUM_CLASSES) -> float:
    """Quadratic weighted kappa, 1 - sum(w O) / sum(w E) with w_ij = (i - j)^2 and
    E = outer(row sums, column sums) / N; NaN when the denominator is 0."""
    cm = confusion(y_true, y_pred, num_classes).astype(np.float64)
    n = cm.sum()
    if n <= 0:
        return float("nan")
    idx = np.arange(num_classes)
    w = (idx[:, None] - idx[None, :]) ** 2
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    den = float((w * expected).sum())
    if den <= 0.0:
        return float("nan")
    return float(1.0 - (w * cm).sum() / den)


def _binary_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC with tie-averaged ranks; NaN without both classes."""
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    change = np.flatnonzero(s[1:] != s[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [s.size]))
    avg_rank = (starts + ends + 1) / 2.0  # 1-based mean rank of each tie group
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.repeat(avg_rank, ends - starts)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auc_ovr(y_true: Any, probs: Any) -> Tuple[np.ndarray, float]:
    """One-vs-rest AUC per class (NaN where a class has no positives or no negatives)
    and the macro average over the defined classes. probs: (N, K) scores."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs must be (N, K), got shape {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("probs contain non-finite values")
    k = p.shape[1]
    t = _labels(y_true, k, "y_true")
    if t.shape[0] != p.shape[0]:
        raise ValueError(f"y_true has {t.shape[0]} rows but probs has {p.shape[0]}")
    per_class = np.array([_binary_auc(p[:, c], t == c) for c in range(k)], dtype=np.float64)
    return per_class, macro_over_defined(per_class)


def softmax(logits: Any) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def summarize(y_true: Any, y_pred: Any, probs: Any, num_classes: int = NUM_CLASSES) -> Dict[str, Any]:
    """Every headline metric for one group of images."""
    rep = per_class_report(y_true, y_pred, num_classes)
    auc_cls, auc_macro = auc_ovr(y_true, probs)
    return {
        "n": int(np.asarray(y_true).shape[0]),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_over_defined(rep["f1"]),
        "qwk": qwk(y_true, y_pred, num_classes),
        "macro_auc": auc_macro,
        "auc": auc_cls,
        "precision": rep["precision"], "recall": rep["recall"], "f1": rep["f1"], "support": rep["support"],
        "confusion": confusion(y_true, y_pred, num_classes),
    }


def default_metric_fns(num_classes: int = NUM_CLASSES) -> Dict[str, Callable[[Any, Any, Any], float]]:
    """The three bootstrapped metrics, each fn(y_true, y_pred, probs) -> float."""
    return {
        "macro_f1": lambda t, p, pr: macro_f1(t, p, num_classes),
        "qwk": lambda t, p, pr: qwk(t, p, num_classes),
        "macro_auc": lambda t, p, pr: auc_ovr(t, pr)[1],
    }
