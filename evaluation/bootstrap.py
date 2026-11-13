"""
evaluation/bootstrap.py: cluster bootstrap for the final evaluation.

Pure (numpy only; pandas is imported lazily by the CSV helper/CLI). Resampling is
by UNIT: a unit is a patient where the manifest has real patient ids and an image
otherwise (make_units). Each resample draws as many units as there are, with
replacement, and takes every row of every drawn unit (with multiplicity).
Intervals are percentile intervals and reflect test-set sampling only, not
training randomness (training seeds, augmentation, DP noise).

  bootstrap_ci(y_true, y_pred, probs, units, metric_fns, n=1000, seed=42, alpha=0.05)
  paired_bootstrap_diff(pred_a, pred_b, units, metric_fn, n=1000, seed=42)

CLI (two predictions.csv files of the same test set):
  python -m evaluation.bootstrap --a A/predictions.csv --b B/predictions.csv --metric macro_f1 [--client POOLED]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from evaluation import metrics as M

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def make_units(clients: Sequence[Any], image_ids: Sequence[Any], patient_ids: Optional[Sequence[Any]] = None
               ) -> Tuple[np.ndarray, Dict[str, str]]:
    """Resampling unit per row, and the unit kind used per client.

    Per client: "patients" when patient ids exist and there are fewer unique units
    than rows, else "images". Unit labels are prefixed with the client, so patient
    ids from different clients never collide. A missing patient id makes that row
    its own unit."""
    clients = np.asarray(clients).astype(str)
    images = np.asarray(image_ids).astype(str)
    if clients.shape != images.shape:
        raise ValueError("clients and image_ids must have the same length")
    pats = None if patient_ids is None else np.asarray(patient_ids, dtype=object)
    if pats is not None and pats.shape != clients.shape:
        raise ValueError("patient_ids must have the same length as clients")
    units = np.empty(clients.shape[0], dtype=object)
    kinds: Dict[str, str] = {}
    for client in dict.fromkeys(clients.tolist()):
        rows = np.flatnonzero(clients == client)
        by_image = [f"{client}|img:{images[i]}" for i in rows]
        by_patient = by_image
        if pats is not None:
            by_patient = [by_image[j] if _missing(pats[i]) else f"{client}|pat:{pats[i]}" for j, i in enumerate(rows)]
        use_patients = pats is not None and len(set(by_patient)) < rows.size
        chosen = by_patient if use_patients else by_image
        kinds[client] = "patients" if use_patients else "images"
        for j, i in enumerate(rows):
            units[i] = chosen[j]
    return units.astype(str), kinds


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and v.strip() == "")


def _codes(units: Any, n_rows: int) -> Tuple[np.ndarray, int]:
    u = np.asarray(units)
    if u.shape != (n_rows,):
        raise ValueError(f"units must have one entry per row ({n_rows}), got shape {u.shape}")
    _vals, codes = np.unique(u, return_inverse=True)
    return codes.astype(np.int64), int(_vals.size)


def _draw_rows(codes: np.ndarray, n_units: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one cluster-bootstrap resample."""
    counts = np.bincount(rng.integers(0, n_units, size=n_units), minlength=n_units)
    return np.repeat(np.arange(codes.size), counts[codes])


def _interval(values: np.ndarray, alpha: float) -> Tuple[float, float, int]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(v, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi), int(v.size)


def _check_run(n: int, alpha: float) -> None:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def bootstrap_ci(y_true: Any, y_pred: Any, probs: Any, units: Any, metric_fns: Mapping[str, MetricFn],
                 n: int = 1000, seed: int = 42, alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Percentile CI per metric: {name: {"estimate","low","high","n_valid","n_resamples"}}.
    metric_fns: {name: fn(y_true, y_pred, probs) -> float}. All metrics use the SAME
    resamples. Resamples where a metric is undefined (NaN) are dropped from that
    metric's interval (n_valid counts the rest)."""
    _check_run(n, alpha)
    if not isinstance(metric_fns, Mapping) or not metric_fns:
        raise ValueError("metric_fns must be a non-empty mapping name -> fn")
    t, p, pr = np.asarray(y_true), np.asarray(y_pred), np.asarray(probs)
    if t.shape != p.shape or t.ndim != 1 or pr.shape[0] != t.shape[0]:
        raise ValueError(f"y_true {t.shape}, y_pred {p.shape} and probs {pr.shape} must align on one axis")
    if t.size == 0:
        raise ValueError("no rows to resample")
    codes, n_units = _codes(units, t.size)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n, dtype=np.float64) for name in metric_fns}
    for b in range(n):
        idx = _draw_rows(codes, n_units, rng)
        for name, fn in metric_fns.items():
            draws[name][b] = fn(t[idx], p[idx], pr[idx])
    out: Dict[str, Dict[str, Any]] = {}
    for name, fn in metric_fns.items():
        lo, hi, n_valid = _interval(draws[name], alpha)
        out[name] = {"estimate": float(fn(t, p, pr)), "low": lo, "high": hi, "n_valid": n_valid, "n_resamples": int(n)}
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def _col(obj: Any, key: str) -> np.ndarray:
    return np.asarray(obj[key])


def _probs(obj: Any, num_classes: int = M.NUM_CLASSES) -> np.ndarray:
    if "probs" in obj:
        return np.asarray(obj["probs"], dtype=np.float64)
    return np.column_stack([np.asarray(obj[f"prob_{c}"], dtype=np.float64) for c in range(num_classes)])


def _aligned(pred_a: Any, pred_b: Any) -> Dict[str, np.ndarray]:
    """Rows of A and B joined on (client, image_id), in A's order."""
    ka = list(zip(_col(pred_a, "client").astype(str), _col(pred_a, "image_id").astype(str)))
    kb = list(zip(_col(pred_b, "client").astype(str), _col(pred_b, "image_id").astype(str)))
    if len(set(ka)) != len(ka) or len(set(kb)) != len(kb):
        raise ValueError("duplicate (client, image_id) keys in a predictions file")
    if set(ka) != set(kb):
        only_a, only_b = len(set(ka) - set(kb)), len(set(kb) - set(ka))
        raise ValueError(f"the two experiments cover different test images ({only_a} only in A, {only_b} only in B)")
    pos_b = {k: i for i, k in enumerate(kb)}
    order_b = np.array([pos_b[k] for k in ka], dtype=np.int64)
    label_a, label_b = _col(pred_a, "label").astype(np.int64), _col(pred_b, "label").astype(np.int64)[order_b]
    if not np.array_equal(label_a, label_b):
        raise ValueError("the two files disagree on the true labels of the same images")
    return {"keys": ka, "label": label_a, "pred_a": _col(pred_a, "pred").astype(np.int64),
            "pred_b": _col(pred_b, "pred").astype(np.int64)[order_b],
            "probs_a": _probs(pred_a), "probs_b": _probs(pred_b)[order_b]}


def paired_bootstrap_diff(pred_a: Any, pred_b: Any, units: Any, metric_fn: MetricFn, n: int = 1000, seed: int = 42,
                          alpha: float = 0.05) -> Dict[str, Any]:
    """Paired bootstrap of metric(A) - metric(B) over the same test images.

    pred_a / pred_b: predictions.csv content (a DataFrame or a mapping of equal-length
    columns) with client, image_id, label, pred and either prob_0..prob_4 or probs
    (N, 5). Rows are joined on (client, image_id); both must cover the same images
    and agree on the labels. units: one resampling unit per row of pred_a (array-like
    in pred_a's order), or a mapping {(client, image_id): unit}. Every resample draws
    units once and applies the SAME drawn units to both experiments.

    Returns {"metric_a","metric_b","diff","low","high","p_two_sided","n_valid",
    "n_resamples","n_rows","n_units"}; the interval is the percentile interval of the
    paired differences, p_two_sided = 2 * min(P(d <= 0), P(d >= 0)) over valid resamples."""
    _check_run(n, alpha)
    j = _aligned(pred_a, pred_b)
    n_rows = j["label"].size
    if isinstance(units, Mapping):
        try:
            unit_arr = np.array([units[k] for k in j["keys"]])
        except KeyError as exc:
            raise ValueError(f"no unit for (client, image_id) {exc.args[0]!r}") from None
    else:
        unit_arr = np.asarray(units)
    codes, n_units = _codes(unit_arr, n_rows)
    y = j["label"]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n, dtype=np.float64)
    for b in range(n):
        idx = _draw_rows(codes, n_units, rng)
        diffs[b] = metric_fn(y[idx], j["pred_a"][idx], j["probs_a"][idx]) - metric_fn(y[idx], j["pred_b"][idx], j["probs_b"][idx])
    lo, hi, n_valid = _interval(diffs, alpha)
    valid = diffs[np.isfinite(diffs)]
    p_two = float(min(1.0, 2.0 * min(np.mean(valid <= 0), np.mean(valid >= 0)))) if valid.size else float("nan")
    ma, mb = float(metric_fn(y, j["pred_a"], j["probs_a"])), float(metric_fn(y, j["pred_b"], j["probs_b"]))
    return {"metric_a": ma, "metric_b": mb, "diff": ma - mb, "low": lo, "high": hi, "p_two_sided": p_two,
            "n_valid": n_valid, "n_resamples": int(n), "n_rows": int(n_rows), "n_units": int(n_units)}


def paired_from_csv(path_a: str, path_b: str, metric: str = "macro_f1", client: str = "POOLED", n: int = 1000,
                    seed: int = 42) -> Dict[str, Any]:
    """paired_bootstrap_diff on two predictions.csv files; units from make_units
    (patients where available, images otherwise). client: one client name or POOLED."""
    import pandas as pd

    fns = M.default_metric_fns()
    if metric not in fns:
        raise ValueError(f"metric must be one of {sorted(fns)}, got {metric!r}")
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    pat = a["patient_id"] if "patient_id" in a.columns else None
    units, kinds = make_units(a["client"], a["image_id"], pat)
    if client != "POOLED":
        keep = (a["client"].astype(str) == client).to_numpy()
        if not keep.any():
            raise ValueError(f"client {client!r} not found; have {sorted(kinds)}")
        a, units = a[keep].reset_index(drop=True), units[keep]
        b = b[(b["client"].astype(str) == client).to_numpy()].reset_index(drop=True)
    out = paired_bootstrap_diff(a, b, units, fns[metric], n=n, seed=seed)
    out.update(metric=metric, client=client, unit_kinds={c: k for c, k in kinds.items() if client in ("POOLED", c)})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Paired bootstrap of a metric difference between two experiments.")
    p.add_argument("--a", required=True, help="predictions.csv of experiment A")
    p.add_argument("--b", required=True, help="predictions.csv of experiment B")
    p.add_argument("--metric", default="macro_f1", choices=sorted(M.default_metric_fns()))
    p.add_argument("--client", default="POOLED", help="a client name or POOLED (default)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    print(json.dumps(paired_from_csv(args.a, args.b, args.metric, args.client, args.n, args.seed), indent=2))
    print("\nIntervals reflect test-set sampling only, not training randomness.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
