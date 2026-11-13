"""
evaluation/evaluate.py: the final test-set evaluation (one run per experiment).

    python -m evaluation.evaluate --experiment-id exp4-fedprox-eps50 \
        --checkpoint-root P --manifest-root P --out P [--device cuda] [--batch-size 64] \
        [--num-workers 8] [--force]

Loads the BEST checkpoint ({root}/{id}/best.pt, or {root}/exp1-centralized/
best_centralized.pt for Experiment 1) with utils.checkpoint.load_checkpoint into a model
rebuilt from the checkpoint's OWN run_config (PEFT-only checkpoints: the backbone comes
from the pretrained weights). Runs each client's test rows (split == "test", the project's
eval transforms, a SequentialSampler) once, then writes under {out}/{id}/:
predictions.csv, per_client.csv, results.json, summary.md and, last,
test_eval_done.json (the test-once guard: a second run refuses unless --force).

The test split is never used for any tuning. Bootstrap intervals reflect test-set
sampling only, not training randomness. Heavy imports (torch, the project modules) are
lazy, so the rest of this module works without them.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from evaluation import bootstrap as B
from evaluation import metrics as M

EVAL_VERSION = "1.0"
BOOTSTRAP_SEED = 42
BOOTSTRAP_ALPHA = 0.05
NUM_CLASSES = M.NUM_CLASSES
CLIENTS_FALLBACK = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")
CLASS_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")
DONE_NAME = "test_eval_done.json"
_ID_RE = re.compile(r"^exp([1-8])-(centralized|fedavg|fedprox)(?:-eps([0-9]+(?:\.[0-9]+)?))?$")
_FL_KEYS = {"lora_rank": "lora-rank", "lora_alpha": "lora-alpha", "img_size": "img-size", "num_classes": "num-classes"}
_CENTRAL_KEYS = {k: k for k in _FL_KEYS}

logger = logging.getLogger("evaluation")


class AlreadyEvaluated(RuntimeError):
    """The test split was already evaluated for this experiment (use --force)."""


class ManifestMismatch(RuntimeError):
    """A manifest differs from the one the checkpoint was trained with."""


def client_names() -> Tuple[str, ...]:
    try:
        from configs.loader import CLIENTS  # same order as the server's partitions
        return tuple(CLIENTS)
    except Exception:
        return CLIENTS_FALLBACK


# ---------------------------------------------------------------------------
# Experiment id, paths, checkpoint contents
# ---------------------------------------------------------------------------

def parse_experiment_id(exp_id: str) -> Dict[str, Any]:
    """exp{N}-{algorithm}[-eps{value}] -> {"n","algorithm","dp","target_epsilon"} (configs.loader naming)."""
    m = _ID_RE.match(exp_id or "")
    if not m:
        raise ValueError(f"experiment id {exp_id!r} does not match exp{{N}}-{{centralized|fedavg|fedprox}}[-eps{{value}}]")
    n, algorithm, eps = int(m.group(1)), m.group(2), m.group(3)
    if (n == 1) != (algorithm == "centralized"):
        raise ValueError(f"{exp_id!r}: only experiment 1 is centralized")
    if eps is not None and (algorithm != "fedprox" or float(eps) <= 0):
        raise ValueError(f"{exp_id!r}: a DP experiment is fedprox with a positive epsilon")
    if n >= 4 and eps is None:
        raise ValueError(f"{exp_id!r}: experiments 4-8 are DP and need -eps{{value}}")
    if n in (2, 3) and eps is not None:
        raise ValueError(f"{exp_id!r}: experiments 2 and 3 are non-DP")
    return {"n": n, "algorithm": algorithm, "dp": eps is not None, "target_epsilon": None if eps is None else float(eps)}


def checkpoint_path(checkpoint_root: Any, exp_id: str) -> Path:
    name = "best_centralized.pt" if parse_experiment_id(exp_id)["algorithm"] == "centralized" else "best.pt"
    return Path(checkpoint_root) / exp_id / name


def file_md5(path: Any) -> str:
    digest = hashlib.md5()  # an identity tag for the guard file, not a security measure
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_pos_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def model_params_from_run_config(rc: Mapping[str, Any]) -> Dict[str, Any]:
    """lora_rank, lora_alpha, img_size, num_classes (+ style) from a checkpoint's run_config.
    FL checkpoints use hyphenated keys, Experiment 1 underscored ones."""
    for style, mapping in (("fl", _FL_KEYS), ("centralized", _CENTRAL_KEYS)):
        if all(src in rc for src in mapping.values()):
            params = {dst: rc[src] for dst, src in mapping.items()}
            bad = [k for k, v in params.items() if not _is_pos_int(v)]
            if bad:
                raise ValueError(f"run_config values must be positive ints: {bad} -> {[params[k] for k in bad]}")
            if params["num_classes"] != NUM_CLASSES:
                raise ValueError(f"run_config num_classes is {params['num_classes']}, the evaluation needs {NUM_CLASSES}")
            if style == "fl" and rc.get("pretrained", True) is not True:
                raise ValueError("run_config says pretrained=False; PEFT-only checkpoints need the pretrained backbone")
            return {**params, "style": style}
    raise ValueError(f"run_config has neither the FL keys {list(_FL_KEYS.values())} nor the centralized keys "
                     f"{list(_CENTRAL_KEYS.values())}; keys present: {sorted(rc)}")


def stored_selection(extra: Mapping[str, Any], style: str) -> Dict[str, Any]:
    """The epoch/round and validation macro-F1 stored INSIDE the best checkpoint."""
    try:
        if style == "fl":
            out = {"kind": "round", "number": int(extra["round"]), "val_macro_f1": float(extra["selection-metric"]),
                   "metric": str(extra.get("selection-metric-name", "macro-f1-pooled")),
                   "recorded_best_round": extra.get("best-round"),
                   "recorded_best_metric": extra.get("best-selection-metric")}
        else:
            out = {"kind": "epoch", "number": int(extra["epoch"]), "val_macro_f1": float(extra["best_val_macro_f1"]),
                   "metric": "val macro-F1 (compute_selection_metric)", "recorded_best_round": None,
                   "recorded_best_metric": None}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"best checkpoint's extra lacks the selection fields ({exc!r}); keys: {sorted(extra)}") from exc
    if not math.isfinite(out["val_macro_f1"]):
        raise ValueError("stored validation macro-F1 is not finite")
    return out


def read_checkpoint_meta(path: Any) -> Dict[str, Any]:
    """Plain read of a checkpoint file (what load_checkpoint itself does first); for run_config,
    extra and privacy info only, never for weights."""
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=False)


@contextlib.contextmanager
def _quiet(name: str) -> Iterator[None]:
    lg = logging.getLogger(name)
    old = lg.level
    lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        lg.setLevel(old)


def load_model(path: Any, device: Any) -> Tuple[Any, Dict[str, Any]]:
    """Rebuild the model from the checkpoint's own run_config and load the PEFT weights with
    utils.checkpoint.load_checkpoint (structural validation, expected_config)."""
    from models.coatnet_lora import create_coatnet_lora_model
    from utils.checkpoint import load_checkpoint

    meta = read_checkpoint_meta(path)
    rc = meta.get("run_config")
    if not isinstance(rc, Mapping):
        raise ValueError(f"{path}: the checkpoint has no run_config; cannot rebuild the model")
    params = model_params_from_run_config(rc)
    model = create_coatnet_lora_model(lora_rank=params["lora_rank"], lora_alpha=params["lora_alpha"],
                                      num_classes=params["num_classes"], img_size=params["img_size"], pretrained=True)
    expected = {src: rc[src] for src in (_FL_KEYS if params["style"] == "fl" else _CENTRAL_KEYS).values()}
    # Experiment 1 checkpoints also carry optimizer/scheduler state, which is deliberately not restored here.
    with _quiet("utils.checkpoint"):
        info = load_checkpoint(str(path), model=model, expected_config=expected, map_location="cpu")
    model.to(device)
    model.eval()
    info["style"] = params["style"]
    info["model_params"] = params
    return model, info


# ---------------------------------------------------------------------------
# Test-once guard
# ---------------------------------------------------------------------------

def read_done(out_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(out_dir) / DONE_NAME
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def check_guard(out_dir: Path, force: bool) -> Tuple[int, List[Dict[str, Any]]]:
    """(force_count, previous_runs) for the run about to start. Refuses (AlreadyEvaluated) if
    the test split was already evaluated and force is off; with force, logs loudly."""
    done = read_done(out_dir)
    if done is None:
        return 0, []
    if not force:
        raise AlreadyEvaluated(
            f"the test split was already evaluated for this experiment ({Path(out_dir) / DONE_NAME}, "
            f"{done.get('timestamp_utc')}); refusing to evaluate again. The test split must not be reused to tune "
            f"anything. Use --force only to regenerate outputs of the SAME checkpoint; every forced run is counted.")
    count = int(done.get("force_count", 0)) + 1
    logger.warning("*" * 78)
    logger.warning("*** --force: EVALUATING THE TEST SPLIT AGAIN (forced run #%d) for %s ***", count, out_dir)
    logger.warning("*** previous evaluation: %s, checkpoint md5 %s ***", done.get("timestamp_utc"), done.get("checkpoint_md5"))
    logger.warning("*" * 78)
    previous = list(done.get("previous_runs", [])) + [
        {"timestamp_utc": done.get("timestamp_utc"), "checkpoint_md5": done.get("checkpoint_md5"), "version": done.get("version")}]
    return count, previous


def write_done(out_dir: Path, *, exp_id: str, ckpt_path: Path, md5: str, force_count: int,
               previous: List[Dict[str, Any]]) -> Dict[str, Any]:
    rec = {"version": EVAL_VERSION, "experiment_id": exp_id, "checkpoint_path": str(ckpt_path), "checkpoint_md5": md5,
           "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "force_count": int(force_count), "previous_runs": previous}
    _atomic_write(Path(out_dir) / DONE_NAME, json.dumps(rec, indent=2, sort_keys=True))
    return rec


# ---------------------------------------------------------------------------
# Manifests and inference
# ---------------------------------------------------------------------------

def manifest_paths(manifest_root: Any) -> Dict[str, Path]:
    return {c: Path(manifest_root) / f"{c}_manifest.csv" for c in client_names()}


def check_manifests(paths: Mapping[str, Path], run_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare each manifest's sha256 with the hashes stored in the checkpoint's run_config (the
    manifests the model was trained and selected with). status: match | mismatch | unavailable."""
    stored = run_config.get("manifest-sha256", run_config.get("manifest_sha256"))
    current = {c: _file_sha256(p) for c, p in paths.items()}
    if not isinstance(stored, Mapping):
        return {"status": "unavailable", "per_client": {c: {"current": h} for c, h in current.items()}}
    per = {c: {"current": h, "stored": stored.get(c), "match": stored.get(c) == h} for c, h in current.items()}
    return {"status": "match" if all(v["match"] for v in per.values()) else "mismatch", "per_client": per}


def run_inference(model: Any, paths: Mapping[str, Path], *, batch_size: int, num_workers: int, device: Any) -> Any:
    """One row per test image of every client: client, image_id, patient_id, label, pred, 5 logits,
    5 softmax probabilities. Uses DRDataset(split="test") with get_eval_transforms(), a
    SequentialSampler and local_trainer.run_validation_epoch (model.eval(), no_grad)."""
    import pandas as pd
    from torch.utils.data import DataLoader, SequentialSampler

    from preprocessing.augmentation import get_eval_transforms
    from preprocessing.dataset import DRDataset
    from training.local_trainer import run_validation_epoch

    need = ("image_id", "cached_image_path", "patient_id", "label", "split")
    frames = []
    for client, path in paths.items():
        df = pd.read_csv(path)
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{client}: manifest {path} lacks columns {missing}")
        dataset = DRDataset(df, split="test", transform=get_eval_transforms())
        loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), num_workers=num_workers)
        logger.info("%s: %d test images", client, len(dataset))
        _loss, logits, targets = run_validation_epoch(model, loader, device)
        data = dataset.data
        if logits.shape != (len(data), NUM_CLASSES) or not np.array_equal(targets, data["label"].to_numpy().astype(targets.dtype)):
            raise RuntimeError(f"{client}: inference output does not line up with the manifest's test rows")
        logits = np.asarray(logits, dtype=np.float64)
        probs = M.softmax(logits)
        out = pd.DataFrame({"client": client, "image_id": data["image_id"].astype(str).to_numpy(),
                            "patient_id": data["patient_id"].to_numpy(), "label": data["label"].to_numpy().astype(int),
                            "pred": logits.argmax(axis=1).astype(int)})
        for c in range(NUM_CLASSES):
            out[f"logit_{c}"] = logits[:, c]
        for c in range(NUM_CLASSES):
            out[f"prob_{c}"] = probs[:, c]
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics and bootstrap over predictions
# ---------------------------------------------------------------------------

def compute_results(pred: Any, *, n_boot: int = 1000, seed: int = BOOTSTRAP_SEED, alpha: float = BOOTSTRAP_ALPHA) -> Dict[str, Any]:
    """Per-client and pooled metrics with bootstrap intervals, plus the unweighted mean of the
    per-client macro-F1. pred: a DataFrame in the predictions.csv layout."""
    clients = list(dict.fromkeys(pred["client"].astype(str).tolist()))
    probs_all = np.column_stack([pred[f"prob_{c}"].to_numpy(dtype=np.float64) for c in range(NUM_CLASSES)])
    y_all, p_all = pred["label"].to_numpy().astype(np.int64), pred["pred"].to_numpy().astype(np.int64)
    pat = pred["patient_id"].to_numpy() if "patient_id" in pred.columns else None
    units, kinds = B.make_units(pred["client"], pred["image_id"], pat)
    fns = M.default_metric_fns()
    names = np.asarray(pred["client"].astype(str))
    groups: Dict[str, np.ndarray] = {c: names == c for c in clients}
    groups["POOLED"] = np.ones(len(pred), dtype=bool)
    metrics: Dict[str, Any] = {}
    boot: Dict[str, Any] = {}
    for g, mask in groups.items():
        metrics[g] = M.summarize(y_all[mask], p_all[mask], probs_all[mask])
        boot[g] = B.bootstrap_ci(y_all[mask], p_all[mask], probs_all[mask], units[mask], fns, n=n_boot, seed=seed, alpha=alpha)
    total = len(pred)
    return {
        "metrics": metrics, "bootstrap": boot,
        "bootstrap_settings": {"n_resamples": int(n_boot), "seed": int(seed), "alpha": alpha, "interval": "percentile"},
        "unit_kind": {**kinds, "POOLED": "patients where available, images otherwise (per client)"},
        "test_images": {g: int(m.sum()) for g, m in groups.items()},
        "test_share": {c: float(groups[c].sum() / total) for c in clients},
        "client_mean_macro_f1": M.macro_over_defined([metrics[c]["macro_f1"] for c in clients]),
        "clients": clients,
    }


# ---------------------------------------------------------------------------
# Privacy (DP runs)
# ---------------------------------------------------------------------------

def _privacy_block(extra: Mapping[str, Any], names: Sequence[str]) -> Optional[Dict[str, Any]]:
    priv = extra.get("privacy")
    if not isinstance(priv, Mapping) or not isinstance(priv.get("epsilon"), Mapping):
        return None

    def per_client(key: str) -> Dict[str, float]:
        raw = priv.get(key) or {}
        return {names[int(pid)] if 0 <= int(pid) < len(names) else f"client{pid}": float(v) for pid, v in raw.items()}

    return {"round": extra.get("round"), "epsilon": per_client("epsilon"), "noise_multiplier": per_client("noise-multiplier"),
            "steps_total": per_client("steps-total"), "max_epsilon": _num(priv.get("max-epsilon")),
            "delta": _num(priv.get("delta")), "accountant": priv.get("accountant")}


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def read_privacy(ckpt_dir: Path, best_extra: Mapping[str, Any], run_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Each client's epsilon at the BEST round (the privacy cost of the released model) and at the
    FINAL round. Both come from the 'privacy' entry the server stores in each round checkpoint's
    extra (federated.server.privacy_report): best.pt for the best round, latest.pt (else the
    highest round_NNN.pt) for the final round."""
    names = client_names()
    best = _privacy_block(best_extra, names)
    latest_path = Path(ckpt_dir) / "latest.pt"
    if not latest_path.is_file():
        rounds = sorted(Path(ckpt_dir).glob("round_*.pt"))
        latest_path = rounds[-1] if rounds else latest_path
    final = None
    if latest_path.is_file():
        final = _privacy_block(read_checkpoint_meta(latest_path).get("extra", {}), names)
    problems = []
    if best is None:
        problems.append("best checkpoint has no privacy info")
    if final is None:
        problems.append("no final-round privacy info found")
    return {"best_round": best, "final_round": final, "target_epsilon": _num(run_config.get("target-epsilon")),
            "delta": _num(run_config.get("delta")), "accountant": run_config.get("accountant"), "problems": problems}


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


def per_client_rows(res: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for g in list(res["clients"]) + ["POOLED"]:
        m, b = res["metrics"][g], res["bootstrap"][g]
        row: Dict[str, Any] = {"client": g, "n": m["n"], "share_of_pooled": res["test_images"][g] / res["test_images"]["POOLED"],
                               "unit": res["unit_kind"][g], "accuracy": m["accuracy"]}
        for name in ("macro_f1", "qwk", "macro_auc"):
            row.update({name: m[name], f"{name}_low": b[name]["low"], f"{name}_high": b[name]["high"]})
        for c in range(NUM_CLASSES):
            row.update({f"support_c{c}": int(m["support"][c]), f"precision_c{c}": m["precision"][c],
                        f"recall_c{c}": m["recall"][c], f"f1_c{c}": m["f1"][c], f"auc_c{c}": m["auc"][c]})
        rows.append(row)
    rows.append({"client": "CLIENT_MEAN", "n": res["test_images"]["POOLED"], "unit": "unweighted mean of per-client macro-F1",
                 "macro_f1": res["client_mean_macro_f1"]})
    return rows


def _f(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None or not math.isfinite(float(x)) else f"{float(x):.{nd}f}"


def _ci(est: float, lo: float, hi: float) -> str:
    return f"{_f(est)} [{_f(lo)}, {_f(hi)}]"


def render_summary(results: Mapping[str, Any]) -> str:
    """summary.md: per-client table is the headline; pooled is EyePACS-dominated."""
    res = results["evaluation"]
    ck, boot = results["checkpoint"], res["bootstrap_settings"]
    clients = res["clients"]
    L: List[str] = [f"# Final test evaluation: {results['experiment_id']}", ""]
    L += [f"- Checkpoint: `{ck['path']}` (md5 {ck['md5']})",
          f"- Selected {ck['stored']['kind']}: {ck['stored']['number']}; validation macro-F1 stored inside the checkpoint: "
          f"{_f(ck['stored']['val_macro_f1'], 4)} ({ck['stored']['metric']}).",
          f"- Intervals: {int((1 - boot['alpha']) * 100)}% percentile bootstrap, {boot['n_resamples']} resamples, seed {boot['seed']}. "
          f"They reflect test-set sampling only, NOT training randomness (training seed, augmentation, DP noise).",
          "- Resampling unit per client: " + ", ".join(f"{c} {res['unit_kind'][c]}" for c in clients) + ".",
          "- Undefined values are NaN (shown n/a): a class with no test images has no recall/F1; AUC needs both positives and "
          "negatives; macro averages are over the DEFINED classes only. The training-time selection metric counted absent "
          "classes as 0 (zero_division=0); this final evaluation does not.", ""]

    L += ["## Headline: per client", "",
          "| client | test images | macro-F1 [95% CI] | QWK [95% CI] | macro AUC [95% CI] | accuracy | **Mild recall** (n) | **Moderate recall** (n) |",
          "|---|---|---|---|---|---|---|---|"]
    for g in clients + ["POOLED"]:
        m, b = res["metrics"][g], res["bootstrap"][g]
        L.append(f"| {g} | {m['n']} | {_ci(m['macro_f1'], b['macro_f1']['low'], b['macro_f1']['high'])} | "
                 f"{_ci(m['qwk'], b['qwk']['low'], b['qwk']['high'])} | {_ci(m['macro_auc'], b['macro_auc']['low'], b['macro_auc']['high'])} | "
                 f"{_f(m['accuracy'])} | **{_f(m['recall'][1])}** ({int(m['support'][1])}) | **{_f(m['recall'][2])}** ({int(m['support'][2])}) |")
    L += ["", f"Unweighted mean of the per-client macro-F1: **{_f(res['client_mean_macro_f1'])}**.", ""]
    share = res["test_share"]
    big = max(share, key=share.get)
    L += [f"The POOLED row is dominated by {big}, which is {100 * share[big]:.1f}% of the pooled test images "
          f"({res['test_images'][big]} of {res['test_images']['POOLED']}); read the per-client rows as the headline.", ""]

    L += ["## Per-class detail (support = test images of that class)", ""]
    for g in clients + ["POOLED"]:
        m = res["metrics"][g]
        L += [f"### {g}", "", "| class | support | precision | recall | F1 | AUC (one-vs-rest) |", "|---|---|---|---|---|---|"]
        for c in range(NUM_CLASSES):
            mark = "**" if c in (1, 2) else ""
            L.append(f"| {mark}{c} {CLASS_NAMES[c]}{mark} | {int(m['support'][c])} | {_f(m['precision'][c])} | "
                     f"{_f(m['recall'][c])} | {_f(m['f1'][c])} | {_f(m['auc'][c])} |")
        L.append("")
    L += ["## Confusion matrices (rows = true class, columns = predicted)", ""]
    for g in clients + ["POOLED"]:
        L += [f"### {g}", "", "| true \\ pred | " + " | ".join(str(c) for c in range(NUM_CLASSES)) + " |", "|---" * (NUM_CLASSES + 1) + "|"]
        for r in range(NUM_CLASSES):
            L.append(f"| {r} | " + " | ".join(str(int(v)) for v in res["metrics"][g]["confusion"][r]) + " |")
        L.append("")

    priv = results.get("privacy")
    if priv:
        L += ["## Privacy cost (DP run)", "",
              f"Target epsilon {_f(priv.get('target_epsilon'), 2)}, delta {priv.get('delta')}, accountant {priv.get('accountant')}. "
              "Epsilon is per client (each client's own accountant).", ""]
        best, final = priv.get("best_round"), priv.get("final_round")
        L += ["| client | epsilon at the best round (the released model) | epsilon at the final round |", "|---|---|---|"]
        for c in clients:
            L.append(f"| {c} | {_f((best or {}).get('epsilon', {}).get(c), 3)} | {_f((final or {}).get('epsilon', {}).get(c), 3)} |")
        L.append(f"| max | {_f((best or {}).get('max_epsilon'), 3)} | {_f((final or {}).get('max_epsilon'), 3)} |")
        L += ["", f"Best round: {(best or {}).get('round')}; final round: {(final or {}).get('round')}."]
        if priv.get("problems"):
            L += ["", "Problems: " + "; ".join(priv["problems"])]
        L.append("")
    mc = results.get("manifest_check", {})
    L += ["## Provenance", "", f"- Manifest hashes vs the checkpoint's run_config: {mc.get('status')}.",
          f"- Test-once guard: forced runs so far {results['guard']['force_count']}."]
    return "\n".join(L) + "\n"


def write_outputs(out_dir: Path, pred: Any, results: Dict[str, Any]) -> None:
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / ".predictions.csv.tmp"
    pred.to_csv(tmp, index=False, float_format="%.9g")
    os.replace(tmp, out_dir / "predictions.csv")
    tmp = out_dir / ".per_client.csv.tmp"
    pd.DataFrame(per_client_rows(results["evaluation"])).to_csv(tmp, index=False, float_format="%.6g")
    os.replace(tmp, out_dir / "per_client.csv")
    _atomic_write(out_dir / "results.json", json.dumps(_jsonable(results), indent=2, sort_keys=True))
    _atomic_write(out_dir / "summary.md", render_summary(results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m evaluation.evaluate", description="Final test-set evaluation of one experiment.")
    p.add_argument("--experiment-id", required=True, help="e.g. exp4-fedprox-eps50 (configs.loader naming)")
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--manifest-root", required=True, help="folder with {client}_manifest.csv")
    p.add_argument("--out", required=True, help="outputs go to {out}/{experiment-id}/")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="evaluate again although the test split was already used (counted, logged loudly)")
    p.add_argument("--bootstrap-n", type=int, default=1000, help="resamples (default 1000; lower only for quick checks)")
    p.add_argument("--skip-manifest-check", action="store_true", help="do not refuse when a manifest differs from the one in the checkpoint")
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.bootstrap_n < 1:
        p.error("--batch-size and --bootstrap-n must be >= 1 and --num-workers >= 0")
    return args


def _resolve_device(name: Optional[str]) -> Any:
    import torch
    if name is None:
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested but CUDA is not available")
    return torch.device(name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    exp_id = args.experiment_id
    try:
        info = parse_experiment_id(exp_id)
        out_dir = Path(args.out) / exp_id
        ckpt = checkpoint_path(args.checkpoint_root, exp_id)
        if not ckpt.is_file():
            raise FileNotFoundError(f"best checkpoint not found: {ckpt}")
        paths = manifest_paths(args.manifest_root)
        absent = [str(p) for p in paths.values() if not p.is_file()]
        if absent:
            raise FileNotFoundError(f"manifests not found: {absent}")
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        force_count, previous = check_guard(ckpt.parent, args.force)
    except AlreadyEvaluated as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    try:
        device = _resolve_device(args.device)
        md5 = file_md5(ckpt)
        model, ck = load_model(ckpt, device)
        rc, extra = ck.get("run_config") or {}, ck.get("extra") or {}
        if ck["style"] == "fl" and rc.get("experiment-id") != exp_id:
            raise ValueError(f"checkpoint {ckpt} belongs to {rc.get('experiment-id')!r}, not {exp_id!r}")
        if ck["style"] == "fl" and bool(rc.get("dp")) != info["dp"]:
            raise ValueError(f"experiment id says dp={info['dp']} but the checkpoint's run_config says dp={rc.get('dp')}")
        stored = stored_selection(extra, ck["style"])
        mcheck = check_manifests(paths, rc)
        if mcheck["status"] == "mismatch" and not args.skip_manifest_check:
            bad = [c for c, v in mcheck["per_client"].items() if not v["match"]]
            raise ManifestMismatch(f"manifests differ from the ones the checkpoint was trained with: {bad}; "
                                   f"the test split may have changed. Use --skip-manifest-check only if you are sure.")
        if mcheck["status"] != "match":
            logger.warning("manifest hash check: %s", mcheck["status"])
        pred = run_inference(model, paths, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
        evaluation = compute_results(pred, n_boot=args.bootstrap_n)
        privacy = read_privacy(ckpt.parent, extra, rc) if info["dp"] else None
    except (ValueError, FileNotFoundError, ManifestMismatch) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results = {
        "experiment_id": exp_id, "version": EVAL_VERSION, "dp": info["dp"], "algorithm": info["algorithm"],
        "checkpoint": {"path": str(ckpt), "md5": md5, "stored": stored, "model_params": ck["model_params"],
                       "run_config": {k: v for k, v in rc.items() if k not in ("manifest-sha256", "manifest_sha256")},
                       "wandb_run_id": ck.get("wandb_run_id")},
        "manifest_check": mcheck, "evaluation": evaluation, "privacy": privacy,
        "guard": {"force_count": force_count, "previous_runs": previous},
        "notes": ["Intervals reflect test-set sampling only, not training randomness.",
                  "Training-time selection counted absent classes as 0 (zero_division=0); this evaluation uses NaN for "
                  "undefined classes and macro-averages over the defined ones."],
    }
    write_outputs(out_dir, pred, results)
    write_done(ckpt.parent, exp_id=exp_id, ckpt_path=ckpt, md5=md5, force_count=force_count, previous=previous)
    print(f"wrote {out_dir}/ (predictions.csv, per_client.csv, results.json, summary.md); test-once guard: {ckpt.parent / DONE_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
