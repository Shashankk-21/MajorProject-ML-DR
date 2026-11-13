"""
notebooks/stepzero/rank_sweep.py: LoRA rank sweep (step zero).

Ranks 4, 8, 16, 32 with alpha = 2 x rank, seed 42, the Experiment 1
hyperparameters on the pooled data (pretrained weights, real augmentation and
class weights), all through train_centralized.run_centralized_training. Schedule
proxy: max_epochs=4, warmup_epochs=0.2, patience=99 (no early stopping). Per rank
the BEST epoch by val macro-F1 is used, with that epoch's pooled QWK; the rank is
then picked by stepzero_lib.select_rank. Nothing is written to base.yaml: the two
lines to paste are printed.

Run from the repo root:
    python notebooks/stepzero/rank_sweep.py --manifest-root P --checkpoint-root P [--ranks 4 8 16 32] [--wandb]

Outputs under {checkpoint-root}/stepzero/: rank{r}.json (per-epoch records and the
result; written after every epoch), rank{r}/ (checkpoints), rank_sweep_summary.json.
Re-running resumes an interrupted rank and skips a finished one.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stepzero_lib as Z  # noqa: E402

logger = logging.getLogger("rank_sweep")

DEFAULT_RANKS = (4, 8, 16, 32)
MAX_EPOCHS = 4
WARMUP_EPOCHS = 0.2
PATIENCE = 99
SEED = 42  # the sweep always uses the decided seed, whatever run-seed says
_EPOCH_KEYS = ("train_loss", "val_loss", "val_macro_f1", "lr_lora", "lr_head")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA rank sweep (step zero).")
    p.add_argument("--manifest-root", default=os.environ.get("DR_MANIFEST_ROOT"), help="folder with {client}_manifest.csv")
    p.add_argument("--checkpoint-root", default=os.environ.get("DR_CHECKPOINT_ROOT"), help="outputs go to {root}/stepzero/")
    p.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS), help="default: 4 8 16 32")
    p.add_argument("--wandb", action="store_true", help="log each rank to W&B with the 'diagnostic' tag (default off)")
    p.add_argument("--config-dir", default=None, help="alternative configs folder (testing)")
    args = p.parse_args(argv)
    for name in ("manifest_root", "checkpoint_root"):
        if not getattr(args, name):
            p.error(f"--{name.replace('_', '-')} is required (or DR_{name.upper()})")
    if len(set(args.ranks)) != len(args.ranks) or any(r < 1 for r in args.ranks):
        p.error("--ranks must be distinct positive integers")
    return args


def _load_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    from configs import loader
    return loader.load_experiment_config(
        1, allow_unset=True, manifest_root=args.manifest_root, checkpoint_root=args.checkpoint_root,
        env={}, config_dir=args.config_dir)


def build_kwargs(cfg: Dict[str, Any], rank: int, rank_dir: Path) -> Dict[str, Any]:
    """Experiment 1 kwargs with this rank/alpha, then the sweep's schedule proxy."""
    from configs import loader
    cfg = dict(cfg)
    cfg["lora-rank"], cfg["lora-alpha"] = rank, 2 * rank
    kwargs = loader.to_centralized_kwargs(cfg)
    kwargs.update(checkpoint_dir=str(rank_dir), max_epochs=MAX_EPOCHS, warmup_epochs=WARMUP_EPOCHS, patience=PATIENCE, seed=SEED)
    return kwargs


def _params_of(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Everything that must match for a stored rank state to be reusable."""
    skip = {"client_manifest_paths", "checkpoint_dir"}
    return {k: v for k, v in sorted(kwargs.items()) if k not in skip}


def _float_or_none(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def best_epoch(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The epoch with the highest val macro-F1 (earliest on ties) that has a QWK."""
    rows = [(int(e), r) for e, r in state["epochs"].items()
            if r.get("val_macro_f1") is not None and r.get("qwk") is not None]
    if not rows:
        return None
    epoch, rec = max(rows, key=lambda er: (er[1]["val_macro_f1"], -er[0]))
    return {"epoch": epoch, **rec}


def _check_best_recorded(state: Dict[str, Any], rank: int, state_path: Path, rank_dir: Path) -> None:
    """The trainer's best val macro-F1 must be among the recorded epochs; otherwise a record was lost
    (crash between the checkpoint write and the callback) and the rank's numbers cannot be trusted."""
    be, got = best_epoch(state), (state.get("result") or {}).get("best_val_macro_f1")
    if be is not None and got is not None and float(got) > be["val_macro_f1"] + 1e-9:
        raise SystemExit(
            f"rank {rank}: the trainer's best val macro-F1 ({float(got):.6f}) is higher than the best recorded epoch "
            f"({be['val_macro_f1']:.6f}): an epoch record was lost (crash between checkpoint and callback). "
            f"Delete {state_path} and {rank_dir} and rerun this rank.")


def run_rank(rank: int, cfg: Dict[str, Any], out_dir: Path, use_wandb: bool) -> Dict[str, Any]:
    from training.train_centralized import run_centralized_training
    from utils.experiment_tracker import quadratic_weighted_kappa

    rank_dir, state_path = out_dir / f"rank{rank}", out_dir / f"rank{rank}.json"
    kwargs = build_kwargs(cfg, rank, rank_dir)
    params = _params_of(kwargs)
    state = Z.read_json(state_path)
    if state is None:
        state = {"rank": rank, "alpha": 2 * rank, "params": params, "epochs": {}, "result": None}
    elif state.get("params") != params:
        raise SystemExit(
            f"rank {rank}: {state_path.name} was written with different settings; delete {state_path} and the folder "
            f"{rank_dir} to rerun it (refusing to mix runs).\n  stored: {state.get('params')}\n  now:    {params}")
    if state["result"] is not None:
        if best_epoch(state) is None:
            raise SystemExit(f"rank {rank}: finished but {state_path.name} has no per-epoch records with a QWK; "
                             f"delete {state_path} and {rank_dir} and rerun this rank.")
        _check_best_recorded(state, rank, state_path, rank_dir)
        print(f"rank {rank}: already finished, skipping (epochs {sorted(map(int, state['epochs']))})")
        return state

    tracker = tracker_cb = None
    if use_wandb:
        try:
            from utils.experiment_tracker import make_epoch_callback, start_centralized
            tracker, run_id = start_centralized(
                {"experiment-id": f"stepzero-rank{rank}", "algorithm": "centralized", "diagnostic": True,
                 "lora-rank": rank, "lora-alpha": 2 * rank, **{k: v for k, v in params.items()}}, rank_dir)
            tracker_cb = make_epoch_callback(tracker)
            kwargs["wandb_run_id"] = run_id
        except Exception as exc:  # logging must not stop a diagnostic
            print(f"rank {rank}: W&B unavailable ({exc!r}); continuing without it")
            tracker = tracker_cb = None

    clock = {"prev": time.perf_counter(), "first": True}

    def on_epoch_end(epoch: int, metrics: Dict[str, Any]) -> None:
        now = time.perf_counter()
        seconds, clock["prev"] = now - clock["prev"], now
        qwk = None
        if metrics.get("val_confusion_matrix") is not None:
            try:
                qwk = float(quadratic_weighted_kappa(metrics["val_confusion_matrix"]))
            except Exception:
                logger.exception("rank %d epoch %s: QWK could not be computed", rank, epoch)
        rec = {k: _float_or_none(metrics.get(k)) for k in _EPOCH_KEYS}
        rec.update(qwk=qwk, seconds=seconds, includes_setup=clock["first"])
        clock["first"] = False
        state["epochs"][str(int(epoch))] = rec
        Z.atomic_write_json(state_path, state)
        print(f"rank {rank} epoch {epoch}: train_loss {rec['train_loss']:.4f} val_f1 {rec['val_macro_f1']:.4f} "
              f"qwk {qwk if qwk is None else round(qwk, 4)} ({seconds:.0f}s)", flush=True)
        if tracker_cb is not None:
            try:
                tracker_cb(epoch, metrics)
            except Exception:
                logger.exception("W&B epoch logging failed; ignoring")

    kwargs["on_epoch_end"] = on_epoch_end
    print(f"\n=== rank {rank} (alpha {2 * rank}) -> {rank_dir} ===", flush=True)
    try:
        result = run_centralized_training(**kwargs)
    except BaseException as exc:
        if tracker is not None:
            tracker.finish(None, exc)
        raise
    if tracker is not None:
        tracker.finish(result, None)
    state["result"] = {k: result[k] for k in ("best_val_macro_f1", "final_epoch", "best_checkpoint_path") if k in result}
    Z.atomic_write_json(state_path, state)
    if best_epoch(state) is None:
        raise SystemExit(
            f"rank {rank}: no per-epoch records with a QWK were captured (the run may have finished in an earlier "
            f"session without them). Delete {state_path} and {rank_dir} and rerun this rank.")
    _check_best_recorded(state, rank, state_path, rank_dir)
    _free_gpu()
    return state


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def print_report(states: Dict[int, Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    ranks = sorted(states)
    rows = []
    for r in ranks:
        for e in sorted(map(int, states[r]["epochs"])):
            d = states[r]["epochs"][str(e)]
            f = lambda v, n=4: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
            rows.append([r, e, f(d["train_loss"]), f(d["val_loss"]), f(d["val_macro_f1"]), f(d["qwk"]),
                         f(d["lr_lora"], 6), f(d["lr_head"], 6), f"{d['seconds']:.0f}" + ("*" if d["includes_setup"] else "")])
    print("\nPer epoch (* = includes model/data setup):")
    print(Z.format_table(["rank", "epoch", "train_loss", "val_loss", "val_f1", "qwk", "lr_lora", "lr_head", "sec"], rows))

    results, summary_rows = {}, []
    for r in ranks:
        b = best_epoch(states[r])
        results[r] = {"val_macro_f1": b["val_macro_f1"], "qwk": b["qwk"]}
        clean = [d["seconds"] for d in states[r]["epochs"].values() if not d["includes_setup"]]
        summary_rows.append([r, 2 * r, b["epoch"], f"{b['val_macro_f1']:.4f}", f"{b['qwk']:.4f}",
                             f"{sorted(clean)[len(clean) // 2]:.0f}" if clean else "-"])
    print("\nBest epoch per rank (by val macro-F1):")
    print(Z.format_table(["rank", "alpha", "best_epoch", "val_macro_f1", "qwk", "sec/epoch"], summary_rows))

    sel = Z.select_rank(results)
    print(f"\nSelection: {sel['reason']}")
    if sorted(ranks) != list(DEFAULT_RANKS):
        print(f"WARNING: partial sweep (ranks {ranks}); do not paste these values until all of "
              f"{list(DEFAULT_RANKS)} have run.")
    if sel["manual_review"]:
        print("\n*** MANUAL REVIEW NEEDED: the chosen rank's QWK is more than 0.02 below the best QWK. "
              "Look at the table above and decide yourself before pasting. ***")
    print("\nPaste into configs/base.yaml (this script never edits it):")
    print(f"lora-rank: {sel['chosen']}")
    print(f"lora-alpha: {2 * sel['chosen']}")
    Z.atomic_write_json(out_dir / "rank_sweep_summary.json", {"results": {str(k): v for k, v in results.items()}, "selection": sel})
    return sel


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = _load_cfg(args)
    out_dir = Path(args.checkpoint_root) / "stepzero"
    out_dir.mkdir(parents=True, exist_ok=True)
    states = {r: run_rank(r, cfg, out_dir, args.wandb) for r in args.ranks}
    print_report(states, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())