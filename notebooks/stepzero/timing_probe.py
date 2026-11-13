"""
notebooks/stepzero/timing_probe.py: step time and GPU memory per physical batch size.

For physical batch sizes 16, 32, 48, 64 on the GPU, for non-DP training and for DP
(functorch, logical batch 64, the project's Opacus path with BatchMemoryManager),
it times 20 steps after 3 warmup steps through the project's own
local_trainer.run_training_epoch and records seconds per step (one step = one
physical micro-batch, optimizer work included) and the peak
torch.cuda.max_memory_allocated. CUDA out-of-memory is caught and reported as OOM.
From the step times and each client's real train size it extrapolates seconds per
epoch (and the pooled set for non-DP) and the DP time of one EyePACS round
(local-epochs x one epoch).

Approximations (stated in the printed notes): synthetic GPU-resident batches, so
data loading and augmentation are NOT included (compare with the measured sec/epoch
from rank_sweep.py); pretrained=False (weights do not change step time); DP noise
multiplier 1.0 and clip 1.0 (irrelevant to timing); rank/alpha from --lora-rank/--lora-alpha
(default 8/16) because this probe runs before the rank is chosen; non-DP accumulation is
ceil(64 / batch), so 48 is timed with 2 (effective 96), which the real config cannot use.

Run from the repo root:
    python notebooks/stepzero/timing_probe.py --manifest-root P --checkpoint-root P [--batch-sizes 16 32 48 64]

Outputs: {root}/stepzero/timing_probe.json (stored results; re-running skips finished
configurations, --force redoes them) and the printed tables.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import stepzero_lib as Z  # noqa: E402

logger = logging.getLogger("timing_probe")

DEFAULT_BATCH_SIZES = (16, 32, 48, 64)
MODES = ("nondp", "dp")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-time and memory probe per physical batch size.")
    p.add_argument("--manifest-root", default=os.environ.get("DR_MANIFEST_ROOT"), help="folder with {client}_manifest.csv")
    p.add_argument("--checkpoint-root", default=os.environ.get("DR_CHECKPOINT_ROOT"), help="outputs go to {root}/stepzero/")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    p.add_argument("--steps", type=int, default=20, help="timed steps (default 20)")
    p.add_argument("--warmup-steps", type=int, default=3, help="untimed warmup steps (default 3)")
    p.add_argument("--lora-rank", type=int, default=8, help="timing only; the rank is not chosen yet (default 8)")
    p.add_argument("--lora-alpha", type=int, default=None, help="default 2 x rank")
    p.add_argument("--force", action="store_true", help="redo stored configurations")
    p.add_argument("--config-dir", default=None, help="alternative configs folder (testing)")
    args = p.parse_args(argv)
    for name in ("manifest_root", "checkpoint_root"):
        if not getattr(args, name):
            p.error(f"--{name.replace('_', '-')} is required (or DR_{name.upper()})")
    if args.steps < 1 or args.warmup_steps < 0 or args.lora_rank < 1 or any(b < 1 for b in args.batch_sizes):
        p.error("--steps >= 1, --warmup-steps >= 0, --lora-rank >= 1 and positive --batch-sizes are required")
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_rank
    return args


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _stamped(source: Iterable[Any], stamps: List[float], needed: int, sync: Any) -> Iterator[Any]:
    """Yield from `source`, recording a synchronized timestamp at every yield and
    stopping after `needed` of them; the gap between two stamps is one step."""
    for item in source:
        sync()
        stamps.append(time.perf_counter())
        yield item
        if len(stamps) >= needed:
            return


def measure_config(mode: str, physical: int, *, cfg: Dict[str, Any], dp_cfg: Dict[str, Any], device: Any,
                   steps: int, warmup: int, rank: int, alpha: int) -> Dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset

    from models.coatnet_lora import create_coatnet_lora_model, freeze_bn_eval, split_trainable_parameters
    from training.local_trainer import run_training_epoch

    img, k = int(cfg["img-size"]), int(cfg["num-classes"])
    logical = int(dp_cfg["dp-logical-batch-size"]) if mode == "dp" else int(cfg["effective-batch-size"])
    needed = warmup + steps + 1
    stamps: List[float] = []
    sync = torch.cuda.synchronize

    _free_gpu()
    torch.cuda.reset_peak_memory_stats()
    model = create_coatnet_lora_model(lora_rank=rank, lora_alpha=alpha, num_classes=k, img_size=img, pretrained=False)
    freeze_bn_eval(model)
    model.to(device)
    lora_params, head_params = split_trainable_parameters(model)
    optimizer = AdamW([{"params": lora_params, "lr": float(cfg["lr-lora-peak"])},
                       {"params": head_params, "lr": float(cfg["lr-head-peak"])}],
                      weight_decay=float(cfg["weight-decay"]))
    class_weights = torch.ones(k, device=device)
    common = dict(class_weights=class_weights, label_smoothing=float(cfg["label-smoothing"]))
    accumulation = 1

    if mode == "nondp":
        accumulation = max(1, math.ceil(logical / physical))
        batch = (torch.randn(physical, 3, img, img, device=device), torch.randint(0, k, (physical,), device=device))
        run_training_epoch(model, _stamped(itertools.repeat(batch), stamps, needed, sync), optimizer, device,
                           accumulation_steps=accumulation, **common)
    else:
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager

        n_logical = math.ceil(needed / math.ceil(logical / physical)) + 2
        n = logical * n_logical
        data = TensorDataset(torch.zeros(1, 3, img, img).expand(n, -1, -1, -1), torch.randint(0, k, (n,)))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Secure RNG turned off")
            engine = PrivacyEngine(accountant=str(dp_cfg["accountant"]))
        dp_model = None
        try:
            dp_model, dp_optimizer, dp_loader = engine.make_private(
                module=model, optimizer=optimizer, data_loader=DataLoader(data, batch_size=logical, num_workers=0),
                noise_multiplier=1.0, max_grad_norm=1.0, poisson_sampling=True,
                grad_sample_mode=str(dp_cfg["grad-sample-mode"]))
            with BatchMemoryManager(data_loader=dp_loader, max_physical_batch_size=physical,
                                    optimizer=dp_optimizer) as safe_loader:
                for _ in range(20):  # normally one epoch is enough
                    if len(stamps) >= needed:
                        break
                    run_training_epoch(dp_model, _stamped(safe_loader, stamps, needed, sync), dp_optimizer, device,
                                       accumulation_steps=1, **common)
        finally:
            remove_hooks = getattr(dp_model, "remove_hooks", None)
            if callable(remove_hooks):
                try:
                    remove_hooks()
                except Exception:
                    logger.debug("remove_hooks() failed", exc_info=True)
    sync()
    peak = torch.cuda.max_memory_allocated()
    if len(stamps) < needed:
        raise RuntimeError(f"{mode} batch {physical}: only {len(stamps)} of {needed} timestamps were collected")
    timed = np.diff(np.asarray(stamps))[warmup:warmup + steps]
    return {"mode": mode, "physical_batch": physical, "status": "ok", "seconds_per_step": float(timed.mean()),
            "median_seconds_per_step": float(np.median(timed)), "n_timed": int(timed.size),
            "peak_mem_gib": peak / 2 ** 30, "accumulation_steps": accumulation, "logical_batch": logical}


def dp_size_is_representative(logical: int, physical: int) -> bool:
    """Opacus splits a logical batch into ceil(L/max) EQUAL chunks, so a max that does not divide L (48 for 64)
    really times smaller chunks (32 + 32)."""
    if physical < 1:
        raise ValueError("physical must be >= 1")
    return logical % physical == 0


def run_one(mode: str, physical: int, **kw: Any) -> Dict[str, Any]:
    try:
        res = measure_config(mode, physical, **kw)
    except Exception as exc:
        if not _is_oom(exc):
            raise
        res = {"mode": mode, "physical_batch": physical, "status": "OOM"}
    _free_gpu()  # outside the except block, so the failed attempt's tensors can be released
    return res


def train_sizes(manifests: Dict[str, Any]) -> Dict[str, int]:
    import pandas as pd
    return {c: int((pd.read_csv(path, usecols=["split"])["split"] == "train").sum()) for c, path in manifests.items()}


def report(store: Dict[str, Any], sizes: Dict[str, int], clients: List[str], local_epochs: int,
           logical_dp: int, gpu: str) -> None:
    results = store["results"]
    rows = []
    for mode in MODES:
        for b in sorted(int(r["physical_batch"]) for r in results.values() if r["mode"] == mode):
            r = results[f"{mode}-{b}"]
            if r["status"] == "ok":
                note = f"accum {r['accumulation_steps']}" + (" (real config cannot use 48)" if mode == "nondp" and b == 48 else "")
                rows.append([mode, b, "ok", f"{r['seconds_per_step']:.4f}", f"{r['peak_mem_gib']:.2f}", note])
            else:
                rows.append([mode, b, r["status"], "-", "-", ""])
    print(f"\nGPU: {gpu}")
    print("\nStep time and peak memory (one step = one physical micro-batch):")
    print(Z.format_table(["mode", "batch", "status", "sec/step", "peak GiB", "note"], rows))

    pooled = sum(sizes[c] for c in clients)
    rows2 = []
    for mode in MODES:
        for b in sorted(int(r["physical_batch"]) for r in results.values() if r["mode"] == mode):
            r = results[f"{mode}-{b}"]
            if r["status"] != "ok":
                continue
            t = r["seconds_per_step"]
            if mode == "nondp":
                per = [Z.epoch_seconds_nondp(sizes[c], b, t) for c in clients]
                rows2.append([mode, b, *(Z.fmt_hms(x) for x in per), Z.fmt_hms(Z.epoch_seconds_nondp(pooled, b, t)), "-"])
            else:
                per = [Z.epoch_seconds_dp(sizes[c], logical_dp, b, t) for c in clients]
                rows2.append([mode, b, *(Z.fmt_hms(x) for x in per), "-", Z.fmt_hms(local_epochs * per[clients.index("EyePACS")])
                              if "EyePACS" in clients else "-"])
    print("\nExtrapolated time per epoch (h:mm:ss), model compute only:")
    print(Z.format_table(["mode", "batch", *clients, "pooled", f"EyePACS DP round ({local_epochs} ep)"], rows2))
    print("train sizes: " + ", ".join(f"{c} {sizes[c]}" for c in clients) + f", pooled {pooled}")

    print("\nLargest physical batch that fits (no OOM):")
    for mode in MODES:
        status = {int(r["physical_batch"]): r["status"] for r in results.values() if r["mode"] == mode}
        best = Z.largest_fitting(status)
        print(f"  {mode}: {best if best is not None else 'none of the tested sizes fit'}")
    print("\nNotes: synthetic GPU-resident batches, so data loading and augmentation are not included; compare with "
          "the measured sec/epoch in rank_sweep.py before trusting wall-clock plans. 'fits' means no OOM in the timed "
          "steps; check peak GiB against the slice size and leave headroom. DP epochs use the expected number of "
          "micro-batches under Poisson sampling (q = 1/ceil(N/64)).")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch
    from configs import loader

    if not torch.cuda.is_available():
        print("error: timing needs the GPU (torch.cuda.is_available() is False).", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    common = dict(allow_unset=True, manifest_root=args.manifest_root, checkpoint_root=args.checkpoint_root,
                  env={}, config_dir=args.config_dir)
    cfg = loader.load_experiment_config(1, **common)
    dp_cfg = loader.load_experiment_config(4, **common)  # DP keys: accountant, grad-sample-mode, dp-logical-batch-size
    cfg_for_paths = dict(cfg, **{"lora-rank": args.lora_rank, "lora-alpha": args.lora_alpha})
    manifests = loader.to_centralized_kwargs(cfg_for_paths)["client_manifest_paths"]
    clients = list(loader.CLIENTS)
    sizes = train_sizes(manifests)

    out_dir = Path(args.checkpoint_root) / "stepzero"
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "timing_probe.json"
    props = torch.cuda.get_device_properties(device)
    params = {"steps": args.steps, "warmup_steps": args.warmup_steps, "lora_rank": args.lora_rank,
              "gpu": props.name, "gpu_mem_gib": round(props.total_memory / 2 ** 30, 1),
              "lora_alpha": args.lora_alpha, "logical_dp": int(dp_cfg["dp-logical-batch-size"]),
              "effective_batch_size": int(cfg["effective-batch-size"]), "img_size": int(cfg["img-size"])}
    store = Z.read_json(store_path)
    if store is None or args.force:
        store = {"params": params, "results": {}}
    elif store.get("params") != params:
        raise SystemExit(f"{store_path.name} was written with different settings; rerun with --force.\n"
                         f"  stored: {store.get('params')}\n  now:    {params}")
    store["train_sizes"] = sizes

    for mode in MODES:
        for b in args.batch_sizes:
            key = f"{mode}-{b}"
            if key in store["results"]:
                print(f"{key}: stored result reused ({store['results'][key]['status']})")
                continue
            if mode == "dp" and not dp_size_is_representative(int(dp_cfg["dp-logical-batch-size"]), b):
                store["results"][key] = {"mode": mode, "physical_batch": b, "status": "skipped"}
                Z.atomic_write_json(store_path, store)
                print(f"{key}: skipped (Opacus splits a logical batch of {dp_cfg['dp-logical-batch-size']} into equal "
                      f"chunks, so a max of {b} would really time smaller batches)")
                continue
            print(f"measuring {key} ...", flush=True)
            res = run_one(mode, b, cfg=cfg, dp_cfg=dp_cfg, device=device, steps=args.steps, warmup=args.warmup_steps,
                          rank=args.lora_rank, alpha=args.lora_alpha)
            store["results"][key] = res
            Z.atomic_write_json(store_path, store)
            print(f"  -> {res['status']}" + (f", {res['seconds_per_step']:.4f} s/step, peak {res['peak_mem_gib']:.2f} GiB"
                                              if res["status"] == "ok" else ""), flush=True)

    props = torch.cuda.get_device_properties(device)
    report(store, sizes, clients, int(dp_cfg["local-epochs"]), int(dp_cfg["dp-logical-batch-size"]),
           f"{props.name}, {props.total_memory / 2 ** 30:.1f} GiB visible")
    return 0


if __name__ == "__main__":
    sys.exit(main())