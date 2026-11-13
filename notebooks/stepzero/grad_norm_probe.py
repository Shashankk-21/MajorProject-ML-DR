"""
notebooks/stepzero/grad_norm_probe.py: per-sample gradient-norm probe for max-grad-norm.

Run AFTER the LoRA rank is chosen: lora-rank and lora-alpha are read from
configs/base.yaml (the script refuses while they are null). It measures the
PRE-CLIP per-sample gradient norms (flat L2 over ALL trainable parameters, as
Opacus clips) under the real training conditions: pretrained weights, the real
loss (weighted_label_smoothed_ce) with each client's OWN class weights
(compute_class_weights on that client's train labels), the real training
augmentation and train loader (get_dataloaders), Opacus functorch per-sample
gradients wrapped exactly as dp_checks does (PrivacyEngine.make_private with
noise 0 and a huge clip, model.train() then freeze_bn_eval), after the
dp_checks gate has passed. No optimizer step is ever taken.

Two phases, ~20 batches PER CLIENT each:
  init    at initialization (pretrained backbone, LoRA B = 0, so LoRA-A gradients are exactly 0)
  epoch1  after one epoch of NON-DP pooled training (run_centralized_training,
          max_epochs=1, into {root}/stepzero/probe_epoch1). This is a PROXY for
          mid-training, not the real DP training state.
Primary phase for the recommendation: epoch1.

Run from the repo root:
    python notebooks/stepzero/grad_norm_probe.py --manifest-root P --checkpoint-root P [--batches 20]

Outputs: {root}/stepzero/grad_norm_probe.json (raw norms per phase and client) and
the printed tables. Re-running reuses stored measurements; use --force to redo them.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import stepzero_lib as Z  # noqa: E402

logger = logging.getLogger("grad_norm_probe")

CANDIDATES = (0.5, 1.0, 2.0, 5.0)
PHASES = ("init", "epoch1")
PRIMARY = "epoch1"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-sample gradient-norm probe for max-grad-norm.")
    p.add_argument("--manifest-root", default=os.environ.get("DR_MANIFEST_ROOT"), help="folder with {client}_manifest.csv")
    p.add_argument("--checkpoint-root", default=os.environ.get("DR_CHECKPOINT_ROOT"), help="outputs go to {root}/stepzero/")
    p.add_argument("--batches", type=int, default=20, help="batches per client and phase (default 20)")
    p.add_argument("--batch-size", type=int, default=None, help="default: physical-batch-size from the config (32)")
    p.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    p.add_argument("--force", action="store_true", help="redo stored measurements")
    p.add_argument("--config-dir", default=None, help="alternative configs folder (testing)")
    args = p.parse_args(argv)
    for name in ("manifest_root", "checkpoint_root"):
        if not getattr(args, name):
            p.error(f"--{name.replace('_', '-')} is required (or DR_{name.upper()})")
    if args.batches < 1 or (args.batch_size is not None and args.batch_size < 2):
        p.error("--batches must be >= 1 and --batch-size >= 2")
    return args


def _seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def measure_clients(model: Any, *, device: Any, cfg: Dict[str, Any], manifests: Dict[str, Any], clients: List[str],
                    batches: int, batch_size: int, grad_sample_mode: str, seed: int) -> Dict[str, List[float]]:
    """Pre-clip flat per-sample norms for `batches` train batches of each client. Wraps `model`
    once the way utils/dp_checks.py does; never steps an optimizer."""
    import torch
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset

    from models.coatnet_lora import freeze_bn_eval
    from preprocessing.dataset import compute_class_weights, get_dataloaders
    from training.local_trainer import weighted_label_smoothed_ce

    img = int(cfg["img-size"])
    model.to(device)
    sgd = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    dummy = DataLoader(TensorDataset(torch.zeros(2, 3, img, img), torch.zeros(2, dtype=torch.long)), batch_size=2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Secure RNG turned off")
        engine = PrivacyEngine(accountant="rdp")
        wrapped, dp_optimizer, _ = engine.make_private(
            module=model, optimizer=sgd, data_loader=dummy, noise_multiplier=0.0, max_grad_norm=1e6,
            poisson_sampling=False, grad_sample_mode=grad_sample_mode)
    out: Dict[str, List[float]] = {}
    try:
        for client in clients:
            _seed_everything(seed)
            loader = get_dataloaders(client, manifest_path=manifests[client], batch_size=batch_size,
                                     num_workers=int(cfg["num-workers"]), seed=seed)[0]
            labels = loader.dataset.data["label"].to_numpy()
            weights = np.asarray(compute_class_weights(labels), dtype=np.float64)  # this client's OWN weights
            class_weights = torch.as_tensor(weights, dtype=torch.float32).to(device)
            wrapped.train()
            freeze_bn_eval(wrapped)  # re-asserted right after train(), as run_training_epoch does
            norms: List[np.ndarray] = []
            iterator = iter(loader)
            for _ in range(batches):
                try:
                    images, targets = next(iterator)
                except StopIteration:
                    break
                images, targets = images.to(device), targets.to(device)
                logits = wrapped(images)
                loss = weighted_label_smoothed_ce(logits, targets, class_weights,
                                                  label_smoothing=float(cfg["label-smoothing"]))
                loss.backward()
                grads = [p.grad_sample for p in wrapped.parameters() if p.requires_grad]
                if any(g is None for g in grads):
                    raise RuntimeError(f"{client}: {sum(g is None for g in grads)} trainable tensors have no grad_sample")
                if any(int(g.shape[0]) != int(targets.shape[0]) for g in grads):
                    raise RuntimeError(f"{client}: grad_sample batch size differs from the batch ({int(targets.shape[0])}); "
                                       f"gradients were not cleared between batches")
                batch_norms = Z.flat_per_sample_norms(grads)
                if not np.isfinite(batch_norms).all():
                    raise RuntimeError(f"{client}: non-finite per-sample gradient norms")
                norms.append(batch_norms)
                del logits, loss, grads
                dp_optimizer.zero_grad(set_to_none=True)
                for p in wrapped.parameters():  # belt and braces: nothing may carry into the next batch
                    if getattr(p, "grad_sample", None) is not None:
                        p.grad_sample = None
            del iterator, loader
            if not norms:
                raise RuntimeError(f"{client}: the train loader yielded no batches")
            out[client] = np.concatenate(norms).astype(np.float64).tolist()
            print(f"  {client}: {len(norms)} batches, {len(out[client])} samples, median norm "
                  f"{float(np.median(out[client])):.4f}", flush=True)
    finally:
        remove_hooks = getattr(wrapped, "remove_hooks", None)
        if callable(remove_hooks):
            try:
                remove_hooks()
            except Exception:
                logger.debug("remove_hooks() failed", exc_info=True)
        del wrapped, dp_optimizer, sgd
        _free_gpu()
    return out


def summarize_phase(client_norms: Dict[str, List[float]]) -> Dict[str, Any]:
    """Percentiles, clipped fractions and the recommendation for one phase (from raw norms)."""
    summaries = {c: Z.percentile_summary(n) for c, n in client_norms.items()}
    pooled = [x for c in client_norms for x in client_norms[c]]  # up to --batches per client (IDRiD has only 11 batches of 32), not data-proportional
    rec = Z.recommend_max_grad_norm({c: s["p50"] for c, s in summaries.items()})
    value = Z.round_sig(rec["value"])
    cands = sorted(set(CANDIDATES) | {value})
    clipped = {c: {str(v): Z.fraction_clipped(n, v) for v in cands} for c, n in client_norms.items()}
    clipped["POOLED"] = {str(v): Z.fraction_clipped(pooled, v) for v in cands}
    return {"summary": {**summaries, "POOLED": Z.percentile_summary(pooled)}, "clipped": clipped,
            "recommendation": {**rec, "value_rounded": value}, "candidates": cands}


def print_phase(name: str, phase: Dict[str, Any]) -> None:
    cands = phase["candidates"]
    rows = []
    for c, s in phase["summary"].items():
        rows.append([c, s["n"], *(f"{s[k]:.4f}" for k in ("p25", "p50", "p75", "p95", "max")),
                     *(f"{100 * phase['clipped'][c][str(v)]:.1f}%" for v in cands)])
    headers = ["client", "n", "p25", "p50", "p75", "p95", "max"] + [f"clip>{v:g}" for v in cands]
    print(f"\nPhase '{name}': pre-clip per-sample norms (flat L2 over all trainable parameters)")
    print(Z.format_table(headers, rows))
    r = phase["recommendation"]
    print(f"  recommendation (geometric mean of per-client medians): {r['value']:.4f} -> {r['value_rounded']:g}; "
          f"client median ratio max/min {r['ratio']:.2f}" + ("  *** FLAG: ratio above 3x ***" if r["flag"] else ""))


def report(store: Dict[str, Any]) -> None:
    phases = {n: summarize_phase({c: d["norms"] for c, d in store["phases"][n]["clients"].items()})
              for n in PHASES if n in store.get("phases", {}) and store["phases"][n].get("clients")}
    for name in PHASES:
        if name in phases:
            print_phase(name, phases[name])
    print("\nNotes: pooled = all collected norms, up to --batches batches per client, so a small client such as IDRiD (330 train images = 11 batches of 32) contributes fewer samples (not data-proportional). "
          "Phase 'init' has LoRA B = 0, so every LoRA-A gradient is exactly 0 there. Phase 'epoch1' is a PROXY for "
          "mid-training (non-DP, one pooled epoch with warmup LR), not the real DP training state.")
    if PRIMARY not in phases:
        print(f"\nPrimary phase '{PRIMARY}' not measured: no value to paste yet.")
        return
    primary = phases[PRIMARY]["recommendation"]
    if "init" in phases:
        pr = Z.phase_ratio(primary["value"], phases["init"]["recommendation"]["value"])
        print(f"\nPhase comparison: epoch1 {primary['value']:.4f} vs init {phases['init']['recommendation']['value']:.4f} "
              f"(ratio {pr['ratio']:.2f})" + ("  *** FLAG: phases differ by more than 3x; think before choosing ***" if pr["flag"] else ""))
    if primary["flag"]:
        print("*** FLAG: after one epoch the per-client medians differ by more than 3x; one global value may suit "
              "some clients badly (see the clipped fractions per client). ***")
    print("\nPaste into configs/base.yaml (this script never edits it):")
    print(f"max-grad-norm: {primary['value_rounded']:g}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from configs import loader

    common = dict(allow_unset=True, manifest_root=args.manifest_root, checkpoint_root=args.checkpoint_root,
                  env={}, config_dir=args.config_dir)
    cfg = loader.load_experiment_config(1, **common)
    rank, alpha = cfg.get("lora-rank"), cfg.get("lora-alpha")
    if rank is None or alpha is None:
        print("error: lora-rank / lora-alpha are still null in configs/base.yaml. Run rank_sweep.py, paste its two "
              "lines into base.yaml, then run this probe.", file=sys.stderr)
        return 2
    grad_sample_mode = loader.load_experiment_config(4, **common)["grad-sample-mode"]
    clients = list(loader.CLIENTS)
    manifests = loader.to_centralized_kwargs(cfg)["client_manifest_paths"]
    seed = int(cfg["run-seed"])
    batch_size = args.batch_size or int(cfg["physical-batch-size"])

    out_dir = Path(args.checkpoint_root) / "stepzero"
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "grad_norm_probe.json"
    params = {"lora_rank": int(rank), "lora_alpha": int(alpha), "batches": args.batches, "batch_size": batch_size,
              "seed": seed, "grad_sample_mode": grad_sample_mode, "label_smoothing": float(cfg["label-smoothing"])}
    store = Z.read_json(store_path)
    if store is None or args.force:
        store = {"params": params, "phases": {}}
    elif store.get("params") != params:
        raise SystemExit(f"{store_path.name} was written with different settings; rerun with --force.\n"
                         f"  stored: {store.get('params')}\n  now:    {params}")

    import torch
    from models.coatnet_lora import create_coatnet_lora_model
    from utils.dp_checks import ensure_per_sample_gradients_verified

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU found; the probe will be very slow and the gate runs on CPU.")

    def factory() -> Any:
        return create_coatnet_lora_model(lora_rank=int(rank), lora_alpha=int(alpha), num_classes=int(cfg["num-classes"]),
                                         img_size=int(cfg["img-size"]), pretrained=True)

    todo = [ph for ph in args.phases
            if [c for c in clients if c not in store["phases"].get(ph, {}).get("clients", {})]]
    if todo:
        print(f"Per-sample-gradient gate (mode {grad_sample_mode!r}, device {device.type}) ...", flush=True)
        gate = ensure_per_sample_gradients_verified(factory, device=device, grad_sample_mode=grad_sample_mode)
        print(f"  gate passed (worst relative error {gate['worst_rel_err']:.2e})")

    for phase in args.phases:
        missing = [c for c in clients if c not in store["phases"].get(phase, {}).get("clients", {})]
        if not missing:
            print(f"\nPhase '{phase}': stored measurements reused.")
            continue
        print(f"\n=== phase '{phase}': measuring {missing} ({args.batches} batches x {batch_size}) ===", flush=True)
        _seed_everything(seed)
        model = factory()
        if phase == "epoch1":
            from training.train_centralized import run_centralized_training
            from utils.checkpoint import load_checkpoint
            kwargs = loader.to_centralized_kwargs(cfg)
            kwargs.update(checkpoint_dir=str(out_dir / "probe_epoch1"), max_epochs=1)
            print("  training one non-DP pooled epoch (resumes or skips if already done) ...", flush=True)
            result = run_centralized_training(**kwargs)
            load_checkpoint(result["best_checkpoint_path"], model=model, map_location="cpu")
            _free_gpu()
        measured = measure_clients(model, device=device, cfg=cfg, manifests=manifests, clients=missing,
                                   batches=args.batches, batch_size=batch_size, grad_sample_mode=grad_sample_mode, seed=seed)
        del model
        _free_gpu()
        entry = store["phases"].setdefault(phase, {"clients": {}})
        for c, norms in measured.items():
            entry["clients"][c] = {"norms": norms}
        Z.atomic_write_json(store_path, store)

    report(store)
    return 0


if __name__ == "__main__":
    sys.exit(main())