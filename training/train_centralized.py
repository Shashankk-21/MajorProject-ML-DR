"""
training/train_centralized.py

Experiment 1: the centralized, non-FL, non-DP baseline, per
training-phase-decisions.md Sections 1-2 and 7, patched against the
discussion chat's spec (apply AFTER the dataset.py patch -- this file
imports compute_class_weights from preprocessing.dataset).

Experiment 1 uses the SAME frozen-backbone + LoRA architecture as every
other experiment -- it calls create_coatnet_lora_model() exactly like
the other 7 do, backbone frozen, LoRA injected -- NOT full fine-tuning.
Experiment 1 exists to isolate "the cost of federation/privacy"; that
comparison only means something if model capacity is held identical
across all 8 experiments.

Pooling: preprocessing/dataset.py is per-client ONLY by explicit design
(no cross-client manifest, split, or DataLoader), so pooling all 4
clients into one centralized training set is this file's job.
build_pooled_dataloaders() reads the 4 already-split FINAL manifests and
concatenates them (splitting happened per-client BEFORE pooling, so
pooling adds no cross-client leakage) and reuses DRDataset directly,
which is dataset-agnostic.

PATCHED (this version) -- what changed and why:

1. Mixup is gone entirely (import, construction, and the mixup_fn=
   argument, which run_training_epoch no longer accepts). Mixup is
   dropped from all 8 experiments: it breaks the per-example
   sensitivity bound DP relies on, and keeping it in Experiments 1-3
   would make the Experiment-3-vs-4 comparison change two things (DP
   on/off AND Mixup on/off) instead of one.
2. Class imbalance is handled by a class-weighted, label-smoothed loss
   (inside local_trainer.run_training_epoch), identical in all 8
   experiments. Class weights are computed ONCE here, from the POOLED
   train labels (Experiment 1 pools counts across the 4 clients), with
   preprocessing.dataset.compute_class_weights, and passed as a
   torch.float32 tensor together with label_smoothing.
3. WeightedRandomSampler is removed from build_pooled_dataloaders (and
   its weights code with it): imbalance now lives in the loss, and
   Opacus would replace any sampler with uniform Poisson sampling for
   the DP experiments anyway. The train loader is now shuffle=True,
   drop_last=False, with an explicit seeded torch.Generator, which is
   checkpointed under extra_generators["train_shuffle"] via
   train_loader.generator (never via train_loader.sampler.generator).
4. Fail-fast validation before anything expensive is built: exact
   client set, manifests present with non-empty train/val/test rows,
   pooled train labels integers in [0, 4] with EVERY class present
   (compute_class_weights itself only warns, because a small FL client
   may legitimately lack a class -- the pooled centralized set must
   not), batch/epoch/patience arguments, effective batch == physical *
   accumulation, and img_size == 224 (the offline cache holds 224x224
   images; the 384 contingency needs a rebuilt cache, not just a
   different argument). A pooled VAL split missing a class only warns.
5. Fresh runs seed random / numpy / torch (and CUDA, if present) before
   the model is built, so LoRA A and the head (which draw from the
   global RNG) initialize deterministically. A resume does not
   re-seed: checkpoint.load_checkpoint() restores the saved global RNG
   state instead.
6. Scheduler math now comes from the loader that actually runs
   (len(train_loader), ceil(len / accumulation_steps)), and the
   warmup start factor is a parameter (warmup_start_factor).
7. The scheduler is checkpointed through checkpoint.py's native
   scheduler= support (it used to be stashed inside `extra`).
   `extra` keeps only epoch, best_val_macro_f1,
   epochs_without_improvement.
8. Every checkpoint carries a run_config (hyperparameters, seed, pooled
   class counts, and the sha256 of each manifest file), and a resume
   passes the freshly built config as expected_config -- ANY changed
   value raises instead of silently resuming a different experiment.
9. A restored run that is already finished (patience exhausted, or all
   max_epochs done) returns its stored result without training. The
   old loop would have trained one more epoch after a restart that
   landed just after an early stop.
10. Per-epoch order is now: metrics -> best_centralized.pt (only if
   improved) -> latest_centralized.pt -> on_epoch_end. The callback is
   wrapped in `except Exception` (logged, training continues -- a
   logging failure must not kill an expensive run); BaseException
   (KeyboardInterrupt, SystemExit) still propagates.

Other locked behavior, unchanged: AdamW with two param groups (LoRA
1e-4, head 1e-3, weight_decay 0.01); step-level warmup+cosine via
SequentialLR, stepped inside run_training_epoch's scheduler= hook;
<= 20 epochs, patience 5, early stopping on val macro-F1 with STRICT
improvement; verify_lora_injection() as an automatic pre-flight gate;
find_latest_checkpoint(pattern="latest_*.pt") for resume; round_offset=0
and no accountant (non-DP, non-FL); no W&B calls here (on_epoch_end
callback instead); no test-set evaluation here (belongs to
evaluation/, run against best_checkpoint_path).

PATCH (tracker support): the on_epoch_end metrics dict also carries lr_lora and
lr_head (the optimizer's learning rates after the epoch's last optimizer step)
and val_confusion_matrix (25 ints, row-major, rows = true class) so the tracker
can log LRs, per-class recall and QWK. Nothing else changed; the matrix is
computed from the validation logits already in hand and a failure to compute it
is logged and skipped, never fatal.

Not added, deliberately: a class-weight cap parameter (that decision is
still open; compute_class_weights is used as-is).
"""

from __future__ import annotations

import hashlib
import logging
import math
import numbers
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from models.coatnet_lora import (
    MODEL_NAME,
    NUM_DR_GRADES,
    create_coatnet_lora_model,
    split_trainable_parameters,
    verify_lora_injection,
)
from training.local_trainer import (
    compute_selection_metric,
    run_training_epoch,
    run_validation_epoch,
)
from utils.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from preprocessing.augmentation import get_eval_transforms, get_train_transforms
from preprocessing.dataset import (
    FINAL_MANIFEST_COLUMNS,
    DRDataset,
    compute_class_weights,
    make_augmentation_worker_init_fn,
)

logger = logging.getLogger(__name__)

__all__ = ["build_pooled_dataloaders", "run_centralized_training"]

REQUIRED_CLIENTS: Tuple[str, ...] = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")
_REQUIRED_SPLITS: Tuple[str, ...] = ("train", "val", "test")
_CACHE_IMG_SIZE = 224  # the offline cache holds 224x224 images
_EXPERIMENT_ID = "exp1"
_SHUFFLE_GENERATOR_KEY = "train_shuffle"
_REQUIRED_EXTRA_KEYS = ("epoch", "best_val_macro_f1", "epochs_without_improvement")


def _is_int(value: Any) -> bool:
    """A real integer -- bool is explicitly excluded (it subclasses int)."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_run_arguments(
    *,
    client_manifest_paths: Any,
    lora_rank: Any,
    lora_alpha: Any,
    checkpoint_dir: Any,
    img_size: Any,
    num_classes: Any,
    physical_batch_size: Any,
    accumulation_steps: Any,
    effective_batch_size: Any,
    num_workers: Any,
    lr_lora: Any,
    lr_head: Any,
    weight_decay: Any,
    warmup_epochs: Any,
    warmup_start_factor: Any,
    max_epochs: Any,
    patience: Any,
    label_smoothing: Any,
    seed: Any,
    wandb_run_id: Any,
    on_epoch_end: Any,
) -> None:
    """Pure argument validation -- no disk access, nothing built. Every
    problem found is reported at once in a single ValueError.
    """
    problems: List[str] = []

    if not isinstance(client_manifest_paths, dict):
        problems.append(
            f"client_manifest_paths must be a dict, got {type(client_manifest_paths).__name__}"
        )
    else:
        actual = set(client_manifest_paths.keys())
        expected = set(REQUIRED_CLIENTS)
        if actual != expected:
            problems.append(
                f"client_manifest_paths keys must be EXACTLY {sorted(expected)}; "
                f"missing {sorted(expected - actual)}, unexpected "
                f"{sorted(actual - expected, key=str)}"
            )

    for name, value in (("lora_rank", lora_rank), ("lora_alpha", lora_alpha)):
        if not _is_int(value) or value <= 0:
            problems.append(f"{name} must be an int > 0, got {value!r}")

    if not isinstance(checkpoint_dir, (str, os.PathLike)) or not str(checkpoint_dir):
        problems.append(f"checkpoint_dir must be a non-empty path, got {checkpoint_dir!r}")

    if not _is_int(img_size) or img_size != _CACHE_IMG_SIZE:
        problems.append(
            f"img_size must be {_CACHE_IMG_SIZE}, got {img_size!r}: the offline "
            f"cache holds {_CACHE_IMG_SIZE}x{_CACHE_IMG_SIZE} images, and the "
            f"384 contingency needs a rebuilt cache, not just a different argument"
        )
    if not _is_int(num_classes) or num_classes != NUM_DR_GRADES:
        problems.append(f"num_classes must be {NUM_DR_GRADES}, got {num_classes!r}")

    ints_ok = True
    for name, value in (
        ("physical_batch_size", physical_batch_size),
        ("accumulation_steps", accumulation_steps),
        ("effective_batch_size", effective_batch_size),
    ):
        if not _is_int(value) or value <= 0:
            problems.append(f"{name} must be an int > 0, got {value!r}")
            ints_ok = False
    if ints_ok and physical_batch_size * accumulation_steps != effective_batch_size:
        problems.append(
            f"physical_batch_size * accumulation_steps "
            f"({physical_batch_size} * {accumulation_steps} = "
            f"{physical_batch_size * accumulation_steps}) must equal "
            f"effective_batch_size ({effective_batch_size})"
        )

    if not _is_int(num_workers) or num_workers < 0:
        problems.append(f"num_workers must be an int >= 0, got {num_workers!r}")
    if not _is_int(max_epochs) or max_epochs < 1:
        problems.append(f"max_epochs must be an int >= 1, got {max_epochs!r}")
    if not _is_int(patience) or patience < 1:
        problems.append(f"patience must be an int >= 1, got {patience!r}")
    # np.random.seed() only accepts [0, 2**32); fail here, not after the
    # data and manifests have already been processed.
    if not _is_int(seed) or not (0 <= seed < 2**32):
        problems.append(f"seed must be an int in [0, 2**32), got {seed!r}")

    for name, value in (("lr_lora", lr_lora), ("lr_head", lr_head), ("weight_decay", weight_decay)):
        if not _is_finite_number(value) or value < 0:
            problems.append(f"{name} must be a finite number >= 0, got {value!r}")
    if not _is_finite_number(warmup_epochs) or warmup_epochs < 0:
        problems.append(f"warmup_epochs must be a finite number >= 0, got {warmup_epochs!r}")
    if not _is_finite_number(warmup_start_factor) or not (0 < warmup_start_factor <= 1):
        problems.append(
            f"warmup_start_factor must be a finite number in (0, 1], got {warmup_start_factor!r}"
        )
    if not _is_finite_number(label_smoothing) or not (0 <= label_smoothing < 1):
        problems.append(f"label_smoothing must be a finite number in [0, 1), got {label_smoothing!r}")

    if wandb_run_id is not None and not isinstance(wandb_run_id, str):
        problems.append(f"wandb_run_id must be None or a str, got {type(wandb_run_id).__name__}")
    if on_epoch_end is not None and not callable(on_epoch_end):
        problems.append("on_epoch_end must be None or callable")

    if problems:
        raise ValueError(
            "run_centralized_training: invalid arguments:\n  " + "\n  ".join(problems)
        )


def _confusion_matrix_flat(logits: Any, targets: Any, num_classes: int) -> List[int]:
    """Row-major flattened confusion matrix (rows = true class) from logits
    [N, K] and integer targets [N]; works on tensors or arrays."""

    def _to_np(x: Any) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            x = x.numpy()
        return np.asarray(x)

    preds = _to_np(logits).argmax(axis=1).astype(np.int64)
    true = _to_np(targets).astype(np.int64).reshape(-1)
    if preds.shape != true.shape:
        raise ValueError(f"logits imply {preds.shape[0]} predictions but there are {true.shape[0]} targets")
    if true.size and (true.min() < 0 or true.max() >= num_classes or preds.max() >= num_classes):
        raise ValueError(f"class ids must be in [0, {num_classes - 1}]")
    return [int(v) for v in np.bincount(true * num_classes + preds, minlength=num_classes * num_classes)]


def _preflight_manifests(client_manifest_paths: Dict[str, Any]) -> Dict[str, str]:
    """Check every client's manifest exists and has non-empty train, val
    and test rows, and return {client: sha256 of the manifest file's
    bytes} for the run_config. Raises ValueError listing every problem.

    This is a pre-flight only; build_pooled_dataloaders() still does the
    actual loading (and keeps its own FileNotFoundError for direct
    callers).
    """
    problems: List[str] = []
    sha256_by_client: Dict[str, str] = {}

    for client in REQUIRED_CLIENTS:
        path = Path(client_manifest_paths[client])
        if not path.is_file():
            problems.append(f"{client}: manifest not found at {path}")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # unreadable/empty/corrupt file
            problems.append(f"{client}: could not read manifest {path}: {exc}")
            continue
        if "split" not in df.columns:
            problems.append(f"{client}: manifest {path} has no 'split' column")
            continue
        for split in _REQUIRED_SPLITS:
            if int((df["split"] == split).sum()) == 0:
                problems.append(f"{client}: manifest {path} has zero '{split}' rows")
        sha256_by_client[client] = _sha256_of_file(path)

    if problems:
        raise ValueError(
            "run_centralized_training: manifest pre-flight failed:\n  " + "\n  ".join(problems)
        )
    return sha256_by_client


def build_pooled_dataloaders(
    client_manifest_paths: Dict[str, Path],
    *,
    batch_size: int = 32,
    num_workers: int = 2,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Pool the clients' FINAL manifests into one centralized dataset.

    See the module docstring for why this exists instead of reusing
    dataset.get_dataloaders() (per-client by explicit design) and why
    reusing DRDataset directly is safe.

    Each client's FINAL manifest already carries its own 'split' column,
    assigned during that client's own preprocessing -- pooling happens
    strictly AFTER splitting, so concatenating introduces no
    cross-client leakage.

    The train loader uses plain shuffle=True (no WeightedRandomSampler --
    class imbalance is handled by the class-weighted loss) with an
    explicit torch.Generator seeded from `seed`; that generator is
    exposed as train_loader.generator so it can be checkpointed.

    Augmentation randomness (albumentations owns its own RNG, which the
    global seeds cannot reach): the train loader gets
    worker_init_fn=make_augmentation_worker_init_fn() from
    preprocessing.dataset, so with num_workers > 0 each worker's train
    transform is reseeded from torch.utils.data.get_worker_info().seed
    (unique per worker and per epoch, derived from the loader's seeded
    generator). With num_workers == 0 no worker ever runs, so the train
    transform is seeded once here with set_random_seed(seed). A train
    transform without set_random_seed raises ValueError at build time.
    The val/test loaders are unchanged (their transform is deterministic).
    """
    frames = []
    for client_name, manifest_path in client_manifest_paths.items():
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{client_name}: final manifest not found at {manifest_path}. "
                f"Run build_dataset.py for this client first."
            )
        df = pd.read_csv(manifest_path)
        missing = set(FINAL_MANIFEST_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"{client_name} manifest at {manifest_path} is missing "
                f"columns {sorted(missing)} -- expected FINAL_MANIFEST_COLUMNS."
            )
        df = df.copy()
        df["source_client"] = client_name  # diagnostic only -- not read by DRDataset
        frames.append(df)

    pooled_df = pd.concat(frames, ignore_index=True)
    logger.info(
        "build_pooled_dataloaders: pooled %d clients, %d total rows",
        len(client_manifest_paths), len(pooled_df),
    )

    if train_transform is None:
        train_transform = get_train_transforms()
    if eval_transform is None:
        eval_transform = get_eval_transforms()
    if not callable(getattr(train_transform, "set_random_seed", None)):
        raise ValueError(
            "build_pooled_dataloaders: the train transform "
            f"({type(train_transform).__name__}) has no set_random_seed(); the "
            "augmentation pipeline cannot be reseeded per worker (or once, when "
            "num_workers == 0), so its randomness would not follow the loader's seed."
        )

    train_dataset = DRDataset(pooled_df, split="train", transform=train_transform)
    val_dataset = DRDataset(pooled_df, split="val", transform=eval_transform)
    test_dataset = DRDataset(pooled_df, split="test", transform=eval_transform)

    shuffle_generator = torch.Generator().manual_seed(seed)
    if num_workers == 0:
        # No worker processes will run worker_init_fn, so seed the train
        # transform's own RNG directly, once, when the loader is built.
        train_transform.set_random_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        generator=shuffle_generator, num_workers=num_workers,
        worker_init_fn=make_augmentation_worker_init_fn(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return train_loader, val_loader, test_loader


def _prepare_class_weights(
    train_loader: DataLoader, val_loader: DataLoader
) -> Tuple[np.ndarray, torch.Tensor]:
    """Validate the pooled labels and compute the class weights.

    Raises ValueError unless the pooled TRAIN labels are integers in
    [0, NUM_DR_GRADES - 1] and EVERY class is present -- compute_class_
    weights() itself only warns about a missing class (a small FL client
    may legitimately lack one), but the pooled centralized set must not.
    A pooled VAL split missing a class only warns.

    Returns (class_counts as an int array of length 5, class_weights as a
    torch.float32 tensor of shape (5,)).
    """
    train_labels = train_loader.dataset.data["label"]
    labels_np = np.asarray(train_labels)

    if labels_np.size == 0:
        raise ValueError("Pooled train split has no rows.")
    if not np.issubdtype(labels_np.dtype, np.integer):
        raise ValueError(
            f"Pooled train labels must be integers, got dtype {labels_np.dtype}."
        )
    if labels_np.min() < 0 or labels_np.max() > NUM_DR_GRADES - 1:
        raise ValueError(
            f"Pooled train labels must lie in [0, {NUM_DR_GRADES - 1}], got "
            f"min={labels_np.min()} max={labels_np.max()}."
        )

    class_counts = np.bincount(labels_np, minlength=NUM_DR_GRADES)
    absent = [int(c) for c in np.where(class_counts == 0)[0]]
    if absent:
        raise ValueError(
            f"Pooled train split has no examples of class(es) {absent} "
            f"(counts={class_counts.tolist()}). The centralized baseline "
            f"needs every DR grade present."
        )

    val_labels = np.asarray(val_loader.dataset.data["label"])
    val_absent = sorted(set(range(NUM_DR_GRADES)) - set(np.unique(val_labels).tolist()))
    if val_absent:
        logger.warning(
            "run_centralized_training: pooled val split has no examples of "
            "class(es) %s -- macro-F1 will count them as 0, so early "
            "stopping and best-checkpoint selection are affected.",
            val_absent,
        )

    # Single call site for the compute_class_weights contract: it takes
    # the pooled train labels and returns one weight per class.
    weights_np = np.asarray(compute_class_weights(train_labels), dtype=np.float64)
    class_weights = torch.as_tensor(weights_np, dtype=torch.float32)

    if tuple(class_weights.shape) != (NUM_DR_GRADES,):
        raise ValueError(
            f"compute_class_weights returned shape {tuple(class_weights.shape)}, "
            f"expected ({NUM_DR_GRADES},)."
        )
    if not bool(torch.isfinite(class_weights).all()) or not bool((class_weights > 0).all()):
        raise ValueError(
            f"compute_class_weights returned non-finite or non-positive "
            f"weights: {class_weights.tolist()}"
        )

    # Locked convention (local_trainer.weighted_label_smoothed_ce): weights
    # are normalised so their expectation under the training label
    # distribution is 1, keeping LR and max_grad_norm comparable to an
    # unweighted run. Check it rather than assume it.
    expected_weight = float(
        (class_counts * class_weights.double().numpy()).sum() / class_counts.sum()
    )
    if abs(expected_weight - 1.0) > 1e-3:
        logger.warning(
            "run_centralized_training: class weights' expectation under the "
            "pooled train label distribution is %.6f, not 1. The locked "
            "convention is expectation == 1 (keeps the locked LR / "
            "max_grad_norm comparable to an unweighted run) -- check "
            "compute_class_weights' normalisation.",
            expected_weight,
        )

    logger.info(
        "run_centralized_training: pooled train class counts=%s class_weights=%s "
        "(expected weight under train distribution=%.6f)",
        class_counts.tolist(),
        [round(w, 6) for w in class_weights.tolist()],
        expected_weight,
    )
    return class_counts, class_weights


def run_centralized_training(
    *,
    client_manifest_paths: Dict[str, Path],
    lora_rank: int,
    lora_alpha: int,
    checkpoint_dir: str,
    img_size: int = 224,
    num_classes: int = NUM_DR_GRADES,
    physical_batch_size: int = 32,
    accumulation_steps: int = 2,
    effective_batch_size: int = 64,
    num_workers: int = 2,
    lr_lora: float = 1e-4,
    lr_head: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_epochs: float = 1.0,
    warmup_start_factor: float = 1e-3,
    max_epochs: int = 20,
    patience: int = 5,
    label_smoothing: float = 0.1,
    seed: int = 42,
    wandb_run_id: Optional[str] = None,
    on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """Run (or resume) Experiment 1 end-to-end: validate, pool data,
    compute class weights, build the frozen-backbone + LoRA model, train
    with early stopping on val macro-F1, checkpoint every epoch, return
    the result.

    Args:
        client_manifest_paths: EXACTLY the 4 clients' FINAL manifest
            CSVs, keyed "APTOS", "IDRiD", "Messidor-2", "EyePACS".
        lora_rank, lora_alpha: no defaults -- must come from the step-zero
            sweep's result.
        checkpoint_dir: where "latest_centralized.pt" / "best_
            centralized.pt" are written, and where resume looks for an
            existing "latest_*.pt" on start.
        img_size: must be 224 (the offline cache holds 224x224 images).
        physical_batch_size, accumulation_steps, effective_batch_size:
            physical * accumulation must equal effective (locked: 64).
        warmup_epochs, warmup_start_factor: LinearLR warmup length (in
            epochs of REAL optimizer steps) and its starting factor.
            warmup_epochs is not confirmed locked in the decisions doc;
            1.0 is an assumed default.
        label_smoothing: passed to the class-weighted loss; locked at 0.1.
        seed: seeds a fresh run's global RNGs and the train-shuffle
            generator. Must be in [0, 2**32).
        on_epoch_end: optional callback(epoch, metrics) for external
            logging. An Exception raised by it is logged and training
            continues; BaseException propagates.

    Resume: if a "latest_*.pt" exists in checkpoint_dir it is restored
    (model, optimizer, scheduler, RNG, train-shuffle generator) and its
    stored run_config must match this call's exactly -- any differing
    hyperparameter, seed, class-count or manifest hash raises ValueError.
    A restored run that already finished (patience exhausted, or all
    max_epochs done) returns its stored result without training.

    Returns:
        {"best_val_macro_f1": float, "final_epoch": int,
         "best_checkpoint_path": str}.
    """
    _validate_run_arguments(
        client_manifest_paths=client_manifest_paths,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        checkpoint_dir=checkpoint_dir,
        img_size=img_size,
        num_classes=num_classes,
        physical_batch_size=physical_batch_size,
        accumulation_steps=accumulation_steps,
        effective_batch_size=effective_batch_size,
        num_workers=num_workers,
        lr_lora=lr_lora,
        lr_head=lr_head,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        max_epochs=max_epochs,
        patience=patience,
        label_smoothing=label_smoothing,
        seed=seed,
        wandb_run_id=wandb_run_id,
        on_epoch_end=on_epoch_end,
    )
    manifest_sha256 = _preflight_manifests(client_manifest_paths)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_path / "latest_centralized.pt"
    best_path = checkpoint_path / "best_centralized.pt"

    latest_ckpt = find_latest_checkpoint(str(checkpoint_path), pattern="latest_*.pt")
    if latest_ckpt is None:
        # Fresh run: LoRA A and the head draw from the global RNG when
        # the model is built, so seed first. A resume does NOT re-seed --
        # load_checkpoint() restores the saved global RNG state instead.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _test_loader = build_pooled_dataloaders(
        client_manifest_paths,
        batch_size=physical_batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    # _test_loader intentionally unused -- test-set evaluation belongs to
    # evaluation/, run later against best_checkpoint_path.

    shuffle_generator = train_loader.generator
    if shuffle_generator is None:
        # Explicit raise, not assert: asserts vanish under `python -O`,
        # and resuming without the shuffle generator's state would
        # silently change the data order.
        raise RuntimeError(
            "train_loader.generator is None -- the train-shuffle generator "
            "cannot be checkpointed. build_pooled_dataloaders must pass an "
            "explicit torch.Generator to the train DataLoader."
        )
    extra_generators = {_SHUFFLE_GENERATOR_KEY: shuffle_generator}

    class_counts, class_weights = _prepare_class_weights(train_loader, val_loader)
    logger.info(
        "run_centralized_training: pooled train=%d val=%d",
        len(train_loader.dataset), len(val_loader.dataset),
    )

    model = create_coatnet_lora_model(
        lora_rank=lora_rank, lora_alpha=lora_alpha,
        num_classes=num_classes, img_size=img_size,
    )
    verify_lora_injection(model, lora_rank=lora_rank, img_size=img_size)
    model.to(device)

    lora_params, head_params = split_trainable_parameters(model)
    optimizer = AdamW(
        [
            {"params": lora_params, "lr": lr_lora},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )

    # Scheduler math from the loader that actually runs.
    micro_batches_per_epoch = len(train_loader)
    real_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = max(max_epochs * real_steps_per_epoch, 1)
    warmup_steps = max(round(warmup_epochs * real_steps_per_epoch), 1)
    logger.info(
        "run_centralized_training: %d micro-batches/epoch, %d real steps/epoch, "
        "%d total steps, %d warmup steps",
        micro_batches_per_epoch, real_steps_per_epoch, total_steps, warmup_steps,
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    run_config: Dict[str, Any] = {
        "experiment_id": _EXPERIMENT_ID,
        "model_name": MODEL_NAME,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "img_size": img_size,
        "num_classes": num_classes,
        "physical_batch_size": physical_batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "lr_lora": lr_lora,
        "lr_head": lr_head,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "warmup_start_factor": warmup_start_factor,
        "max_epochs": max_epochs,
        "patience": patience,
        "label_smoothing": label_smoothing,
        "seed": seed,
        "pooled_train_class_counts": [int(c) for c in class_counts],
        "manifest_sha256": dict(manifest_sha256),
    }

    start_epoch = 0
    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    if latest_ckpt is not None:
        restored = load_checkpoint(
            latest_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            extra_generators=extra_generators,
            expected_config=run_config,
            map_location=device,
        )
        extra = restored["extra"]
        missing_extra = [k for k in _REQUIRED_EXTRA_KEYS if k not in extra]
        if missing_extra:
            raise ValueError(
                f"Checkpoint {latest_ckpt} is missing resume-state key(s) "
                f"{missing_extra} in its `extra` -- refusing to resume with "
                f"silently defaulted epoch/patience state."
            )
        start_epoch = extra["epoch"]
        best_val_macro_f1 = extra["best_val_macro_f1"]
        epochs_without_improvement = extra["epochs_without_improvement"]
        wandb_run_id = restored["wandb_run_id"] or wandb_run_id
        logger.info(
            "run_centralized_training: resumed from %s at epoch %d "
            "(best_val_macro_f1=%.4f, epochs_without_improvement=%d)",
            latest_ckpt, start_epoch, best_val_macro_f1, epochs_without_improvement,
        )

        if epochs_without_improvement >= patience or start_epoch >= max_epochs:
            logger.info(
                "run_centralized_training: restored run is already finished "
                "(epoch=%d, epochs_without_improvement=%d, patience=%d, "
                "max_epochs=%d) -- returning the stored result without training.",
                start_epoch, epochs_without_improvement, patience, max_epochs,
            )
            return {
                "best_val_macro_f1": best_val_macro_f1,
                "final_epoch": start_epoch,
                "best_checkpoint_path": str(best_path),
            }

    final_epoch = start_epoch
    for epoch in range(start_epoch, max_epochs):
        train_loss = run_training_epoch(
            model, train_loader, optimizer, device,
            accumulation_steps=accumulation_steps,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            scheduler=scheduler,
        )
        val_loss, val_logits, val_targets = run_validation_epoch(model, val_loader, device)
        val_macro_f1 = compute_selection_metric(val_logits, val_targets)

        improved = val_macro_f1 > best_val_macro_f1  # strict improvement
        if improved:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        final_epoch = epoch + 1
        metrics = {
            "epoch": final_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
        }
        metrics["lr_lora"] = float(optimizer.param_groups[0]["lr"])
        metrics["lr_head"] = float(optimizer.param_groups[1]["lr"])
        logger.info("run_centralized_training: %s", metrics)
        try:  # tracker support only; never fatal
            metrics["val_confusion_matrix"] = _confusion_matrix_flat(val_logits, val_targets, num_classes)
        except Exception:
            logger.exception("run_centralized_training: could not build the validation confusion matrix")

        extra = {
            "epoch": final_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "epochs_without_improvement": epochs_without_improvement,
        }
        # Order: best (only if improved) -> latest -> callback.
        if improved:
            save_checkpoint(
                str(best_path), model=model, optimizer=optimizer, scheduler=scheduler,
                round_offset=0, opacus_accountant_states=None,
                extra_generators=extra_generators, wandb_run_id=wandb_run_id,
                run_config=run_config, extra=extra,
            )
        save_checkpoint(
            str(latest_path), model=model, optimizer=optimizer, scheduler=scheduler,
            round_offset=0, opacus_accountant_states=None,
            extra_generators=extra_generators, wandb_run_id=wandb_run_id,
            run_config=run_config, extra=extra,
        )

        if on_epoch_end is not None:
            try:
                on_epoch_end(final_epoch, metrics)
            except Exception:
                # A logging failure must not kill an expensive run. Only
                # Exception is caught -- KeyboardInterrupt/SystemExit
                # (BaseException) still propagate.
                logger.exception(
                    "run_centralized_training: on_epoch_end callback raised at "
                    "epoch %d; continuing training.", final_epoch,
                )

        if epochs_without_improvement >= patience:
            logger.info(
                "run_centralized_training: early stopping at epoch %d "
                "(patience=%d exhausted, best_val_macro_f1=%.4f)",
                final_epoch, patience, best_val_macro_f1,
            )
            break

    return {
        "best_val_macro_f1": best_val_macro_f1,
        "final_epoch": final_epoch,
        "best_checkpoint_path": str(best_path),
    }