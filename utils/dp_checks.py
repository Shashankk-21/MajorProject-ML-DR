"""
utils/dp_checks.py

The standing per-sample-gradient gate for every DP experiment (4-8).

WHY THIS EXISTS. Opacus's default grad_sample_mode="hooks" returns
silently WRONG per-sample gradients for this project's qkv LoRA layers
(timm's LayerNorm2d leaves channels-last strides, and Opacus's unfold2d
as_strided misreads them -- 32 of 66 trainable tensors were wrong in
validation). Nothing errors: training runs, epsilon is reported, and
the clipped/noised update is simply not the one DP-SGD is supposed to
compute. grad_sample_mode="functorch" matches plain autograd to ~1e-6.
So "functorch is used" is not enough on its own; this module PROVES, on
the real architecture and the real device, that the per-sample gradients
Opacus hands to the optimizer equal a loop of single-sample autograd
backward passes.

WHAT IT CHECKS (verify_per_sample_gradients):
  Reference: for each sample i, a single-sample forward+backward through
    a plain peft model gives that sample's gradient for every trainable
    tensor.
  Test: an identical second model is wrapped by Opacus (PrivacyEngine
    with accountant "rdp", noise_multiplier=0, a huge max_grad_norm,
    poisson_sampling=False, the requested grad_sample_mode), one batched
    forward+backward is run, and p.grad_sample[i] is compared with the
    reference for EVERY trainable tensor and EVERY sample.
  A relative error ||a-b||/||a|| must be < rel_tol (when ||a|| == 0, b
  must be exactly zero). The check also requires at least one non-zero
  LoRA-A gradient and one non-zero head gradient in the reference, so it
  cannot pass vacuously (LoRA B is initialised to zero, which makes every
  LoRA-A gradient exactly zero -- so lora_B is set to small random values
  in BOTH models first).

SIDE EFFECTS ARE CONTAINED, on purpose:
  - The global torch RNG (CPU and the checked CUDA device) is forked and
    restored. The gate calls torch.manual_seed, and client.py seeds the
    RNGs per round for determinism; a gate that leaked its seeding would
    make a round's randomness depend on whether the gate happened to run
    earlier in that process.
  - On CUDA, TF32 is switched off for the duration of the check and
    restored afterwards. cuDNN convolutions use TF32 by default
    (~1e-3 relative precision) and a batch-of-1 backward and a batch-of-4
    backward can pick different algorithms, so TF32 rounding alone could
    exceed rel_tol and fail a perfectly correct mode. The gate is meant
    to catch structurally wrong gradients (relative error of order 1),
    not float precision.
  - GPU memory is released (gc + empty_cache) after every run.

ensure_per_sample_gradients_verified runs the check once per process per
(grad_sample_mode, device type) and caches the outcome. A failure is
cached too and re-raised on EVERY later call -- the gate can never
silently pass after having failed.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.metadata
import logging
import math
import numbers
import threading
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.coatnet_lora import NUM_DR_GRADES, freeze_bn_eval
from training.local_trainer import weighted_label_smoothed_ce

logger = logging.getLogger(__name__)

__all__ = [
    "DPGradientCheckError",
    "verify_per_sample_gradients",
    "ensure_per_sample_gradients_verified",
    "clear_verification_cache",
]

_IMG_SIZE = 224
_LABEL_SMOOTHING = 0.1  # the locked value local_trainer uses
_LORA_B_STD = 0.02
_NUM_WORST_REPORTED = 8


class DPGradientCheckError(RuntimeError):
    """Raised when Opacus's per-sample gradients do not match the
    single-sample autograd reference (or the check cannot be run
    meaningfully). A DP experiment must not start after this.
    """


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _as_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    raise ValueError(f"device must be a torch.device or str, got {type(device).__name__}")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def _validate_check_arguments(
    model_factory: Any, grad_sample_mode: Any, batch_size: Any, rel_tol: Any, seed: Any
) -> None:
    problems: List[str] = []
    if not callable(model_factory):
        problems.append("model_factory must be a zero-argument callable")
    if not isinstance(grad_sample_mode, str) or not grad_sample_mode:
        problems.append(f"grad_sample_mode must be a non-empty string, got {grad_sample_mode!r}")
    if not _is_int(batch_size) or batch_size < 2:
        problems.append(
            f"batch_size must be an int >= 2 (a batch of 1 cannot distinguish "
            f"per-sample from batch gradients), got {batch_size!r}"
        )
    if (
        not isinstance(rel_tol, numbers.Real)
        or isinstance(rel_tol, bool)
        or not math.isfinite(rel_tol)
        or rel_tol <= 0
    ):
        problems.append(f"rel_tol must be a finite number > 0, got {rel_tol!r}")
    if not _is_int(seed) or seed < 0:
        problems.append(f"seed must be an int >= 0, got {seed!r}")
    if problems:
        raise ValueError("verify_per_sample_gradients: invalid arguments:\n  " + "\n  ".join(problems))


def _randomize_lora_b(model: torch.nn.Module, seed: int) -> int:
    """Fill every lora_B tensor with small seeded random values (same
    values for any model built identically, since parameter order is
    identical). Returns how many tensors were filled.
    """
    generator = torch.Generator().manual_seed(seed)
    count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                values = torch.randn(param.shape, generator=generator) * _LORA_B_STD
                param.copy_(values.to(device=param.device, dtype=param.dtype))
                count += 1
    return count


def _build_checked_model(
    model_factory: Callable[[], torch.nn.Module], device: torch.device, seed: int
) -> torch.nn.Module:
    torch.manual_seed(seed)  # identical init for both models
    model = model_factory().to(device)
    if _randomize_lora_b(model, seed + 1) == 0:
        raise DPGradientCheckError(
            "model_factory returned a model with no lora_B parameters -- the "
            "gate needs the real LoRA architecture to be meaningful."
        )
    model.train()
    freeze_bn_eval(model)  # frozen BatchNorm must stay in eval, as in real training
    return model


@contextlib.contextmanager
def _tf32_disabled(device: torch.device) -> Iterator[None]:
    """On CUDA, switch TF32 off for the duration of the check (see module
    docstring) and restore the previous settings afterwards. No-op on CPU.
    """
    if device.type != "cuda":
        yield
        return
    previous_cudnn = torch.backends.cudnn.allow_tf32
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous_cudnn
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul


def _run_check(
    model_factory: Callable[[], torch.nn.Module],
    device: torch.device,
    grad_sample_mode: str,
    batch_size: int,
    rel_tol: float,
    seed: int,
) -> Dict[str, Any]:
    try:
        from opacus import PrivacyEngine
    except ImportError as exc:
        raise DPGradientCheckError(
            "Could not import opacus.PrivacyEngine -- the per-sample-gradient "
            "gate cannot run, so DP training must not proceed."
        ) from exc

    generator = torch.Generator().manual_seed(seed + 2)
    x_cpu = torch.randn(batch_size, 3, _IMG_SIZE, _IMG_SIZE, generator=generator)
    y_cpu = torch.randint(0, NUM_DR_GRADES, (batch_size,), generator=generator)
    class_weights = (torch.rand(NUM_DR_GRADES, generator=generator) + 0.5).to(device)
    x, y = x_cpu.to(device), y_cpu.to(device)

    # ---- Reference: a loop of single-sample autograd backward passes.
    reference_model = _build_checked_model(model_factory, device, seed)
    named = [(n, p) for n, p in reference_model.named_parameters() if p.requires_grad]
    if not named:
        raise DPGradientCheckError("model_factory returned a model with no trainable parameters.")
    names = [n for n, _ in named]
    reference_params = [p for _, p in named]

    rows: List[List[torch.Tensor]] = [[] for _ in reference_params]
    for i in range(batch_size):
        for p in reference_params:
            p.grad = None
        logits = reference_model(x[i : i + 1])
        loss = weighted_label_smoothed_ce(
            logits, y[i : i + 1], class_weights, label_smoothing=_LABEL_SMOOTHING
        )
        loss.backward()
        for k, p in enumerate(reference_params):
            rows[k].append(torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
    reference = [torch.stack(r) for r in rows]
    del rows, reference_model, reference_params, named

    # ---- Test: the same architecture wrapped by Opacus.
    test_model = _build_checked_model(model_factory, device, seed)
    optimizer = torch.optim.SGD([p for p in test_model.parameters() if p.requires_grad], lr=0.0)
    loader = DataLoader(TensorDataset(x_cpu, y_cpu), batch_size=batch_size, shuffle=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Secure RNG turned off")
        engine = PrivacyEngine(accountant="rdp")
        wrapped, _dp_optimizer, _dp_loader = engine.make_private(
            module=test_model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=0.0,
            max_grad_norm=1e6,
            poisson_sampling=False,
            grad_sample_mode=grad_sample_mode,
        )
    try:
        wrapped.train()
        freeze_bn_eval(wrapped)
        logits = wrapped(x)
        loss = weighted_label_smoothed_ce(
            logits, y, class_weights, label_smoothing=_LABEL_SMOOTHING
        )
        loss.backward()

        test_params = [p for p in wrapped.parameters() if p.requires_grad]
        if len(test_params) != len(reference):
            raise DPGradientCheckError(
                f"Wrapped model has {len(test_params)} trainable tensors but the "
                f"reference model has {len(reference)}."
            )

        errors: List[Tuple[float, str, int]] = []  # (relative error, tensor name, sample)
        structural: List[str] = []
        lora_a_nonzero = head_nonzero = False
        n_comparisons = 0

        for name, ref, param in zip(names, reference, test_params):
            grad_sample = getattr(param, "grad_sample", None)
            if grad_sample is None:
                structural.append(f"{name}: no grad_sample after backward")
                continue
            if tuple(grad_sample.shape) != (batch_size,) + tuple(param.shape):
                structural.append(
                    f"{name}: grad_sample shape {tuple(grad_sample.shape)}, "
                    f"expected {(batch_size,) + tuple(param.shape)}"
                )
                continue
            for i in range(batch_size):
                a = ref[i].double()
                b = grad_sample[i].detach().to(a.device).double()
                a_norm = float(torch.linalg.vector_norm(a))
                if a_norm > 0.0:
                    err = float(torch.linalg.vector_norm(a - b)) / a_norm
                    if "lora_A" in name:
                        lora_a_nonzero = True
                    if "modules_to_save" in name:
                        head_nonzero = True
                else:
                    err = 0.0 if float(b.abs().max()) == 0.0 else float("inf")
                if not math.isfinite(err):
                    err = float("inf")  # NaN/inf in either tensor is a failure
                errors.append((err, name, i))
                n_comparisons += 1
    finally:
        remove_hooks = getattr(wrapped, "remove_hooks", None)
        if callable(remove_hooks):
            try:
                remove_hooks()
            except Exception:  # cleanup only; never mask the real outcome
                logger.debug("dp_checks: remove_hooks() failed", exc_info=True)

    problems: List[str] = list(structural)
    failing = sorted((e for e in errors if not (e[0] < rel_tol)), reverse=True)
    if failing:
        failing_tensors = sorted({name for _, name, _ in failing})
        worst = "; ".join(
            f"{name} (sample {i}): rel_err={err:.3e}"
            for err, name, i in failing[:_NUM_WORST_REPORTED]
        )
        problems.append(
            f"{len(failing_tensors)} of {len(names)} trainable tensors exceed "
            f"rel_tol={rel_tol:g}; worst: {worst}"
        )
    if not lora_a_nonzero:
        problems.append(
            "no non-zero LoRA-A reference gradient was found -- the check would "
            "pass vacuously for the LoRA matrices it exists to protect"
        )
    if not head_nonzero:
        problems.append("no non-zero head reference gradient was found")
    if problems:
        raise DPGradientCheckError(
            f"Per-sample-gradient gate FAILED for grad_sample_mode="
            f"{grad_sample_mode!r} on {device.type}: " + " | ".join(problems)
            + " -- do not run DP training with this configuration."
        )

    worst_rel_err = max((e[0] for e in errors), default=0.0)
    return {
        "worst_rel_err": worst_rel_err,
        "n_tensors": len(names),
        "n_comparisons": n_comparisons,
        "grad_sample_mode": grad_sample_mode,
        "device_type": device.type,
        "torch_version": str(torch.__version__),
        "opacus_version": _package_version("opacus"),
        "rel_tol": float(rel_tol),
        "batch_size": int(batch_size),
    }


def verify_per_sample_gradients(
    model_factory: Callable[[], torch.nn.Module],
    *,
    device: Any,
    grad_sample_mode: str = "functorch",
    batch_size: int = 4,
    rel_tol: float = 1e-3,
    seed: int = 0,
) -> Dict[str, Any]:
    """Prove Opacus's per-sample gradients equal single-sample autograd.

    Args:
        model_factory: zero-argument callable returning a FRESH peft model
            of the real architecture (e.g. a closure over
            create_coatnet_lora_model with the experiment's rank/alpha;
            pretrained=True on the pod, False for CPU unit tests). Called
            twice, after torch.manual_seed(seed) each time, so both models
            are identical.
        device: torch.device (or str) to run on.
        grad_sample_mode: the Opacus mode under test. "functorch" is the
            project's locked mode; "hooks" is expected to FAIL on this
            architecture (that is what the gate exists to catch).
        batch_size: samples compared (>= 2).
        rel_tol: maximum allowed relative error per tensor per sample.
        seed: seeds model init, lora_B fill, data, labels, class weights.

    Returns a summary dict: worst_rel_err, n_tensors, n_comparisons,
    grad_sample_mode, device_type, torch_version, opacus_version, rel_tol,
    batch_size.

    Raises DPGradientCheckError on any mismatch (listing the worst
    tensors), on a vacuous check, or if the check cannot be run;
    ValueError for invalid arguments. Leaves the global RNG state and the
    TF32 settings as it found them.
    """
    device = _as_device(device)
    _validate_check_arguments(model_factory, grad_sample_mode, batch_size, rel_tol, seed)

    fork_devices: List[int] = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    try:
        with torch.random.fork_rng(devices=fork_devices), _tf32_disabled(device):
            summary = _run_check(model_factory, device, grad_sample_mode, batch_size, rel_tol, seed)
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(
        "dp_checks: per-sample gradients verified (mode=%s, device=%s, "
        "worst_rel_err=%.3e over %d comparisons, opacus=%s, torch=%s)",
        summary["grad_sample_mode"], summary["device_type"], summary["worst_rel_err"],
        summary["n_comparisons"], summary["opacus_version"], summary["torch_version"],
    )
    return summary


# ---------------------------------------------------------------------------
# Process-level cache
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
# key -> ("ok", summary dict) | ("error", exception instance)
_VERIFICATION_CACHE: Dict[Tuple[str, str], Tuple[str, Any]] = {}


def ensure_per_sample_gradients_verified(
    model_factory: Callable[[], torch.nn.Module],
    *,
    device: Any,
    grad_sample_mode: str = "functorch",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run verify_per_sample_gradients at most once per process for each
    (grad_sample_mode, device.type), and return the cached summary on
    later calls (a copy -- callers cannot mutate the cache).

    If the first run RAISED (any Exception), that exception is cached and
    re-raised on EVERY later call with the same key: the gate can never
    silently pass after having failed. Extra keyword arguments
    (batch_size, rel_tol, seed) are forwarded to the first run only; a
    later call with different values still gets the cached outcome.
    """
    device = _as_device(device)
    key = (str(grad_sample_mode), device.type)
    with _CACHE_LOCK:
        cached = _VERIFICATION_CACHE.get(key)
        if cached is None:
            try:
                summary = verify_per_sample_gradients(
                    model_factory, device=device, grad_sample_mode=grad_sample_mode, **kwargs
                )
            except Exception as exc:  # not BaseException: Ctrl-C must not be cached
                _VERIFICATION_CACHE[key] = ("error", exc)
                raise
            _VERIFICATION_CACHE[key] = ("ok", summary)
            return dict(summary)

    status, value = cached
    if status == "error":
        raise value
    return dict(value)


def clear_verification_cache() -> None:
    """Forget every cached outcome (used by tests; not for production
    code, which should rely on the once-per-process guarantee)."""
    with _CACHE_LOCK:
        _VERIFICATION_CACHE.clear()