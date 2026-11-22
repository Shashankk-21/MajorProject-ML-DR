from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import timm

from federated.client import _build_model
from models.coatnet_lora import (
    freeze_bn_eval,
    split_trainable_parameters,
    verify_lora_injection,
)


def print_result(ok: bool, check: str, detail: str = "") -> bool:
    suffix = f": {detail}" if detail else ""
    print(("PASS " if ok else "FAIL ") + check + suffix)
    return ok


def recursive_cache_info(hf_home: str) -> tuple[int, int]:
    root = Path(hf_home)
    if not root.is_dir():
        raise OSError(f"HF_HOME is not an inspectable directory: {root}")

    total_bytes = 0
    file_count = 0
    for path in root.rglob("*"):
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
    return total_bytes, file_count


def format_bytes(size_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0


def underlying_timm_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        candidate = model.base_model.model
        if hasattr(candidate, "named_modules"):
            return candidate
    if hasattr(model, "get_base_model"):
        candidate = model.get_base_model()
        if hasattr(candidate, "base_model") and hasattr(candidate.base_model, "model"):
            candidate = candidate.base_model.model
        if hasattr(candidate, "named_modules"):
            return candidate
    raise RuntimeError(
        "Could not locate the underlying timm backbone inside the PeftModel"
    )


def pretrained_cfg(base_model: torch.nn.Module) -> dict[str, Any]:
    cfg = getattr(base_model, "pretrained_cfg", None)
    if isinstance(cfg, dict):
        return cfg
    if cfg is not None:
        try:
            return dict(cfg)
        except (TypeError, ValueError):
            pass

    pretrained_cfgs = getattr(base_model, "pretrained_cfgs", None)
    if isinstance(pretrained_cfgs, dict) and pretrained_cfgs:
        first_cfg = next(iter(pretrained_cfgs.values()))
        if isinstance(first_cfg, dict):
            return first_cfg
    raise RuntimeError("Underlying timm backbone has no resolved pretrained_cfg")


def qualified_classifier_name(base_model: torch.nn.Module) -> str:
    classifier = base_model.get_classifier()
    if classifier is None:
        raise RuntimeError("Underlying timm backbone get_classifier() returned None")
    for name, module in base_model.named_modules():
        if name and module is classifier:
            return name
    raise RuntimeError("Could not find the classifier module in named_modules")


def classifier_backbone_filter(
    key: str,
    classifier_name: str,
) -> bool:
    key_lower = key.lower()
    if "lora_a" in key_lower or "lora_b" in key_lower:
        return False
    if "modules_to_save" in key_lower:
        return False
    if classifier_name and (
        key == classifier_name
        or key.startswith(classifier_name + ".")
        or key.endswith("." + classifier_name)
        or (".modules_to_save." in key and classifier_name in key)
    ):
        return False
    return True


def backbone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    base_model = underlying_timm_model(model)
    classifier_name = qualified_classifier_name(base_model)
    result: dict[str, torch.Tensor] = {}
    for key, tensor in model.state_dict().items():
        if not classifier_backbone_filter(key, classifier_name):
            continue
        normalized = key
        marker = ".base_model.model."
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]
        elif normalized.startswith("base_model.model."):
            normalized = normalized[len("base_model.model.") :]
        elif normalized.startswith("base_model."):
            normalized = normalized[len("base_model.") :]
        if normalized.startswith(classifier_name + "."):
            continue
        if normalized == classifier_name:
            continue
        if "lora_" in normalized or "modules_to_save" in normalized:
            continue
        result[normalized] = tensor.detach().cpu()
    return result


def batchnorm_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(
            module,
            (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
        )
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        module = sys.modules.get(name)
        return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    mandatory_failures: list[str] = []

    print(f"Python: {platform.python_version()} ({sys.version.split()[0]})")
    print(f"PyTorch: {torch.__version__}")
    print(f"timm: {package_version('timm')}")
    print(f"PEFT: {package_version('peft')}")

    hf_home = os.environ.get("HF_HOME")
    hf_home_ok = bool(hf_home)
    if not print_result(hf_home_ok, "HF_HOME", hf_home if hf_home else "unset"):
        mandatory_failures.append("HF_HOME")
    if hf_home:
        print(f"HF cache location: {hf_home}")

    rank4_cfg = {
        "lora-rank": 4,
        "lora-alpha": 8,
        "num-classes": 5,
        "img-size": 224,
        "pretrained": True,
    }
    rank8_cfg = {
        "lora-rank": 8,
        "lora-alpha": 16,
        "num-classes": 5,
        "img-size": 224,
        "pretrained": True,
    }

    cache_before = None
    cache_before_files = None
    if hf_home:
        try:
            cache_before, cache_before_files = recursive_cache_info(hf_home)
        except (OSError, RuntimeError) as exc:
            print_result(False, "HF cache inspection", str(exc))
            mandatory_failures.append("HF cache inspection")

    first_model = None
    second_model = None
    first_time = None
    second_time = None

    try:
        start = time.perf_counter()
        first_model = _build_model(rank4_cfg)
        first_time = time.perf_counter() - start
    except Exception as exc:
        print_result(False, "rank-4 first build", f"{type(exc).__name__}: {exc}")
        mandatory_failures.append("rank-4 first build")

    cache_after_first = None
    cache_after_first_files = None
    if hf_home:
        try:
            cache_after_first, cache_after_first_files = recursive_cache_info(hf_home)
        except (OSError, RuntimeError) as exc:
            print_result(False, "HF cache inspection after first build", str(exc))
            mandatory_failures.append("HF cache inspection")

    try:
        start = time.perf_counter()
        second_model = _build_model(rank4_cfg)
        second_time = time.perf_counter() - start
    except Exception as exc:
        print_result(False, "rank-4 second build", f"{type(exc).__name__}: {exc}")
        mandatory_failures.append("rank-4 second build")

    print(f"First rank-4 build time: {first_time:.6f} seconds" if first_time is not None else "First rank-4 build time: unavailable")
    print(f"Second rank-4 build time: {second_time:.6f} seconds" if second_time is not None else "Second rank-4 build time: unavailable")

    cache_after = None
    cache_after_files = None
    cache_population_detected = False
    if hf_home:
        try:
            cache_after, cache_after_files = recursive_cache_info(hf_home)
            print_result(
                True,
                "HF cache inspection",
                f"{cache_after_files} files, {format_bytes(cache_after)}",
            )
            if cache_before is not None and cache_after_first is not None:
                cache_population_detected = (
                    cache_after_first != cache_before
                    or cache_after_first_files != cache_before_files
                )
        except (OSError, RuntimeError) as exc:
            print_result(False, "HF cache inspection", str(exc))
            mandatory_failures.append("HF cache inspection")

    print(f"HF cache location: {hf_home if hf_home else 'unset'}")
    print(
        f"HF cache size: {format_bytes(cache_after) if cache_after is not None else 'unavailable'} "
        f"({cache_after_files if cache_after_files is not None else 'unavailable'} files)"
    )

    if (
        first_time is not None
        and second_time is not None
        and first_model is not None
        and second_model is not None
    ):
        if cache_population_detected:
            marginal = second_time <= first_time + max(0.05, first_time * 0.05)
            if second_time < first_time:
                ok = print_result(
                    True,
                    "second build cache timing",
                    f"first={first_time:.6f}s, second={second_time:.6f}s",
                )
            elif marginal:
                ok = print_result(
                    True,
                    "second build cache timing",
                    f"no material slowdown; first={first_time:.6f}s, "
                    f"second={second_time:.6f}s",
                )
            else:
                ok = print_result(
                    False,
                    "second build cache timing",
                    f"first={first_time:.6f}s, second={second_time:.6f}s",
                )
                mandatory_failures.append("second build cache timing")
        else:
            print_result(
                True,
                "second build cache timing",
                f"cache was already populated or unchanged; "
                f"first={first_time:.6f}s, second={second_time:.6f}s",
            )

    backbone_equal = False
    if first_model is not None and second_model is not None:
        try:
            first_state = backbone_state(first_model)
            second_state = backbone_state(second_model)
            key_sets_equal = set(first_state) == set(second_state)
            tensor_equal = key_sets_equal and all(
                torch.equal(first_state[key], second_state[key])
                for key in first_state
            )
            backbone_equal = key_sets_equal and tensor_equal
            detail = (
                f"{len(first_state)} backbone tensors, matching key sets={key_sets_equal}, "
                f"bit-identical={tensor_equal}"
            )
            print_result(
                backbone_equal,
                "backbone tensor equality",
                detail,
            )
            if not backbone_equal:
                mandatory_failures.append("backbone tensor equality")
        except Exception as exc:
            print_result(
                False,
                "backbone tensor equality",
                f"{type(exc).__name__}: {exc}",
            )
            mandatory_failures.append("backbone tensor equality")
    else:
        print_result(False, "backbone tensor equality", "rank-4 builds unavailable")
        mandatory_failures.append("backbone tensor equality")

    base_cfg = None
    if second_model is not None:
        try:
            base_model = underlying_timm_model(second_model)
            base_cfg = pretrained_cfg(base_model)
            hf_hub_id = base_cfg.get("hf_hub_id", "missing")
            tag = base_cfg.get("tag", "missing")
            print(f"Resolved pretrained_cfg hf_hub_id: {hf_hub_id}")
            print(f"Resolved pretrained_cfg tag: {tag}")
            print_result(
                hf_hub_id != "missing" and tag != "missing",
                "resolved pretrained configuration",
                f"hf_hub_id={hf_hub_id}, tag={tag}",
            )
            if hf_hub_id == "missing" or tag == "missing":
                mandatory_failures.append("resolved pretrained configuration")
        except Exception as exc:
            print_result(
                False,
                "resolved pretrained configuration",
                f"{type(exc).__name__}: {exc}",
            )
            mandatory_failures.append("resolved pretrained configuration")
    else:
        print_result(False, "resolved pretrained configuration", "rank-4 build unavailable")
        mandatory_failures.append("resolved pretrained configuration")

    bn_equal = False
    if second_model is not None:
        try:
            rank4_base = underlying_timm_model(second_model)
            direct_model = timm.create_model(
                "coatnet_1_rw_224",
                pretrained=True,
                num_classes=5,
                img_size=224,
            )
            direct_bn = batchnorm_modules(direct_model)
            rank4_bn = batchnorm_modules(rank4_base)
            name_sets_equal = set(direct_bn) == set(rank4_bn)
            counts_equal = len(direct_bn) == len(rank4_bn)
            mean_equal = True
            var_equal = True
            tracked_equal = True

            if name_sets_equal:
                for name in direct_bn:
                    direct = direct_bn[name]
                    rank4 = rank4_bn[name]
                    if getattr(direct, "running_mean", None) is None or getattr(rank4, "running_mean", None) is None:
                        mean_equal = mean_equal and getattr(direct, "running_mean", None) is getattr(rank4, "running_mean", None)
                    else:
                        mean_equal = mean_equal and torch.equal(direct.running_mean, rank4.running_mean)
                    if getattr(direct, "running_var", None) is None or getattr(rank4, "running_var", None) is None:
                        var_equal = var_equal and getattr(direct, "running_var", None) is getattr(rank4, "running_var", None)
                    else:
                        var_equal = var_equal and torch.equal(direct.running_var, rank4.running_var)
                    if getattr(direct, "num_batches_tracked", None) is None or getattr(rank4, "num_batches_tracked", None) is None:
                        tracked_equal = tracked_equal and getattr(direct, "num_batches_tracked", None) is getattr(rank4, "num_batches_tracked", None)
                    else:
                        tracked_equal = tracked_equal and torch.equal(direct.num_batches_tracked, rank4.num_batches_tracked)

            bn_equal = name_sets_equal and counts_equal and mean_equal and var_equal and tracked_equal
            print_result(
                bn_equal,
                "BatchNorm agreement",
                f"names={name_sets_equal}, count={len(direct_bn)}/{len(rank4_bn)}, "
                f"running_mean={mean_equal}, running_var={var_equal}, "
                f"num_batches_tracked={tracked_equal}",
            )
            if not bn_equal:
                mandatory_failures.append("BatchNorm agreement")
        except Exception as exc:
            print_result(
                False,
                "BatchNorm agreement",
                f"{type(exc).__name__}: {exc}",
            )
            mandatory_failures.append("BatchNorm agreement")
    else:
        print_result(False, "BatchNorm agreement", "rank-4 build unavailable")
        mandatory_failures.append("BatchNorm agreement")

    structural_models: dict[int, torch.nn.Module] = {}
    for rank, cfg in ((4, rank4_cfg), (8, rank8_cfg)):
        try:
            model = _build_model(cfg) if rank == 8 else (second_model if second_model is not None else _build_model(cfg))
            if model is None:
                raise RuntimeError("rank-4 model unavailable")

            verify_lora_injection(model, lora_rank=rank, img_size=224)

            lora_params, head_params = split_trainable_parameters(model)
            lora_count = sum(param.numel() for param in lora_params)
            head_count = sum(param.numel() for param in head_params)
            all_trainable = [param for param in model.parameters() if param.requires_grad]
            split_ids = {id(param) for param in lora_params + head_params}
            all_ids = {id(param) for param in all_trainable}
            accounted = split_ids == all_ids
            total_trainable = sum(param.numel() for param in all_trainable)

            print(
                f"Rank {rank}: LoRA parameters={lora_count}, "
                f"head parameters={head_count}, total trainable={total_trainable}"
            )
            if not print_result(bool(lora_params and head_params), f"rank-{rank} parameter groups"):
                mandatory_failures.append(f"rank-{rank} parameter groups")
            if not print_result(accounted, f"rank-{rank} trainable parameter accounting"):
                mandatory_failures.append(f"rank-{rank} trainable parameter accounting")
            if not print_result(True, f"rank-{rank} verify_lora_injection"):
                mandatory_failures.append(f"rank-{rank} verify_lora_injection")

            model.train()
            freeze_bn_eval(model)
            frozen_bn_ok = True
            frozen_bn_count = 0
            for module in model.modules():
                if isinstance(
                    module,
                    (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
                ):
                    params = list(module.parameters(recurse=False))
                    if params and all(not param.requires_grad for param in params):
                        frozen_bn_count += 1
                        if module.training:
                            frozen_bn_ok = False
            if not print_result(
                frozen_bn_ok,
                f"rank-{rank} frozen BatchNorm eval",
                f"{frozen_bn_count} frozen BatchNorm modules",
            ):
                mandatory_failures.append(f"rank-{rank} frozen BatchNorm eval")

            structural_models[rank] = model
        except Exception as exc:
            print_result(
                False,
                f"rank-{rank} LoRA/structural verification",
                f"{type(exc).__name__}: {exc}",
            )
            mandatory_failures.append(f"rank-{rank} LoRA/structural verification")

    offline_ok = False
    offline_output = ""
    offline_status = "not run"
    repo_root = Path(__file__).resolve().parents[1]
    offline_code = (
        "from federated.client import _build_model; "
        "cfg={'lora-rank':4,'lora-alpha':8,'num-classes':5,"
        "'img-size':224,'pretrained':True}; "
        "model=_build_model(cfg); "
        "print('offline rank-4 build passed')"
    )
    offline_env = os.environ.copy()
    offline_env["HF_HUB_OFFLINE"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", offline_code],
            cwd=str(repo_root),
            env=offline_env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        offline_status = str(completed.returncode)
        offline_output = (completed.stdout + completed.stderr).strip()
        offline_ok = completed.returncode == 0
        print("Offline subprocess output:")
        print(offline_output if offline_output else "(no output)")
        print(f"Offline subprocess exit status: {offline_status}")
        print_result(offline_ok, "offline cache proof")
        if not offline_ok:
            mandatory_failures.append("offline cache proof")
    except Exception as exc:
        print("Offline subprocess output:")
        print(f"{type(exc).__name__}: {exc}")
        print("Offline subprocess exit status: unavailable")
        print_result(False, "offline cache proof", f"{type(exc).__name__}: {exc}")
        mandatory_failures.append("offline cache proof")

    cuda_ok = False
    cuda_shape = "unavailable"
    cuda_finite = False
    cuda_model = structural_models.get(4)
    if cuda_model is None:
        cuda_model = second_model
    try:
        cuda_available = torch.cuda.is_available()
        print_result(cuda_available, "CUDA availability")
        if not cuda_available:
            mandatory_failures.append("CUDA forward")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU name: {gpu_name}")
            cuda_model = cuda_model.to("cuda")
            cuda_model.eval()
            batch = torch.randn(2, 3, 224, 224, device="cuda")
            with torch.inference_mode():
                output = cuda_model(batch)
            torch.cuda.synchronize()
            cuda_shape = str(tuple(output.shape))
            shape_ok = tuple(output.shape) == (2, 5)
            cuda_finite = bool(torch.isfinite(output).all().item())
            cuda_ok = shape_ok and cuda_finite
            print_result(shape_ok, "CUDA forward shape", f"shape={cuda_shape}, expected=(2, 5)")
            print_result(cuda_finite, "CUDA forward finiteness")
            if not cuda_ok:
                mandatory_failures.append("CUDA forward")
    except Exception as exc:
        print_result(
            False,
            "CUDA forward",
            f"{type(exc).__name__}: {exc}",
        )
        mandatory_failures.append("CUDA forward")

    if torch.cuda.is_available():
        try:
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    if mandatory_failures:
        print("PRETRAINED MODEL VALIDATION FAILED")
        return 1
    print("ALL PRETRAINED MODEL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
