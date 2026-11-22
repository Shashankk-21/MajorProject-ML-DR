from __future__ import annotations

import importlib
import importlib.metadata
import os
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
failed = False


def emit(level: str, check: str, detail: str = "") -> None:
    global failed
    if level == "FAIL":
        failed = True
    suffix = f": {detail}" if detail else ""
    print(f"{level} {check}{suffix}")


def under_data(value: str) -> bool:
    if not value:
        return False
    try:
        return Path(value).resolve().is_relative_to(Path("/data"))
    except (OSError, RuntimeError):
        return False


# 1. Python environment
if sys.version_info[:2] == (3, 12):
    emit("PASS", "Python", f"{sys.version.split()[0]}")
else:
    emit("FAIL", "Python", f"expected 3.12, found {sys.version.split()[0]}")

if sys.prefix == "/data/venv":
    emit("PASS", "sys.prefix", sys.prefix)
else:
    emit("FAIL", "sys.prefix", f"expected /data/venv, found {sys.prefix}")


# 2. Environment variables
env_checks = (
    ("FLWR_HOME", "/data/.flwr"),
    ("FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION", "1"),
)
for name, expected in env_checks:
    actual = os.environ.get(name)
    if actual == expected:
        emit("PASS", name, expected)
    else:
        emit("FAIL", name, f"expected {expected!r}, found {actual!r}")

for name in ("HF_HOME", "TORCH_HOME", "WANDB_DIR"):
    value = os.environ.get(name)
    if under_data(value or ""):
        emit("PASS", name, value)
    else:
        emit("FAIL", name, f"expected a path under /data, found {value!r}")

pip_constraint = os.environ.get("PIP_CONSTRAINT")
if pip_constraint == "":
    emit("PASS", "PIP_CONSTRAINT", "empty")
else:
    emit("FAIL", "PIP_CONSTRAINT", f"expected empty, found {pip_constraint!r}")


# 3. Package versions
expected_versions = {
    "torch": "2.7.0a0+79aa17489c.nv25.04",
    "numpy": "1.26.4",
    "opencv": "4.10.0",
    "albumentations": "2.0.8",
    "flwr": "1.37.0",
    "opacus": "1.6.0",
    "peft": "0.21.0",
    "timm": "1.0.29",
    "wandb": "0.30.0",
    "ray": "2.55.1",
    "typer": "0.19.2",
}
distribution_names = {
    "opencv": "opencv-python",
}
for package, expected in expected_versions.items():
    try:
        if package == "torch":
            module = importlib.import_module("torch")
            actual = getattr(module, "__version__", "")
        elif package == "opencv":
            module = importlib.import_module("cv2")
            actual = getattr(module, "__version__", "")
        else:
            actual = importlib.metadata.version(distribution_names.get(package, package))
        if actual == expected:
            emit("PASS", f"version {package}", actual)
        else:
            emit("FAIL", f"version {package}", f"expected {expected}, found {actual}")
    except (ImportError, importlib.metadata.PackageNotFoundError, AttributeError) as exc:
        emit("FAIL", f"version {package}", f"not available: {exc}")


# 4. CUDA
try:
    import torch

    cuda_ok = bool(torch.cuda.is_available())
    if cuda_ok:
        emit("PASS", "CUDA available", "yes")

        name = torch.cuda.get_device_name(0)
        if "MIG" in name.upper() or "H200" in name.upper():
            emit("PASS", "CUDA device name", name)
        else:
            emit("FAIL", "CUDA device name", f"expected MIG or H200, found {name}")

        total_bytes = torch.cuda.get_device_properties(0).total_memory
        total_gib = total_bytes / (1024 ** 3)
        if 14 <= total_gib <= 18:
            emit("PASS", "CUDA memory", f"{total_gib:.2f} GiB")
        else:
            emit("FAIL", "CUDA memory", f"expected 14 to 18 GiB, found {total_gib:.2f} GiB")

        try:
            torch.cuda.synchronize()
            result = torch.matmul(torch.ones((16, 16), device="cuda"), torch.ones((16, 16), device="cuda"))
            torch.cuda.synchronize()
            if result.shape == (16, 16):
                emit("PASS", "CUDA matmul", "16x16 matmul completed")
            else:
                emit("FAIL", "CUDA matmul", f"unexpected result shape {tuple(result.shape)}")
        except Exception as exc:
            emit("FAIL", "CUDA matmul", str(exc))
    else:
        emit("FAIL", "CUDA available", "no")
        emit("FAIL", "CUDA device name", "CUDA unavailable")
        emit("FAIL", "CUDA memory", "CUDA unavailable")
        emit("FAIL", "CUDA matmul", "CUDA unavailable")
except Exception as exc:
    emit("FAIL", "CUDA", str(exc))


# 5. Ray
try:
    import ray # type: ignore

    ray_started = False
    try:
        ray.init(include_dashboard=False)
        ray_started = True
        resources = ray.cluster_resources()
        gpu_count = float(resources.get("GPU", 0))
        if gpu_count >= 1:
            emit("PASS", "Ray GPU resources", f"{gpu_count:g}")
        else:
            emit("FAIL", "Ray GPU resources", f"GPU={gpu_count:g}")

        @ray.remote(num_gpus=1)
        def ray_gpu_probe():
            import torch
            return bool(torch.cuda.is_available()), torch.cuda.get_device_name(0)

        available, device_name = ray.get(ray_gpu_probe.remote())
        if available:
            emit("PASS", "Ray remote GPU task", device_name)
        else:
            emit("FAIL", "Ray remote GPU task", f"CUDA unavailable; device={device_name}")
    except Exception as exc:
        emit("FAIL", "Ray", str(exc))
    finally:
        if ray_started:
            ray.shutdown()
except Exception as exc:
    emit("FAIL", "Ray", str(exc))


# 6. Storage
try:
    import shutil

    shm = shutil.disk_usage("/dev/shm")
    shm_gib = shm.total / (1024 ** 3)
    if shm_gib >= 8:
        emit("PASS", "/dev/shm", f"{shm_gib:.2f} GiB")
    else:
        emit("FAIL", "/dev/shm", f"expected at least 8 GiB, found {shm_gib:.2f} GiB")
except Exception as exc:
    emit("FAIL", "/dev/shm", str(exc))

try:
    result = subprocess.run(
        ["du", "-s", "-B1", "/data"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        emit("FAIL", "/data usage", result.stderr.strip() or f"du exited {result.returncode}")
    else:
        data_bytes = int(result.stdout.strip().split()[0])
        data_gib = data_bytes / (1024 ** 3)
        if data_gib < 40:
            emit("PASS", "/data usage", f"{data_gib:.2f} GiB")
        elif data_gib <= 45:
            emit("WARN", "/data usage", f"{data_gib:.2f} GiB (40-45 GiB)")
        else:
            emit("FAIL", "/data usage", f"{data_gib:.2f} GiB (above 45 GiB)")
except Exception as exc:
    emit("FAIL", "/data usage", str(exc))

checkpoint_dir = Path("/data/checkpoints")
try:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fd, probe_name = tempfile.mkstemp(prefix=".write-probe-", dir=checkpoint_dir)
    os.close(fd)
    Path(probe_name).unlink()
    emit("PASS", "/data/checkpoints writable", str(checkpoint_dir))
except Exception as exc:
    emit("FAIL", "/data/checkpoints writable", str(exc))


# 7. Flower
try:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from notebooks.runner.run_queue import validate_flower_config

    config_path = Path("/data/.flwr/config.toml")
    problems = validate_flower_config(config_path)
    if not problems:
        emit("PASS", "Flower configuration", str(config_path))
    else:
        emit("FAIL", "Flower configuration", "; ".join(problems))
except Exception as exc:
    emit("FAIL", "Flower configuration", str(exc))

superlink_found = False
superlink_env_ok = True
for proc_dir in Path("/proc").glob("[0-9]*"):
    try:
        comm = (proc_dir / "comm").read_text(encoding="utf-8", errors="replace").strip()
        cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ")
        is_superlink = "flower-superlink" in comm or b"flower-superlink" in cmdline
        if not is_superlink:
            continue
        superlink_found = True
        environ = (proc_dir / "environ").read_bytes()
        env_map = {}
        for entry in environ.split(b"\0"):
            if b"=" in entry:
                key, value = entry.split(b"=", 1)
                env_map[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
        if "FLWR_HOME" not in env_map:
            superlink_env_ok = False
        elif env_map["FLWR_HOME"] != "/data/.flwr":
            superlink_env_ok = False
    except (OSError, PermissionError):
        continue

if not superlink_found:
    emit("PASS", "flower-superlink process", "not running")
elif superlink_env_ok:
    emit("PASS", "flower-superlink process", "FLWR_HOME=/data/.flwr")
else:
    emit("FAIL", "flower-superlink process", "FLWR_HOME missing or incorrect")


# 8. Datasets and API key
dataset_counts = {
    "aptos": 3662,
    "idrid": 516,
    "messidor2": 1744,
    "eyepacs": 88702,
}
for name, expected in dataset_counts.items():
    folder = Path("/data/datasets") / name
    try:
        if not folder.is_dir():
            emit("FAIL", f"dataset {name}", "directory missing")
            continue
        count = sum(
            1
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".png"
        )
        if count == expected:
            emit("PASS", f"dataset {name}", f"{count} PNG files")
        else:
            emit("FAIL", f"dataset {name}", f"expected {expected} PNG files, found {count}")
    except Exception as exc:
        emit("FAIL", f"dataset {name}", str(exc))

api_key = Path("/data/.wandb_api_key")
try:
    mode = stat.S_IMODE(api_key.stat().st_mode)
    if api_key.is_file() and mode == 0o600:
        emit("PASS", "/data/.wandb_api_key", "exists with mode 600")
    elif not api_key.is_file():
        emit("FAIL", "/data/.wandb_api_key", "missing")
    else:
        emit("FAIL", "/data/.wandb_api_key", f"mode {mode:03o}, expected 600")
except Exception as exc:
    emit("FAIL", "/data/.wandb_api_key", str(exc))


# 9. Git
try:
    fetch = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "fetch", "--quiet", "origin", "main"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if fetch.returncode != 0:
        emit("FAIL", "git fetch", fetch.stderr.strip() or f"exited {fetch.returncode}")
    else:
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        origin = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "refs/remotes/origin/main"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if head and head == origin:
            emit("PASS", "git HEAD", "HEAD equals origin/main")
        else:
            emit("FAIL", "git HEAD", f"HEAD={head or 'unknown'}, origin/main={origin or 'unknown'}")

    tracked_status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--quiet", "HEAD", "--"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked_status.returncode == 0:
        emit("PASS", "tracked tree", "clean")
    else:
        emit("FAIL", "tracked tree", "tracked modifications detected")
except Exception as exc:
    emit("FAIL", "git", str(exc))

try:
    lock_check = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "verify_locks.py")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if lock_check.returncode == 0:
        emit("PASS", "verify_locks", "exit 0")
    else:
        emit("FAIL", "verify_locks", f"exit {lock_check.returncode}")
except Exception as exc:
    emit("FAIL", "verify_locks", str(exc))

# 10. Network
def reachable(host: str) -> bool:
    url = f"https://{host}/"
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status < 400
    except urllib.error.HTTPError:
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status < 400
        except Exception:
            return False
    except Exception:
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status < 400
        except Exception:
            return False


# Hugging Face: generic HTTPS connectivity check
try:
    if reachable("huggingface.co"):
        emit("PASS", "network huggingface.co", "reachable")
    else:
        emit("FAIL", "network huggingface.co", "not reachable")
except Exception as exc:
    emit("FAIL", "network huggingface.co", str(exc))


# Weights & Biases: verify using an authenticated API request
try:
    import wandb

    key_path = Path("/data/.wandb_api_key")
    api_key = key_path.read_text(encoding="utf-8").strip()

    if not api_key:
        emit("FAIL", "W&B API authentication", "API key is empty")
    else:
        username = wandb.Api(api_key=api_key).viewer.username
        if username:
            emit(
                "PASS",
                "W&B API authentication",
                f"authenticated as {username}",
            )
        else:
            emit("FAIL", "W&B API authentication", "viewer username missing")

except Exception as exc:
    emit("FAIL", "W&B API authentication", str(exc))


raise SystemExit(1 if failed else 0)
