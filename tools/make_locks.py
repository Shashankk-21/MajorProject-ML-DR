from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MANIFEST_PATH = REPO_ROOT / "locks" / "locks.json"

LOCKED_FILES: tuple[str, ...] = (
    "federated/server.py",
    "federated/client.py",
    "training/train_centralized.py",
    "training/local_trainer.py",
    "models/coatnet_lora.py",
    "utils/checkpoint.py",
    "utils/dp_checks.py",
    "utils/experiment_tracker.py",
    "utils/privacy_calibration.py",
    "preprocessing/dataset.py",
    "preprocessing/augmentation.py",
    "configs/loader.py",
    "configs/base.yaml",
    "configs/exp1.yaml",
    "configs/exp2.yaml",
    "configs/exp3.yaml",
    "configs/exp4.yaml",
    "configs/exp5.yaml",
    "configs/exp6.yaml",
    "configs/exp7.yaml",
    "configs/exp8.yaml",
    "pyproject.toml",
    "notebooks/stepzero/grad_norm_probe.py",
    "notebooks/stepzero/rank_sweep.py",
    "notebooks/stepzero/stepzero_lib.py",
    "notebooks/stepzero/timing_probe.py",
    "evaluation/bootstrap.py",
    "evaluation/evaluate.py",
    "evaluation/metrics.py",
    "notebooks/runner/run_queue.py",
    "notebooks/runner/run_experiment1.py",
)


def normalize_file_text(text: str) -> bytes:
    """Apply the strict lock-normalization algorithm."""
    normalized_lines = [
        line.rstrip()
        for line in text.splitlines()
    ]
    normalized = "\n".join(normalized_lines).rstrip("\r\n")
    return (normalized + "\n").encode("utf-8")


def strict_md5(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{path}: file is not valid UTF-8: {exc}") from exc
    return hashlib.md5(normalize_file_text(text)).hexdigest().upper()


def validate_locked_list(paths: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_absolute():
            raise RuntimeError(f"LOCKED_FILES entries must be repository-relative: {raw}")
        normalized = path.as_posix()
        if normalized in seen:
            raise RuntimeError(f"duplicate locked file entry: {normalized}")
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def build_manifest(repo_root: Path, locked_files: Sequence[str]) -> Dict[str, object]:
    files: Dict[str, str] = {}
    missing: list[str] = []
    for relative in locked_files:
        path = repo_root / Path(relative)
        if not path.is_file():
            missing.append(relative)
            continue
        files[relative] = strict_md5(path)
    if missing:
        raise RuntimeError(
            "Cannot generate locks/locks.json; missing locked file(s):\n  "
            + "\n  ".join(missing)
        )
    return {
        "algorithm": "strict-md5-v1",
        "files": files,
    }


def write_manifest(path: Path, manifest: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temp.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the repository strict-MD5 lock manifest."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: the directory containing tools/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest output (default: <repo-root>/locks/locks.json)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    output = args.output.resolve() if args.output else (repo_root / "locks" / "locks.json")
    try:
        locked_files = validate_locked_list(LOCKED_FILES)
        manifest = build_manifest(repo_root, locked_files)
        write_manifest(output, manifest)
    except (RuntimeError, OSError) as exc:
        print(f"lock manifest generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"generated {output.relative_to(repo_root) if output.is_relative_to(repo_root) else output}")
    print(f"locked files: {len(locked_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
