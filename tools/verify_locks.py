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


def load_manifest(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"lock manifest missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"lock manifest is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"lock manifest must contain a JSON object: {path}")
    if payload.get("algorithm") != "strict-md5-v1":
        raise RuntimeError(
            f"unsupported or missing manifest algorithm in {path}: "
            f"{payload.get('algorithm')!r}"
        )
    files = payload.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"lock manifest 'files' must be a JSON object: {path}")
    return payload


def verify(repo_root: Path, manifest_path: Path, locked_files: Sequence[str]) -> list[str]:
    payload = load_manifest(manifest_path)
    entries = payload["files"]
    if not isinstance(entries, dict):
        raise RuntimeError("lock manifest 'files' must be a JSON object")

    failures: list[str] = []
    expected = set(locked_files)
    actual = set(entries.keys())

    for relative in sorted(expected - actual):
        failures.append(f"missing manifest entry: {relative}")
    for relative in sorted(actual - expected):
        failures.append(f"unexpected manifest entry: {relative}")

    for relative in locked_files:
        path = repo_root / Path(relative)
        if not path.is_file():
            failures.append(f"missing locked file: {relative}")
            continue
        if relative not in entries:
            continue
        expected_hash = entries[relative]
        if not isinstance(expected_hash, str) or len(expected_hash) != 32:
            failures.append(f"invalid manifest hash: {relative}")
            continue
        if expected_hash.upper() != expected_hash:
            failures.append(f"manifest hash must be uppercase: {relative}")
            continue
        try:
            actual_hash = strict_md5(path)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        if actual_hash != expected_hash:
            failures.append(
                f"{relative}: expected {expected_hash}, found {actual_hash}"
            )
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the repository strict-MD5 lock manifest."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: the directory containing tools/)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path (default: <repo-root>/locks/locks.json)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else (repo_root / "locks" / "locks.json")
    try:
        locked_files = validate_locked_list(LOCKED_FILES)
        failures = verify(repo_root, manifest_path, locked_files)
    except (RuntimeError, OSError) as exc:
        print(f"lock verification failed: {exc}", file=sys.stderr)
        return 1

    if failures:
        print("lock verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"lock verification passed: {len(locked_files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
