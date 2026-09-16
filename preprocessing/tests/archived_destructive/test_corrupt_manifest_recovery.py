"""
Corrupt-manifest recovery test.

Run from the project root with dr_project activated:
    python -m preprocessing.test_corrupt_manifest_recovery
"""
import shutil
import subprocess
import sys
from pathlib import Path

CLIENT = "EyePACS"
DATA_ROOT = r"C:\MajorProject - Datasets"
MAX_SAMPLES = 50
MANIFEST_PATH = Path("data/splits/EyePACS_manifest.csv")
CACHE_DIR = Path("data/cache/EyePACS")

BUILD_CMD = [
    sys.executable, "-m", "preprocessing.build_dataset",
    "--client", CLIENT,
    "--data-root", DATA_ROOT,
    "--max-samples", str(MAX_SAMPLES),
]


def clear_eyepacs_artifacts():
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)


def main():
    print("=" * 60)
    print("CORRUPT-MANIFEST RECOVERY TEST")
    print("=" * 60)

    print("Building a small clean manifest first...")
    clear_eyepacs_artifacts()
    result = subprocess.run(BUILD_CMD)
    assert result.returncode == 0, "Setup build failed -- fix that before testing recovery."

    print("Truncating the manifest file mid-line to corrupt it...")
    original = MANIFEST_PATH.read_text()
    assert len(original) > 200, "Manifest too short to truncate meaningfully -- increase MAX_SAMPLES."
    MANIFEST_PATH.write_text(original[: len(original) // 2])

    print("Re-running the identical build command against the corrupted manifest...")
    result = subprocess.run(BUILD_CMD, capture_output=True, text=True)

    print(f"Exit code: {result.returncode}")
    print("--- stdout (tail) ---")
    print("\n".join(result.stdout.strip().splitlines()[-15:]))
    print("--- stderr (tail) ---")
    print("\n".join(result.stderr.strip().splitlines()[-15:]))

    assert result.returncode != 0, (
        "FAIL: build_dataset.py exited 0 against a corrupted manifest -- it either "
        "silently ignored the corruption or silently rebuilt without telling you."
    )
    print("\nPASS (partial): non-zero exit on corruption. Manually confirm the error "
          "above actually names the manifest/corruption as the cause, not an unrelated crash.")


if __name__ == "__main__":
    main()