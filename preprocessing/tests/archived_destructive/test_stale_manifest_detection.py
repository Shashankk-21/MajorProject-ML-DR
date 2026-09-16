"""
Stale-manifest detection test.

This is literally the transition you're about to make for real (a small
--max-samples cap -> --max-samples none) -- if this fails silently, so
could tomorrow's real launch.

Run from the project root with dr_project activated:
    python -m preprocessing.test_stale_manifest_detection
"""
import shutil
import subprocess
import sys
from pathlib import Path

CLIENT = "EyePACS"
DATA_ROOT = r"C:\MajorProject - Datasets"
FIRST_MAX_SAMPLES = 50
SECOND_MAX_SAMPLES = 150  # deliberately different
MANIFEST_PATH = Path("data/splits/EyePACS_manifest.csv")
CACHE_DIR = Path("data/cache/EyePACS")


def clear_eyepacs_artifacts():
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)


def build(max_samples, capture=False):
    cmd = [
        sys.executable, "-m", "preprocessing.build_dataset",
        "--client", CLIENT,
        "--data-root", DATA_ROOT,
        "--max-samples", str(max_samples),
    ]
    return subprocess.run(cmd, capture_output=capture, text=True)


def main():
    print("=" * 60)
    print("STALE-MANIFEST DETECTION TEST")
    print("=" * 60)

    print(f"Building with --max-samples {FIRST_MAX_SAMPLES}...")
    clear_eyepacs_artifacts()
    result = build(FIRST_MAX_SAMPLES)
    assert result.returncode == 0, "Setup build failed -- fix that before testing stale detection."

    print(f"\nAttempting to resume with a DIFFERENT --max-samples {SECOND_MAX_SAMPLES}...")
    result = build(SECOND_MAX_SAMPLES, capture=True)

    print(f"Exit code: {result.returncode}")
    print("--- stdout (tail) ---")
    print("\n".join(result.stdout.strip().splitlines()[-15:]))
    print("--- stderr (tail) ---")
    print("\n".join(result.stderr.strip().splitlines()[-15:]))

    assert result.returncode != 0, (
        "FAIL: mismatched --max-samples was accepted silently against an existing "
        "manifest -- exactly the scenario you're about to hit for real."
    )
    print("\nPASS (partial): mismatch not silently accepted. Manually confirm the "
          "error above actually names the max-samples mismatch as the cause.")


if __name__ == "__main__":
    main()