"""
Interrupt/resume test for build_dataset.py.

Uses a moderate EyePACS max-samples cap so there's real processing time to
interrupt into after zip reconstruction (reconstruction happens once per run
regardless of cap, so this costs a few minutes of zip-rebuild overhead on
each run -- expected).

Run from the project root with dr_project activated:
    python -m preprocessing.test_interrupt_resume
"""
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

CLIENT = "EyePACS"
DATA_ROOT = r"C:\MajorProject - Datasets"
MAX_SAMPLES = 300  # big enough to leave processing time after zip reconstruction
MANIFEST_PATH = Path("data/splits/EyePACS_manifest.csv")
CACHE_DIR = Path("data/cache/EyePACS")
ZIP_WORK_DIR = Path("data/cache/_eyepacs_zip_work")

BUILD_CMD = [
    sys.executable, "-m", "preprocessing.build_dataset",
    "--client", CLIENT,
    "--data-root", DATA_ROOT,
    "--max-samples", str(MAX_SAMPLES),
]


def manifest_row_count() -> int:
    if not MANIFEST_PATH.exists():
        return 0
    with open(MANIFEST_PATH, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def cached_png_count() -> int:
    if not CACHE_DIR.exists():
        return 0
    return len(list(CACHE_DIR.rglob("*.png")))


def clear_eyepacs_artifacts():
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)


def main():
    print("=" * 60)
    print("INTERRUPT/RESUME TEST")
    print("=" * 60)

    print("Clearing any existing EyePACS dev artifacts before starting...")
    clear_eyepacs_artifacts()

    print(f"Launching: {' '.join(BUILD_CMD)}")
    proc = subprocess.Popen(BUILD_CMD)

    # Poll for partial progress rather than guessing a fixed sleep -- kills
    # once PNGs start appearing, which only happens after zip reconstruction
    # fully completes (per your own log ordering), so this reliably lands
    # mid-processing rather than mid-reconstruction.
    deadline = time.time() + 15 * 60
    killed_with_partial_progress = False
    while time.time() < deadline:
        time.sleep(5)
        if proc.poll() is not None:
            print("Process finished before we got a chance to interrupt it -- "
                  "rerun with a larger MAX_SAMPLES to leave more processing time.")
            break
        n_cached = cached_png_count()
        if n_cached > 5:
            print(f"Partial progress detected: {n_cached} PNGs cached. Killing process now.")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            killed_with_partial_progress = True
            break

    if not killed_with_partial_progress:
        print("FAIL: never observed partial progress to interrupt into.")
        return

    partial_rows = manifest_row_count()
    partial_cached = cached_png_count()
    print(f"After kill: manifest rows={partial_rows}, cached PNGs={partial_cached}")

    print("\nResuming with the identical command...")
    result = subprocess.run(BUILD_CMD)
    if result.returncode != 0:
        print(f"FAIL: resumed run exited non-zero ({result.returncode}).")
        return

    final_rows = manifest_row_count()
    final_cached = cached_png_count()
    print(f"After resume: manifest rows={final_rows}, cached PNGs={final_cached}")

    with open(MANIFEST_PATH, newline="") as f:
        ids = [row["image_id"] for row in csv.DictReader(f)]
    duplicates = len(ids) - len(set(ids))

    leftover = list(ZIP_WORK_DIR.glob("*")) if ZIP_WORK_DIR.exists() else []
    if leftover:
        print(f"NOTE: {len(leftover)} file(s) left in {ZIP_WORK_DIR} after resume: "
              f"{[p.name for p in leftover]} -- worth checking whether that's "
              f"intended reuse or a cleanup gap.")
    else:
        print(f"{ZIP_WORK_DIR} is clean after resume.")

    assert duplicates == 0, f"FAIL: {duplicates} duplicate image_id(s) in resumed manifest."
    assert final_rows >= partial_rows, "FAIL: resumed manifest has fewer rows than the interrupted one."
    assert final_cached == final_rows, (
        f"FAIL: cached PNG count ({final_cached}) doesn't match manifest row count ({final_rows})."
    )

    print(f"\nPASS: resume completed cleanly. 0 duplicates, {final_rows} rows, {final_cached} cached PNGs.")


if __name__ == "__main__":
    main()