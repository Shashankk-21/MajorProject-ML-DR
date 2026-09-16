from pathlib import Path
import csv
import zipfile

import numpy as np
from PIL import Image

from preprocessing.dataset import build_manifest
from preprocessing.image_processing import (
    circle_crop,
    blur_score,
    pad_to_square,
    resize_224,
    DEFAULT_BLUR_FLAG_THRESHOLD,
)

# EyePACS's zip-half reconstruction/parsing is non-trivial and already
# vetted in build_dataset.py -- reused here via import rather than
# duplicated. Nothing in build_dataset.py is modified.
from preprocessing.build_dataset import (
    _reconstruct_eyepacs_zip,
    _load_raw_image_from_zip,
    _parse_eyepacs_locator,
    _EYEPACS_ZIP_HALVES,
)


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path(r"C:\MajorProject - Datsets")
OUTPUT_ROOT = Path("data") / "_blur_cross_client"
EYEPACS_ZIP_WORK_DIR = OUTPUT_ROOT / "_eyepacs_zip_work"

RANDOM_STATE = 42
NUM_IMAGES_PER_CLIENT = 200

FILESYSTEM_CLIENTS = ("APTOS", "IDRiD", "Messidor-2")  # normal on-disk images
ALL_CLIENTS = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")

PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 100]
PERCENTILE_LABELS = {
    0: "min", 5: "5%", 10: "10%", 25: "25%",
    50: "median", 75: "75%", 90: "90%", 95: "95%", 100: "max",
}
THRESHOLDS = [50, 60, 80, 100, 120, 150]


# ============================================================
# SETUP
# ============================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# POST-RESIZE BLUR SCORE
# ============================================================

def compute_post_resize_blur_score(raw_image: np.ndarray) -> float:
    """Blur score measured AFTER resize_224, instead of run_offline_pipeline()'s
    own pre-resize value.

    run_offline_pipeline() calls blur_score() on the cropped-but-still-full-
    resolution image (see its docstring: "measured post-crop, pre-resize").
    That's the value cached in the real manifest and used for
    low_quality_flag. This function instead calls the same public
    circle_crop -> pad_to_square -> resize_224 chain, in the same order,
    with the same defaults, and measures blur_score() one step later --
    right after resize_224, before apply_clahe. This isolates what
    downsizing to 224x224 alone does to the Laplacian-variance metric,
    which is what actually determines whether DEFAULT_BLUR_FLAG_THRESHOLD
    is a meaningful cutoff for the images the model actually trains on.

    image_processing.py itself is not modified -- this only calls its
    already-exported public functions in the documented pipeline order.
    """
    cropped, _was_cropped = circle_crop(raw_image)
    squared = pad_to_square(cropped)
    resized = resize_224(squared)
    return blur_score(resized)


# ============================================================
# PER-CLIENT SAMPLE + SCORE COLLECTION
# ============================================================

def sample_client_manifest(client_name: str) -> "pd.DataFrame":  # noqa: F821
    manifest = build_manifest(client_name, data_root=DATA_ROOT)
    print(f"{client_name}: {len(manifest):,} labeled images available")
    sample = manifest.sample(
        n=min(NUM_IMAGES_PER_CLIENT, len(manifest)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)
    print(f"{client_name}: sampled {len(sample)} images (random_state={RANDOM_STATE})")
    return sample


def process_filesystem_client(client_name: str, sample) -> list:
    """APTOS, IDRiD, Messidor-2: raw_image_locator is a normal on-disk path."""
    results = []
    for i, row in enumerate(sample.itertuples(), start=1):
        image_id = str(row.image_id)
        raw_path = Path(row.raw_image_locator)
        print(f"  [{i:03d}/{len(sample):03d}] {image_id}")
        raw_image = np.array(Image.open(raw_path).convert("RGB"))
        score = compute_post_resize_blur_score(raw_image)
        results.append({"client": client_name, "image_id": image_id, "label": int(row.label), "blur_score": score})
    return results


def process_eyepacs_client(sample) -> list:
    """EyePACS: raw_image_locator is "{zip_half}::{member}". Reconstruct only
    the zip half(s) actually needed for the 200 sampled rows, read just
    those members via zipfile (no extraction), then delete the
    reconstructed zip -- never touch the other ~88,500 images.
    """
    results = []
    sample = sample.copy()
    sample["_zip_half"] = sample["raw_image_locator"].apply(lambda loc: _parse_eyepacs_locator(loc)[0])

    for zip_half in _EYEPACS_ZIP_HALVES:
        half_rows = sample[sample["_zip_half"] == zip_half]
        if half_rows.empty:
            print(f"  EyePACS ({zip_half}): 0 sampled rows -- skipping, no reconstruction needed.")
            continue

        print(f"  EyePACS ({zip_half}): reconstructing zip locally for {len(half_rows)} sampled row(s)...")
        zip_path = _reconstruct_eyepacs_zip(DATA_ROOT, zip_half, work_dir=EYEPACS_ZIP_WORK_DIR)

        try:
            with zipfile.ZipFile(zip_path) as zf:
                for i, row in enumerate(half_rows.itertuples(), start=1):
                    image_id = str(row.image_id)
                    _, member_name = _parse_eyepacs_locator(row.raw_image_locator)
                    print(f"    [{zip_half} {i:03d}/{len(half_rows):03d}] {image_id}")
                    raw_image = _load_raw_image_from_zip(zf, member_name)
                    score = compute_post_resize_blur_score(raw_image)
                    results.append(
                        {"client": "EyePACS", "image_id": image_id, "label": int(row.label), "blur_score": score}
                    )
        finally:
            zip_path.unlink(missing_ok=True)
            print(f"  EyePACS ({zip_half}): deleted reconstructed {zip_path.name} to free disk space.")

    return results


# ============================================================
# STATS
# ============================================================

def print_client_stats(client_name: str, results: list) -> None:
    scores = np.array([r["blur_score"] for r in results], dtype=float)
    n = len(scores)

    print("\n" + "=" * 70)
    print(f"CLIENT: {client_name}  (n={n})")
    print("=" * 70)

    print("Post-resize blur_score percentiles:")
    for p in PERCENTILES:
        val = float(np.percentile(scores, p))
        print(f"  {PERCENTILE_LABELS[p]:7s}: {val:9.3f}")

    print("\nCumulative %% below threshold:".replace("%%", "%"))
    for t in THRESHOLDS:
        count = int((scores < t).sum())
        pct = 100.0 * count / n
        print(f"  <{t:<5d}: {pct:5.1f}%  (n={count})")

    flagged_count = int((scores < DEFAULT_BLUR_FLAG_THRESHOLD).sum())
    flagged_pct = 100.0 * flagged_count / n
    print(
        f"\nPercentage flagged at threshold={DEFAULT_BLUR_FLAG_THRESHOLD:.1f} "
        f"(DEFAULT_BLUR_FLAG_THRESHOLD): {flagged_pct:.1f}%  (n={flagged_count})"
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CROSS-CLIENT POST-RESIZE BLUR SCORE AUDIT")
print(f"{NUM_IMAGES_PER_CLIENT} images/client, random_state={RANDOM_STATE}")
print("=" * 70)

all_results = []

for client_name in ALL_CLIENTS:
    print("\n" + "-" * 70)
    print(f"Sampling {client_name}")
    print("-" * 70)

    sample = sample_client_manifest(client_name)

    if client_name in FILESYSTEM_CLIENTS:
        client_results = process_filesystem_client(client_name, sample)
    else:
        client_results = process_eyepacs_client(sample)

    all_results.extend(client_results)
    print_client_stats(client_name, client_results)


# ============================================================
# SAVE RAW PER-IMAGE SCORES (bonus -- not required, cheap to keep for
# later histogram/threshold-tuning work)
# ============================================================

csv_path = OUTPUT_ROOT / "blur_cross_client_scores.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["client", "image_id", "label", "blur_score"])
    writer.writeheader()
    for r in all_results:
        writer.writerow(r)

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)
print(f"Total images scored: {len(all_results)} ({NUM_IMAGES_PER_CLIENT} x {len(ALL_CLIENTS)} clients)")
print(f"Raw per-image scores saved to: {csv_path}")
print()
print("No full EyePACS extraction was performed.")
print("Only the reconstructed zip half(s) actually needed were built, then deleted.")
print("image_processing.py and build_dataset.py were not modified.")
