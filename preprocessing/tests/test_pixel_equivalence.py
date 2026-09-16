from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from preprocessing.dataset import build_manifest
from preprocessing.image_processing import run_offline_pipeline


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path(r"C:\MajorProject - Datasets")

# Same paths test_build_dataset_smoke.py writes to -- this script reads
# whatever is already sitting there, it does not regenerate it.
TEST_ROOT = Path("data") / "_smoke_test_build_dataset"
MANIFEST_PATH = TEST_ROOT / "APTOS_smoke_manifest.csv"

EXPECTED_NUM_IMAGES = 5


# ============================================================
# 1. LOAD THE EXISTING SMOKE-TEST MANIFEST
# ============================================================

print("=" * 70)
print("1. LOADING EXISTING SMOKE-TEST MANIFEST")
print("=" * 70)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"{MANIFEST_PATH} not found. Run "
        "`python -m preprocessing.test_build_dataset_smoke` first so there is "
        "an existing cached manifest + PNGs to compare against."
    )

existing_df = pd.read_csv(MANIFEST_PATH)
existing_df["image_id"] = existing_df["image_id"].astype(str)

print(f"Existing smoke-test manifest rows: {len(existing_df)}")
print(existing_df[["image_id", "cached_image_path", "label", "split"]].to_string(index=False))

if len(existing_df) != EXPECTED_NUM_IMAGES:
    print(
        f"\nWARNING: expected {EXPECTED_NUM_IMAGES} rows, found {len(existing_df)}. "
        "Continuing anyway, comparing whatever is actually in the manifest."
    )

image_ids = set(existing_df["image_id"])


# ============================================================
# 2. RESOLVE RAW IMAGE PATHS FOR THESE EXACT 5 IMAGES
# ============================================================

# FINAL_MANIFEST_COLUMNS (build_dataset.py) doesn't carry raw_image_locator,
# so it's re-derived from the real APTOS source manifest and filtered down
# to exactly these image_ids -- the SAME images already cached here, not a
# fresh/different sample.

print("\n" + "=" * 70)
print("2. RESOLVING RAW IMAGE PATHS FOR THESE IMAGES")
print("=" * 70)

source_manifest = build_manifest("APTOS", data_root=DATA_ROOT)
source_manifest["image_id"] = source_manifest["image_id"].astype(str)

matched = source_manifest[source_manifest["image_id"].isin(image_ids)]

missing = image_ids - set(matched["image_id"])
if missing:
    raise RuntimeError(
        f"{len(missing)} image_id(s) from {MANIFEST_PATH} were not found in the "
        f"current APTOS source manifest: {sorted(missing)}"
    )

raw_locator_by_id = dict(zip(matched["image_id"], matched["raw_image_locator"]))
print(f"Resolved raw_image_locator for {len(raw_locator_by_id)}/{len(image_ids)} image(s).")


# ============================================================
# 3. RE-RUN run_offline_pipeline() AND COMPARE
#    new run_offline_pipeline(raw_image)  VS  existing cached PNG
# ============================================================

print("\n" + "=" * 70)
print("3. COMPARING NEW run_offline_pipeline(raw_image) VS EXISTING CACHED PNG")
print("=" * 70)

failed_pixels = []
failed_meta = []

for _, row in existing_df.iterrows():
    image_id = row["image_id"]
    cached_path = Path(row["cached_image_path"])
    raw_path = Path(raw_locator_by_id[image_id])

    print(f"\n--- {image_id} ---")

    if not cached_path.exists():
        print(f"  PIXELS: FAIL -- cached PNG missing at {cached_path}")
        failed_pixels.append(image_id)
        continue

    raw_image = np.array(Image.open(raw_path).convert("RGB"))
    new_image, new_meta = run_offline_pipeline(raw_image)
    cached_image = np.array(Image.open(cached_path).convert("RGB"))

    # --- pixel comparison ---
    if new_image.shape != cached_image.shape:
        print(f"  PIXELS: FAIL -- shape mismatch: new={new_image.shape} cached={cached_image.shape}")
        failed_pixels.append(image_id)
    elif np.array_equal(new_image, cached_image):
        print(f"  PIXELS: PASS -- byte-for-byte identical ({new_image.shape[1]}x{new_image.shape[0]})")
    else:
        diff = new_image.astype(np.int16) - cached_image.astype(np.int16)
        n_diff_pixels = int(np.any(diff != 0, axis=-1).sum())
        total_pixels = new_image.shape[0] * new_image.shape[1]
        max_abs_diff = int(np.abs(diff).max())
        mean_abs_diff = float(np.abs(diff).mean())
        print(
            f"  PIXELS: FAIL -- {n_diff_pixels}/{total_pixels} pixel(s) differ "
            f"({100 * n_diff_pixels / total_pixels:.2f}%), "
            f"max_abs_diff={max_abs_diff}, mean_abs_diff={mean_abs_diff:.4f}"
        )
        failed_pixels.append(image_id)

    # --- metadata comparison (secondary check, same manifest row) ---
    meta_checks = [
        ("blur_score", float(row["blur_score"]), float(new_meta["blur_score"]), "float"),
        ("low_quality_flag", bool(row["low_quality_flag"]), bool(new_meta["low_quality_flag"]), "bool"),
        ("was_cropped", bool(row["was_cropped"]), bool(new_meta["was_cropped"]), "bool"),
        ("original_height", int(row["original_height"]), int(new_meta["original_height"]), "int"),
        ("original_width", int(row["original_width"]), int(new_meta["original_width"]), "int"),
    ]

    for name, old_val, new_val, kind in meta_checks:
        ok = abs(old_val - new_val) < 1e-9 if kind == "float" else old_val == new_val
        print(f"  META {name:18s}: {'PASS' if ok else 'FAIL'}  (cached={old_val}, new={new_val})")
        if not ok:
            failed_meta.append((image_id, name))


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(f"Images compared : {len(existing_df)}")
print(f"Pixel failures  : {len(failed_pixels)}  {failed_pixels if failed_pixels else ''}")
print(f"Metadata failures: {len(failed_meta)}  {failed_meta if failed_meta else ''}")

if not failed_pixels and not failed_meta:
    print("\nPIXEL + METADATA EQUIVALENCE: PASS")
    print("run_offline_pipeline() is deterministic and matches the existing cache exactly.")
else:
    print("\nPIXEL + METADATA EQUIVALENCE: FAIL")
    print("This usually means image_processing.py (or one of its default parameters) changed")
    print("since data/_smoke_test_build_dataset/ was built, and the cache is now stale.")
    raise AssertionError(
        f"Pixel/metadata equivalence failed for {len(set(failed_pixels) | {i for i, _ in failed_meta})} "
        "image(s) -- see report above."
    )
