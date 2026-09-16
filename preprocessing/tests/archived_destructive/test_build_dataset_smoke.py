from pathlib import Path
import shutil
import pandas as pd
from PIL import Image

from preprocessing.dataset import build_manifest, split_manifest
from preprocessing.build_dataset import run_offline_preprocessing


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path(r"C:\MajorProject - Datsets")

TEST_ROOT = Path("data") / "_smoke_test_build_dataset"
CACHE_ROOT = TEST_ROOT / "cache"
MANIFEST_PATH = TEST_ROOT / "APTOS_smoke_manifest.csv"

NUM_IMAGES = 5


# ============================================================
# CLEAN OLD TEST OUTPUT
# ============================================================

if TEST_ROOT.exists():
    shutil.rmtree(TEST_ROOT)

TEST_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. BUILD REAL APTOS SOURCE MANIFEST
# ============================================================

print("=" * 70)
print("1. BUILDING REAL APTOS SOURCE MANIFEST")
print("=" * 70)

manifest = build_manifest(
    "APTOS",
    data_root=DATA_ROOT,
)

manifest = split_manifest(
    "APTOS",
    manifest,
    random_state=42,
)

print(f"Full APTOS rows: {len(manifest):,}")


# ============================================================
# 2. MAKE A TINY TEST MANIFEST
# ============================================================

# Pick 5 images from different locations while preserving their
# already-established official split assignment.
#
# This is ONLY a smoke test; we are not creating a new ML split.

selected = []

for split_name in ("train", "val", "test"):
    subset = manifest[manifest["split"] == split_name]

    # Take up to 2 train, 2 val, 1 test.
    count = {"train": 2, "val": 2, "test": 1}[split_name]

    selected.append(subset.head(count))

smoke_manifest = pd.concat(selected, ignore_index=True)

print("\nSelected smoke-test rows:")
print(
    smoke_manifest[
        ["image_id", "label", "split", "raw_image_locator"]
    ].to_string(index=False)
)

assert len(smoke_manifest) == NUM_IMAGES


# ============================================================
# 3. TEMPORARILY PATCH build_manifest/split_manifest
# ============================================================

# run_offline_preprocessing normally builds the complete source
# manifest internally. For this smoke test we replace those
# functions inside the imported module so ONLY these 5 real images
# are processed.
#
# These patches must stay active through BOTH the first
# preprocessing call and the resume call below, since the resume
# call also triggers run_offline_preprocessing's internal
# build_manifest/split_manifest calls. If we restore the real
# implementations before the resume call, it silently rebuilds the
# full 3,662-image APTOS manifest instead of reusing these same 5
# rows, which is what caused the previous hang.

import preprocessing.build_dataset as bd

original_build_manifest = bd.build_manifest
original_split_manifest = bd.split_manifest

bd.build_manifest = lambda client_name, data_root: smoke_manifest.copy()

bd.split_manifest = (
    lambda client_name, manifest_df, random_state=42:
    manifest_df.copy()
)

try:
    # ============================================================
    # 4. RUN REAL OFFLINE PREPROCESSING
    # ============================================================

    print("\n" + "=" * 70)
    print("2. RUNNING OFFLINE PREPROCESSING ON 5 REAL APTOS IMAGES")
    print("=" * 70)

    stats = run_offline_preprocessing(
        "APTOS",
        data_root=DATA_ROOT,
        cache_root=CACHE_ROOT,
        manifest_path=MANIFEST_PATH,
        max_samples=6000,
        random_state=42,
    )

    print("\nBuildStats:")
    print(stats)

    # ============================================================
    # 5. CHECK FINAL MANIFEST
    # ============================================================

    print("\n" + "=" * 70)
    print("3. CHECKING FINAL MANIFEST")
    print("=" * 70)

    assert MANIFEST_PATH.exists(), "Final manifest was not created"

    final_df = pd.read_csv(MANIFEST_PATH)

    print(final_df.to_string(index=False))

    assert len(final_df) == NUM_IMAGES

    required_columns = [
        "image_id",
        "cached_image_path",
        "patient_id",
        "label",
        "split",
        "blur_score",
        "low_quality_flag",
        "was_cropped",
        "original_height",
        "original_width",
    ]

    for column in required_columns:
        assert column in final_df.columns, (
            f"Missing final manifest column: {column}"
        )

    print("\nFinal manifest schema: PASS")

    # ============================================================
    # 6. CHECK CACHED IMAGES
    # ============================================================

    print("\n" + "=" * 70)
    print("4. CHECKING CACHED PNG IMAGES")
    print("=" * 70)

    for _, row in final_df.iterrows():
        path = Path(row["cached_image_path"])

        assert path.exists(), f"Cached image missing: {path}"
        assert path.suffix.lower() == ".png"

        with Image.open(path) as img:
            print(
                f"{row['image_id']}: "
                f"size={img.size}, mode={img.mode}, format={img.format}"
            )

            assert img.size == (224, 224)
            assert img.mode == "RGB"
            assert img.format == "PNG"

    print("\nCached PNG validation: PASS")

    # ============================================================
    # 7. CHECK METADATA
    # ============================================================

    print("\n" + "=" * 70)
    print("5. CHECKING OFFLINE METADATA")
    print("=" * 70)

    assert final_df["blur_score"].notna().all()
    assert final_df["low_quality_flag"].notna().all()
    assert final_df["was_cropped"].notna().all()
    assert final_df["original_height"].notna().all()
    assert final_df["original_width"].notna().all()

    print(
        final_df[
            [
                "image_id",
                "blur_score",
                "low_quality_flag",
                "was_cropped",
                "original_height",
                "original_width",
            ]
        ].to_string(index=False)
    )

    print("\nOffline metadata preservation: PASS")

    # ============================================================
    # 8. RESUME TEST
    # ============================================================

    print("\n" + "=" * 70)
    print("6. TESTING RESUME BEHAVIOR")
    print("=" * 70)

    stats_resume = run_offline_preprocessing(
        "APTOS",
        data_root=DATA_ROOT,
        cache_root=CACHE_ROOT,
        manifest_path=MANIFEST_PATH,
        max_samples=6000,
        random_state=42,
    )

    print("\nResume BuildStats:")
    print(stats_resume)

    assert stats_resume.already_cached == NUM_IMAGES
    assert stats_resume.newly_processed == 0
    assert stats_resume.failed == 0

    final_df_after_resume = pd.read_csv(MANIFEST_PATH)

    assert len(final_df_after_resume) == NUM_IMAGES

    print("\nResume behavior: PASS")
finally:
    # Only restore the real build_manifest/split_manifest now that
    # both the initial run AND the resume run are done with them.
    bd.build_manifest = original_build_manifest
    bd.split_manifest = original_split_manifest


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("ALL BUILD_DATASET SMOKE TESTS PASSED")
print()
print(f"Temporary test output: {TEST_ROOT}")
print("This test processed exactly 5 real APTOS images.")
print("No EyePACS ZIP archives were touched.")