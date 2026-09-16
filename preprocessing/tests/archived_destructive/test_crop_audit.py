from pathlib import Path
import random
import csv
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageOps

from preprocessing.dataset import build_manifest
from preprocessing.image_processing import run_offline_pipeline


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path(r"C:\MajorProject - Datsets")
OUTPUT_ROOT = Path("data") / "_crop_audit"

RANDOM_STATE = 42
NUM_IMAGES_TO_PROCESS = 40
NUM_IMAGES_TO_DISPLAY = 20


# ============================================================
# SETUP
# ============================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_STATE)


# ============================================================
# BUILD REAL APTOS MANIFEST
# ============================================================

print("=" * 70)
print("1. BUILDING APTOS MANIFEST")
print("=" * 70)

manifest = build_manifest(
    "APTOS",
    data_root=DATA_ROOT,
)

print(f"APTOS images available: {len(manifest):,}")


# Take a deterministic sample, spread across the full manifest.
sample = manifest.sample(
    n=min(NUM_IMAGES_TO_PROCESS, len(manifest)),
    random_state=RANDOM_STATE,
).reset_index(drop=True)

print(f"Processing {len(sample)} real images...")


# ============================================================
# RUN OFFLINE PIPELINE
# ============================================================

results = []

for i, row in sample.iterrows():

    image_id = str(row["image_id"])
    raw_path = Path(row["raw_image_locator"])

    print(f"[{i + 1:02d}/{len(sample):02d}] {image_id}")

    raw_image = np.array(Image.open(raw_path).convert("RGB"))

    processed_array, meta = run_offline_pipeline(
        raw_image
    )

    processed_image = Image.fromarray(processed_array)

    results.append(
        {
            "image_id": image_id,
            "raw_path": str(raw_path),
            "label": int(row["label"]),
            "was_cropped": bool(meta["was_cropped"]),
            "blur_score": float(meta["blur_score"]),
            "low_quality_flag": bool(meta["low_quality_flag"]),
            "original_height": int(meta["original_height"]),
            "original_width": int(meta["original_width"]),
            "raw_image": raw_image.copy(),
            "processed_image": processed_image.copy(),
        }
    )



# ============================================================
# SUMMARY
# ============================================================

cropped = [r for r in results if r["was_cropped"]]
not_cropped = [r for r in results if not r["was_cropped"]]

print("\n" + "=" * 70)
print("2. CROP SUMMARY")
print("=" * 70)

print(f"Images processed : {len(results)}")
print(f"was_cropped=True : {len(cropped)}")
print(f"was_cropped=False: {len(not_cropped)}")


# ============================================================
# SELECT DISPLAY SET
# ============================================================

# Prefer a balanced set of cropped / non-cropped images.
display_results = []

display_results.extend(
    cropped[: NUM_IMAGES_TO_DISPLAY // 2]
)

display_results.extend(
    not_cropped[: NUM_IMAGES_TO_DISPLAY // 2]
)

# If one category is too small, fill from the other.
remaining = [
    r for r in results
    if r not in display_results
]

for r in remaining:
    if len(display_results) >= NUM_IMAGES_TO_DISPLAY:
        break
    display_results.append(r)


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

print("\n" + "=" * 70)
print("3. CREATING VISUAL AUDIT")
print("=" * 70)

CARD_W = 620
CARD_H = 300
MARGIN = 20

columns = 2
rows = (len(display_results) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (
        columns * CARD_W + (columns + 1) * MARGIN,
        rows * CARD_H + (rows + 1) * MARGIN,
    ),
    "white",
)

draw = ImageDraw.Draw(sheet)


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    return ImageOps.contain(
        img,
        (box_w, box_h),
    )


for idx, result in enumerate(display_results):

    col = idx % columns
    row_idx = idx // columns

    x0 = MARGIN + col * CARD_W
    y0 = MARGIN + row_idx * CARD_H

    draw.rectangle(
        [x0, y0, x0 + CARD_W, y0 + CARD_H],
        outline="black",
        width=2,
    )

    draw.text(
        (x0 + 10, y0 + 8),
        (
            f"{result['image_id']}   "
            f"label={result['label']}   "
            f"cropped={result['was_cropped']}"
        ),
        fill="black",
    )

    # Raw
    raw_display = fit_image(
    Image.fromarray(result["raw_image"]),
    285,
    225,
    )

    raw_x = x0 + 10
    raw_y = y0 + 45

    sheet.paste(
        raw_display,
        (
            raw_x,
            raw_y,
        ),
    )

    draw.text(
        (raw_x, raw_y + 228),
        "RAW",
        fill="black",
    )

    # Processed
    processed_display = fit_image(
        result["processed_image"],
        285,
        225,
    )

    proc_x = x0 + 315
    proc_y = y0 + 45

    sheet.paste(
        processed_display,
        (
            proc_x,
            proc_y,
        ),
    )

    draw.text(
        (proc_x, proc_y + 228),
        "PROCESSED 224x224",
        fill="black",
    )


sheet_path = OUTPUT_ROOT / "crop_audit.png"
sheet.save(sheet_path)

print(f"Saved visual audit to: {sheet_path}")


# ============================================================
# SAVE METADATA CSV
# ============================================================

csv_path = OUTPUT_ROOT / "crop_audit_metadata.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id",
            "raw_path",
            "label",
            "was_cropped",
            "blur_score",
            "low_quality_flag",
            "original_height",
            "original_width",
        ],
    )

    writer.writeheader()

    for result in results:
        writer.writerow(
            {
                key: result[key]
                for key in writer.fieldnames
            }
        )


print(f"Saved metadata to: {csv_path}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("CROP AUDIT COMPLETE")
print()
print(f"Visual audit : {sheet_path}")
print(f"Metadata CSV : {csv_path}")
print()
print("No original images were modified.")
print("No cached dataset was modified.")