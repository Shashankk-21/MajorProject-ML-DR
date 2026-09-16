from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from preprocessing.dataset import build_manifest
from preprocessing.image_processing import run_offline_pipeline


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path(r"C:\MajorProject - Datsets")
OUTPUT_ROOT = Path("data") / "_blur_audit"

RANDOM_STATE = 42
NUM_IMAGES_TO_SAMPLE = 200
NUM_EXTREMES_PER_SIDE = 10  # 10 lowest + 10 highest = 20 total


# ============================================================
# SETUP
# ============================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# BUILD REAL APTOS MANIFEST + DETERMINISTIC 200-IMAGE SAMPLE
# ============================================================

print("=" * 70)
print("1. BUILDING APTOS MANIFEST")
print("=" * 70)

manifest = build_manifest(
    "APTOS",
    data_root=DATA_ROOT,
)

print(f"APTOS images available: {len(manifest):,}")

# Same deterministic sampling approach as the crop audit: a plain
# manifest.sample() at random_state=42, no split_manifest() involved.
sample = manifest.sample(
    n=min(NUM_IMAGES_TO_SAMPLE, len(manifest)),
    random_state=RANDOM_STATE,
).reset_index(drop=True)

print(f"Sampled {len(sample)} images (random_state={RANDOM_STATE})")


# ============================================================
# RECOMPUTE BLUR SCORE FOR ALL 200 IMAGES
# ============================================================

# run_offline_pipeline() is called exactly as the real pipeline calls
# it (circle_crop -> blur_score -> pad_to_square -> resize_224 ->
# apply_clahe internally). We only read meta["blur_score"] out of it --
# no separate/manual cropping step is added here, and image_processing.py
# is not touched.

print("\n" + "=" * 70)
print("2. RECOMPUTING BLUR SCORE FOR 200 REAL APTOS IMAGES")
print("=" * 70)

results = []

for i, row in sample.iterrows():

    image_id = str(row["image_id"])
    raw_path = Path(row["raw_image_locator"])

    print(f"[{i + 1:03d}/{len(sample):03d}] {image_id}")

    raw_image = np.array(Image.open(raw_path).convert("RGB"))

    _, meta = run_offline_pipeline(raw_image)

    results.append(
        {
            "image_id": image_id,
            "raw_path": str(raw_path),
            "label": int(row["label"]),
            "blur_score": float(meta["blur_score"]),
            "low_quality_flag": bool(meta["low_quality_flag"]),
            "raw_image": raw_image,
        }
    )


# ============================================================
# SUMMARY
# ============================================================

scores_sorted = sorted(r["blur_score"] for r in results)
print("\n" + "=" * 70)
print("3. BLUR SCORE SUMMARY (all 200)")
print("=" * 70)
print(f"min    : {scores_sorted[0]:.3f}")
print(f"median : {scores_sorted[len(scores_sorted) // 2]:.3f}")
print(f"max    : {scores_sorted[-1]:.3f}")


# ============================================================
# SELECT THE 10 LOWEST AND 10 HIGHEST
# ============================================================

by_blur = sorted(results, key=lambda r: r["blur_score"])

lowest = by_blur[:NUM_EXTREMES_PER_SIDE]                      # blurriest first
highest = list(reversed(by_blur[-NUM_EXTREMES_PER_SIDE:]))    # sharpest first

selected = lowest + highest

print("\nSelected 10 lowest (blurriest) blur scores:")
for r in lowest:
    print(f"  {r['image_id']}  blur_score={r['blur_score']:.6f}")

print("\nSelected 10 highest (sharpest) blur scores:")
for r in highest:
    print(f"  {r['image_id']}  blur_score={r['blur_score']:.6f}")


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

print("\n" + "=" * 70)
print("4. CREATING BLUR EXTREMES CONTACT SHEET")
print("=" * 70)

COLUMNS = 5
CARD_W = 300
CARD_H = 340
IMG_BOX_W = 280
IMG_BOX_H = 250
MARGIN = 20
HEADER_H = 40          # room for the section title band above each block
ROWS_PER_SECTION = NUM_EXTREMES_PER_SIDE // COLUMNS  # 2 rows of 5 per section

section_h = HEADER_H + ROWS_PER_SECTION * CARD_H

sheet_w = COLUMNS * CARD_W + (COLUMNS + 1) * MARGIN
sheet_h = 2 * section_h + 3 * MARGIN

sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
draw = ImageDraw.Draw(sheet)


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    return ImageOps.contain(img, (box_w, box_h))


def draw_section(results_subset, section_index, title, border_color):
    section_y0 = MARGIN + section_index * (section_h + MARGIN)

    draw.text((MARGIN, section_y0 + 5), title, fill=border_color)

    grid_y0 = section_y0 + HEADER_H

    for idx, result in enumerate(results_subset):
        col = idx % COLUMNS
        row_idx = idx // COLUMNS

        x0 = MARGIN + col * CARD_W
        y0 = grid_y0 + row_idx * CARD_H

        draw.rectangle(
            [x0, y0, x0 + CARD_W - MARGIN, y0 + CARD_H - 10],
            outline=border_color,
            width=2,
        )

        raw_display = fit_image(
            Image.fromarray(result["raw_image"]),
            IMG_BOX_W,
            IMG_BOX_H,
        )

        img_x = x0 + 10
        img_y = y0 + 10

        sheet.paste(raw_display, (img_x, img_y))

        text_y = y0 + IMG_BOX_H + 15
        draw.text((img_x, text_y), f"id={result['image_id']}", fill="black")
        draw.text((img_x, text_y + 14), f"label={result['label']}", fill="black")
        draw.text(
            (img_x, text_y + 28),
            f"blur_score={result['blur_score']:.4f}",
            fill="black",
        )


draw_section(
    lowest,
    section_index=0,
    title=f"LOWEST {NUM_EXTREMES_PER_SIDE} BLUR SCORES (most blurred)",
    border_color=(200, 30, 30),
)

draw_section(
    highest,
    section_index=1,
    title=f"HIGHEST {NUM_EXTREMES_PER_SIDE} BLUR SCORES (sharpest)",
    border_color=(30, 140, 30),
)

sheet_path = OUTPUT_ROOT / "blur_extremes.png"
sheet.save(sheet_path)

print(f"Saved contact sheet to: {sheet_path}")


# ============================================================
# SAVE METADATA CSV
# ============================================================

csv_path = OUTPUT_ROOT / "blur_extremes_metadata.csv"

fieldnames = [
    "group",
    "image_id",
    "raw_path",
    "label",
    "blur_score",
    "low_quality_flag",
]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for r in lowest:
        writer.writerow({"group": "lowest", **{k: r[k] for k in fieldnames if k != "group"}})

    for r in highest:
        writer.writerow({"group": "highest", **{k: r[k] for k in fieldnames if k != "group"}})

print(f"Saved metadata to: {csv_path}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("BLUR VISUAL AUDIT COMPLETE")
print()
print(f"Sampled          : {len(sample)} real APTOS images (random_state={RANDOM_STATE})")
print(f"Contact sheet     : {sheet_path}")
print(f"Metadata CSV      : {csv_path}")
print()
print("No cropping beyond run_offline_pipeline()'s own internal circle_crop was performed.")
print("image_processing.py was not modified.")
print("No cached dataset or final manifest was modified.")
