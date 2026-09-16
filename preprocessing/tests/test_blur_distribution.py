from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from preprocessing.dataset import build_manifest
from preprocessing.image_processing import run_offline_pipeline


DATA_ROOT = Path(r"C:\MajorProject - Datsets")
NUM_IMAGES = 200
RANDOM_STATE = 42


manifest = build_manifest(
    "APTOS",
    data_root=DATA_ROOT,
)

sample = manifest.sample(
    n=min(NUM_IMAGES, len(manifest)),
    random_state=RANDOM_STATE,
).reset_index(drop=True)

scores = []

print("=" * 70)
print("APTOS BLUR-SCORE DISTRIBUTION")
print("=" * 70)
print(f"Images sampled: {len(sample)}")

for i, row in sample.iterrows():
    path = Path(row["raw_image_locator"])

    image = np.array(Image.open(path).convert("RGB"))

    _, meta = run_offline_pipeline(image)

    scores.append(float(meta["blur_score"]))

    print(
        f"[{i + 1:03d}/{len(sample):03d}] "
        f"{row['image_id']} -> {meta['blur_score']:.4f}"
    )


scores = np.asarray(scores)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

series = pd.Series(scores)

print(f"Min    : {series.min():.4f}")
print(f"5%     : {series.quantile(0.05):.4f}")
print(f"10%    : {series.quantile(0.10):.4f}")
print(f"25%    : {series.quantile(0.25):.4f}")
print(f"Median : {series.median():.4f}")
print(f"75%    : {series.quantile(0.75):.4f}")
print(f"90%    : {series.quantile(0.90):.4f}")
print(f"95%    : {series.quantile(0.95):.4f}")
print(f"Max    : {series.max():.4f}")

print("\nThreshold counts:")
for threshold in [10, 20, 30, 40, 50, 60, 80, 100]:
    count = int((scores < threshold).sum())
    pct = count / len(scores) * 100
    print(f"< {threshold:3d}: {count:3d} / {len(scores)} ({pct:5.1f}%)")

flagged = scores < 100.0

print(
    f"\nCurrent threshold (<100): "
    f"{flagged.sum()} / {len(scores)} "
    f"({flagged.mean() * 100:.1f}%) flagged"
)