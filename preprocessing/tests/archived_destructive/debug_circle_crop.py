"""
debug_circle_crop.py

Standalone diagnostic: shows EXACTLY what circle_crop() detects on a raw
source image, before any cropping/padding/resizing happens. Saves an
overlay PNG (raw image + detected bounding box drawn on it, plus the
computed crop ratio) so you can tell apart:

  (a) the raw photo already has the fundus circle flush against one edge
      (nothing wrong with the pipeline), vs.
  (b) something bright (glare/artifact) near the edge is getting merged
      into the "largest contour" and pulling the box past the true
      retinal tissue (a real bug worth fixing).

Usage:
    python debug_circle_crop.py "C:\\MajorProject - Datsets\\APTOS-2019-Kaggle\\train_images\\train_images\\1ae8c165fd53.png"
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Reuse the exact same defaults as image_processing.py so this matches
# what the real pipeline does.
from preprocessing.image_processing import (
    DEFAULT_CIRCLE_CROP_THRESHOLD,
    DEFAULT_CROP_PADDING_FRACTION,
    DEFAULT_MIN_CROP_AREA_RATIO,
    DEFAULT_MAX_CROP_AREA_RATIO,
)


def inspect(path: Path, threshold: int = DEFAULT_CIRCLE_CROP_THRESHOLD):
    image = np.array(Image.open(path).convert("RGB"))
    h, w = image.shape[:2]
    original_area = float(h * w)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contour found at all -- circle_crop would fall back to the original image.")
        return

    largest = max(contours, key=cv2.contourArea)
    x, y, box_w, box_h = cv2.boundingRect(largest)

    pad = int(round(max(box_w, box_h) * DEFAULT_CROP_PADDING_FRACTION))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + box_w + pad)
    y1 = min(h, y + box_h + pad)

    crop_area = float((x1 - x0) * (y1 - y0))
    crop_ratio = crop_area / original_area

    print(f"Image: {path.name}")
    print(f"Original size: {w}x{h}")
    print(f"Detected contour bounding box: x={x}, y={y}, box_w={box_w}, box_h={box_h}")
    print(f"After {DEFAULT_CROP_PADDING_FRACTION*100:.1f}% padding, clamped to image bounds:")
    print(f"  x0={x0}, y0={y0}, x1={x1}, y1={y1}")
    print(f"Crop ratio: {crop_ratio:.3f}  (accepted range: "
          f"[{DEFAULT_MIN_CROP_AREA_RATIO}, {DEFAULT_MAX_CROP_AREA_RATIO}])")

    hit_left = x0 == 0
    hit_top = y0 == 0
    hit_right = x1 == w
    hit_bottom = y1 == h
    print(f"Clamped against raw image edge -> left:{hit_left} top:{hit_top} "
          f"right:{hit_right} bottom:{hit_bottom}")
    print(
        "(If exactly one side is clamped, that's your zero-margin side. Check the "
        "overlay image below: if the drawn box visibly wraps a bright artifact/glare "
        "near that edge rather than hugging just the retinal tissue, the threshold is "
        "being fooled. If the box genuinely tracks the tissue right up to the raw "
        "image's edge, the source photo itself is off-center -- not a pipeline bug.)"
    )

    # Draw overlay: mask contour + bounding box + final crop box, all on the raw image.
    overlay = image.copy()
    cv2.drawContours(overlay, [largest], -1, (0, 255, 0), 4)          # green: raw contour
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (255, 255, 0), 4)  # yellow: bounding box
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 6)        # red: final crop box (after padding+clamp)

    out_path = path.with_name(path.stem + "_circle_crop_debug.png")
    Image.fromarray(overlay).save(out_path)
    print(f"\nSaved overlay to: {out_path}")
    print("Green = raw detected contour | Yellow = its bounding box | Red = final crop region")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <path_to_raw_image>")
        sys.exit(1)
    inspect(Path(sys.argv[1]))
