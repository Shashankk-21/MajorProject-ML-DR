"""
preprocessing/image_processing.py

Core, deterministic, OFFLINE image-processing utilities for the Federated
DR Detection preprocessing pipeline.

Finalized pipeline order (Preprocessing Design Doc, Part 2 -- "OFFLINE"
stage; applies once per image, cached to disk, identical for every split):

    circle_crop -> pad_to_square -> resize_224 -> blur_score (quality
    flag, logged only, measured at the final 224x224 scale) -> apply_clahe
    (fixed clipLimit=2.0, tileGridSize=8x8)

Note: blur_score was originally measured immediately after circle_crop
(post-crop, pre-resize). Following a blur-score audit, it was moved to
run on the final 224x224, pre-CLAHE image instead, so the metric reflects
the same fixed scale every image is actually trained on, rather than
each image's variable post-crop resolution. This is a measurement-point
change only -- circle_crop/pad_to_square/resize_224/apply_clahe's own
behavior and the final processed pixel output are unaffected; only the
number reported in meta_dict["blur_score"] (and, downstream,
low_quality_flag) changes.

Scope boundary (Design Doc Part 3 / chat Section 9): this module contains
ONLY the per-image, deterministic pixel operations above. It intentionally
does not contain:
    - augmentation (see preprocessing/augmentation.py)
    - manifests, splits, DRDataset, DataLoaders (see preprocessing/dataset.py)
    - file I/O, disk caching, resumability, or a CLI (see
      preprocessing/build_dataset.py)
    - any model / training-loop code

Image convention (implementation default -- not numerically specified in
the finalized design, stated here per the project's own rule for
unspecified details)
------------------------------------------------------------------------
Every function expects and returns images as RGB, uint8 NumPy arrays of
shape (H, W, 3). RGB (not BGR) was chosen because the project's EDA
notebook loads images via PIL (`Image.open(...)`), which is natively RGB.
Callers (build_dataset.py) must respect this convention:
    - PIL:    `np.array(Image.open(path).convert("RGB"))` is already RGB.
    - OpenCV: `cv2.imread(path)` returns BGR and MUST be converted with
              `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before calling into
              this module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ImageProcessingError",
    "circle_crop",
    "blur_score",
    "pad_to_square",
    "resize_224",
    "apply_clahe",
    "run_offline_pipeline",
    "TARGET_SIZE",
    "DEFAULT_CLAHE_CLIP_LIMIT",
    "DEFAULT_CLAHE_TILE_GRID_SIZE",
]

# ---------------------------------------------------------------------------
# Parameters
#
# Values marked "fixed" come directly from Part 1 of the preprocessing
# design doc. Values marked "assumption" are not numerically specified
# there ("a grayscale threshold", "2-3% padding", "quality flag") -- a
# sensible default is chosen and stated here per the project's own rule:
# "if something genuinely is not specified, choose a sensible
# implementation default, clearly state the assumption, and proceed."
# All are exposed as function parameters so they can be tuned without
# touching this module's logic.
# ---------------------------------------------------------------------------

# Fixed (Part 1, Step 3): CoAtNet-1's input size.
TARGET_SIZE: int = 224

# Fixed (Part 1, Step 2): offline-pass CLAHE parameters. clip_limit is a
# parameter specifically so augmentation.py can reuse apply_clahe() with a
# randomized value for the separate, train-only, on-the-fly re-application.
DEFAULT_CLAHE_CLIP_LIMIT: float = 2.0
DEFAULT_CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)

# Assumption: grayscale cutoff separating the fundus disc from a
# near-black background. 7 is a conservative, widely-used value for this
# kind of fundus circle-crop across varied camera sources.
DEFAULT_CIRCLE_CROP_THRESHOLD: int = 7

# Fixed range (Part 1, Step 1): "fallback to original if crop ratio is
# outside 30-98%". Ratio = (cropped bounding-box area) / (original image
# area).
DEFAULT_MIN_CROP_AREA_RATIO: float = 0.30
DEFAULT_MAX_CROP_AREA_RATIO: float = 0.98

# Assumption: "2-3% padding" -> midpoint, expressed as a fraction of the
# detected bounding box's larger side.
DEFAULT_CROP_PADDING_FRACTION: float = 0.025

# Assumption: Laplacian-variance cutoff below which an image is flagged as
# low quality. 100.0 is a commonly used heuristic for this metric. This is
# a LOGGED FLAG ONLY (Part 1, Step 10: "Flag, don't auto-drop") -- no image
# is ever altered or dropped by this module based on it.
DEFAULT_BLUR_FLAG_THRESHOLD: float = 100.0


class ImageProcessingError(ValueError):
    """Raised when an input image fails validation for this module's functions."""


def _validate_image(image: np.ndarray, *, allow_grayscale: bool = False) -> None:
    """Validate that `image` is a well-formed RGB (or optionally grayscale) uint8 array.

    Args:
        image: The array to validate.
        allow_grayscale: If True, a 2-D (H, W) uint8 array is also accepted.

    Raises:
        ImageProcessingError: if `image` is not a NumPy array, is empty,
            is not uint8, or does not have shape (H, W, 3) (and, if
            `allow_grayscale`, is also not shape (H, W)).
    """
    if not isinstance(image, np.ndarray):
        raise ImageProcessingError(f"Expected a numpy.ndarray, got {type(image).__name__}")
    if image.size == 0:
        raise ImageProcessingError("Received an empty image array")
    if image.dtype != np.uint8:
        raise ImageProcessingError(f"Expected dtype=uint8, got {image.dtype}")
    if allow_grayscale and image.ndim == 2:
        return
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageProcessingError(f"Expected an (H, W, 3) RGB array, got shape {image.shape}")


def circle_crop(
    image: np.ndarray,
    *,
    threshold: int = DEFAULT_CIRCLE_CROP_THRESHOLD,
    padding_fraction: float = DEFAULT_CROP_PADDING_FRACTION,
    min_crop_area_ratio: float = DEFAULT_MIN_CROP_AREA_RATIO,
    max_crop_area_ratio: float = DEFAULT_MAX_CROP_AREA_RATIO,
) -> Tuple[np.ndarray, bool]:
    """Crop a fundus photo tightly around the circular retinal field of view.

    Pipeline (Part 1, Step 1): grayscale threshold -> largest external
    contour -> bounding box -> pad by `padding_fraction` of the box's
    larger dimension -> crop. Falls back to returning the ORIGINAL,
    uncropped image whenever the detected region looks implausible (no
    contour found, a degenerate box, or a crop-to-original area ratio
    outside [min_crop_area_ratio, max_crop_area_ratio]) rather than risk
    cropping out clinically relevant tissue.

    Args:
        image: RGB uint8 array, shape (H, W, 3).
        threshold: Grayscale intensity cutoff separating the fundus disc
            from background (see DEFAULT_CIRCLE_CROP_THRESHOLD).
        padding_fraction: Extra margin added around the detected bounding
            box, as a fraction of the box's larger side.
        min_crop_area_ratio: Minimum acceptable (crop area / image area);
            below this, the crop is rejected.
        max_crop_area_ratio: Maximum acceptable (crop area / image area);
            above this, the crop is rejected (image was already tight).

    Returns:
        A tuple (cropped_or_original_image, was_cropped).

    Raises:
        ImageProcessingError: if `image` fails validation.
    """
    _validate_image(image)
    h, w = image.shape[:2]
    original_area = float(h * w)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.debug("circle_crop: no contour found above threshold=%d; using original image", threshold)
        return image, False

    largest = max(contours, key=cv2.contourArea)
    x, y, box_w, box_h = cv2.boundingRect(largest)

    if box_w <= 0 or box_h <= 0:
        logger.debug("circle_crop: degenerate bounding box; using original image")
        return image, False

    pad = int(round(max(box_w, box_h) * padding_fraction))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + box_w + pad)
    y1 = min(h, y + box_h + pad)

    crop_area = float((x1 - x0) * (y1 - y0))
    crop_ratio = crop_area / original_area if original_area > 0 else 0.0

    if crop_ratio < min_crop_area_ratio or crop_ratio > max_crop_area_ratio:
        logger.debug(
            "circle_crop: crop ratio %.3f outside [%.2f, %.2f]; using original image",
            crop_ratio, min_crop_area_ratio, max_crop_area_ratio,
        )
        return image, False

    return image[y0:y1, x0:x1], True


def blur_score(image: np.ndarray) -> float:
    """Compute a Laplacian-variance blur/sharpness score for `image`.

    Higher values indicate a sharper image; lower values indicate a
    blurrier one. This is a DIAGNOSTIC MEASUREMENT ONLY (Part 1, Step 10:
    "Quality flag ... flag, don't auto-drop") -- thresholding/flagging is
    the caller's decision (see `blur_flag_threshold` in
    run_offline_pipeline and the manifest written by build_dataset.py).

    Args:
        image: RGB (H, W, 3) or grayscale (H, W) uint8 array.

    Returns:
        The variance of the Laplacian of the grayscale image.

    Raises:
        ImageProcessingError: if `image` fails validation.
    """
    _validate_image(image, allow_grayscale=True)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pad_to_square(image: np.ndarray, *, fill_value: int = 0) -> np.ndarray:
    """Pad `image` to a square canvas with a constant (default black) border.

    Padding is split as evenly as possible between both sides of the
    shorter dimension so the original content stays centered. This MUST
    run before resize_224 (Part 1, Step 3) to avoid aspect-ratio
    distortion.

    Args:
        image: RGB uint8 array, shape (H, W, 3).
        fill_value: Constant fill value for the padded border on every
            channel (0 = black, matching the finalized design).

    Returns:
        A square RGB uint8 array of shape (max(H, W), max(H, W), 3).

    Raises:
        ImageProcessingError: if `image` fails validation.
    """
    _validate_image(image)
    h, w = image.shape[:2]
    side = max(h, w)

    pad_h = side - h
    pad_w = side - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        image, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(fill_value, fill_value, fill_value),
    )


def resize_224(image: np.ndarray, *, size: int = TARGET_SIZE) -> np.ndarray:
    """Resize `image` to `size` x `size` using area interpolation.

    Part 1, Step 3 (mandatory): cv2.INTER_AREA is the correct choice for
    downsizing, which is the overwhelmingly common case here (most source
    images across all 4 clients exceed 224x224). Must be called AFTER
    pad_to_square so no aspect-ratio distortion is introduced.

    Args:
        image: RGB uint8 array, shape (H, W, 3). Should already be square
            (via pad_to_square) -- a non-square input is resized anyway,
            but will be stretched, and a warning is logged.
        size: Target side length in pixels (defaults to the CoAtNet-1
            input size, 224).

    Returns:
        An RGB uint8 array of shape (size, size, 3).

    Raises:
        ImageProcessingError: if `image` fails validation.
    """
    _validate_image(image)
    if image.shape[0] != image.shape[1]:
        logger.warning(
            "resize_224: input is %dx%d (not square) -- this will distort aspect "
            "ratio. Call pad_to_square() first.",
            image.shape[0], image.shape[1],
        )
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def apply_clahe(
    image: np.ndarray,
    *,
    clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = DEFAULT_CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """Apply CLAHE to the L-channel of `image` in LAB color space only.

    Part 1, Step 2: contrast-limited adaptive histogram equalization is
    applied to the LAB L (lightness) channel only -- not per RGB channel,
    which would distort color balance. `clip_limit` (and `tile_grid_size`)
    are parameters specifically so augmentation.py can re-invoke this
    function with a randomized clip_limit for its separate, train-only,
    on-the-fly augmentation step, distinct from this module's fixed
    offline-pass value.

    Args:
        image: RGB uint8 array, shape (H, W, 3).
        clip_limit: CLAHE contrast-limiting threshold. Fixed at 2.0 for
            the offline pass (DEFAULT_CLAHE_CLIP_LIMIT).
        tile_grid_size: CLAHE tile grid size (rows, cols). Fixed at 8x8
            for the offline pass (DEFAULT_CLAHE_TILE_GRID_SIZE).

    Returns:
        An RGB uint8 array, same shape as `image`.

    Raises:
        ImageProcessingError: if `image` fails validation.
    """
    _validate_image(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_equalized = clahe.apply(l_channel)

    merged = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def run_offline_pipeline(
    image: np.ndarray,
    *,
    circle_crop_threshold: int = DEFAULT_CIRCLE_CROP_THRESHOLD,
    circle_crop_padding_fraction: float = DEFAULT_CROP_PADDING_FRACTION,
    min_crop_area_ratio: float = DEFAULT_MIN_CROP_AREA_RATIO,
    max_crop_area_ratio: float = DEFAULT_MAX_CROP_AREA_RATIO,
    target_size: int = TARGET_SIZE,
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    clahe_tile_grid_size: Tuple[int, int] = DEFAULT_CLAHE_TILE_GRID_SIZE,
    blur_flag_threshold: float = DEFAULT_BLUR_FLAG_THRESHOLD,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run the full, finalized OFFLINE preprocessing chain on one image.

    Fixed order (Design Doc Part 2, "OFFLINE" stage -- do not reorder):
        circle_crop -> pad_to_square -> resize_224 -> blur_score (logged
        only, measured at the final 224x224 scale) -> apply_clahe (fixed
        clip_limit / tile_grid_size)

    This function performs no file I/O and is a pure image transform.
    preprocessing/build_dataset.py (File 4) is responsible for loading the
    source image into the RGB uint8 convention this module expects,
    calling this function, saving `processed_image` to the per-client
    cache directory, and writing the returned `meta_dict` into that
    client's manifest row.

    Args:
        image: RGB uint8 array, shape (H, W, 3), as loaded from disk.
        circle_crop_threshold: See circle_crop().
        circle_crop_padding_fraction: See circle_crop().
        min_crop_area_ratio: See circle_crop().
        max_crop_area_ratio: See circle_crop().
        target_size: See resize_224().
        clahe_clip_limit: Fixed CLAHE clip limit for the offline pass.
        clahe_tile_grid_size: Fixed CLAHE tile grid for the offline pass.
        blur_flag_threshold: Laplacian-variance cutoff below which
            `meta_dict["low_quality_flag"]` is set True. Diagnostic only --
            this function never drops or alters the image because of it.

    Returns:
        A tuple (processed_image, meta_dict):
            - processed_image: (target_size, target_size, 3) RGB uint8 array.
            - meta_dict: {
                  "was_cropped": bool,
                  "blur_score": float,        # measured at 224x224, pre-CLAHE
                  "low_quality_flag": bool,
                  "original_height": int,
                  "original_width": int,
              }

    Raises:
        ImageProcessingError: if `image` fails validation.
    """
    _validate_image(image)
    original_height, original_width = image.shape[:2]

    cropped, was_cropped = circle_crop(
        image,
        threshold=circle_crop_threshold,
        padding_fraction=circle_crop_padding_fraction,
        min_crop_area_ratio=min_crop_area_ratio,
        max_crop_area_ratio=max_crop_area_ratio,
    )

    squared = pad_to_square(cropped)
    resized = resize_224(squared, size=target_size)

    # Measured here (final 224x224, pre-CLAHE) rather than right after
    # circle_crop -- see module docstring's blur-score-audit note. This is
    # purely a measurement-point change: it does not affect resized, which
    # apply_clahe() below still receives unmodified.
    score = blur_score(resized)
    low_quality_flag = score < blur_flag_threshold

    final_image = apply_clahe(
        resized,
        clip_limit=clahe_clip_limit,
        tile_grid_size=clahe_tile_grid_size,
    )

    meta_dict: Dict[str, Any] = {
        "was_cropped": was_cropped,
        "blur_score": score,
        "low_quality_flag": low_quality_flag,
        "original_height": original_height,
        "original_width": original_width,
    }
    return final_image, meta_dict