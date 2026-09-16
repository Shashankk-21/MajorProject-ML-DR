"""
preprocessing/augmentation.py

On-the-fly, TRAIN-ONLY augmentation for the Federated DR Detection
preprocessing pipeline, plus the shared eval-time normalization transform.

Finalized scope (Design Doc Part 1, Step 4 & Part 2, "DATALOADER" stage):
    - get_train_transforms(): the "heavy augmentation" step -- applied to
      the TRAIN split only, never to validation or test, for any client.
    - get_eval_transforms(): normalization only -- applied to ALL splits
      (val, test, and train, in addition to get_train_transforms()).
    - IMAGENET_MEAN / IMAGENET_STD: defined once here (Part 3), imported
      by dataset.py / build_dataset.py wherever normalization stats are
      needed.

Boundary: this module contains only per-image transform *definitions*
(Albumentations Compose objects). It has no manifest, split, Dataset, or
DataLoader logic (see dataset.py) and does no file I/O (see
build_dataset.py). It imports `apply_clahe` from image_processing.py and
reuses it directly, rather than reimplementing CLAHE, so the offline and
on-the-fly CLAHE steps stay identical except for the randomized clip_limit
called out in Part 1, Step 4, fix 4.

---------------------------------------------------------------------------
FLAGGED: dependency and version note (read before running)
---------------------------------------------------------------------------
Part 1, Step 4's four fixes ("border_mode=reflect_101", "HueSaturationValue"
-style hue/sat, "CoarseDropout") are Albumentations-specific vocabulary, so
this module is built on the `albumentations` library. Two real,
non-cosmetic issues came up implementing it, both resolved below rather
than left as blocking questions:

1. LICENSING. The actively-developed package on PyPI today is
   `albumentationsx` (AGPL-3.0-only / commercial dual license). The
   original `albumentations` package is now a frozen, MIT-licensed,
   archived release, last published as 2.0.8 (May 2025) -- unmaintained,
   but functionally complete for everything this pipeline needs. Because
   this repo includes a `webapp/` (a network-facing service), pulling in
   AGPL-licensed code is a real licensing consideration, not just a
   style choice -- AGPL's "network use" clause can require you to release
   your modified source if the webapp is deployed as a network service.
   requirements.txt should therefore pin the last MIT release explicitly:
       albumentations==2.0.8
   If that constraint doesn't matter for your situation (e.g. the webapp
   stays local/non-distributed) and you'd rather have an actively
   maintained library, swap in `albumentationsx` instead -- it's a
   documented drop-in replacement for the same `import albumentations as
   A` API, so nothing else in this file would need to change.

2. PARAMETER RENAMES. Starting with the 2.1.x line (i.e. after 2.0.8),
   Albumentations renamed several scalar "_limit" constructor args to
   tuple-only "_range" args (Rotate.limit -> angle_range,
   RandomBrightnessContrast.{brightness,contrast}_limit -> *_range,
   ShiftScaleRotate.{shift,scale,rotate}_limit -> *_range). CoarseDropout
   is unaffected -- confirmed against a real installed 2.0.8 that its
   num_holes_range/hole_height_range/hole_width_range/fill names already
   work as-is. Pinning 2.0.8 means the *_limit names are what's actually
   installed, so that's what this file targets -- but a small
   `_construct_versioned()` helper (below) tries the modern kwargs first
   and falls back to the legacy ones, so this file also keeps working
   unmodified if you opt into `albumentationsx` or a newer pin instead.

   Verified against a real installed albumentations==2.0.8: passing the
   modern names (e.g. `angle_range=`) to a 2.0.8 transform does NOT raise
   TypeError -- it raises a UserWarning ("Argument(s) 'angle_range' are
   not valid for transform Rotate") and silently falls back to that
   transform's OWN built-in defaults for the rejected arguments (e.g.
   Rotate's default limit=90, or ShiftScaleRotate's default
   rotate_limit=45) rather than erroring. A bare `except TypeError` never
   catches this, and would silently produce a rotation range of only
   +/-90 degrees, or leave ShiftScaleRotate's rotation component at its
   default +/-45 degrees, instead of the finalized values below. This
   module's `_construct_versioned()` therefore captures warnings during
   the modern-kwargs attempt and treats an Albumentations
   "not valid for transform" warning the same as a hard failure, forcing
   the legacy-kwargs path.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import random
import warnings
from typing import Any, Dict, Tuple

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from .image_processing import DEFAULT_CLAHE_TILE_GRID_SIZE, apply_clahe

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "get_train_transforms",
    "get_eval_transforms",
    "RandomClahe",
]

# ---------------------------------------------------------------------------
# Shared normalization constants (Part 1, Step 6: "ImageNet mean/std --
# keep, don't compute custom stats"). Defined once here; import these from
# dataset.py / build_dataset.py rather than redefining them.
# ---------------------------------------------------------------------------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Default augmentation parameters.
#
# Four of these are named, finalized "fixes" in Part 1, Step 4:
#   1. Rotation border_mode -> DEFAULT_ROTATE_BORDER_MODE
#   2. Hue/sat range narrowed -> DEFAULT_HUE/SAT/VAL_SHIFT_LIMIT
#   3. CoarseDropout capped to 1-2 holes -> DEFAULT_DROPOUT_NUM_HOLES_RANGE
#   4. Aug-time CLAHE clip_limit randomized -> DEFAULT_AUG_CLAHE_CLIP_LIMIT_RANGE
#
# The remaining transforms/values (flip, shift/scale, brightness/contrast,
# Gaussian noise, and their probabilities) fill out the "full 9-transform
# stack" mentioned in Part 3 but are not individually specified there -- a
# sensible default fundus-photography augmentation policy, stated as an
# assumption per the project's own rule for unspecified details.
# ---------------------------------------------------------------------------

# Fix 1: reflect padding instead of black fill, so rotation never
# introduces artificial black wedges the model could latch onto.
DEFAULT_ROTATE_BORDER_MODE: int = cv2.BORDER_REFLECT_101
# Fix, Design Doc Part 1 Step 4: rotation range is +/-30 degrees, not a
# wide/arbitrary range.[cite: 1]
DEFAULT_ROTATION_LIMIT_DEGREES: float = 30.0
DEFAULT_ROTATE_P: float = 0.7

# Assumption: standard "heavy" brightness/contrast jitter.
DEFAULT_BRIGHTNESS_LIMIT: float = 0.2
DEFAULT_CONTRAST_LIMIT: float = 0.2
DEFAULT_BRIGHTNESS_CONTRAST_P: float = 0.7

# Fix 2: narrowed from Albumentations' own defaults (hue=20, sat=30,
# val=20) -- large hue shifts in particular can push healthy fundus tissue
# into implausible colors that don't occur in real fundus photography.
DEFAULT_HUE_SHIFT_LIMIT: int = 10
DEFAULT_SAT_SHIFT_LIMIT: int = 15
DEFAULT_VAL_SHIFT_LIMIT: int = 10
DEFAULT_HUE_SAT_VAL_P: float = 0.5

# Assumption: the kernel is kept small because fine lesion detail is already 
# only a few pixels wide at 224x224.
DEFAULT_GAUSS_BLUR_LIMIT: Tuple[int, int] = (3, 5)
DEFAULT_GAUSS_BLUR_P: float = 0.3

# Assumption, revised after a visual augmentation audit: mild sensor-noise
# simulation. Albumentations' own default, std_range=(0.2, 0.44), expresses
# noise std as a FRACTION of max_pixel_value (255 for uint8) -- i.e. a std
# of ~51-112 intensity levels, 20-44% of the full dynamic range. That is
# large enough to bury small, low-contrast DR lesions (microaneurysms,
# dot/blot hemorrhages are often only a few pixels wide and just a modest
# number of intensity levels darker than surrounding tissue) under random
# per-pixel speckle -- confirmed directly by the audit: several augmented
# samples came back with retinal structure completely obscured by colored
# static. std_range=(0.01, 0.03) targets a std of ~2.6-7.7 out of 255
# instead -- in line with the mild noise-variance band (var~10-50, i.e.
# std~3-7) commonly used specifically for retinal-image augmentation. This
# still provides genuine regularization against real sensor/compression
# noise and the hardware variation across this project's 4 different
# camera sources, without erasing few-pixel-wide, low-contrast lesion
# detail at 224x224.
DEFAULT_GAUSS_NOISE_STD_RANGE: Tuple[float, float] = (0.01, 0.03)
DEFAULT_GAUSS_NOISE_P: float = 0.3

# Fix 3: "cap to 1-2 small holes" -- num_holes_range=(1, 2), hole sizes
# expressed as a fraction of image side (3-8% of 224px ~= 7-18px) so a
# hole never covers a large fraction of the retina.
DEFAULT_DROPOUT_NUM_HOLES_RANGE: Tuple[int, int] = (1, 2)
DEFAULT_DROPOUT_HOLE_HEIGHT_RANGE: Tuple[float, float] = (0.03, 0.08)
DEFAULT_DROPOUT_HOLE_WIDTH_RANGE: Tuple[float, float] = (0.03, 0.08)
DEFAULT_DROPOUT_P: float = 0.3

# Fix 4: randomized clip_limit, deliberately NOT the offline pipeline's
# fixed 2.0 (image_processing.DEFAULT_CLAHE_CLIP_LIMIT). tile_grid_size is
# reused unchanged from image_processing.py -- only clip_limit is
# specified as randomized in Part 1, Step 4.
DEFAULT_AUG_CLAHE_CLIP_LIMIT_RANGE: Tuple[float, float] = (1.0, 4.0)
DEFAULT_AUG_CLAHE_P: float = 0.5


_INVALID_ARG_WARNING_MARKER = "not valid for transform"


def _construct_versioned(
    transform_cls: Any,
    modern_kwargs: Dict[str, Any],
    legacy_kwargs: Dict[str, Any],
) -> Any:
    """Instantiate an Albumentations transform across the 2.0.x/2.1.x+ API split.

    Tries `modern_kwargs` (the post-2.1.x "_range"-style names) first;
    falls back to `legacy_kwargs` (the pre-2.1.x names, matching the
    2.0.8 pin this module targets -- see module docstring) if that
    attempt fails to actually take effect.

    "Fails" is deliberately checked two ways, because Albumentations
    2.0.8 does NOT raise on an unrecognized keyword argument -- it emits
    a UserWarning ("Argument(s) 'x' are not valid for transform Y") and
    silently constructs the transform using its own built-in defaults
    for the rejected argument(s) instead:
        1. TypeError: a hard failure (e.g. on a stricter future release).
        2. A captured UserWarning containing "not valid for transform":
           the soft-failure path actually seen against a real installed
           albumentations==2.0.8 -- the call *succeeds* but silently
           ignored our intended values, which is worse than a TypeError
           because nothing would otherwise signal that the finalized
           parameters (e.g. Fix 1's rotation range, or disabling
           ShiftScaleRotate's rotation component) never actually applied.

    Args:
        transform_cls: An Albumentations transform class, e.g. A.Rotate.
        modern_kwargs: Keyword arguments using current (>=2.1.x) names.
        legacy_kwargs: Keyword arguments using legacy (<=2.0.8) names.

    Returns:
        An instantiated transform, constructed from whichever of
        `modern_kwargs` / `legacy_kwargs` actually took effect.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            instance = transform_cls(**modern_kwargs)
        except TypeError:
            return transform_cls(**legacy_kwargs)

    if any(_INVALID_ARG_WARNING_MARKER in str(w.message) for w in caught):
        return transform_cls(**legacy_kwargs)
    return instance


class RandomClahe(A.ImageOnlyTransform):
    """Albumentations wrapper reusing image_processing.apply_clahe with a randomized clip_limit.

    Implements Part 1, Step 4, fix 4 ("Aug-time CLAHE: randomize
    clipLimit -- don't repeat Step 2's fixed value"). Deliberately calls
    the SAME apply_clahe() function used by the offline pipeline
    (image_processing.py) rather than reimplementing LAB/CLAHE logic, so
    the two stages can never silently drift apart -- only clip_limit
    differs, and it differs by construction (drawn fresh per call from
    `clip_limit_range`, so it should essentially never land exactly on
    the offline pass's fixed 2.0).
    """

    def __init__(
        self,
        clip_limit_range: Tuple[float, float] = DEFAULT_AUG_CLAHE_CLIP_LIMIT_RANGE,
        tile_grid_size: Tuple[int, int] = DEFAULT_CLAHE_TILE_GRID_SIZE,
        p: float = DEFAULT_AUG_CLAHE_P,
    ) -> None:
        super().__init__(p=p)
        if clip_limit_range[0] <= 0 or clip_limit_range[1] < clip_limit_range[0]:
            raise ValueError(f"Invalid clip_limit_range: {clip_limit_range}")
        self.clip_limit_range = clip_limit_range
        self.tile_grid_size = tile_grid_size

    def get_params(self) -> Dict[str, Any]:
        return {"clip_limit": random.uniform(*self.clip_limit_range)}

    def apply(self, img: Any, clip_limit: float = 2.0, **params: Any) -> Any:
        return apply_clahe(img, clip_limit=clip_limit, tile_grid_size=self.tile_grid_size)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("clip_limit_range", "tile_grid_size")


def get_train_transforms(
    *,
    rotation_limit_degrees: float = DEFAULT_ROTATION_LIMIT_DEGREES,
    brightness_limit: float = DEFAULT_BRIGHTNESS_LIMIT,
    contrast_limit: float = DEFAULT_CONTRAST_LIMIT,
    hue_shift_limit: int = DEFAULT_HUE_SHIFT_LIMIT,
    sat_shift_limit: int = DEFAULT_SAT_SHIFT_LIMIT,
    val_shift_limit: int = DEFAULT_VAL_SHIFT_LIMIT,
    gauss_blur_limit: Tuple[int, int] = DEFAULT_GAUSS_BLUR_LIMIT,
    gauss_noise_std_range: Tuple[float, float] = DEFAULT_GAUSS_NOISE_STD_RANGE,
    dropout_num_holes_range: Tuple[int, int] = DEFAULT_DROPOUT_NUM_HOLES_RANGE,
    dropout_hole_height_range: Tuple[float, float] = DEFAULT_DROPOUT_HOLE_HEIGHT_RANGE,
    dropout_hole_width_range: Tuple[float, float] = DEFAULT_DROPOUT_HOLE_WIDTH_RANGE,
    clahe_clip_limit_range: Tuple[float, float] = DEFAULT_AUG_CLAHE_CLIP_LIMIT_RANGE,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """Build the TRAIN-ONLY heavy-augmentation pipeline (Part 1, Step 4).

    Never apply this to validation or test splits, for any client (Part 2:
    "heavy augmentation [TRAIN ONLY]"; Section 6 evaluation framework:
    "Validation sets are never touched by ... training-time augmentation
    for any client").

    The "full 9-transform stack" (Part 3) is, in order:
        1. HorizontalFlip
        2. VerticalFlip
        3. Rotate                    -- fix 1: border_mode=REFLECT_101
        4. RandomBrightnessContrast
        5. HueSaturationValue        -- fix 2: narrowed hue/sat/val ranges
        6. GaussianBlur
        7. GaussNoise                 -- mild sensor-noise simulation,
                                         explicit std_range (see
                                         DEFAULT_GAUSS_NOISE_STD_RANGE)
        8. CoarseDropout             -- fix 3: capped to 1-2 small holes
        9. RandomClahe               -- fix 4: randomized clip_limit,
                                         reusing image_processing.apply_clahe
    followed by Normalize + ToTensorV2 (shared with get_eval_transforms(),
    not counted among the 9).

    Note on pipeline order: Design Doc Part 2 lists the DataLoader-time
    order as "ToTensor([0,1]) -> heavy augmentation -> normalize". Taken
    fully literally this conflicts with how Albumentations transforms
    operate -- CoarseDropout, HueSaturationValue, Rotate's border_mode,
    and CLAHE are all designed to run on uint8/array images, not on an
    already-tensorized float. This function implements the equivalent,
    Albumentations-native ordering that reaches the same end state (an
    augmented, ImageNet-normalized tensor): augmentation operates on the
    raw cached uint8 image, then Normalize (which itself divides by 255
    to reach [0,1] before subtracting mean/dividing by std) runs, then
    ToTensorV2 does the final HWC->CHW tensor conversion.

    Args:
        rotation_limit_degrees: Max +/- rotation angle in degrees.
        brightness_limit: Max +/- brightness jitter.
        contrast_limit: Max +/- contrast jitter.
        hue_shift_limit: Max +/- hue shift (Fix 2).
        sat_shift_limit: Max +/- saturation shift (Fix 2).
        val_shift_limit: Max +/- value/brightness (HSV) shift (Fix 2).
        gauss_blur_limit: Max kernel size for Gaussian Blur.
        gauss_noise_std_range: (min, max) range to sample the per-image
            Gaussian noise std from, expressed as a fraction of
            max_pixel_value (255) -- see DEFAULT_GAUSS_NOISE_STD_RANGE for
            the reasoning behind the default.
        dropout_num_holes_range: (min, max) number of dropout holes (Fix 3).
        dropout_hole_height_range: (min, max) hole height, as a fraction
            of image height.
        dropout_hole_width_range: (min, max) hole width, as a fraction of
            image width.
        clahe_clip_limit_range: (min, max) range to sample the aug-time
            CLAHE clip_limit from (Fix 4).
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.

    Returns:
        An albumentations.Compose ready to call as
        `transform(image=img)["image"]`.
    """
    rotate = _construct_versioned(
        A.Rotate,
        modern_kwargs=dict(
            angle_range=(-rotation_limit_degrees, rotation_limit_degrees),
            border_mode=DEFAULT_ROTATE_BORDER_MODE,
            p=DEFAULT_ROTATE_P,
        ),
        legacy_kwargs=dict(
            limit=rotation_limit_degrees,
            border_mode=DEFAULT_ROTATE_BORDER_MODE,
            p=DEFAULT_ROTATE_P,
        ),
    )

    brightness_contrast = _construct_versioned(
        A.RandomBrightnessContrast,
        modern_kwargs=dict(
            brightness_range=(-brightness_limit, brightness_limit),
            contrast_range=(-contrast_limit, contrast_limit),
            p=DEFAULT_BRIGHTNESS_CONTRAST_P,
        ),
        legacy_kwargs=dict(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=DEFAULT_BRIGHTNESS_CONTRAST_P,
        ),
    )

    coarse_dropout = _construct_versioned(
        A.CoarseDropout,
        modern_kwargs=dict(
            num_holes_range=dropout_num_holes_range,
            hole_height_range=dropout_hole_height_range,
            hole_width_range=dropout_hole_width_range,
            fill=0,
            p=DEFAULT_DROPOUT_P,
        ),
        legacy_kwargs=dict(
            num_holes_range=dropout_num_holes_range,
            hole_height_range=dropout_hole_height_range,
            hole_width_range=dropout_hole_width_range,
            fill_value=0,
            p=DEFAULT_DROPOUT_P,
        ),
    )

    transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        rotate,
        brightness_contrast,
        A.HueSaturationValue(
            hue_shift_limit=hue_shift_limit,
            sat_shift_limit=sat_shift_limit,
            val_shift_limit=val_shift_limit,
            p=DEFAULT_HUE_SAT_VAL_P,
        ),
        A.GaussianBlur(
            blur_limit=gauss_blur_limit, 
            p=DEFAULT_GAUSS_BLUR_P
        ),
        A.GaussNoise(std_range=gauss_noise_std_range, p=DEFAULT_GAUSS_NOISE_P),
        coarse_dropout,
        RandomClahe(clip_limit_range=clahe_clip_limit_range, p=DEFAULT_AUG_CLAHE_P),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def get_eval_transforms(
    *,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """Build the normalization-only pipeline shared by ALL splits.

    Applies to validation and test always, and to train in addition to
    get_train_transforms() (Part 3: "get_eval_transforms() -> Compose --
    ToTensor + Normalize only").

    Args:
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.

    Returns:
        An albumentations.Compose ready to call as
        `transform(image=img)["image"]`.
    """
    return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])