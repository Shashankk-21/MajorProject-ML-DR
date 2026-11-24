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
build_dataset.py). Offline CLAHE is performed once by image_processing.py;
the active training transform does not apply CLAHE again.

---------------------------------------------------------------------------
FLAGGED: dependency and version note (read before running)
---------------------------------------------------------------------------
The version-dependent arguments for Rotate and
RandomBrightnessContrast are Albumentations-specific vocabulary, so this
module is built on the `albumentations` library. Two real,
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
   RandomBrightnessContrast.{brightness,contrast}_limit -> *_range).
   Pinning 2.0.8 means the *_limit names are what's actually
   installed, so that's what this file targets -- but a small
   `_construct_versioned()` helper (below) tries the modern kwargs first
   and falls back to the legacy ones, so this file also keeps working
   unmodified if you opt into `albumentationsx` or a newer pin instead.

   Verified against a real installed albumentations==2.0.8: passing the
   modern names (e.g. `angle_range=`) to a 2.0.8 transform does NOT raise
   TypeError -- it raises a UserWarning ("Argument(s) 'angle_range' are
   not valid for transform Rotate") and silently falls back to that
   transform's OWN built-in defaults for the rejected arguments (e.g.
   Rotate's default limit=90) rather than erroring. A bare
   `except TypeError` never
   catches this, and would silently produce a rotation range of only
   +/-90 degrees instead of the intended value below. This module's
   `_construct_versioned()` therefore captures warnings during the
   modern-kwargs attempt and treats an Albumentations
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
# Parameters for the active conservative training pipeline.
# ---------------------------------------------------------------------------

# Fix 1: reflect padding instead of black fill, so rotation never
# introduces artificial black wedges the model could latch onto.
DEFAULT_ROTATE_BORDER_MODE: int = cv2.BORDER_REFLECT_101
DEFAULT_ROTATION_LIMIT_DEGREES: float = 15.0
DEFAULT_ROTATE_P: float = 0.5

DEFAULT_BRIGHTNESS_LIMIT: float = 0.15
DEFAULT_CONTRAST_LIMIT: float = 0.15
DEFAULT_BRIGHTNESS_CONTRAST_P: float = 0.5

DEFAULT_HUE_SHIFT_LIMIT: int = 5
DEFAULT_SAT_SHIFT_LIMIT: int = 10
DEFAULT_VAL_SHIFT_LIMIT: int = 10
DEFAULT_HUE_SAT_VAL_P: float = 0.3

# Retained legacy defaults are not used by get_train_transforms().
DEFAULT_GAUSS_BLUR_LIMIT: Tuple[int, int] = (3, 5)
DEFAULT_GAUSS_BLUR_P: float = 0.3

DEFAULT_GAUSS_NOISE_STD_RANGE: Tuple[float, float] = (0.01, 0.03)
DEFAULT_GAUSS_NOISE_P: float = 0.3

DEFAULT_DROPOUT_NUM_HOLES_RANGE: Tuple[int, int] = (1, 2)
DEFAULT_DROPOUT_HOLE_HEIGHT_RANGE: Tuple[float, float] = (0.03, 0.08)
DEFAULT_DROPOUT_HOLE_WIDTH_RANGE: Tuple[float, float] = (0.03, 0.08)
DEFAULT_DROPOUT_P: float = 0.3

# CLAHE defaults are retained for the reusable RandomClahe class, which is
# not included in the active training pipeline.
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
           because nothing would otherwise signal that the intended
           rotation or brightness/contrast values never actually applied.

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
    """Reusable Albumentations CLAHE wrapper, not used by the train pipeline.

    When explicitly used, it calls the same apply_clahe() helper as the
    offline pipeline. The active training transform leaves this out so
    offline CLAHE is applied only once.
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
        return {"clip_limit": self.py_random.uniform(*self.clip_limit_range)}

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
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """Build the conservative TRAIN-ONLY augmentation pipeline.

    Never apply this to validation or test splits, for any client (Part 2:
    "training augmentation [TRAIN ONLY]"; Section 6 evaluation framework:
    "Validation sets are never touched by ... training-time augmentation
    for any client").

    The stochastic transforms are, in order:
        1. HorizontalFlip
        2. Rotate                    -- +/-15 degrees, border_mode=REFLECT_101
        3. RandomBrightnessContrast -- brightness/contrast limits +/-0.15
        4. HueSaturationValue        -- hue=5, saturation=10, value=10
    followed by ImageNet Normalize + ToTensorV2. CLAHE is not applied here;
    it has already been performed once by the offline preprocessing pipeline.

    Note on pipeline order: Design Doc Part 2 lists the DataLoader-time
    order as "ToTensor([0,1]) -> heavy augmentation -> normalize". Taken
    fully literally this conflicts with how Albumentations transforms
    operate -- HueSaturationValue and Rotate's border_mode are designed to
    run on uint8/array images, not on an already-tensorized float. This
    function implements the equivalent,
    Albumentations-native ordering that reaches the same end state (an
    augmented, ImageNet-normalized tensor): augmentation operates on the
    raw cached uint8 image, then Normalize (which itself divides by 255
    to reach [0,1] before subtracting mean/dividing by std) runs, then
    ToTensorV2 does the final HWC->CHW tensor conversion.

    Args:
        rotation_limit_degrees: Max +/- rotation angle in degrees.
        brightness_limit: Max +/- brightness jitter.
        contrast_limit: Max +/- contrast jitter.
        hue_shift_limit: Max +/- hue shift.
        sat_shift_limit: Max +/- saturation shift.
        val_shift_limit: Max +/- value/brightness (HSV) shift.
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

    transforms = [
        A.HorizontalFlip(p=0.5),
        rotate,
        brightness_contrast,
        A.HueSaturationValue(
            hue_shift_limit=hue_shift_limit,
            sat_shift_limit=sat_shift_limit,
            val_shift_limit=val_shift_limit,
            p=DEFAULT_HUE_SAT_VAL_P,
        ),
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