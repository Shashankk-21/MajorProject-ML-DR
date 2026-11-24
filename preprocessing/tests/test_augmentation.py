import cv2
import numpy as np
import torch

from preprocessing.augmentation import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_eval_transforms,
    get_train_transforms,
)


def _sample_image() -> np.ndarray:
    y, x = np.indices((224, 224))
    image = np.stack(
        ((x * 3 + y) % 256, (x + y * 2) % 256, (x * 2 + y * 3) % 256),
        axis=-1,
    ).astype(np.uint8)
    image[35:48, 41:54] = (220, 30, 45)
    image[142:151, 155:164] = (20, 15, 10)
    return image


def test_train_pipeline_has_only_conservative_transforms() -> None:
    transforms = get_train_transforms().transforms
    names = [type(transform).__name__ for transform in transforms]
    assert names == [
        "HorizontalFlip",
        "Rotate",
        "RandomBrightnessContrast",
        "HueSaturationValue",
        "Normalize",
        "ToTensorV2",
    ]

    rotate, brightness_contrast, hue_saturation_value = transforms[1:4]
    assert transforms[0].p == 0.5
    assert rotate.limit == (-15.0, 15.0)
    assert rotate.p == 0.5
    assert rotate.border_mode == cv2.BORDER_REFLECT_101
    assert brightness_contrast.brightness_limit == (-0.15, 0.15)
    assert brightness_contrast.contrast_limit == (-0.15, 0.15)
    assert brightness_contrast.p == 0.5
    assert hue_saturation_value.hue_shift_limit == 5
    assert hue_saturation_value.sat_shift_limit == 10
    assert hue_saturation_value.val_shift_limit == 10
    assert hue_saturation_value.p == 0.3

    excluded = {
        "GaussianBlur",
        "GaussNoise",
        "CoarseDropout",
        "VerticalFlip",
        "RandomClahe",
    }
    assert excluded.isdisjoint(names)


def test_train_transform_returns_stochastic_float32_tensors() -> None:
    transform = get_train_transforms()
    image = _sample_image()
    outputs = [transform(image=image)["image"] for _ in range(8)]

    assert all(output.shape == (3, 224, 224) for output in outputs)
    assert all(output.dtype == torch.float32 for output in outputs)
    assert any(not torch.equal(outputs[0], output) for output in outputs[1:])


def test_eval_transform_remains_normalization_only_and_deterministic() -> None:
    transform = get_eval_transforms()
    names = [type(operation).__name__ for operation in transform.transforms]
    assert names == ["Normalize", "ToTensorV2"]
    assert tuple(transform.transforms[0].mean) == IMAGENET_MEAN
    assert tuple(transform.transforms[0].std) == IMAGENET_STD

    image = _sample_image()
    first = transform(image=image)["image"]
    second = transform(image=image)["image"]

    assert first.shape == (3, 224, 224)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
