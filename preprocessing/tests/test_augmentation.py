from pathlib import Path

import numpy as np
import torch
from PIL import Image

from preprocessing.image_processing import run_offline_pipeline
from preprocessing.augmentation import (
    get_train_transforms,
    get_eval_transforms,
)


# ============================================================
# CHANGE ONLY THIS PATH
# ============================================================
IMAGE_PATH = Path(
    r"C:\MajorProject - Datasets\APTOS-2019-Kaggle\train_images\train_images\1b8ad0afe9fb.png"
)
# ============================================================


def main() -> None:
    print("=" * 60)
    print("1. CHECKING REAL IMAGE")
    print("=" * 60)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found:\n{IMAGE_PATH}\n"
            "\nFix IMAGE_PATH at the top of test_augmentation.py."
        )

    print("Image found:")
    print(IMAGE_PATH)

    # --------------------------------------------------------
    # Load raw image exactly as build_dataset.py will later do.
    # --------------------------------------------------------
    raw_image = np.array(
        Image.open(IMAGE_PATH).convert("RGB")
    )

    print("\nRAW IMAGE")
    print("Shape :", raw_image.shape)
    print("Dtype :", raw_image.dtype)
    print("Min   :", raw_image.min())
    print("Max   :", raw_image.max())

    # --------------------------------------------------------
    # Run the finalized OFFLINE preprocessing stage.
    # This should produce 224x224 RGB uint8.
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. OFFLINE PREPROCESSING")
    print("=" * 60)

    processed_image, metadata = run_offline_pipeline(raw_image)

    print("Processed image:")
    print("  Shape :", processed_image.shape)
    print("  Dtype :", processed_image.dtype)
    print("  Min   :", processed_image.min())
    print("  Max   :", processed_image.max())

    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # --------------------------------------------------------
    # Verify offline output.
    # --------------------------------------------------------
    assert processed_image.shape == (224, 224, 3), (
        f"Expected offline output (224, 224, 3), "
        f"got {processed_image.shape}"
    )

    assert processed_image.dtype == np.uint8, (
        f"Expected uint8, got {processed_image.dtype}"
    )

    print("\nOffline preprocessing: PASS")

    # --------------------------------------------------------
    # Build transforms.
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. BUILDING AUGMENTATION TRANSFORMS")
    print("=" * 60)

    train_transform = get_train_transforms()
    eval_transform = get_eval_transforms()

    print("Train transform: OK")
    print("Eval transform : OK")

    # --------------------------------------------------------
    # Apply TRAIN transform multiple times.
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. TRAIN TRANSFORM")
    print("=" * 60)

    train_1 = train_transform(
        image=processed_image
    )["image"]

    train_2 = train_transform(
        image=processed_image
    )["image"]

    train_3 = train_transform(
        image=processed_image
    )["image"]

    for i, tensor in enumerate(
        [train_1, train_2, train_3],
        start=1,
    ):
        print(f"\nTrain {i}:")
        print("  Shape :", tuple(tensor.shape))
        print("  Dtype :", tensor.dtype)
        print("  Min   :", tensor.min().item())
        print("  Max   :", tensor.max().item())

    # --------------------------------------------------------
    # Check train shape/dtype.
    # --------------------------------------------------------
    train_shape_ok = train_1.shape == (3, 224, 224)
    train_dtype_ok = train_1.dtype == torch.float32

    # --------------------------------------------------------
    # Check train stochasticity.
    # --------------------------------------------------------
    diff_12 = torch.mean(
        torch.abs(train_1 - train_2)
    ).item()

    diff_23 = torch.mean(
        torch.abs(train_2 - train_3)
    ).item()

    print("\nTrain 1 vs Train 2 difference:", diff_12)
    print("Train 2 vs Train 3 difference:", diff_23)

    train_random_ok = diff_12 > 0 or diff_23 > 0

    if train_random_ok:
        print("Train stochasticity: PASS")
    else:
        print("Train stochasticity: FAIL")

    # --------------------------------------------------------
    # Apply EVAL transform twice.
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. EVAL TRANSFORM")
    print("=" * 60)

    eval_1 = eval_transform(
        image=processed_image
    )["image"]

    eval_2 = eval_transform(
        image=processed_image
    )["image"]

    print("Eval 1:")
    print("  Shape :", tuple(eval_1.shape))
    print("  Dtype :", eval_1.dtype)
    print("  Min   :", eval_1.min().item())
    print("  Max   :", eval_1.max().item())

    print("\nEval 2:")
    print("  Shape :", tuple(eval_2.shape))
    print("  Dtype :", eval_2.dtype)
    print("  Min   :", eval_2.min().item())
    print("  Max   :", eval_2.max().item())

    # --------------------------------------------------------
    # Check eval shape/dtype/determinism.
    # --------------------------------------------------------
    eval_shape_ok = eval_1.shape == (3, 224, 224)
    eval_dtype_ok = eval_1.dtype == torch.float32

    eval_identical = torch.equal(eval_1, eval_2)

    eval_difference = torch.mean(
        torch.abs(eval_1 - eval_2)
    ).item()

    print("\nEval tensors identical:", eval_identical)
    print(
        "Eval mean absolute difference:",
        eval_difference,
    )

    eval_deterministic_ok = (
        eval_identical and eval_difference == 0.0
    )

    if eval_deterministic_ok:
        print("Eval determinism: PASS")
    else:
        print("Eval determinism: FAIL")

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("6. FINAL RESULT")
    print("=" * 60)

    checks = {
        "Offline output = (224, 224, 3)": (
            processed_image.shape == (224, 224, 3)
        ),
        "Offline output = uint8": (
            processed_image.dtype == np.uint8
        ),
        "Train output = (3, 224, 224)": train_shape_ok,
        "Eval output = (3, 224, 224)": eval_shape_ok,
        "Train dtype = float32": train_dtype_ok,
        "Eval dtype = float32": eval_dtype_ok,
        "Train augmentation is stochastic": train_random_ok,
        "Eval transform is deterministic": eval_deterministic_ok,
    }

    all_passed = True

    for name, passed in checks.items():
        print(
            f"{'PASS' if passed else 'FAIL'}: {name}"
        )

        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("ONE OR MORE TESTS FAILED")


if __name__ == "__main__":
    main()