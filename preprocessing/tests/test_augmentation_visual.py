from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from preprocessing.augmentation import get_train_transforms
from preprocessing.image_processing import run_offline_pipeline


# ============================================================
# REAL APTOS IMAGES
# Change only these paths if needed.
# ============================================================

IMAGE_PATHS = APTOS_TRAIN_DIR = Path(
    r"C:\MajorProject - Datsets\APTOS-2019-Kaggle\train_images\train_images"
)

IMAGE_PATHS = sorted(APTOS_TRAIN_DIR.glob("*.png"))[:5]
OUTPUT_DIR = Path("data/_augmentation_visual_audit")
OUTPUT_FILE = OUTPUT_DIR / "augmentation_contact_sheet.png"

# Number of stochastic augmentations per image.
N_AUGMENTATIONS = 8


def load_rgb(path: Path) -> np.ndarray:
    """Load a real image as RGB uint8."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found:\n{path}")

    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def tensor_to_display_image(tensor) -> np.ndarray:
    """
    Convert ImageNet-normalized CHW tensor back to displayable HWC RGB.

    This is ONLY for visualization. It does not modify the actual pipeline.
    """
    arr = tensor.detach().cpu().numpy()

    if arr.shape != (3, 224, 224):
        raise ValueError(f"Unexpected tensor shape: {arr.shape}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    arr = arr * std + mean
    arr = np.clip(arr, 0.0, 1.0)

    arr = np.transpose(arr, (1, 2, 0))

    return arr


def main() -> None:
    print("=" * 70)
    print("AUGMENTATION VISUAL AUDIT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_transform = get_train_transforms()

    # --------------------------------------------------------
    # Load + offline preprocess
    # --------------------------------------------------------

    processed_images: list[np.ndarray] = []

    for path in IMAGE_PATHS:
        raw = load_rgb(path)
        processed, metadata = run_offline_pipeline(raw)

        print(f"\n{path.name}")
        print(f"  Raw shape       : {raw.shape}")
        print(f"  Processed shape : {processed.shape}")
        print(f"  Was cropped     : {metadata['was_cropped']}")
        print(f"  Blur score      : {metadata['blur_score']:.4f}")
        print(f"  Low quality     : {metadata['low_quality_flag']}")

        if processed.shape != (224, 224, 3):
            raise AssertionError(
                f"{path.name}: unexpected processed shape {processed.shape}"
            )

        if processed.dtype != np.uint8:
            raise AssertionError(
                f"{path.name}: unexpected dtype {processed.dtype}"
            )

        processed_images.append(processed)

    # --------------------------------------------------------
    # Contact sheet:
    # each row = one real image
    # col 0 = original
    # cols 1..8 = stochastic augmentations
    # --------------------------------------------------------

    rows = len(processed_images)
    cols = 1 + N_AUGMENTATIONS

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(18, 3.2 * rows),
        squeeze=False,
    )

    for row_idx, (path, processed) in enumerate(
        zip(IMAGE_PATHS, processed_images)
    ):
        # Original
        ax = axes[row_idx][0]
        ax.imshow(processed)
        ax.set_title(f"{path.stem}\nORIGINAL", fontsize=9)
        ax.axis("off")

        # Augmented versions
        for aug_idx in range(N_AUGMENTATIONS):
            transformed = train_transform(image=processed)["image"]
            display_img = tensor_to_display_image(transformed)

            ax = axes[row_idx][aug_idx + 1]
            ax.imshow(display_img)
            ax.set_title(f"AUG {aug_idx + 1}", fontsize=9)
            ax.axis("off")

    fig.suptitle(
        "APTOS — Training Augmentation Visual Audit",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(
        OUTPUT_FILE,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Images audited       : {len(processed_images)}")
    print(f"Augmentations/image  : {N_AUGMENTATIONS}")
    print(f"Total augmented imgs : {len(processed_images) * N_AUGMENTATIONS}")
    print(f"Contact sheet saved   : {OUTPUT_FILE}")
    print("\nVISUAL AUDIT GENERATED SUCCESSFULLY")


if __name__ == "__main__":
    main()