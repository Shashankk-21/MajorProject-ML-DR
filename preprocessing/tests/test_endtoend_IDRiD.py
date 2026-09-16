import pandas as pd
import torch
from pathlib import Path
from PIL import Image

from preprocessing.dataset import build_manifest, split_manifest, get_dataloaders
from preprocessing.augmentation import get_train_transforms, get_eval_transforms

DATA_ROOT = Path(r"C:\MajorProject - Datasets")
MANIFEST_PATH = Path("data/splits/IDRiD_manifest.csv")

def run_end_to_end_test():
    print("=" * 70)
    print("1. IN-MEMORY BUILD & SPLIT")
    print("=" * 70)
    # Re-run the exact logic that built the dataset
    source_df = build_manifest("IDRiD", data_root=DATA_ROOT)
    split_df = split_manifest("IDRiD", source_df, random_state=42)
    split_df["image_id"] = split_df["image_id"].astype(str)
    print(f"In-memory manifest generated. Rows: {len(split_df)}")

    print("\n" + "=" * 70)
    print("2. COMPARE AGAINST CACHED DISK MANIFEST")
    print("=" * 70)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing {MANIFEST_PATH}. Did you delete it?")
        
    disk_df = pd.read_csv(MANIFEST_PATH)
    disk_df["image_id"] = disk_df["image_id"].astype(str)
    
    # Verify row counts match exactly
    assert len(split_df) == len(disk_df), f"Row count mismatch: Mem={len(split_df)}, Disk={len(disk_df)}"
    
    # Merge on image_id to ensure the exact same images received the exact same splits and labels
    merged = split_df.merge(disk_df, on="image_id", suffixes=("_mem", "_disk"))
    
    assert (merged["split_mem"] == merged["split_disk"]).all(), "FAIL: Split assignments drifted between memory and disk."
    assert (merged["label_mem"] == merged["label_disk"]).all(), "FAIL: Label assignments drifted between memory and disk."
    print("PASS: In-memory split logic perfectly matches the actual cached manifest on disk.")

    print("\n" + "=" * 70)
    print("3. DATALOADER INITIALIZATION & BATCH EXTRACTION")
    print("=" * 70)
    train_loader, val_loader, test_loader = get_dataloaders(
        "IDRiD",
        manifest_path=MANIFEST_PATH,
        batch_size=8,
        num_workers=0,
        train_transform=get_train_transforms(),
        eval_transform=get_eval_transforms(),
        seed=42
    )
    print("DataLoaders initialized successfully.")
    
    # Pull a real batch
    images, labels = next(iter(train_loader))
    assert images.shape == (8, 3, 224, 224), f"Unexpected batch shape: {images.shape}"
    assert labels.shape == (8,), f"Unexpected labels shape: {labels.shape}"
    assert images.dtype == torch.float32, "Tensor is not float32!"
    
    print(f"PASS: Real batch pulled. Shape: {tuple(images.shape)}, Dtype: {images.dtype}")
    print(f"Batch Labels: {labels.tolist()}")

    print("\n" + "=" * 70)
    print("4. METADATA & CACHE SPOT-CHECK")
    print("=" * 70)
    # Pick a random row from the disk manifest to verify physical properties
    sample_row = disk_df.sample(1, random_state=123).iloc[0]
    cached_path = Path(sample_row["cached_image_path"])
    
    assert cached_path.exists(), f"FAIL: Cached image missing at {cached_path}"
    
    img = Image.open(cached_path)
    img_array = np.array(img)
    
    print(f"Spot-checking Image ID : {sample_row['image_id']}")
    print(f"Physical Path          : {cached_path}")
    print(f"Physical Cache Shape   : {img_array.shape} (Expected: (224, 224, 3))")
    print(f"Metadata - Split       : {sample_row['split']}")
    print(f"Metadata - Label       : {sample_row['label']}")
    print(f"Metadata - Cropped?    : {sample_row['was_cropped']}")
    print(f"Metadata - Blur Score  : {sample_row['blur_score']:.2f}")
    print(f"Metadata - Orig Size   : {sample_row['original_height']}x{sample_row['original_width']}")
    
    assert img_array.shape == (224, 224, 3), "FAIL: Cached image is not 224x224x3!"
    print("\nPASS: Metadata perfectly aligns with the physical cached PNG.")

if __name__ == "__main__":
    import numpy as np
    run_end_to_end_test()