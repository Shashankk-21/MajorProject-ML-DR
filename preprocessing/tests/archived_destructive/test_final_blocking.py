import pandas as pd
from pathlib import Path
from preprocessing.dataset import build_manifest, split_manifest, get_dataloaders
from preprocessing.augmentation import get_train_transforms, get_eval_transforms

DATA_ROOT = Path(r"C:\MajorProject - Datasets")

print("=" * 60)
print("1. CROSS-CLIENT ISOLATION ASSERTION")
print("=" * 60)
# Build manifests for two separate clients and ensure absolutely zero overlap
idrid = build_manifest("IDRiD", data_root=DATA_ROOT)
messidor = build_manifest("Messidor-2", data_root=DATA_ROOT)

overlap_images = set(idrid["image_id"]) & set(messidor["image_id"])
overlap_patients = set(idrid["patient_id"]) & set(messidor["patient_id"])
overlap_paths = set(idrid["raw_image_locator"]) & set(messidor["raw_image_locator"])

assert not overlap_images, f"FAIL: Image ID overlap detected: {overlap_images}"
assert not overlap_patients, f"FAIL: Patient ID overlap detected: {overlap_patients}"
assert not overlap_paths, f"FAIL: Raw path overlap detected: {overlap_paths}"
print("PASS: IDRiD and Messidor-2 share zero IDs, patients, or file paths.")

print("\n" + "=" * 60)
print("2. REPRODUCIBILITY CHECK")
print("=" * 60)
# Assert that running the split logic twice with the same seed yields byte-identical output
aptos_base = build_manifest("APTOS", data_root=DATA_ROOT)
split_1 = split_manifest("APTOS", aptos_base.copy(), random_state=42)
split_2 = split_manifest("APTOS", aptos_base.copy(), random_state=42)

pd.testing.assert_frame_equal(split_1, split_2)
print("PASS: split_manifest is perfectly deterministic for a given random_state.")

print("\n" + "=" * 60)
print("3. SAMPLER BEHAVIOR CHECK")
print("=" * 60)
# APTOS is highly imbalanced: Class 3 (Severe) is only ~5% of the raw dataset.
manifest_path = Path("data/splits/APTOS_manifest.csv")
if not manifest_path.exists():
    print("SKIPPED: APTOS manifest not found. Run a build_dataset for APTOS first.")
else:
    train_loader, _, _ = get_dataloaders(
        "APTOS",
        manifest_path=manifest_path,
        batch_size=32,
        num_workers=0,
        train_transform=get_train_transforms(),
        eval_transform=get_eval_transforms(),
        seed=42
    )

    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    batches_to_pull = 10

    for i, (_, labels) in enumerate(train_loader):
        if i >= batches_to_pull:
            break
        for label in labels.tolist():
            class_counts[label] += 1

    total_samples = sum(class_counts.values())
    class_3_pct = (class_counts[3] / total_samples) * 100

    print(f"Pulled {total_samples} samples. Class frequencies: {class_counts}")
    print(f"Class 3 (Severe) representation: {class_3_pct:.1f}% (Raw is ~5.2%)")

    assert class_3_pct > 10.0, "FAIL: WeightedRandomSampler is not actively oversampling minority classes."
    print("PASS: WeightedRandomSampler is actively reweighting the distribution.")