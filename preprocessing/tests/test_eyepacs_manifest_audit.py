"""
Validates the real, full-scale EyePACS production manifest.

Run from the project root with dr_project activated:
    python -m preprocessing.test_eyepacs_manifest_audit
"""
import pandas as pd
from pathlib import Path

MANIFEST_PATH = Path("data/splits/EyePACS_manifest.csv")
CACHE_DIR = Path("data/cache/EyePACS")
EXPECTED_TOTAL = 88702

print("=" * 60)
print("EYEPACS PRODUCTION MANIFEST AUDIT")
print("=" * 60)

df = pd.read_csv(MANIFEST_PATH)
print(f"Total rows: {len(df)} (expected {EXPECTED_TOTAL})")
assert len(df) == EXPECTED_TOTAL, "FAIL: row count doesn't match the full labeled pool."

dupes = len(df) - df["image_id"].nunique()
print(f"Duplicate image_id count: {dupes}")
assert dupes == 0, "FAIL: duplicate image_id present."

missing_labels = df["label"].isna().sum()
print(f"Missing labels: {missing_labels}")
assert missing_labels == 0, "FAIL: rows with missing labels."

leak = df.groupby("patient_id")["split"].nunique()
bad = leak[leak > 1]
print(f"Unique patients: {df['patient_id'].nunique()}")
print(f"Patients spanning multiple splits: {len(bad)}")
assert len(bad) == 0, f"FAIL: patient leakage detected for {list(bad.index)}"

missing_files = [p for p in df["cached_image_path"] if not Path(p).exists()]
print(f"Manifest rows pointing to a missing cached file: {len(missing_files)}")
assert len(missing_files) == 0, f"FAIL: {len(missing_files)} cached file(s) missing on disk."

print("\nPASS: manifest is complete, duplicate-free, leakage-free, and every cached file exists.")