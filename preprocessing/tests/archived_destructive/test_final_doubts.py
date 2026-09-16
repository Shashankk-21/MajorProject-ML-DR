import pandas as pd
from pathlib import Path
from preprocessing.dataset import build_manifest, split_manifest

DATA_ROOT = Path(r"C:\MajorProject - Datasets")
MANIFEST_PATH = Path("data/splits/EyePACS_manifest.csv")

print("=" * 60)
print("1. EYEPACS PATIENT LEAKAGE CHECK (Post-Resume)")
print("=" * 60)
if MANIFEST_PATH.exists():
    df = pd.read_csv(MANIFEST_PATH)
    leak = df.groupby("patient_id")["split"].nunique()
    bad = leak[leak > 1]
    
    print(f"Total rows in manifest: {len(df)}")
    print(f"Patients spanning multiple splits: {len(bad)}")
    if len(bad) > 0:
        print("FAIL: Patient leakage detected!")
        print(bad)
    else:
        print("PASS: Zero patient leakage. The pipeline safely aligned splits.")
else:
    print("SKIPPED: EyePACS manifest not found.")


print("\n" + "=" * 60)
print("2. MESSIDOR-2 REPRODUCIBILITY CHECK")
print("=" * 60)
messidor_base = build_manifest("Messidor-2", data_root=DATA_ROOT)
split_1 = split_manifest("Messidor-2", messidor_base.copy(), random_state=42)
split_2 = split_manifest("Messidor-2", messidor_base.copy(), random_state=42)

try:
    pd.testing.assert_frame_equal(split_1, split_2)
    print("PASS: split_manifest is perfectly deterministic for StratifiedGroupKFold.")
except AssertionError as e:
    print("FAIL: DataFrames are not equal!")
    print(e)