import pandas as pd
from pathlib import Path

EXPECTED_COLUMNS = {
    "image_id", "cached_image_path", "patient_id", "label", "split", 
    "blur_score", "low_quality_flag", "was_cropped", "original_height", "original_width"
}

EXPECTED_TOTALS = {
    "APTOS": 3662,
    "IDRiD": 516,
    "Messidor-2": 1744,
    "EyePACS": 88702
}

OFFICIAL_SPLITS = {
    "APTOS": {"train": 2930, "val": 366, "test": 366},
    "IDRiD": {"train": 330, "val": 83, "test": 103}
}

def run_audit():
    print("=" * 70)
    print("FULL 4-CLIENT MANIFEST AUDIT")
    print("=" * 70)

    for client, expected_total in EXPECTED_TOTALS.items():
        manifest_path = Path(f"data/splits/{client}_manifest.csv")
        if not manifest_path.exists():
            print(f"SKIPPED: {client} manifest not found.")
            continue
            
        df = pd.read_csv(manifest_path)
        errors = []
        
        # 1. Total counts (proves no blur-based exclusions occurred)
        if len(df) != expected_total:
            errors.append(f"Row count {len(df)} != expected {expected_total} (Images were incorrectly excluded)")
            
        # 2. Column integrity
        missing_cols = EXPECTED_COLUMNS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing expected columns: {missing_cols}")
            
        # 3. Duplicate IDs
        dupes = df["image_id"].duplicated().sum()
        if dupes > 0:
            errors.append(f"{dupes} duplicate image_ids detected")
            
        # 4. Patient Leakage
        leak = df.groupby("patient_id")["split"].nunique()
        leaked_patients = len(leak[leak > 1])
        if leaked_patients > 0:
            errors.append(f"{leaked_patients} patients span multiple splits (Data Leakage!)")
            
        # 5. Official Splits constraint (APTOS & IDRiD only)
        if client in OFFICIAL_SPLITS:
            counts = df["split"].value_counts().to_dict()
            if counts != OFFICIAL_SPLITS[client]:
                errors.append(f"Official split altered! Expected {OFFICIAL_SPLITS[client]}, got {counts}")
                
        # Output results
        if errors:
            print(f"[{client}] FAIL:")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"[{client:10s}] PASS: {len(df):>5} rows. Columns, splits, totals, and isolation perfect.")

if __name__ == "__main__":
    run_audit()