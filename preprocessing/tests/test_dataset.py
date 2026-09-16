from pathlib import Path
import sys

import pandas as pd

from preprocessing.dataset import (
    build_manifest,
    apply_eyepacs_max_samples_cap,
    split_manifest,
)


# ============================================================
# CONFIG
# ============================================================

# CHANGE THIS to the folder containing:
#   APTOS-2019-Kaggle/
#   Disease Grading Images - IDRiD/
#   Messidor-2-Dataset/
#   EYEPACS - Kaggle/
#
# Based on your existing Windows layout, this is likely:
# C:\MajorProject - Datasets
DATA_ROOT = Path(r"C:\MajorProject - Datasets")

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_manifest_basic(manifest: pd.DataFrame, client: str) -> None:
    required = {
        "image_id",
        "raw_image_locator",
        "patient_id",
        "label",
        "source_split_hint",
    }

    missing = required - set(manifest.columns)
    assert not missing, f"{client}: missing columns: {missing}"

    assert not manifest.empty, f"{client}: manifest is empty"

    assert manifest["label"].between(0, 4).all(), (
        f"{client}: found label outside 0-4"
    )

    assert manifest["image_id"].notna().all(), (
        f"{client}: null image_id found"
    )

    assert manifest["patient_id"].notna().all(), (
        f"{client}: null patient_id found"
    )

    print(f"Rows       : {len(manifest):,}")
    print(f"Columns    : {list(manifest.columns)}")
    print("Class count:")
    print(manifest["label"].value_counts().sort_index().to_string())


def check_split_basic(
    manifest: pd.DataFrame,
    client: str,
) -> None:
    assert "split" in manifest.columns, f"{client}: split column missing"

    counts = manifest["split"].value_counts()

    print("\nSplit counts:")
    print(counts.to_string())

    print("\nSplit percentages:")
    print(
        (manifest["split"].value_counts(normalize=True) * 100)
        .round(2)
        .to_string()
    )

    assert set(manifest["split"].dropna().unique()).issubset(
        {"train", "val", "test"}
    ), f"{client}: invalid split value found"

    for split in ("train", "val", "test"):
        assert (manifest["split"] == split).any(), (
            f"{client}: {split} split is empty"
        )


def check_no_patient_leakage(
    manifest: pd.DataFrame,
    client: str,
) -> None:
    overlaps = {}

    for a, b in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        a_ids = set(
            manifest.loc[manifest["split"] == a, "patient_id"]
        )
        b_ids = set(
            manifest.loc[manifest["split"] == b, "patient_id"]
        )

        overlap = a_ids & b_ids
        overlaps[(a, b)] = overlap

        assert not overlap, (
            f"{client}: PATIENT LEAKAGE between {a} and {b}: "
            f"{sorted(list(overlap))[:10]}"
        )

    print("Patient leakage check: PASS")


# ============================================================
# TEST 1 — APTOS
# ============================================================

print_header("TEST 1 — APTOS")

try:
    aptos = build_manifest(
        "APTOS",
        data_root=DATA_ROOT,
    )

    check_manifest_basic(aptos, "APTOS")

    print("\nsource_split_hint:")
    print(aptos["source_split_hint"].value_counts().sort_index().to_string())

    aptos = split_manifest(
        "APTOS",
        aptos,
        random_state=RANDOM_STATE,
    )

    check_split_basic(aptos, "APTOS")

    expected = {
        "train": 2930,
        "val": 366,
        "test": 366,
    }

    actual = aptos["split"].value_counts().to_dict()

    assert actual == expected, (
        f"APTOS official split mismatch.\n"
        f"Expected: {expected}\n"
        f"Actual:   {actual}"
    )

    check_no_patient_leakage(aptos, "APTOS")

    print("APTOS: PASS")

except Exception as exc:
    print(f"\nAPTOS: FAIL\n{type(exc).__name__}: {exc}")
    raise


# ============================================================
# TEST 2 — IDRiD
# ============================================================

print_header("TEST 2 — IDRiD")

try:
    idrid = build_manifest(
        "IDRiD",
        data_root=DATA_ROOT,
    )

    check_manifest_basic(idrid, "IDRiD")

    print("\nsource_split_hint:")
    print(idrid["source_split_hint"].value_counts().sort_index().to_string())

    idrid = split_manifest(
        "IDRiD",
        idrid,
        random_state=RANDOM_STATE,
    )

    check_split_basic(idrid, "IDRiD")

    # Official test set MUST remain exactly 103.
    test_count = int((idrid["split"] == "test").sum())

    assert test_count == 103, (
        f"IDRiD official test size changed: expected 103, got {test_count}"
    )

    # Validation comes only from the original training portion.
    val_rows = idrid[idrid["split"] == "val"]
    assert (val_rows["source_split_hint"] == "train").all(), (
        "IDRiD: validation contains something from the original test set"
    )

    train_rows = idrid[idrid["split"] == "train"]
    assert (train_rows["source_split_hint"] == "train").all(), (
        "IDRiD: training contains something from the original test set"
    )

    test_rows = idrid[idrid["split"] == "test"]
    assert (test_rows["source_split_hint"] == "test").all(), (
        "IDRiD: test contains something outside official test set"
    )

    # Expected validation fraction = 20% of 413 = approximately 83.
    val_count = len(val_rows)

    assert 80 <= val_count <= 85, (
        f"IDRiD validation size unexpected: {val_count}"
    )

    print("IDRiD patient-leakage check: SKIPPED (no real patient ID available)")

    print("IDRiD official test boundary: PASS")
    print("IDRiD stratified validation carve-out: PASS")
    print("IDRiD: PASS")

except Exception as exc:
    print(f"\nIDRiD: FAIL\n{type(exc).__name__}: {exc}")
    raise


# ============================================================
# TEST 3 — MESSIDOR-2
# ============================================================

print_header("TEST 3 — MESSIDOR-2")

try:
    messidor = build_manifest(
        "Messidor-2",
        data_root=DATA_ROOT,
    )

    check_manifest_basic(messidor, "Messidor-2")

    # We expect 1744 gradable images after removing 4 ungradable rows.
    assert len(messidor) == 1744, (
        f"Messidor-2 row count unexpected: expected 1744, "
        f"got {len(messidor)}"
    )

    # There should be 870? patient/exam groups after the 4 ungradable
    # images are removed from 874 pairing rows.
    print(f"Unique patient/exam groups: {messidor['patient_id'].nunique():,}")

    messidor = split_manifest(
        "Messidor-2",
        messidor,
        random_state=RANDOM_STATE,
    )

    check_split_basic(messidor, "Messidor-2")
    check_no_patient_leakage(messidor, "Messidor-2")

    # Ensure an exam/patient group was never split.
    group_split_counts = (
        messidor.groupby("patient_id")["split"].nunique()
    )

    assert (group_split_counts == 1).all(), (
        "Messidor-2: at least one patient/exam group appears "
        "in multiple splits"
    )

    print("Messidor-2 grouped split: PASS")
    print("Messidor-2: PASS")

except Exception as exc:
    print(f"\nMessidor-2: FAIL\n{type(exc).__name__}: {exc}")
    raise


# ============================================================
# TEST 4 — EYEPACS MANIFEST
# ============================================================

print_header("TEST 4 — EYEPACS MANIFEST")

try:
    eyepacs = build_manifest(
        "EyePACS",
        data_root=DATA_ROOT,
    )

    check_manifest_basic(eyepacs, "EyePACS")

    assert len(eyepacs) == 88702, (
        f"EyePACS manifest size unexpected: expected 88702, "
        f"got {len(eyepacs)}"
    )

    # Ensure patient parsing worked.
    left_mask = eyepacs["image_id"].str.endswith("_left")
    right_mask = eyepacs["image_id"].str.endswith("_right")

    assert left_mask.any(), "EyePACS: no _left images detected"
    assert right_mask.any(), "EyePACS: no _right images detected"

    print(
        f"Unique EyePACS patients: "
        f"{eyepacs['patient_id'].nunique():,}"
    )

    # Verify that every image has the expected archive-style locator.
    assert eyepacs["raw_image_locator"].str.contains("::").all(), (
        "EyePACS: one or more raw_image_locator values do not use "
        "the expected archive locator format"
    )

    print("EyePACS manifest count: PASS")
    print("EyePACS patient parsing: PASS")
    print("EyePACS archive locator format: PASS")

except Exception as exc:
    print(f"\nEyePACS manifest: FAIL\n{type(exc).__name__}: {exc}")
    raise


# ============================================================
# TEST 5 — EYEPACS 6000 CAP
# ============================================================

print_header("TEST 5 — EYEPACS 6000-IMAGE WORKING-POOL CAP")

try:
    capped = apply_eyepacs_max_samples_cap(
        eyepacs,
        max_samples=6000,
        random_state=RANDOM_STATE,
    )

    print(f"Requested target : 6000")
    print(f"Actual rows      : {len(capped):,}")
    print(
        f"Unique patients  : "
        f"{capped['patient_id'].nunique():,}"
    )

    # Whole-patient integrity.
    rows_per_patient = capped.groupby("patient_id").size()

    assert set(rows_per_patient.unique()).issubset({1, 2}), (
        "EyePACS cap produced an unexpected number of rows per patient"
    )

    # Any patient in original EyePACS should have BOTH eyes selected
    # whenever both eyes exist in the original dataset.
    original_counts = eyepacs.groupby("patient_id").size()

    for patient_id, count in rows_per_patient.items():
        if original_counts.loc[patient_id] == 2:
            assert count == 2, (
                f"EyePACS cap split a two-eye patient: {patient_id}"
            )

    # Because groups are indivisible, actual count is allowed to be
    # slightly above 6000.
    assert len(capped) >= 6000, (
        f"EyePACS capped pool unexpectedly undershot target: {len(capped)}"
    )

    # But it should not overshoot by an absurd amount.
    assert len(capped) < 6100, (
        f"EyePACS cap overshot target unexpectedly: {len(capped)}"
    )

    # Check reproducibility.
    capped_again = apply_eyepacs_max_samples_cap(
        eyepacs,
        max_samples=6000,
        random_state=RANDOM_STATE,
    )

    assert set(capped["image_id"]) == set(capped_again["image_id"]), (
        "EyePACS cap is not reproducible with the same random_state"
    )

    print("Whole-patient cap integrity: PASS")
    print("Cap reproducibility: PASS")
    print("EyePACS 6000 cap: PASS")

except Exception as exc:
    print(f"\nEyePACS cap: FAIL\n{type(exc).__name__}: {exc}")
    raise


# ============================================================
# TEST 6 — EYEPACS FINAL GROUPED SPLIT
# ============================================================

print_header("TEST 6 — EYEPACS GROUPED 80/10/10 SPLIT")

try:
    eyepacs_capped_split = split_manifest(
        "EyePACS",
        capped,
        random_state=RANDOM_STATE,
    )

    check_split_basic(
        eyepacs_capped_split,
        "EyePACS",
    )

    check_no_patient_leakage(
        eyepacs_capped_split,
        "EyePACS",
    )

    group_split_counts = (
        eyepacs_capped_split.groupby("patient_id")["split"].nunique()
    )

    assert (group_split_counts == 1).all(), (
        "EyePACS: at least one patient appears in multiple splits"
    )

    print("EyePACS patient-group integrity: PASS")
    print("EyePACS grouped split: PASS")


except Exception as exc:
    print(f"\nEyePACS grouped split: FAIL\n{type(exc).__name__}: {exc}")
    raise


# ============================================================
# FINAL
# ============================================================

print_header("FINAL RESULT")

print("ALL REAL-DATASET DATASET.PY TESTS PASSED")
print()
print("No images were opened or preprocessed by this test.")
print("EyePACS ZIP archives were NOT extracted or reconstructed.")
print("This test validates manifests, labels, grouping, splitting,")
print("patient leakage, and the EyePACS working-pool cap only.")