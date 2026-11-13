"""
preprocessing/dataset.py

Per-client (never pooled) manifest building, splitting, PyTorch Dataset,
and DataLoader construction for the Federated DR Detection preprocessing
pipeline.

Absolute constraint (unchanged throughout this project): every function
here is scoped to ONE client at a time. There is no cross-client manifest,
no cross-client split, and no cross-client DataLoader. Calling code (the
future federated training loop) is expected to call build_manifest() /
split_manifest() / get_dataloaders() once per client and keep the four
results independent.

---------------------------------------------------------------------------
TWO-STAGE MANIFEST ARCHITECTURE (read before extending this file)
---------------------------------------------------------------------------
There are two distinct manifests in this pipeline, and this file only
produces the first one:

1. SOURCE manifest (this file's build_manifest() + split_manifest()):
   one row per RAW image, columns image_id / raw_image_locator /
   patient_id / label / source_split_hint / split. Points at raw,
   not-yet-offline-processed images. This is what build_dataset.py (File
   4, not yet implemented) will iterate over to run
   image_processing.run_offline_pipeline() on each image, cache the
   result to disk, and write the SECOND manifest.

2. FINAL manifest (written by build_dataset.py, read by this file's
   DRDataset / get_dataloaders): one row per successfully cached image,
   with the schema in FINAL_MANIFEST_COLUMNS below -- the cached image's
   path plus every field of run_offline_pipeline()'s meta_dict
   (blur_score, low_quality_flag, was_cropped, original_height,
   original_width), preserved unmodified per this project's explicit
   requirement that offline metadata survive into the manifest and stay
   diagnostic-only (never an auto-drop condition).

build_manifest()/split_manifest() do not touch pixels and do not require
the offline pipeline to have run yet. DRDataset/get_dataloaders assume it
HAS already run (they load cached, already-224x224 images) and know
nothing about raw source layouts.

---------------------------------------------------------------------------
FLAGGED IMPLEMENTATION DECISIONS (confirmed with the project owner, or
resolved defensively where the finalized doc + EDA materials were silent)
---------------------------------------------------------------------------
- Patient-level stratification label (confirmed decision): where a single
  label representing an entire patient/exam group is needed -- the
  EyePACS max_samples cap, and the label fed to the grouped-stratified
  splitter for Messidor-2/EyePACS -- this file uses the MAXIMUM DR grade
  across that patient's rows. Implemented as the single, standalone
  compute_patient_labels() function (not inlined ad hoc in multiple
  places) specifically so this policy can be swapped later without
  touching the splitting logic that consumes it.

- EyePACS/Messidor-2/APTOS/IDRiD split ratios and mechanism (confirmed):
  APTOS keeps its official 2,930/366/366 split untouched. IDRiD keeps its
  official 413/103 boundary untouched -- the 103 test images are only
  ever labeled 'test' and this file never re-splits them; a stratified
  ~20% validation subset (IDRID_VAL_FRACTION) is carved from the 413
  training images only, via plain (ungrouped) stratified sampling, since
  IDRiD has no real patient ID (Part 1, Step 9: "no patient ID available
  -- split at image level"). Messidor-2 and EyePACS both use an
  approximate 80/10/10 split via StratifiedGroupKFold (grouped by
  patient_id, stratified by the patient-level label above) -- chosen
  specifically over a plain GroupShuffleSplit because GroupShuffleSplit
  ignores label balance entirely.

- EyePACS max_samples cap (confirmed): applied as an explicit, separate
  step (apply_eyepacs_max_samples_cap()) between build_manifest() and
  split_manifest() -- NOT inside either of them -- because the cap
  controls working-pool size and is not itself a split. It selects whole
  patients (both eyes always move together) via the same per-stratum
  proportional-budget approach, so it stays class-representative without
  ever splitting a patient's eyes.

- IDRiD groundtruths CSV filenames: the exact filenames of IDRiD's two
  label CSVs were not confirmed in the materials available while writing
  this file (the EDA notebook's own path-discovery output was truncated
  before showing the literal filename). Rather than hardcode a guessed
  name, _build_manifest_idrid() reuses the EDA notebook's own proven
  approach: a case-insensitive substring search for a file containing
  ("idrid", "train"/"test", "label") under the Groundtruths folder,
  raising an informative error (listing what WAS found) if that search
  doesn't resolve to exactly one file.

- Messidor-2 CSV schemas: CONFIRMED directly against the real uploaded
  files (not guessed). messidor-2.csv is SEMICOLON-separated with columns
  `left` / `right` (one row per exam/patient, 874 rows); the well-known
  jpg/png case mismatch (gotcha #4) is real -- 690 of 1,748 filenames in
  this pairing file use `.JPG` where messidor_data.csv's `image_id` uses
  lowercase `.jpg` for the same files -- confirmed by direct inspection,
  so the join is done case-insensitively on the full filename. Because
  even a case-insensitive CSV join isn't a guarantee the exact string
  matches what's on disk, _build_manifest_messidor2() additionally builds
  a real on-disk index (by lowercased filename stem) from the IMAGES
  folder and resolves through THAT, exactly mirroring the approach the
  project's own EDA notebook already validated end-to-end (0
  missing/corrupt across all 1,748 images).

- EyePACS raw image locator (assumption, since raw bytes live inside zip
  archives that must be reconstructed -- see build_dataset.py's future
  Colab/Drive handling, not this file's job): build_manifest() does NOT
  open or reconstruct the train.zip / test.zip archives. It constructs a
  deterministic locator string `"{zip_half}::{zip_half}/{image_id}.jpeg"`
  (e.g. `"train::train/10003_left.jpeg"`) directly from which label CSV
  the row came from, matching the exact internal naming convention
  (`{split}/{image_id}.jpeg`) already confirmed by the project's own EDA
  notebook when it indexed the real archives. build_dataset.py must
  special-case any `raw_image_locator` containing "::" as
  "{zip_half}::{member_name}" and resolve it via zipfile against a
  (cached, resumable) reconstructed archive, rather than treating it as a
  plain filesystem path like the other three clients' locators.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from .augmentation import get_eval_transforms, get_train_transforms

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetError",
    "VALID_SPLITS",
    "FINAL_MANIFEST_COLUMNS",
    "compute_patient_labels",
    "build_manifest",
    "apply_eyepacs_max_samples_cap",
    "split_manifest",
    "DRDataset",
    "compute_class_weights",
    "DEFAULT_CLASS_WEIGHT_CAP",
    "make_augmentation_worker_init_fn",
    "get_dataloaders",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_SPLITS: Tuple[str, str, str] = ("train", "val", "test")
NUM_DR_GRADES: int = 5  # 0=No DR .. 4=Proliferative DR
# Absolute upper bound on any class loss weight (same for all 8 experiments).
DEFAULT_CLASS_WEIGHT_CAP: float = 10.0

# The FINAL manifest schema build_dataset.py (File 4) must produce and this
# file's DRDataset/get_dataloaders consume. `cached_image_path` points at
# the already offline-processed (224x224 RGB uint8) image on disk. The
# five metadata columns are passed through from run_offline_pipeline()'s
# meta_dict UNCHANGED -- diagnostic only, never used here to drop a row.
FINAL_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "image_id",
    "cached_image_path",
    "patient_id",
    "label",
    "split",
    "blur_score",
    "low_quality_flag",
    "was_cropped",
    "original_height",
    "original_width",
)

# Assumption (not specified in the finalized doc): "~15-20%" val fraction
# for IDRiD -- upper end chosen, trading a little more training data for a
# less noisy monitoring signal on the 25-image Mild class, per the design
# doc's own stated reasoning for that range.
IDRID_VAL_FRACTION: float = 0.20

# Confirmed decision: approximate 80/10/10 for Messidor-2 and EyePACS.
DEFAULT_VAL_FRACTION: float = 0.10
DEFAULT_TEST_FRACTION: float = 0.10

# Confirmed decision.
EYEPACS_DEFAULT_MAX_SAMPLES: Optional[int] = 6000

DEFAULT_SPLIT_RANDOM_STATE: int = 42


class DatasetError(ValueError):
    """Raised for any manifest-building, splitting, or dataset-construction failure in this module."""


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    """Raise DatasetError with an informative message if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetError(f"{context}: missing expected column(s) {missing}; found {list(df.columns)}")


def _find_file(directory: Path, must_contain: Sequence[str], *, max_depth: int = 3) -> Path:
    """Case-insensitive substring search for exactly one file under `directory`.

    Mirrors the EDA notebook's own proven find_file() approach (used there
    to resolve IDRiD's exact label filenames without hardcoding a guess).
    Used here for the same reason: this file was written without a
    confirmed literal filename for IDRiD's two Groundtruths CSVs.

    Args:
        directory: Root directory to search (recursively, depth-limited).
        must_contain: Substrings (case-insensitive) that must ALL appear
            in a candidate filename.
        max_depth: Maximum directory depth below `directory` to search.

    Returns:
        The single matching file's Path.

    Raises:
        DatasetError: if zero or more than one file matches.
    """
    if not directory.exists():
        raise DatasetError(f"Directory not found: {directory}")
    must_contain_lower = [s.lower() for s in must_contain]
    root_depth = len(directory.parts)
    matches = []
    all_files = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if len(path.parts) - root_depth > max_depth:
            continue
        all_files.append(path)
        if all(sub in path.name.lower() for sub in must_contain_lower):
            matches.append(path)

    if len(matches) == 0:
        names = sorted(p.name for p in all_files)
        raise DatasetError(
            f"No file matching all of {must_contain} found under {directory}. "
            f"Files present: {names[:20]}{' ...' if len(names) > 20 else ''}"
        )
    if len(matches) > 1:
        raise DatasetError(f"Multiple files matching {must_contain} found under {directory}: {matches}")
    return matches[0]


def compute_patient_labels(
    df: pd.DataFrame, *, patient_col: str = "patient_id", label_col: str = "label"
) -> pd.Series:
    """Collapse each patient's (possibly multiple) row-level labels into one stratification label.

    IMPLEMENTATION CHOICE (confirmed with the project owner, since the
    finalized design doc does not specify this): uses the MAXIMUM DR grade
    across a patient's rows, so a patient with one healthy eye and one
    Moderate eye is treated as a Moderate patient for stratification
    purposes -- keeping patients relevant to the project's stated
    Mild/Moderate priority from being diluted by a healthier fellow eye.

    This is the single place this policy is implemented; both
    apply_eyepacs_max_samples_cap() and the grouped splitter for
    Messidor-2/EyePACS call this function rather than reimplementing the
    collapse logic, so the policy can be changed here alone later.

    For clients with no real patient ID (APTOS, IDRiD), patient_id is set
    to the image's own id at build_manifest() time, so every "patient"
    group has exactly one row and this function is a no-op passthrough.

    Args:
        df: A manifest DataFrame with `patient_col` and `label_col`.
        patient_col: Column identifying the patient/exam group.
        label_col: Column holding each row's own DR grade.

    Returns:
        A Series aligned to df's index: every row belonging to the same
        patient gets that patient's collapsed (max) label.
    """
    return df.groupby(patient_col)[label_col].transform("max")


# ---------------------------------------------------------------------------
# Per-client manifest builders (raw images, NOT yet offline-processed)
# ---------------------------------------------------------------------------
def _build_manifest_aptos(data_root: Path) -> pd.DataFrame:
    """Build APTOS's raw-image manifest from its 3 official CSVs.

    Handles the confirmed doubled-folder Drive-upload gotcha
    (train_images/train_images/...). No real patient ID exists, so
    patient_id == image_id (Part 1, Step 9: image-level split).
    """
    root = data_root / "APTOS-2019-Kaggle"
    csv_to_split = {"train_1.csv": "train", "valid.csv": "val", "test.csv": "test"}
    folder_by_split = {"train": "train_images", "val": "val_images", "test": "test_images"}

    frames = []
    for csv_name, split_name in csv_to_split.items():
        csv_path = root / csv_name
        if not csv_path.exists():
            raise DatasetError(f"APTOS: label file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        _require_columns(df, ["id_code", "diagnosis"], context=str(csv_path))

        folder = folder_by_split[split_name]
        image_dir = root / folder / folder  # doubled folder, confirmed gotcha
        frames.append(
            pd.DataFrame(
                {
                    "image_id": df["id_code"].astype(str),
                    "raw_image_locator": [str(image_dir / f"{stem}.png") for stem in df["id_code"]],
                    "patient_id": df["id_code"].astype(str),
                    "label": df["diagnosis"].astype(int),
                    "source_split_hint": split_name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _build_manifest_idrid(data_root: Path) -> pd.DataFrame:
    """Build IDRiD's raw-image manifest from its Training/Testing Groundtruths CSVs.

    Label CSV filenames resolved defensively via _find_file() -- see
    module docstring. Handles the confirmed stray trailing empty columns
    in the training CSV.

    CONFIRMED BUG FIX: IDRiD's Training Set and Testing Set CSVs each
    independently number images starting from 1 (IDRiD_001-413 for train,
    IDRiD_001-103 for test) -- a real quirk of the dataset itself, not a
    naming coincidence. All 103 "Image name" values in the test CSV
    collide with train CSV entries, and 81 of those 103 collisions point
    to genuinely different images with genuinely different grades
    (confirmed directly against the real label CSVs). Using "Image name"
    as image_id unprefixed silently merged these into a single identity
    downstream (dict(zip(image_id, ...)) keeps only the last value for a
    repeated key), corrupting cache paths and split assignment. Both
    image_id and patient_id are therefore prefixed with `split_name`
    (e.g. "train_IDRiD_001" vs "test_IDRiD_001") so the two pools can
    never collide -- this preserves the existing "no real patient ID ->
    patient_id == image_id" design invariant, just fixes it at the root
    so both are actually unique. raw_image_locator is NOT prefixed: the
    on-disk files are still literally named "IDRiD_001.jpg" etc., already
    disambiguated by which of the two Training/Testing Set folders they
    live in, so the original (unprefixed) name is what must be used to
    locate them.
    """
    root = data_root / "Disease Grading Images - IDRiD"
    groundtruths_dir = root / "B. Disease Grading" / "2. Groundtruths"
    train_csv = _find_file(groundtruths_dir, must_contain=("idrid", "train", "label"))
    test_csv = _find_file(groundtruths_dir, must_contain=("idrid", "test", "label"))

    image_dirs = {
        "train": root / "B. Disease Grading" / "1. Original Images" / "a. Training Set",
        "test": root / "B. Disease Grading" / "1. Original Images" / "b. Testing Set",
    }

    frames = []
    for split_name, csv_path in (("train", train_csv), ("test", test_csv)):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        _require_columns(df, ["Image name", "Retinopathy grade"], context=str(csv_path))
        df = df[["Image name", "Retinopathy grade"]].dropna(subset=["Image name"])

        image_dir = image_dirs[split_name]
        prefixed_id = split_name + "_" + df["Image name"].astype(str)
        frames.append(
            pd.DataFrame(
                {
                    "image_id": prefixed_id,
                    "raw_image_locator": [str(image_dir / f"{name}.jpg") for name in df["Image name"]],
                    "patient_id": prefixed_id,
                    "label": df["Retinopathy grade"].astype(int),
                    "source_split_hint": split_name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _build_manifest_messidor2(data_root: Path) -> pd.DataFrame:
    """Build Messidor-2's raw-image manifest from the pairing + grades CSVs.

    Schemas CONFIRMED against the real files (see module docstring):
    messidor-2.csv is ';'-separated with columns left/right (874 exam
    rows); archive/messidor_data.csv has image_id/adjudicated_dr_grade/
    adjudicated_dme/adjudicated_gradable. Join is case-insensitive on the
    filename (confirmed real .JPG/.jpg case mismatch for 690 files). The 4
    ungradable images (adjudicated_gradable == 0) are excluded -- each has
    a gradable partner eye in the same exam (confirmed), so no patient is
    fully orphaned by this exclusion. Raw paths are resolved through a
    real on-disk filename index (by lowercased stem), not trusted from
    either CSV's literal string, mirroring the EDA notebook's proven
    approach.
    """
    root = data_root / "Messidor-2-Dataset"
    pairing_path = root / "messidor-2.csv"
    grades_path = root / "archive" / "messidor_data.csv"
    images_dir = root / "IMAGES"

    if not pairing_path.exists():
        raise DatasetError(f"Messidor-2: pairing file not found: {pairing_path}")
    if not grades_path.exists():
        raise DatasetError(f"Messidor-2: grades file not found: {grades_path}")
    if not images_dir.exists():
        raise DatasetError(f"Messidor-2: images folder not found: {images_dir}")

    pairing = pd.read_csv(pairing_path, sep=";")
    pairing.columns = [c.strip() for c in pairing.columns]
    _require_columns(pairing, ["left", "right"], context=str(pairing_path))

    grades = pd.read_csv(grades_path)
    _require_columns(
        grades, ["image_id", "adjudicated_dr_grade", "adjudicated_gradable"], context=str(grades_path)
    )
    grades_lookup: Dict[str, Tuple[float, int]] = {
        str(img).lower(): (grade, int(gradable))
        for img, grade, gradable in zip(
            grades["image_id"], grades["adjudicated_dr_grade"], grades["adjudicated_gradable"]
        )
    }

    on_disk_index = {p.stem.lower(): p for p in images_dir.iterdir() if p.is_file()}

    rows = []
    excluded_ungradable = 0
    for exam_idx, prow in pairing.iterrows():
        patient_id = f"messidor_exam_{exam_idx:04d}"
        for eye_filename in (prow["left"], prow["right"]):
            key = str(eye_filename).lower()
            lookup = grades_lookup.get(key)
            if lookup is None:
                raise DatasetError(
                    f"Messidor-2: '{eye_filename}' from the pairing file has no matching "
                    f"row in {grades_path.name} (case-insensitive lookup)."
                )
            grade, gradable = lookup
            if gradable != 1 or pd.isna(grade):
                excluded_ungradable += 1
                continue

            stem = Path(str(eye_filename)).stem.lower()
            resolved_path = on_disk_index.get(stem)
            if resolved_path is None:
                raise DatasetError(
                    f"Messidor-2: no on-disk file found for '{eye_filename}' "
                    f"(looked up by stem '{stem}') in {images_dir}"
                )
            rows.append(
                {
                    "image_id": str(eye_filename),
                    "raw_image_locator": str(resolved_path),
                    "patient_id": patient_id,
                    "label": int(grade),
                    "source_split_hint": None,
                }
            )
    logger.info("Messidor-2: excluded %d ungradable image(s) with no usable label.", excluded_ungradable)
    return pd.DataFrame(rows)


_EYEPACS_EYE_SUFFIX_RE = re.compile(r"_(left|right)$", flags=re.IGNORECASE)


def _parse_eyepacs_patient_id(image_id: str) -> str:
    """Strip the trailing _left/_right suffix from an EyePACS image_id to get its patient ID."""
    return _EYEPACS_EYE_SUFFIX_RE.sub("", image_id)


def _build_manifest_eyepacs(data_root: Path) -> pd.DataFrame:
    """Build EyePACS's raw-image manifest by concatenating its two label CSVs.

    Client-local concatenation ONLY (trainLabels.csv, 35,126 rows +
    retinopathy_solution.csv, 53,576 rows = 88,702) -- explicitly not a
    cross-client pool. Original Kaggle train/test boundary is discarded as
    a split decision, but which physical zip archive (train.zip vs
    test.zip) each image's bytes live in is an immutable fact tracked via
    raw_image_locator (see module docstring for its "{zip_half}::{member}"
    format -- build_dataset.py must special-case it). patient_id is parsed
    from the filename by stripping _left/_right.
    """
    root = data_root / "EYEPACS - Kaggle"
    # Confirmed doubled-folder layout from the EDA notebook.
    train_csv = root / "trainLabels.csv" / "trainLabels.csv"
    test_csv = root / "testLabels.csv" / "retinopathy_solution.csv"

    frames = []
    for zip_half, csv_path in (("train", train_csv), ("test", test_csv)):
        if not csv_path.exists():
            raise DatasetError(f"EyePACS: label file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        _require_columns(df, ["image", "level"], context=str(csv_path))
        image_ids = df["image"].astype(str)
        frames.append(
            pd.DataFrame(
                {
                    "image_id": image_ids,
                    "raw_image_locator": [f"{zip_half}::{zip_half}/{iid}.jpeg" for iid in image_ids],
                    "patient_id": [_parse_eyepacs_patient_id(iid) for iid in image_ids],
                    "label": df["level"].astype(int),
                    "source_split_hint": None,
                }
            )
        )
    manifest = pd.concat(frames, ignore_index=True)
    return manifest


_CLIENT_BUILDERS: Dict[str, Callable[[Path], pd.DataFrame]] = {
    "APTOS": _build_manifest_aptos,
    "IDRiD": _build_manifest_idrid,
    "Messidor-2": _build_manifest_messidor2,
    "EyePACS": _build_manifest_eyepacs,
}


def build_manifest(client_name: str, *, data_root: Path) -> pd.DataFrame:
    """Build one client's raw-image SOURCE manifest (no split column yet).

    Per Design Doc Part 3: "one of 4 client-specific builders, each
    handling that client's own filename/label quirks." Never pools across
    clients -- each call is fully independent.

    Args:
        client_name: One of "APTOS", "IDRiD", "Messidor-2", "EyePACS".
        data_root: Path to the directory containing the 4 raw dataset
            folders (e.g. the mounted "MajorProject - Datasets" root).
            Required, never hardcoded, per the project's platform-agnostic
            path-handling rule -- caller (build_dataset.py / a notebook /
            a test script) supplies it.

    Returns:
        A DataFrame with columns: image_id, raw_image_locator,
        patient_id, label, source_split_hint.

    Raises:
        DatasetError: for an unknown client_name, missing files, missing
            expected CSV columns, an empty result, an out-of-range label
            value, or a duplicate image_id within this client's manifest.
    """
    if client_name not in _CLIENT_BUILDERS:
        raise DatasetError(f"Unknown client_name {client_name!r}; expected one of {sorted(_CLIENT_BUILDERS)}")

    data_root = Path(data_root)
    manifest = _CLIENT_BUILDERS[client_name](data_root)

    if manifest.empty:
        raise DatasetError(f"build_manifest({client_name!r}) produced an empty manifest -- check data_root={data_root}")

    manifest["label"] = manifest["label"].astype(int)
    out_of_range = ~manifest["label"].between(0, NUM_DR_GRADES - 1)
    if out_of_range.any():
        bad = sorted(manifest.loc[out_of_range, "label"].unique().tolist())
        raise DatasetError(f"{client_name}: found out-of-range label value(s) {bad}; expected 0-{NUM_DR_GRADES - 1}")

    # Promoted here (was previously EyePACS-only, guarding just its two
    # concatenated label CSVs) after a confirmed real bug: IDRiD's
    # Training Set and Testing Set CSVs each independently number images
    # starting from 1, so unprefixed "Image name" values collided across
    # the two pools -- 103 collisions, 81 of them pointing to genuinely
    # different images with different grades. That specific case is now
    # fixed at the source in _build_manifest_idrid() (split-prefixed
    # image_id), but a duplicate image_id is a serious enough correctness
    # hazard (silent cache-path collisions, silent split-assignment loss
    # downstream in build_dataset.py's dict(zip(image_id, split))) that
    # this check now runs for every client automatically, not only
    # EyePACS -- this exact bug class should never again be something
    # only a manual smoke test catches.
    dup_mask = manifest["image_id"].duplicated()
    if dup_mask.any():
        example_ids = sorted(manifest.loc[dup_mask, "image_id"].unique().tolist())[:10]
        raise DatasetError(
            f"{client_name}: {int(dup_mask.sum())} duplicate image_id(s) found in the manifest "
            f"(e.g. {example_ids}) -- image_id must be unique within a client's manifest."
        )

    return manifest.reset_index(drop=True)


# ---------------------------------------------------------------------------
# EyePACS max_samples cap (explicit, separate step -- NOT itself a split)
# ---------------------------------------------------------------------------
def apply_eyepacs_max_samples_cap(
    manifest_df: pd.DataFrame,
    *,
    max_samples: Optional[int] = EYEPACS_DEFAULT_MAX_SAMPLES,
    random_state: int = DEFAULT_SPLIT_RANDOM_STATE,
) -> pd.DataFrame:
    """Cap EyePACS's working pool to (approximately) `max_samples` images.

    Confirmed requirements: class-stratified (via compute_patient_labels'
    max-grade policy), must never split a patient's two eyes, and is
    applied to the EyePACS-local manifest only -- this is not a
    train/val/test split, just a pre-split pool-size control, meant to be
    called between build_manifest("EyePACS", ...) and
    split_manifest("EyePACS", ...).

    Selection method: allocate an image budget to each patient-label
    stratum proportional to that stratum's share of total images, then
    greedily add whole patients (in random order) from that stratum until
    its budget is met -- keeping both of a selected patient's eyes
    together by construction (whole patients are the unit of selection).

    Because the budget check happens BEFORE adding each candidate patient,
    the patient that crosses the threshold is still added in full, so the
    actual selected count is typically slightly ABOVE `max_samples` (true
    for the default max_samples=6000, a small fraction of EyePACS's
    ~88,702 images). This only holds when `max_samples` is meaningfully
    smaller than the client's full size, though: for `max_samples` values
    close to the full size, a small stratum can run out of patients
    before reaching its proportional budget, giving a slight UNDERSHOOT
    for that stratum instead.

    Args:
        manifest_df: EyePACS's raw-image manifest (from build_manifest).
        max_samples: Target pool size. None means "use everything" (the
            manifest is returned unchanged). Defaults to 6000.
        random_state: Seed for reproducible patient selection.

    Returns:
        A copy of manifest_df, filtered to the selected patients' rows.

    Raises:
        ValueError: if max_samples is not None and <= 0.
    """
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be a positive integer or None, got {max_samples}")

    if max_samples is None or max_samples >= len(manifest_df):
        return manifest_df.copy()

    df = manifest_df.copy()
    df["_patient_stratify_label"] = compute_patient_labels(df)

    patients = (
        df.groupby("patient_id")
        .agg(_patient_stratify_label=("_patient_stratify_label", "first"), _image_count=("patient_id", "size"))
        .reset_index()
    )
    total_images = len(df)
    rng = np.random.default_rng(random_state)

    selected_ids = []
    for _, stratum in patients.groupby("_patient_stratify_label"):
        stratum_image_share = stratum["_image_count"].sum() / total_images
        stratum_budget = int(round(max_samples * stratum_image_share))
        shuffled = stratum.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        running = 0
        for _, prow in shuffled.iterrows():
            if running >= stratum_budget:
                break
            selected_ids.append(prow["patient_id"])
            running += int(prow["_image_count"])

    capped = df[df["patient_id"].isin(selected_ids)].drop(columns=["_patient_stratify_label"])
    capped = capped.reset_index(drop=True)
    logger.info(
        "EyePACS max_samples cap: requested=%s, selected=%d images across %d patients (from %d total).",
        max_samples,
        len(capped),
        len(selected_ids),
        total_images,
    )
    return capped


# ---------------------------------------------------------------------------
# Per-client split logic
# ---------------------------------------------------------------------------
def _split_aptos(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """APTOS: keep the official split untouched -- source_split_hint IS the split."""
    df = manifest_df.copy()
    hints = set(df["source_split_hint"].unique())
    if df["source_split_hint"].isna().any() or not hints.issubset(set(VALID_SPLITS)):
        raise DatasetError(
            f"APTOS: source_split_hint contains unexpected/missing values {hints}; "
            f"expected only {VALID_SPLITS}."
        )
    df["split"] = df["source_split_hint"]
    return df


def _split_idrid(manifest_df: pd.DataFrame, *, val_fraction: float, random_state: int) -> pd.DataFrame:
    """IDRiD: official 413/103 boundary untouched; stratified val carved from the 413 only.

    The 103 official test images are labeled 'test' directly from
    source_split_hint and never enter this function's split computation
    -- they can never leak into training, validation, or early stopping.
    """
    df = manifest_df.copy()
    hints = set(df["source_split_hint"].unique())
    if not hints.issubset({"train", "test"}):
        raise DatasetError(f"IDRiD: source_split_hint contains unexpected values {hints}; expected 'train'/'test'.")

    is_train = df["source_split_hint"] == "train"
    train_part = df[is_train]
    labels = train_part["label"].to_numpy()

    try:
        train_index_values, val_index_values = train_test_split(
            train_part.index.to_numpy(),
            test_size=val_fraction,
            stratify=labels,
            random_state=random_state,
        )
    except ValueError as exc:
        counts = train_part["label"].value_counts().sort_index().to_dict()
        raise DatasetError(
            f"IDRiD: stratified train/val split failed at val_fraction={val_fraction} "
            f"(likely a class with too few images in the 413-image training portion to "
            f"stratify -- e.g. the known-scarce Mild class). Per-class counts in the "
            f"training portion: {counts}. Original error: {exc}"
        ) from exc

    df["split"] = pd.NA
    df.loc[train_index_values, "split"] = "train"
    df.loc[val_index_values, "split"] = "val"
    df.loc[df["source_split_hint"] == "test", "split"] = "test"
    return df


def _safe_n_splits(labels: np.ndarray, groups: np.ndarray, requested_n_splits: int) -> int:
    """Cap n_splits at the rarest class's distinct-group count, so StratifiedGroupKFold never hard-fails.

    StratifiedGroupKFold requires n_splits no larger than the number of
    distinct groups within the smallest class. Rather than let that raise
    an opaque error on a rare class (a real risk here -- e.g. Messidor-2's
    smallest patient-level class has only a few dozen patients), this
    reduces n_splits and logs a warning, at the cost of a less exact
    80/10/10 ratio for that client -- consistent with the confirmed
    "approximate 80/10/10" framing.
    """
    label_to_groups: Dict[int, set] = {}
    for lbl, grp in zip(labels, groups):
        label_to_groups.setdefault(lbl, set()).add(grp)
    min_groups = min(len(g) for g in label_to_groups.values())
    n = max(2, min(requested_n_splits, min_groups))
    if n < requested_n_splits:
        logger.warning(
            "Reducing StratifiedGroupKFold n_splits from %d to %d because the rarest "
            "class only has %d distinct patient group(s); the resulting split ratio "
            "will be less exact for this client.",
            requested_n_splits,
            n,
            min_groups,
        )
    return n


def _stratified_group_split(
    df: pd.DataFrame,
    *,
    label_col: str,
    group_col: str,
    val_fraction: float,
    test_fraction: float,
    random_state: int,
    true_label_col: str = "label",
) -> pd.Series:
    """Approximate 80/10/10 (or given fractions) split via two chained StratifiedGroupKFold passes.

    First carves out ~test_fraction as 'test', then splits the remainder
    into 'train'/'val' so ~val_fraction of the ORIGINAL total ends up in
    'val'. A group (patient) never appears in more than one output split,
    by construction of StratifiedGroupKFold.

    Note on exactness: StratifiedGroupKFold optimizes for label balance
    across folds, not for hitting an exact size percentage -- and since
    splitting happens at the whole-patient level, an exact 80/10/10 was
    never actually guaranteed here regardless. The resulting split sizes
    are logged (not just assumed) so any drift from the target ratio is
    visible rather than silently accepted.

    Args:
        df: Manifest with `label_col`, `group_col`, and (for logging)
            `true_label_col`.
        label_col: Per-row stratification label (patient-level, per
            compute_patient_labels()) -- what StratifiedGroupKFold
            actually balances across folds.
        group_col: Column identifying the group (patient_id).
        val_fraction: Target fraction of the total in 'val'.
        test_fraction: Target fraction of the total in 'test'.
        random_state: Seed for reproducibility.
        true_label_col: Column holding each row's own actual DR grade
            (0-4), used only for the post-split diagnostic logging below
            -- distinct from `label_col`, which may be a collapsed
            patient-level proxy. This is deliberately checked separately
            from `label_col`: the patient-level label is what the
            algorithm balances, but the per-image grade distribution is
            what actually matters for training, and the two are not
            guaranteed identical (a patient's collapsed max-grade label
            can differ from either of their individual eyes' own grade).
            Skipped if not present in `df`.

    Returns:
        A Series aligned to df's index with values in VALID_SPLITS.

    Raises:
        DatasetError: if StratifiedGroupKFold still fails even after
            _safe_n_splits' adaptive reduction (e.g. a class with fewer
            than 2 distinct patients).
    """
    labels = df[label_col].to_numpy()
    groups = df[group_col].to_numpy()
    x_placeholder = np.zeros(len(df))

    n_test_splits = _safe_n_splits(labels, groups, max(2, round(1.0 / test_fraction)))
    sgkf_test = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=random_state)
    try:
        trainval_pos, test_pos = next(sgkf_test.split(x_placeholder, labels, groups))
    except ValueError as exc:
        raise DatasetError(
            f"Grouped stratified split failed while carving out the test split "
            f"(n_splits={n_test_splits}). This usually means a class has fewer than 2 "
            f"distinct patient groups. Original error: {exc}"
        ) from exc

    trainval_labels = labels[trainval_pos]
    trainval_groups = groups[trainval_pos]
    trainval_x = x_placeholder[trainval_pos]

    remaining_fraction = 1.0 - test_fraction
    val_fraction_of_remaining = val_fraction / remaining_fraction
    n_val_splits = _safe_n_splits(
        trainval_labels, trainval_groups, max(2, round(1.0 / val_fraction_of_remaining))
    )
    sgkf_val = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=random_state)
    try:
        train_local_pos, val_local_pos = next(sgkf_val.split(trainval_x, trainval_labels, trainval_groups))
    except ValueError as exc:
        raise DatasetError(
            f"Grouped stratified split failed while carving out the validation split "
            f"(n_splits={n_val_splits}). Original error: {exc}"
        ) from exc

    val_pos = trainval_pos[val_local_pos]
    test_pos_arr = test_pos  # already positions into the original df

    split = pd.Series("train", index=df.index, dtype=object, name="split")
    split.iloc[val_pos] = "val"
    split.iloc[test_pos_arr] = "test"

    split_sizes = split.value_counts()
    split_pcts = (split.value_counts(normalize=True) * 100).round(1)
    logger.info(
        "Grouped stratified split sizes: %s (%s)",
        split_sizes.to_dict(),
        {k: f"{v}%" for k, v in split_pcts.to_dict().items()},
    )

    # Size proportions are the less important thing to verify here -- the
    # whole reason StratifiedGroupKFold was chosen over plain
    # GroupShuffleSplit was to preserve per-grade label balance across
    # splits. Report the ACTUAL per-image grade (0-4) distribution per
    # split, not just the patient-level proxy label_col was stratified
    # on, so that goal is checked directly rather than assumed.
    if true_label_col in df.columns:
        grade_counts = pd.crosstab(split, df[true_label_col])
        grade_pcts = pd.crosstab(split, df[true_label_col], normalize="index") * 100
        logger.info(
            "Grouped stratified split per-grade (0-%d) image counts:\n%s",
            NUM_DR_GRADES - 1,
            grade_counts.to_string(),
        )
        logger.info(
            "Grouped stratified split per-grade (0-%d) distribution (%% within each split):\n%s",
            NUM_DR_GRADES - 1,
            grade_pcts.round(1).to_string(),
        )
    return split


def _split_grouped(
    manifest_df: pd.DataFrame, *, val_fraction: float, test_fraction: float, random_state: int
) -> pd.DataFrame:
    """Shared Messidor-2 / EyePACS split: patient-grouped + patient-label-stratified, ~80/10/10."""
    df = manifest_df.copy()
    df["_patient_stratify_label"] = compute_patient_labels(df)
    split_series = _stratified_group_split(
        df,
        label_col="_patient_stratify_label",
        group_col="patient_id",
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    df["split"] = split_series
    return df.drop(columns=["_patient_stratify_label"])


def split_manifest(
    client_name: str,
    manifest_df: pd.DataFrame,
    *,
    random_state: int = DEFAULT_SPLIT_RANDOM_STATE,
    idrid_val_fraction: float = IDRID_VAL_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> pd.DataFrame:
    """Add a 'split' column to a client's manifest, using that client's finalized split logic.

    See module docstring / this turn's confirmed decisions for exactly
    what each client does. EyePACS's max_samples cap is NOT applied
    here -- call apply_eyepacs_max_samples_cap() on the manifest first if
    a capped pool is wanted, then pass the result to this function.

    Args:
        client_name: One of "APTOS", "IDRiD", "Messidor-2", "EyePACS".
        manifest_df: That client's manifest, as returned by
            build_manifest() (optionally passed through
            apply_eyepacs_max_samples_cap() for EyePACS).
        random_state: Seed for reproducibility (IDRiD's stratified split
            and the grouped splitter for Messidor-2/EyePACS).
        idrid_val_fraction: Fraction of IDRiD's 413 official training
            images to carve into 'val'.
        val_fraction: Target 'val' fraction for Messidor-2/EyePACS.
        test_fraction: Target 'test' fraction for Messidor-2/EyePACS.

    Returns:
        manifest_df with an added 'split' column (values in VALID_SPLITS).

    Raises:
        DatasetError: for an unknown client_name or any split failure.
    """
    if client_name == "APTOS":
        return _split_aptos(manifest_df)
    if client_name == "IDRiD":
        return _split_idrid(manifest_df, val_fraction=idrid_val_fraction, random_state=random_state)
    if client_name in ("Messidor-2", "EyePACS"):
        return _split_grouped(
            manifest_df, val_fraction=val_fraction, test_fraction=test_fraction, random_state=random_state
        )
    raise DatasetError(f"Unknown client_name {client_name!r}; expected one of {sorted(_CLIENT_BUILDERS)}")


# ---------------------------------------------------------------------------
# DRDataset + get_dataloaders (operate on the FINAL, post-offline manifest)
# ---------------------------------------------------------------------------
class DRDataset(Dataset):
    """A single client's single split, reading already offline-processed (cached) images.

    Expects `manifest_df` to already have a 'split' column and a
    'cached_image_path' column pointing at 224x224 RGB uint8 images on
    disk (i.e. this is the FINAL manifest written by build_dataset.py,
    not the SOURCE manifest from build_manifest()/split_manifest() in
    this file).
    """

    def __init__(self, manifest_df: pd.DataFrame, split: str, transform: Callable) -> None:
        """Filter `manifest_df` to `split` and store it for __getitem__.

        Args:
            manifest_df: A client's FINAL manifest (see FINAL_MANIFEST_COLUMNS).
            split: One of VALID_SPLITS.
            transform: An Albumentations-style callable applied as
                `transform(image=np.ndarray)["image"]` (e.g.
                augmentation.get_train_transforms() or
                augmentation.get_eval_transforms()).

        Raises:
            DatasetError: if `split` is invalid, required columns are
                missing, or no rows match `split`.
        """
        if split not in VALID_SPLITS:
            raise DatasetError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        _require_columns(manifest_df, ("cached_image_path", "label", "split"), context="DRDataset manifest")

        self.split = split
        self.transform = transform
        self.data = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        if self.data.empty:
            raise DatasetError(f"DRDataset: no rows found for split={split!r} in the given manifest")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.data.iloc[idx]
        image_path = Path(row["cached_image_path"])
        if not image_path.exists():
            raise FileNotFoundError(
                f"DRDataset: cached image not found for image_id={row.get('image_id', '?')}: "
                f"{image_path}. Was build_dataset.py's offline pass run to completion?"
            )
        image = np.array(Image.open(image_path).convert("RGB"))
        label = int(row["label"])
        transformed = self.transform(image=image)["image"]
        return transformed, label


def compute_class_weights(
    train_labels: np.ndarray,
    num_classes: int = NUM_DR_GRADES,
    *,
    cap: float = DEFAULT_CLASS_WEIGHT_CAP,
) -> np.ndarray:
    """Compute capped per-class loss weights from a client's own train-split labels.

    SUPERSEDES the old WeightedRandomSampler-based imbalance handling
    that used to live in get_dataloaders() (see that function's
    docstring for why). Per training-phase-decisions.md, Section 2:
    class imbalance is now handled via a class-weighted loss, identical
    across all 8 experiments, computed by the CALLER (training code)
    from this client's own train-split labels and applied in the loss
    function -- never in the DataLoader.

    Basis: w_c = N / (K * n_c), K = num_classes, which has expected weight
    1 under the client's own label distribution p_c = n_c / N when every
    class is present (keeps LR/max_grad_norm comparable to an unweighted
    run). The weights are then limited to the ABSOLUTE value `cap`:

    * Not cap-then-renormalize (that pushes capped classes, and sometimes
      other classes, back above the cap). Instead one scale s is solved by
      bisection so that sum_c p_c * min(s * w_c, cap) == 1 over the
      PRESENT classes. s >= 1, the result has mean 1 (to ~1e-12) and
      max <= cap, and rarer classes never get a smaller weight.
    *Classes with zero samples (p_c = 0): weight = 0 (the existing contract), excluded from the mean (logged).
    *No inf/NaN anywhere.
    * If the cap never engages, s = K / K_present (exactly 1 when all
      classes are present) and the output is the plain formula
      N / (K_present * n_c) over the present classes, identical to the
      uncapped version of this function.

    Args:
        train_labels: 1-D array-like of this client's train-split
            integer DR grades (e.g. a DRDataset's filtered
            `.data["label"]`, or `train_dataset.data["label"].to_numpy()`
            from get_dataloaders()).
        num_classes: Number of DR grade classes. Defaults to
            NUM_DR_GRADES (5).
        cap: Absolute maximum weight, finite and >= 1. Defaults to
            DEFAULT_CLASS_WEIGHT_CAP (10.0). cap == 1 gives every PRESENT class weight 1 (absent classes stay 0).

    Returns:
        A float64 array of shape (num_classes,): per-class loss weight,
        directly indexable by a label (weights[label]).

    Raises:
        DatasetError: if train_labels is not 1-D, is empty, is not
            integer-valued, or contains a value outside
            [0, num_classes); or if cap is not a finite number >= 1.
    """
    labels = np.asarray(train_labels)
    if labels.ndim != 1:
        raise DatasetError(f"compute_class_weights: train_labels must be 1-D, got shape {labels.shape}")
    if labels.size == 0:
        raise DatasetError("compute_class_weights: train_labels is empty")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.mod(labels, 1) == 0):
            raise DatasetError(
                f"compute_class_weights: train_labels must be integer-valued, got dtype {labels.dtype}"
            )
        labels = labels.astype(np.int64)
    if labels.min() < 0 or labels.max() >= num_classes:
        raise DatasetError(
            f"compute_class_weights: train_labels must be in [0, {num_classes}), got range "
            f"[{int(labels.min())}, {int(labels.max())}]"
        )
    if (
        isinstance(cap, (bool, np.bool_))
        or not isinstance(cap, (int, float, np.integer, np.floating))
        or not np.isfinite(cap)
        or cap < 1.0
    ):
        raise DatasetError(f"compute_class_weights: cap must be a finite number >= 1, got {cap!r}")
    cap = float(cap)

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    n_total = float(labels.size)
    present = counts > 0
    k_present = int(present.sum())

    zero_classes = np.where(~present)[0]
    if zero_classes.size:
           logger.warning(
               "compute_class_weights: class(es) %s have zero training samples; their weight is set to 0 "
               "and they are excluded from the mean.",
               zero_classes.tolist(),
           )

    p = counts[present] / n_total
    raw = n_total / (num_classes * counts[present])  # w_c = N / (K * n_c)
    scale = num_classes / k_present  # uncapped mean == 1; exactly 1.0 when all classes are present

    weights = np.zeros(num_classes, dtype=np.float64)
    if scale * float(raw.max()) <= cap:
        weights[present] = n_total / (k_present * counts[present])  # cap never engages
        return weights

    # Cap engages: f(s) = sum_c p_c * min(s * w_c, cap) is nondecreasing, f(scale) < 1 and
    # f(cap / min w) = cap >= 1, so the root is bracketed.
    lo, hi = scale, cap / float(raw.min())
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float((p * np.minimum(mid * raw, cap)).sum()) < 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    weights[present] = np.minimum(hi * raw, cap)
    logger.info(
        "compute_class_weights: cap %g engaged for class(es) %s (raw weights %s); scale s=%.6f",
        cap, np.where(present)[0][(hi * raw) >= cap].tolist(),
        [round(float(w), 4) for w in raw], hi,
    )
    return weights


# ---------------------------------------------------------------------------
# Augmentation seeding (TRAIN loader only)
# ---------------------------------------------------------------------------
def _seed_augmentation_transform(transform: Callable, seed: int) -> None:
    """Re-seed every random draw inside an Albumentations Compose from one integer seed. In place.

    Deliberately uses Compose.set_random_state() with ONE shared numpy
    Generator and ONE shared python random.Random built from `seed`, NOT
    Compose.set_random_seed(seed). The latter gives every child transform
    its own generator seeded with the SAME value, so children that share a
    probability (e.g. HorizontalFlip and VerticalFlip, both p=0.5) draw
    identical sequences and fire in lockstep -- verified on
    albumentations==2.0.8: in 400 calls a flip pair never flipped on only
    one axis, versus roughly half the calls when unseeded. With shared
    generators the children consume a single stream in sequence, so they
    stay independent of one another while the whole pipeline remains
    reproducible from `seed`.

    Args:
        transform: An albumentations Compose (anything exposing
            set_random_state(numpy_generator, python_random)).
        seed: Non-negative integer.
    """
    transform.set_random_state(np.random.default_rng(seed), random.Random(seed))


def _augmentation_worker_init(worker_id: int) -> None:
    """DataLoader worker_init_fn body; see make_augmentation_worker_init_fn().

    Defined at module level (not as a closure) so it is picklable, which
    DataLoader needs for multiprocessing start methods that spawn rather
    than fork worker processes (e.g. Windows).
    """
    info = torch.utils.data.get_worker_info()
    _seed_augmentation_transform(info.dataset.transform, int(info.seed) % 2**32)


def make_augmentation_worker_init_fn() -> Callable[[int], None]:
    """Return a DataLoader `worker_init_fn` that gives each worker its own augmentation RNG state.

    Albumentations transforms own their numpy / python random generators
    (entropy-seeded unless seeded, independent of torch/numpy/python global
    seeds), and each DataLoader worker receives a COPY of the dataset and
    therefore of those generators -- so without this hook workers duplicate
    each other's augmentation sequence and every epoch replays it.

    The returned function runs inside each worker as
    worker_init(worker_id): it reads torch.utils.data.get_worker_info() and
    reseeds `info.dataset.transform` from int(info.seed) % 2**32.
    info.seed = base_seed + worker_id, where base_seed is drawn from the
    loader's own generator each time an epoch's iterator is created, so the
    seed is unique per worker and per epoch and reproducible from a seeded
    generator.

    Requirements (all hold for get_dataloaders()'s train loader):
        - `info.dataset` is the DRDataset itself (no Subset or other
          wrapper), exposing the transform as `.transform`.
        - persistent_workers is False. With persistent workers,
          worker_init would run once per worker rather than once per
          epoch, and the "unique per epoch" property would be lost.
        - The transform has set_random_state() (every Albumentations Compose does).

    Opacus's DPDataLoader is expected to carry worker_init_fn over when it
    rebuilds a DP loader, so no DP-specific code is added; this has not been
    verified by this repository's tests.

    Returns:
        A picklable function worker_init(worker_id) suitable for
        DataLoader(worker_init_fn=...).
    """
    return _augmentation_worker_init


def get_dataloaders(
    client_name: str,
    *,
    manifest_path: Path,
    batch_size: int = 32,
    num_workers: int = 2,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    seed: int = DEFAULT_SPLIT_RANDOM_STATE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build one client's independent train/val/test DataLoaders.

    SUPERSEDED for loader construction (training-phase-decisions.md,
    Section 2): the original preprocessing requirement here was "Part 1,
    Step 7: WeightedRandomSampler weight = 1/class_count, train only".
    That is no longer what this function does -- validated against
    Opacus 1.6.0 source, DPDataLoader.from_data_loader rebuilds a DP
    training loader from its .dataset using Opacus's own uniform Poisson
    sampling and discards whatever sampler the loader had, so a sampler
    here would have been silently dropped in the 5 DP experiments (4-8)
    while staying active in the 3 non-DP ones (1-3), making the 8
    experiments non-comparable. Class imbalance is now handled by a
    class-weighted loss instead, identical across all 8 experiments --
    see compute_class_weights(), called by the training code on this
    same client's train-split labels and applied in the loss function,
    not here. Manifests, splits, the offline cache, and augmentation are
    unaffected by this change.

    Train loader now uses a plain shuffle (shuffle=True) with its own
    seeded generator; val/test loaders are unchanged -- a fixed,
    non-shuffled order.

    AUGMENTATION SEEDING (train loader only): Albumentations transforms own
    their random generators (entropy-seeded, independent of global seeds)
    and every DataLoader worker gets a copy, so unseeded workers duplicate
    each other's augmentation sequence and each epoch replays it. The train
    loader is therefore built with
    worker_init_fn=make_augmentation_worker_init_fn(), which reseeds the
    transform inside each worker from torch's per-worker seed
    (base_seed + worker_id; base_seed is drawn from the loader's seeded
    generator for every epoch's iterator) -- unique per worker and per
    epoch, reproducible from `seed`. Validation/test loaders, eval
    transforms, the signature and the return value are unchanged. This
    relies on persistent_workers staying False (this function never sets
    it): with persistent workers worker_init would run once per worker,
    not once per epoch.

    With num_workers == 0 no worker_init_fn is ever invoked, so the train
    transform is instead seeded once, here, from `seed`. In that mode the
    transform's generator state is not part of any checkpoint, so exact
    augmentation resume holds only with num_workers > 0.

    SIDE EFFECT: `train_transform` (the caller's object, or the default one
    built here) has its random state replaced in place when num_workers == 0.
    Callers that share one transform object across clients or rounds should
    pass a per-client / per-round `seed` (or separate transform objects) so
    those streams do not coincide.

    DP NOTE: if this loader will be wrapped by Opacus for a DP experiment
    (4-8), `batch_size` MUST be set to that experiment's configured
    dp_logical_batch_size (locked at 64), not this function's default of
    32 -- Opacus derives sample_rate = 1/len(loader) from however many
    batches this loader actually produces, so the wrong batch_size here
    silently miscalibrates that client's privacy accounting.

    Args:
        client_name: One of "APTOS", "IDRiD", "Messidor-2", "EyePACS"
            (used only for error messages here -- the manifest itself,
            not this argument, determines which images are loaded).
        manifest_path: Path to this client's FINAL manifest CSV (written
            by build_dataset.py), with columns FINAL_MANIFEST_COLUMNS.
        batch_size: Batch size for all 3 loaders. See DP NOTE above if
            building a DP experiment's loader.
        num_workers: DataLoader worker count for all 3 loaders. See
            AUGMENTATION SEEDING above for what 0 vs > 0 means for the
            train transform.
        train_transform: Defaults to augmentation.get_train_transforms().
            Must expose set_random_state() (every albumentations Compose
            does); a plain callable is rejected with DatasetError.
        eval_transform: Defaults to augmentation.get_eval_transforms(),
            used for both val and test.
        seed: Seeds the train loader's shuffle generator and, when
            num_workers == 0, the train transform's random state (as
            int(seed) % 2**32).

    Returns:
        (train_loader, val_loader, test_loader).

    Raises:
        DatasetError: if the manifest file is missing, missing required
            columns, any split has zero rows, or train_transform has no
            set_random_state() method.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise DatasetError(
            f"{client_name}: final manifest not found at {manifest_path}. "
            f"Run build_dataset.py for this client first."
        )
    manifest_df = pd.read_csv(manifest_path)
    _require_columns(manifest_df, FINAL_MANIFEST_COLUMNS, context=f"{client_name} final manifest at {manifest_path}")

    if train_transform is None:
        train_transform = get_train_transforms()
    if eval_transform is None:
        eval_transform = get_eval_transforms()

    if not callable(getattr(train_transform, "set_random_state", None)):
        raise DatasetError(
            f"{client_name}: train_transform ({type(train_transform).__name__}) has no "
            f"set_random_state() method; it must be an albumentations Compose (or expose the "
            f"same method) so the augmentation RNG can be seeded per DataLoader worker."
        )

    train_dataset = DRDataset(manifest_df, split="train", transform=train_transform)
    val_dataset = DRDataset(manifest_df, split="val", transform=eval_transform)
    test_dataset = DRDataset(manifest_df, split="test", transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)
    if num_workers == 0:
        # worker_init_fn is not invoked without worker processes, so seed the
        # transform once here instead (see AUGMENTATION SEEDING in the docstring).
        _seed_augmentation_transform(train_transform, int(seed) % 2**32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
        worker_init_fn=make_augmentation_worker_init_fn(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return train_loader, val_loader, test_loader