"""
preprocessing/build_dataset.py

Offline preprocessing driver: consumes a client's SOURCE manifest (from
dataset.py's build_manifest()/split_manifest()), runs
image_processing.run_offline_pipeline() on each raw image, caches the
224x224 result to disk, and incrementally writes the FINAL manifest
(dataset.FINAL_MANIFEST_COLUMNS) that dataset.py's DRDataset /
get_dataloaders() consume at train time.

One client per invocation. Never reads or writes anything for more than
one client at a time -- consistent with every other file in this
pipeline, there is no cross-client state here.

---------------------------------------------------------------------------
FLAGGED DECISIONS (confirmed with the project owner)
---------------------------------------------------------------------------
- Cached image format: PNG (lossless), not JPEG. Storage cost is
  negligible at 224x224 either way (~15-20GB total against a 5TB Drive
  budget); Drive-FUSE's per-file-open latency dominates over raw byte
  size at this scale anyway (training reads should be staged to local
  disk each session regardless of format); JPEG's default chroma
  subsampling risks exactly the lesion-color fidelity this project
  already treats carefully elsewhere (see augmentation.py's narrowed
  hue/sat range); and PNG->JPEG is a cheap, reversible downgrade later if
  a *measured* throughput problem ever justifies it, while the reverse
  isn't possible once detail is discarded.

- Resumability: a row is considered done if, and ONLY if, its image_id is
  already present in the final manifest CSV -- checked once at startup,
  not per-file-existence-on-disk (a manifest row is the single source of
  truth for "done"; an orphaned cached PNG with no manifest row, e.g. from
  a process that died between saving the image and appending its manifest
  row, is harmless and gets silently regenerated/overwritten). Already-
  recorded rows are NEVER reprocessed or rewritten on a resumed run.

- Split-assignment stability: this file computes a fresh split for every
  row currently in the (possibly max_samples-capped) source manifest via
  dataset.split_manifest(), but that fresh computation is used ONLY to
  assign a split to rows NOT already in the final manifest. This is a
  structural guarantee, not a lookup-and-override: because already-
  recorded rows are skipped outright (see above) and the manifest CSV is
  only ever appended to, never rewritten, it is not possible for a row's
  recorded split to change across resumed runs -- even if a fresh
  split_manifest() call would compute something different for that row
  (e.g. after a library version bump changes StratifiedGroupKFold's exact
  fold assignment, or filesystem directory-listing order shifts
  Messidor-2/EyePACS's groupby/sample order). This matters because
  silently reassigning an already-cached image from train to test (or
  vice versa) after training has already used it would be a real
  correctness bug, not a cosmetic one.

- EyePACS zip handling: raw bytes live inside Kaggle's split-zip parts
  (train.zip.001-005, test.zip.001-007). This file reconstructs each half
  locally via simple binary concatenation -- exactly the approach the
  project's own EDA notebook already validated -- reads members directly
  from the reconstructed archive via `zipfile` (no full extraction to
  individual files), and deletes the reconstructed archive once that
  half's pending rows are done. Rows are processed in two whole-half
  passes (all pending 'train' rows, then all pending 'test' rows), never
  interleaved, so each archive is reconstructed at most once per run. If
  a complete reconstructed archive from an interrupted prior run is still
  sitting in the work directory (size matches the sum of its source
  parts), reconstruction is skipped entirely.

- No multiprocessing in this version: sequential processing, one image at
  a time. Simplest to reason about correctness for a resumable pipeline
  (no risk of two workers racing on the same manifest-append or the same
  reconstructed-zip lifecycle). Revisit if EyePACS's full 88,702-image
  run proves too slow for the available Colab/GPU session budget.

- Offline pipeline parameters (circle-crop threshold, blur-flag
  threshold, CLAHE clip limit, etc.) are NOT exposed as CLI flags here --
  this file calls image_processing.run_offline_pipeline() with its
  defaults, unmodified. Those parameters were already finalized/tunable
  in image_processing.py itself (File 1); this file's job is to drive
  that pipeline across a client's images, not to re-expose or re-decide
  its parameters.
---------------------------------------------------------------------------

CLI usage:
    python -m preprocessing.build_dataset --client APTOS --data-root "/content/drive/MyDrive/MajorProject - Datasets"
    python -m preprocessing.build_dataset --client EyePACS --data-root <root> --max-samples 6000
    python -m preprocessing.build_dataset --client EyePACS --data-root <root> --max-samples none

Programmatic usage (e.g. a Colab cell):
    from preprocessing.build_dataset import main
    stats = main(["--client", "APTOS", "--data-root", DATA_ROOT])
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .dataset import (
    EYEPACS_DEFAULT_MAX_SAMPLES,
    FINAL_MANIFEST_COLUMNS,
    DEFAULT_SPLIT_RANDOM_STATE,
    apply_eyepacs_max_samples_cap,
    build_manifest,
    split_manifest,
)
from .image_processing import ImageProcessingError, run_offline_pipeline

logger = logging.getLogger(__name__)

__all__ = ["BuildDatasetError", "BuildStats", "run_offline_preprocessing", "build_arg_parser", "main"]


class BuildDatasetError(ValueError):
    """Raised for any build_dataset.py-level failure (distinct from dataset.py's DatasetError)."""


_PROGRESS_LOG_INTERVAL = 500
_EYEPACS_ZIP_HALVES: Tuple[str, str] = ("train", "test")


@dataclass
class BuildStats:
    """Summary of one run_offline_preprocessing() call, for logging and Colab-cell inspection."""

    client_name: str
    total_rows: int
    already_cached: int
    newly_processed: int
    failed: int
    manifest_path: Path

    def __str__(self) -> str:
        return (
            f"client={self.client_name} total={self.total_rows} "
            f"already_cached={self.already_cached} newly_processed={self.newly_processed} "
            f"failed={self.failed} manifest={self.manifest_path}"
        )


def _require_columns(df: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise BuildDatasetError(f"{context}: missing expected column(s) {missing}; found {list(df.columns)}")


# ---------------------------------------------------------------------------
# Cache path / atomic write helpers
# ---------------------------------------------------------------------------
def _cache_path_for(cache_root: Path, split: str, image_id: str) -> Path:
    """Deterministic cache path for one image: {cache_root}/{split}/{stem}.png.

    Uses Path(image_id).stem rather than image_id directly, since some
    clients' image_id already embeds an extension (e.g. Messidor-2's
    "20051109_57451_0400_PP.png") -- stem strips it so we never end up
    with a doubled ".png.png".
    """
    safe_stem = Path(str(image_id)).stem
    return cache_root / split / f"{safe_stem}.png"


def _atomic_save_png(image: np.ndarray, dest_path: Path) -> None:
    """Save `image` as PNG at `dest_path`, atomically (write to a temp file, then os.replace).

    os.replace() is atomic on both POSIX and Windows, so a process killed
    mid-write can never leave a corrupt file sitting at `dest_path` --
    either the old version (if any) remains, or the fully-written new one
    does; nothing in between is ever observable.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    Image.fromarray(image).save(tmp_path, format="PNG")
    os.replace(tmp_path, dest_path)


def _write_manifest_row(writer: "csv.DictWriter[str]", manifest_file, row: dict) -> None:
    """Append one row and force it to physical disk before returning.

    flush() + fsync() so a completed row survives even a hard interruption
    immediately after this call -- the whole point of incremental,
    per-row appends over batching everything until the end.
    """
    writer.writerow(row)
    manifest_file.flush()
    os.fsync(manifest_file.fileno())


# ---------------------------------------------------------------------------
# Raw image loading (per-client locator formats -- see dataset.py's
# build_manifest() docstring for the "{zip_half}::{member}" EyePACS format)
# ---------------------------------------------------------------------------
def _load_raw_image_from_path(locator: str) -> np.ndarray:
    path = Path(locator)
    if not path.exists():
        raise FileNotFoundError(f"raw image not found: {path}")
    return np.array(Image.open(path).convert("RGB"))


def _load_raw_image_from_zip(zf: "zipfile.ZipFile", member_name: str) -> np.ndarray:
    with zf.open(member_name) as f:
        data = f.read()
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def _parse_eyepacs_locator(locator: str) -> Tuple[str, str]:
    """Split "{zip_half}::{member_name}" into its two parts."""
    zip_half, _, member_name = locator.partition("::")
    if not member_name:
        raise BuildDatasetError(f"EyePACS: malformed raw_image_locator (expected '{{zip_half}}::{{member}}'): {locator!r}")
    return zip_half, member_name


def _reconstruct_eyepacs_zip(data_root: Path, zip_half: str, *, work_dir: Path) -> Path:
    """Reconstruct EyePACS's train.zip or test.zip from its numbered Kaggle-split parts.

    Mirrors the project's own EDA notebook approach exactly: simple binary
    concatenation of the parts, in order. Skips reconstruction if a
    complete local copy already sits in `work_dir` (its size matches the
    sum of the source parts) -- e.g. left over from an interrupted prior
    run. That size check is a cheap sanity check, not a full integrity
    check; zipfile.ZipFile() will still fail loudly and informatively if
    the reused file turns out to be genuinely corrupt.

    Args:
        data_root: Root directory containing "EYEPACS - Kaggle/".
        zip_half: "train" or "test".
        work_dir: Directory to reconstruct the zip into (and reuse from).

    Returns:
        Path to the reconstructed (or reused) zip file.

    Raises:
        BuildDatasetError: if no parts are found for `zip_half`.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    eyepacs_root = data_root / "EYEPACS - Kaggle"
    parts = sorted(eyepacs_root.glob(f"{zip_half}.zip.*"))
    if not parts:
        raise BuildDatasetError(f"EyePACS: no {zip_half}.zip.NNN part files found under {eyepacs_root}")

    output_path = work_dir / f"{zip_half}.zip"
    expected_size = sum(p.stat().st_size for p in parts)

    if output_path.exists() and output_path.stat().st_size == expected_size:
        logger.info(
            "EyePACS: reusing existing reconstructed %s (%d bytes matches the %d source part(s)).",
            output_path,
            expected_size,
            len(parts),
        )
        return output_path

    logger.info(
        "EyePACS: reconstructing %s from %d part(s) (~%.1f GB total) -- this can take a while.",
        zip_half,
        len(parts),
        expected_size / 1e9,
    )
    start = time.time()
    tmp_output = output_path.with_suffix(output_path.suffix + ".building")
    with open(tmp_output, "wb") as outfile:
        for part in parts:
            with open(part, "rb") as infile:
                shutil.copyfileobj(infile, outfile, length=64 * 1024 * 1024)
    os.replace(tmp_output, output_path)
    elapsed_min = (time.time() - start) / 60
    logger.info("EyePACS: reconstructed %s in %.1f min.", output_path, elapsed_min)
    return output_path


# ---------------------------------------------------------------------------
# Per-row processing (shared by both the plain-path and zip-based loaders)
# ---------------------------------------------------------------------------
def _process_and_write(
    *,
    image_id: str,
    patient_id: str,
    label: int,
    split: str,
    raw_image: np.ndarray,
    cache_root: Path,
    writer: "csv.DictWriter[str]",
    manifest_file,
) -> bool:
    """Run the offline pipeline on one raw image, cache it, and append its manifest row.

    Catches and logs (rather than propagates) any failure -- a single bad
    image must never abort an otherwise multi-hour client run. The EDA
    already confirmed zero corrupt files across all four clients, so this
    is a defensive backstop, not an expected code path.

    Returns:
        True if the row was successfully cached and recorded, False if it
        was skipped due to an error (already logged).
    """
    try:
        processed_image, meta = run_offline_pipeline(raw_image)
    except ImageProcessingError as exc:
        logger.error("%s: offline pipeline failed, skipping: %s", image_id, exc)
        return False
    except Exception as exc:  # noqa: BLE001 -- intentional: see docstring above.
        logger.error("%s: unexpected error in offline pipeline, skipping: %r", image_id, exc)
        return False

    cache_path = _cache_path_for(cache_root, split, image_id)
    try:
        _atomic_save_png(processed_image, cache_path)
    except OSError as exc:
        logger.error("%s: failed to save cached image to %s: %s", image_id, cache_path, exc)
        return False

    row = {
        "image_id": image_id,
        "cached_image_path": str(cache_path),
        "patient_id": patient_id,
        "label": int(label),
        "split": split,
        "blur_score": meta["blur_score"],
        "low_quality_flag": meta["low_quality_flag"],
        "was_cropped": meta["was_cropped"],
        "original_height": meta["original_height"],
        "original_width": meta["original_width"],
    }
    try:
        _write_manifest_row(writer, manifest_file, row)
    except OSError as exc:
        logger.error("%s: failed to write manifest row (image was cached at %s): %s", image_id, cache_path, exc)
        return False
    return True


def _log_progress(client_name: str, done: int, total_pending: int, start_time: float) -> None:
    if done == 0 or (done % _PROGRESS_LOG_INTERVAL != 0 and done != total_pending):
        return
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = total_pending - done
    eta_min = (remaining / rate / 60) if rate > 0 else float("nan")
    logger.info(
        "%s: %d/%d new image(s) processed (%.1f img/s, ETA ~%.0f min)",
        client_name,
        done,
        total_pending,
        rate,
        eta_min,
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def run_offline_preprocessing(
    client_name: str,
    *,
    data_root: Path,
    cache_root: Path,
    manifest_path: Path,
    max_samples: Optional[int] = EYEPACS_DEFAULT_MAX_SAMPLES,
    random_state: int = DEFAULT_SPLIT_RANDOM_STATE,
    zip_work_dir: Optional[Path] = None,
) -> BuildStats:
    """Run (or resume) the offline preprocessing pass for one client.

    Args:
        client_name: One of "APTOS", "IDRiD", "Messidor-2", "EyePACS".
        data_root: Directory containing the 4 raw dataset folders.
        cache_root: Directory to write cached, offline-processed PNGs
            into (organized as {cache_root}/{split}/{image_id}.png).
        manifest_path: Path to this client's FINAL manifest CSV -- read
            (if it already exists) to determine what's already done, then
            appended to as new rows complete.
        max_samples: EyePACS only; ignored (with a log message) for other
            clients. None means use the full client.
        random_state: Seed forwarded to dataset.split_manifest() and
            dataset.apply_eyepacs_max_samples_cap().
        zip_work_dir: Where to reconstruct EyePACS's zip archives.
            Defaults to `cache_root.parent / "_eyepacs_zip_work"`. Ignored
            for other clients.

    Returns:
        A BuildStats summary of this run.

    Raises:
        BuildDatasetError: for a corrupted/unparseable existing manifest,
            missing EyePACS zip parts, or other setup failures.
        DatasetError: propagated from dataset.py's build_manifest() /
            split_manifest() for missing source files, bad label data,
            etc.
    """
    data_root = Path(data_root)
    cache_root = Path(cache_root)
    manifest_path = Path(manifest_path)
    if zip_work_dir is None:
        zip_work_dir = cache_root.parent / "_eyepacs_zip_work"

    if not data_root.exists():
        raise BuildDatasetError(f"{client_name}: data_root does not exist: {data_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("%s: building source manifest from %s", client_name, data_root)
    source_manifest = build_manifest(client_name, data_root=data_root)

    if client_name == "EyePACS":
        source_manifest = apply_eyepacs_max_samples_cap(
            source_manifest, max_samples=max_samples, random_state=random_state
        )
    elif max_samples != EYEPACS_DEFAULT_MAX_SAMPLES:
        logger.info("%s: --max-samples only applies to EyePACS; ignoring for this client.", client_name)

    # Existing final manifest (if resuming). A row's image_id being present
    # here is the ONLY thing that marks it done -- see module docstring's
    # "Resumability" / "Split-assignment stability" sections for why this
    # single check is sufficient to guarantee an already-recorded row's
    # split can never silently change across resumed runs.
    existing_image_ids: Set[str] = set()
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(manifest_path)
        except pd.errors.ParserError as exc:
            raise BuildDatasetError(
                f"{client_name}: existing manifest at {manifest_path} could not be parsed -- it may "
                f"have been left in a partially-written state by an earlier interrupted run. Inspect "
                f"(and if needed, truncate the incomplete last line of) or remove it before resuming. "
                f"Original error: {exc}"
            ) from exc
        _require_columns(existing_df, FINAL_MANIFEST_COLUMNS, context=f"existing manifest at {manifest_path}")
        existing_image_ids = set(existing_df["image_id"].astype(str))

        # Guard against resuming into a manifest built from a DIFFERENT
        # working pool/configuration (e.g. a prior run used a different
        # --max-samples, or --data-root pointed somewhere else). If the
        # existing manifest has any image_id not present in the current
        # (post-cap, for EyePACS) source_manifest, appending to it would
        # silently mix two incompatible manifests together -- fail loudly
        # instead of doing that.
        current_image_ids = set(source_manifest["image_id"].astype(str))
        stale_image_ids = existing_image_ids - current_image_ids
        if stale_image_ids:
            example_ids = sorted(stale_image_ids)[:10]
            raise BuildDatasetError(
                f"{client_name}: existing manifest at {manifest_path} contains "
                f"{len(stale_image_ids)} image_id(s) not present in the current source "
                f"manifest (e.g. {example_ids}). This means the existing manifest was built "
                f"from a different working pool/configuration (e.g. a different --max-samples "
                f"or --data-root) and should not be appended to -- use a different "
                f"--manifest-out, or remove/archive the existing file, before resuming."
            )

        logger.info(
            "%s: resuming -- %d row(s) already recorded in %s", client_name, len(existing_image_ids), manifest_path
        )

    # Freshly split the FULL (possibly capped) source manifest. Only used
    # below for rows NOT already in existing_image_ids -- already-recorded
    # rows are never reprocessed, so this computation can never touch them.
    split_source_manifest = split_manifest(client_name, source_manifest, random_state=random_state)
    split_by_image_id: Dict[str, str] = dict(
        zip(split_source_manifest["image_id"].astype(str), split_source_manifest["split"])
    )

    total_rows = len(source_manifest)
    already_cached = 0
    newly_processed = 0
    failed = 0

    fieldnames = list(FINAL_MANIFEST_COLUMNS)
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0

    with open(manifest_path, "a", newline="", encoding="utf-8") as manifest_file:
        writer: "csv.DictWriter[str]" = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            manifest_file.flush()
            os.fsync(manifest_file.fileno())

        start_time = time.time()

        if client_name == "EyePACS":
            for zip_half in _EYEPACS_ZIP_HALVES:
                half_df = split_source_manifest[
                    split_source_manifest["raw_image_locator"].str.startswith(f"{zip_half}::")
                ]
                pending_df = half_df[~half_df["image_id"].astype(str).isin(existing_image_ids)]
                already_cached += len(half_df) - len(pending_df)
                if pending_df.empty:
                    logger.info(
                        "%s: %s half -- all %d row(s) already cached, skipping zip reconstruction entirely.",
                        client_name,
                        zip_half,
                        len(half_df),
                    )
                    continue

                zip_path = _reconstruct_eyepacs_zip(data_root, zip_half, work_dir=zip_work_dir)
                done_this_half = 0
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        for row in pending_df.itertuples():
                            image_id = str(row.image_id)
                            _, member_name = _parse_eyepacs_locator(row.raw_image_locator)
                            try:
                                raw_image = _load_raw_image_from_zip(zf, member_name)
                            except (KeyError, OSError) as exc:
                                logger.error(
                                    "%s: failed to read %s from %s: %s", client_name, member_name, zip_path.name, exc
                                )
                                failed += 1
                                done_this_half += 1
                                continue

                            ok = _process_and_write(
                                image_id=image_id,
                                patient_id=str(row.patient_id),
                                label=int(row.label),
                                split=split_by_image_id[image_id],
                                raw_image=raw_image,
                                cache_root=cache_root,
                                writer=writer,
                                manifest_file=manifest_file,
                            )
                            newly_processed += int(ok)
                            failed += int(not ok)
                            done_this_half += 1
                            _log_progress(f"{client_name} ({zip_half})", done_this_half, len(pending_df), start_time)
                finally:
                    zip_path.unlink(missing_ok=True)
                    logger.info("%s: deleted reconstructed %s to free disk space.", client_name, zip_path.name)
        else:
            pending_df = split_source_manifest[~split_source_manifest["image_id"].astype(str).isin(existing_image_ids)]
            already_cached = len(split_source_manifest) - len(pending_df)
            done = 0
            for row in pending_df.itertuples():
                image_id = str(row.image_id)
                try:
                    raw_image = _load_raw_image_from_path(row.raw_image_locator)
                except FileNotFoundError as exc:
                    logger.error("%s: %s", client_name, exc)
                    failed += 1
                    done += 1
                    continue

                ok = _process_and_write(
                    image_id=image_id,
                    patient_id=str(row.patient_id),
                    label=int(row.label),
                    split=split_by_image_id[image_id],
                    raw_image=raw_image,
                    cache_root=cache_root,
                    writer=writer,
                    manifest_file=manifest_file,
                )
                newly_processed += int(ok)
                failed += int(not ok)
                done += 1
                _log_progress(client_name, done, len(pending_df), start_time)

    stats = BuildStats(
        client_name=client_name,
        total_rows=total_rows,
        already_cached=already_cached,
        newly_processed=newly_processed,
        failed=failed,
        manifest_path=manifest_path,
    )
    logger.info("%s: run complete. %s", client_name, stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_max_samples(value: str) -> Optional[int]:
    if value.strip().lower() == "none":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--max-samples must be an integer or 'none', got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"--max-samples must be a positive integer or 'none', got {parsed}")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline preprocessing driver: caches one client's images and writes its final manifest."
    )
    parser.add_argument("--client", required=True, choices=["APTOS", "IDRiD", "Messidor-2", "EyePACS"])
    parser.add_argument(
        "--data-root", required=True, type=Path, help="Directory containing the 4 raw dataset folders."
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Where to write cached PNGs. Defaults to './data/cache/{client}' relative to the "
        "current working directory.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Path to this client's final manifest CSV. Defaults to "
        "'./data/splits/{client}_manifest.csv' relative to the current working directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=_parse_max_samples,
        default=EYEPACS_DEFAULT_MAX_SAMPLES,
        help="EyePACS only: cap the working pool to this many images (stratified, patient-preserving). "
        "Pass 'none' for the full 88,702. Ignored for other clients. Default: 6000.",
    )
    parser.add_argument("--random-state", type=int, default=DEFAULT_SPLIT_RANDOM_STATE)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> BuildStats:
    """CLI / programmatic entry point.

    Usable both as `python -m preprocessing.build_dataset ...` and from a
    Colab cell as `main(["--client", "APTOS", "--data-root", ROOT])`.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    cache_root = args.cache_root or (Path("data") / "cache" / args.client)
    manifest_path = args.manifest_out or (Path("data") / "splits" / f"{args.client}_manifest.csv")

    return run_offline_preprocessing(
        args.client,
        data_root=args.data_root,
        cache_root=cache_root,
        manifest_path=manifest_path,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()