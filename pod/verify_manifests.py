from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import random
import tempfile
from pathlib import Path
from typing import Iterable

from PIL import Image


CLIENTS = (
    ("APTOS", "aptos", 2930, 366, 366, 3662),
    ("IDRiD", "idrid", 330, 83, 103, 516),
    ("Messidor-2", "messidor2", 1396, 174, 174, 1744),
    ("EyePACS", "eyepacs", 70960, 8872, 8870, 88702),
)


class ValidationFailure(Exception):
    pass


def fail(condition: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    raise ValidationFailure(f"FAIL {condition}{suffix}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(path: Path) -> tuple[list[str], list[list[str]], int]:
    if not path.is_file():
        fail("source manifest", f"missing file: {path}")

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        fail("source manifest", f"cannot read {path}: {exc}")

    if not rows:
        fail("source manifest", f"empty CSV: {path}")

    header = rows[0]
    data_rows = rows[1:]
    if len(header) != len(set(header)):
        fail("source manifest", f"duplicate column name in {path}")
    if len([name for name in header if name == "cached_image_path"]) != 1:
        fail("source manifest", f"cached_image_path must occur exactly once in {path}")
    for name in ("image_id", "label", "split"):
        if name not in header:
            fail(f"required column {name}", f"missing from {path}")

    cached_index = header.index("cached_image_path")
    image_id_index = header.index("image_id")
    label_index = header.index("label")
    split_index = header.index("split")

    for row_number, row in enumerate(data_rows, start=2):
        if len(row) != len(header):
            fail(
                "source row shape",
                f"{path}: row {row_number} has {len(row)} cells; "
                f"expected {len(header)}",
            )
        if not row[image_id_index]:
            fail("unique image_id", f"{path}: row {row_number} has an empty image_id")
        if row[label_index] not in {"0", "1", "2", "3", "4"}:
            fail(
                "labels 0..4",
                f"{path}: row {row_number} label={row[label_index]!r}",
            )
        if row[split_index] not in {"train", "val", "test"}:
            fail(
                "splits",
                f"{path}: row {row_number} split={row[split_index]!r}",
            )
        if not row[cached_index]:
            fail(
                "source path",
                f"{path}: row {row_number} has an empty cached_image_path",
            )

    print("PASS source manifest")
    return header, data_rows, cached_index


def remap_source_path(
    source_path: str,
    client: str,
    split_value: str,
    datasets_root: Path,
    folder: str,
) -> Path:
    normalized = source_path.replace("\\", "/")

    if normalized.startswith("/") or len(normalized) >= 3 and normalized[1:3] in (":/", ":\\"):
        fail("source path", f"absolute path is not allowed: {source_path!r}")

    components = normalized.split("/")
    if ".." in components:
        fail("source path", f"'..' traversal is not allowed: {source_path!r}")
    if not components or "" in components:
        fail("source path", f"empty or malformed path: {source_path!r}")

    prefix = ["data", "cache", client]
    if len(components) != 5 or components[:3] != prefix:
        fail(
            "source path",
            f"unexpected prefix or directory depth: {source_path!r}; "
            f"expected data/cache/{client}/{{split}}/{{filename}}",
        )

    split_directory = components[3]
    filename = components[4]
    if split_directory not in {"train", "val", "test"}:
        fail("source path", f"unexpected split directory: {source_path!r}")
    if split_directory != split_value:
        fail(
            "splits and source path consistency",
            f"CSV split={split_value!r}, path split={split_directory!r}; "
            f"path={source_path!r}",
        )
    if filename == "":
        fail("source path", f"empty filename: {source_path!r}")
    if not filename.lower().endswith(".png"):
        fail("source path", f"non-PNG filename: {source_path!r}")
    if Path(filename).name != filename:
        fail("source path", f"unexpected nested filename structure: {source_path!r}")

    dataset_directory = (datasets_root / folder).resolve(strict=False)
    remapped = (dataset_directory / split_directory / filename).resolve(strict=False)
    try:
        remapped.relative_to(dataset_directory)
    except ValueError:
        fail(
            "source path",
            f"remapped path leaves client dataset directory: {source_path!r}",
        )

    if remapped.parent != (dataset_directory / split_directory).resolve(strict=False):
        fail("source path", f"unexpected directory depth: {source_path!r}")
    if not remapped.is_file():
        fail("remapped image path", f"file does not exist: {remapped}")

    return remapped


def png_file_list(dataset_directory: Path) -> list[Path]:
    if not dataset_directory.is_dir():
        fail("dataset directory", f"missing: {dataset_directory}")
    files = [
        path.resolve()
        for path in dataset_directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    ]
    files.sort(key=lambda item: str(item))
    return files


def validate_client(
    client: str,
    folder: str,
    train_count: int,
    val_count: int,
    test_count: int,
    total_count: int,
    src_root: Path,
    datasets_root: Path,
) -> tuple[list[str], list[list[str]], list[str], bytes, str]:
    print(f"\n=== {client} ===")
    source_path = src_root / f"{client}_manifest.csv"
    header, rows, cached_index = read_source(source_path)

    split_index = header.index("split")
    image_id_index = header.index("image_id")

    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        counts[row[split_index]] += 1
    if len(rows) != total_count or any(
        counts[split_name] != expected
        for split_name, expected in (
            ("train", train_count),
            ("val", val_count),
            ("test", test_count),
        )
    ):
        fail(
            "row counts",
            f"expected train={train_count} val={val_count} "
            f"test={test_count} total={total_count}; "
            f"found train={counts['train']} val={counts['val']} "
            f"test={counts['test']} total={len(rows)}",
        )
    print(
        f"PASS row counts: train={counts['train']} val={counts['val']} "
        f"test={counts['test']} total={len(rows)}"
    )

    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        image_id = row[image_id_index]
        if image_id in seen_ids:
            fail(
                "unique image_id",
                f"{source_path}: row {row_number} duplicate image_id={image_id!r}",
            )
        seen_ids.add(image_id)
    print("PASS unique image_id")
    print("PASS labels 0..4")

    dataset_directory = (datasets_root / folder).resolve(strict=False)
    remapped_paths: list[Path] = []
    remapped_strings: list[str] = []
    seen_paths: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        remapped = remap_source_path(
            row[cached_index],
            client,
            row[split_index],
            datasets_root,
            folder,
        )
        remapped_key = str(remapped)
        if remapped_key in seen_paths:
            fail(
                "no duplicate paths",
                f"{source_path}: row {row_number} duplicate remapped path={remapped_key}",
            )
        seen_paths.add(remapped_key)
        remapped_paths.append(remapped)
        remapped_strings.append(str(remapped))

    print("PASS splits and source path consistency")
    print(f"PASS all {len(rows)} remapped image paths exist")
    print("PASS no duplicate paths")

    png_files = png_file_list(dataset_directory)
    unreferenced = len(png_files) - len(seen_paths)
    if len(png_files) != total_count:
        fail(
            "dataset PNG count",
            f"{dataset_directory}: found {len(png_files)} PNG files, "
            f"expected {total_count}; unreferenced or excess PNGs: "
            f"{max(0, unreferenced)}",
        )
    if len(png_files) != len(rows):
        fail(
            "dataset completeness",
            f"{dataset_directory}: found {len(png_files)} PNG files, "
            f"manifest has {len(rows)} rows",
        )
    print(f"PASS dataset PNG count: {len(png_files)}")

    if len(png_files) < 300:
        fail("sampled images", f"only {len(png_files)} PNG files; need 300")
    sampled = random.Random(42).sample(png_files, 300)
    for image_path in sampled:
        try:
            with Image.open(image_path) as image:
                image.load()
                if image.size != (224, 224):
                    fail(
                        "sampled images",
                        f"{image_path}: size={image.size}, expected (224, 224)",
                    )
                if image.mode != "RGB":
                    fail(
                        "sampled images",
                        f"{image_path}: mode={image.mode}, expected RGB",
                    )
        except ValidationFailure:
            raise
        except Exception as exc:
            fail("sampled images", f"cannot open/validate {image_path}: {exc}")
    print("PASS sampled images: 300/300 are 224x224 RGB")

    output_rows = [
        [
            remapped_strings[row_number]
            if column_name == "cached_image_path"
            else row[column_index]
            for column_index, column_name in enumerate(header)
        ]
        for row_number, row in enumerate(rows)
    ]

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(output_rows)
    output_bytes = buffer.getvalue().encode("utf-8")

    return header, rows, remapped_strings, output_bytes, str(source_path)


def target_matches(path: Path, expected: bytes) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        fail("destination manifest", f"target exists but is not a regular file: {path}")
    try:
        actual = path.read_bytes()
    except OSError as exc:
        fail("destination manifest", f"cannot read {path}: {exc}")
    return actual == expected


def write_temp_file(destination: Path, data: bytes) -> Path:
    fd, temp_name = tempfile.mkstemp(prefix=".verify-manifest-", dir=str(destination))
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o444)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def publish_new_file(destination: Path, data: bytes) -> str:
    temp_path = write_temp_file(destination.parent, data)
    try:
        try:
            os.link(temp_path, destination)
        except FileExistsError:
            if target_matches(destination, data):
                print("PASS destination manifest (already identical)")
                return "identical"
            fail(
                "destination conflict",
                f"existing file differs and will not be overwritten: {destination}",
            )
        except OSError as exc:
            fail(
                "atomic publication",
                f"cannot publish {destination}: {exc}",
            )
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

    print("PASS destination manifest")
    return "created"


def preflight_destination_conflicts(
    dst: Path,
    manifest_bytes: dict[str, bytes],
) -> None:
    for client, _, _, _, _, _ in CLIENTS:
        target = dst / f"{client}_manifest.csv"
        if target.exists():
            if not target.is_file():
                fail("destination manifest", f"target is not a regular file: {target}")
            try:
                existing = target.read_bytes()
            except OSError as exc:
                fail("destination manifest", f"cannot read {target}: {exc}")
            if existing != manifest_bytes[client]:
                fail(
                    "destination conflict",
                    f"existing file differs and will not be overwritten: {target}",
                )


def readback_verify(
    dst: Path,
    client: str,
    source_header: list[str],
    source_rows: list[list[str]],
    remapped_paths: list[str],
    expected_sha256: str,
) -> None:
    target = dst / f"{client}_manifest.csv"
    try:
        with target.open("r", encoding="utf-8", newline="") as handle:
            output_rows = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        fail("read-back verification", f"cannot read {target}: {exc}")

    if not output_rows:
        fail("read-back verification", f"empty destination file: {target}")
    if output_rows[0] != source_header:
        fail("read-back verification", f"{target}: column order changed")
    if len(output_rows) - 1 != len(source_rows):
        fail("read-back verification", f"{target}: row count changed")

    for row_number, (source_row, output_row) in enumerate(
        zip(source_rows, output_rows[1:]),
        start=2,
    ):
        if len(output_row) != len(source_header):
            fail(
                "read-back verification",
                f"{target}: row {row_number} has unexpected column count",
            )
        for column_index, column_name in enumerate(source_header):
            expected = (
                remapped_paths[row_number - 2]
                if column_name == "cached_image_path"
                else source_row[column_index]
            )
            if output_row[column_index] != expected:
                fail(
                    "read-back verification",
                    f"{target}: row {row_number}, column {column_name!r} changed; "
                    f"expected {expected!r}, found {output_row[column_index]!r}",
                )

    actual_sha256 = sha256_file(target)
    if actual_sha256 != expected_sha256:
        fail(
                "read-back verification",
                f"{target}: SHA-256 mismatch; expected {expected_sha256}, "
                f"found {actual_sha256}",
        )

    print("PASS successful read-back verification")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate source manifests and publish immutable Pod manifests."
    )
    parser.add_argument("--src", default="data/splits")
    parser.add_argument("--dst", default="/data/manifests")
    parser.add_argument("--datasets-root", default="/data/datasets")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    src = Path(args.src).resolve(strict=False)
    dst = Path(args.dst).resolve(strict=False)
    datasets_root = Path(args.datasets_root).resolve(strict=False)

    print(f"Effective source: {src}")
    print(f"Effective destination: {dst}")
    print(f"Effective datasets root: {datasets_root}")

    results: dict[str, tuple[list[str], list[list[str]], list[str], bytes, str]] = {}
    try:
        for client, folder, train_count, val_count, test_count, total_count in CLIENTS:
            results[client] = validate_client(
                client,
                folder,
                train_count,
                val_count,
                test_count,
                total_count,
                src,
                datasets_root,
            )
    except ValidationFailure as exc:
        print(exc)
        print("\nVALIDATION FAILED - NO MANIFESTS PUBLISHED")
        return 1
    except Exception as exc:
        print(f"FAIL unexpected validation error: {exc}")
        print("\nVALIDATION FAILED - NO MANIFESTS PUBLISHED")
        return 1

    manifest_bytes = {
        client: results[client][3]
        for client, _, _, _, _, _ in CLIENTS
    }

    try:
        dst.mkdir(parents=True, exist_ok=True)
        if not dst.is_dir():
            fail("destination directory", f"not a directory: {dst}")

        preflight_destination_conflicts(dst, manifest_bytes)

        checksum_lines = []
        for client, _, _, _, _, _ in CLIENTS:
            filename = f"{client}_manifest.csv"
            digest = sha256_bytes(manifest_bytes[client])
            checksum_lines.append(f"{digest}  {filename}\n")
        checksum_bytes = "".join(checksum_lines).encode("utf-8")

        checksum_target = dst / "SHA256SUMS"
        if checksum_target.exists():
            if not checksum_target.is_file():
                fail("SHA256SUMS", f"target is not a regular file: {checksum_target}")
            existing = checksum_target.read_bytes()
            if existing != checksum_bytes:
                fail(
                    "SHA256SUMS conflict",
                    f"existing file differs and will not be overwritten: "
                    f"{checksum_target}",
                )

        for client, _, _, _, _, _ in CLIENTS:
            filename = f"{client}_manifest.csv"
            target = dst / filename
            if target.exists():
                if target_matches(target, manifest_bytes[client]):
                    print(f"PASS destination manifest: {target}")
                    print("PASS existing manifest is identical")
                else:
                    fail(
                        "destination conflict",
                        f"existing file differs and will not be overwritten: {target}",
                    )
            else:
                publish_new_file(target, manifest_bytes[client])

        if checksum_target.exists():
            if checksum_target.read_bytes() == checksum_bytes:
                print("PASS SHA256SUMS (already identical)")
        else:
            publish_result = publish_new_file(checksum_target, checksum_bytes)
            if publish_result == "created":
                print(f"SHA256SUMS written to {checksum_target}")

        for client, _, _, _, _, _ in CLIENTS:
            filename = f"{client}_manifest.csv"
            digest = sha256_bytes(manifest_bytes[client])
            print(f"SHA256 {filename}: {digest}")
            source_header, source_rows, remapped_paths, _, _ = results[client]
            readback_verify(
                dst,
                client,
                source_header,
                source_rows,
                remapped_paths,
                digest,
            )

        print("\n=== SUMMARY ===")
        for client, _, _, _, _, _ in CLIENTS:
            print(f"{client:<12} PASS")
        print("\nALL FOUR MANIFESTS VERIFIED")
        print(f"SHA256SUMS: {checksum_target}")
        return 0
    except ValidationFailure as exc:
        print(exc)
        print("\nPUBLICATION FAILED")
        return 1
    except Exception as exc:
        print(f"FAIL unexpected publication error: {exc}")
        print("\nPUBLICATION FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
