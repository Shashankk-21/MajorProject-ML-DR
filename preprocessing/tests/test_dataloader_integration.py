"""
preprocessing/tests/test_dataloader_integration.py

Integration tests for the per-client FINAL-manifest Dataset/DataLoader layer.

Run from the project root:
    python -m preprocessing.tests.test_dataloader_integration
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from preprocessing.augmentation import get_eval_transforms, get_train_transforms
from preprocessing.dataset import (
    FINAL_MANIFEST_COLUMNS,
    DRDataset,
    compute_class_weights,
    get_dataloaders,
)


PROJECT_ROOT = Path(".")
MANIFEST_ROOT = PROJECT_ROOT / "data" / "splits"
CLIENTS = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")
BATCH_SIZE = 4
NUM_WORKERS = 0
SEED = 42


def manifest_path_for(client_name: str) -> Path:
    return MANIFEST_ROOT / f"{client_name}_manifest.csv"


def check_manifest_exists_and_schema(
    client_name: str,
    manifest_path: Path,
) -> pd.DataFrame:
    assert manifest_path.exists(), (
        f"{client_name}: manifest not found at {manifest_path}. "
        "Run build_dataset.py for this client first."
    )

    manifest_df = pd.read_csv(manifest_path)

    missing = [
        column
        for column in FINAL_MANIFEST_COLUMNS
        if column not in manifest_df.columns
    ]

    assert not missing, (
        f"{client_name}: manifest {manifest_path} "
        f"is missing required columns: {missing}"
    )

    assert not manifest_df.empty, f"{client_name}: manifest is empty."

    return manifest_df


def check_split_sizes(
    client_name: str,
    manifest_path: Path,
    manifest_df: pd.DataFrame,
) -> Tuple[int, int, int]:
    train_size = int((manifest_df["split"] == "train").sum())
    val_size = int((manifest_df["split"] == "val").sum())
    test_size = int((manifest_df["split"] == "test").sum())

    # Strong total-row check:
    # split counts must equal the actual number of rows in the manifest.
    manifest_row_count = len(pd.read_csv(manifest_path))

    assert train_size + val_size + test_size == manifest_row_count, (
        f"{client_name}: split counts do not sum to manifest row count: "
        f"train={train_size}, val={val_size}, test={test_size}, "
        f"manifest_rows={manifest_row_count}"
    )

    assert train_size > 0, f"{client_name}: train split is empty."
    assert val_size > 0, f"{client_name}: val split is empty."
    assert test_size > 0, f"{client_name}: test split is empty."

    return train_size, val_size, test_size


def build_loaders_for_client(
    client_name: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    manifest_path = manifest_path_for(client_name)

    manifest_df = check_manifest_exists_and_schema(
        client_name,
        manifest_path,
    )

    check_split_sizes(
        client_name,
        manifest_path,
        manifest_df,
    )

    train_transform = get_train_transforms()
    eval_transform = get_eval_transforms()

    train_loader, val_loader, test_loader = get_dataloaders(
        client_name,
        manifest_path=manifest_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transform=train_transform,
        eval_transform=eval_transform,
        seed=SEED,
    )

    return train_loader, val_loader, test_loader, manifest_df


def check_class_weights() -> None:
    print("\n[1] compute_class_weights() synthetic contract")

    # Class 3 is deliberately absent.
    labels = np.array(
        [0, 0, 1, 1, 1, 2, 4, 4],
        dtype=np.int64,
    )

    weights = compute_class_weights(
        labels,
        num_classes=5,
    )

    assert weights.shape == (5,), (
        f"Expected shape (5,), got {weights.shape}"
    )

    assert weights.dtype == np.float64, (
        f"Expected float64, got {weights.dtype}"
    )

    assert weights[3] == 0.0, (
        f"Absent class 3 must have weight 0, got {weights[3]}"
    )

    for class_id in (0, 1, 2, 4):
        assert weights[class_id] > 0.0, (
            f"Present class {class_id} must have positive weight, "
            f"got {weights[class_id]}"
        )

    counts = np.bincount(
        labels,
        minlength=5,
    ).astype(np.float64)

    weighted_mean = float(
        np.sum((counts / labels.size) * weights)
    )

    assert np.isclose(
        weighted_mean,
        1.0,
        atol=1e-12,
    ), (
        "Expected sum(count/N * weight)=1.0, "
        f"got {weighted_mean}"
    )

    print("PASS: shape=(5,)")
    print("PASS: dtype=float64")
    print("PASS: zero-count class gets weight 0")
    print("PASS: present classes get positive weights")
    print("PASS: sum(count/N * weight)=1")


def check_actual_client_class_weights() -> None:
    print("\n[2] actual-client class-weight contract")

    for client_name in CLIENTS:
        manifest_path = manifest_path_for(client_name)

        manifest_df = check_manifest_exists_and_schema(
            client_name,
            manifest_path,
        )

        train_labels = manifest_df.loc[
            manifest_df["split"] == "train",
            "label",
        ].to_numpy(dtype=np.int64)

        weights = compute_class_weights(
            train_labels,
            num_classes=5,
        )

        assert weights.shape == (5,), (
            f"{client_name}: expected class-weight shape "
            f"(5,), got {weights.shape}"
        )

        assert weights.dtype == np.float64, (
            f"{client_name}: expected float64 weights, "
            f"got {weights.dtype}"
        )

        counts = np.bincount(
            train_labels,
            minlength=5,
        ).astype(np.float64)

        weighted_mean = float(
            np.sum((counts / train_labels.size) * weights)
        )

        assert np.isclose(
            weighted_mean,
            1.0,
            atol=1e-12,
        ), (
            f"{client_name}: empirical weighted mean must be 1.0, "
            f"got {weighted_mean}"
        )

        zero_classes = np.where(counts == 0)[0]

        if zero_classes.size:
            assert np.all(
                weights[zero_classes] == 0.0
            ), (
                f"{client_name}: zero-count classes "
                f"{zero_classes.tolist()} must have zero weight."
            )

        print(
            f"PASS: {client_name} "
            f"(train={train_labels.size}, "
            f"weights={np.round(weights, 6).tolist()})"
        )


def check_dataset_round_trip() -> None:
    print("\n[3] DRDataset + DataLoader round-trip")

    for client_name in CLIENTS:
        manifest_path = manifest_path_for(client_name)

        manifest_df = check_manifest_exists_and_schema(
            client_name,
            manifest_path,
        )

        check_split_sizes(
            client_name,
            manifest_path,
            manifest_df,
        )

        train_transform = get_train_transforms()
        eval_transform = get_eval_transforms()

        train_dataset = DRDataset(
            manifest_df,
            split="train",
            transform=train_transform,
        )

        val_dataset = DRDataset(
            manifest_df,
            split="val",
            transform=eval_transform,
        )

        test_dataset = DRDataset(
            manifest_df,
            split="test",
            transform=eval_transform,
        )

        assert len(train_dataset) == int(
            (manifest_df["split"] == "train").sum()
        )

        assert len(val_dataset) == int(
            (manifest_df["split"] == "val").sum()
        )

        assert len(test_dataset) == int(
            (manifest_df["split"] == "test").sum()
        )

        train_image, train_label = train_dataset[0]
        val_image, val_label = val_dataset[0]
        test_image, test_label = test_dataset[0]

        for split_name, image, label in (
            ("train", train_image, train_label),
            ("val", val_image, val_label),
            ("test", test_image, test_label),
        ):
            assert isinstance(
                image,
                torch.Tensor,
            ), (
                f"{client_name}/{split_name}: "
                f"expected torch.Tensor, got {type(image)}"
            )

            assert image.shape == (3, 224, 224), (
                f"{client_name}/{split_name}: "
                f"expected (3,224,224), got {tuple(image.shape)}"
            )

            assert image.dtype == torch.float32, (
                f"{client_name}/{split_name}: "
                f"expected float32, got {image.dtype}"
            )

            assert torch.isfinite(image).all(), (
                f"{client_name}/{split_name}: "
                "tensor contains non-finite values."
            )

            assert isinstance(label, int), (
                f"{client_name}/{split_name}: "
                f"expected Python int label, got {type(label)}"
            )

            assert 0 <= label < 5, (
                f"{client_name}/{split_name}: "
                f"invalid label {label}"
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            drop_last=False,
            generator=torch.Generator().manual_seed(SEED),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            drop_last=False,
        )

        train_images, train_labels = next(
            iter(train_loader)
        )

        val_images, val_labels = next(
            iter(val_loader)
        )

        test_images, test_labels = next(
            iter(test_loader)
        )

        for split_name, images, labels in (
            ("train", train_images, train_labels),
            ("val", val_images, val_labels),
            ("test", test_images, test_labels),
        ):
            assert images.ndim == 4, (
                f"{client_name}/{split_name}: "
                "expected 4-D image batch"
            )

            assert images.shape[1:] == (
                3,
                224,
                224,
            ), (
                f"{client_name}/{split_name}: "
                f"unexpected batch shape {tuple(images.shape)}"
            )

            assert images.dtype == torch.float32, (
                f"{client_name}/{split_name}: "
                f"expected float32 batch, got {images.dtype}"
            )

            assert labels.ndim == 1, (
                f"{client_name}/{split_name}: "
                "expected 1-D labels"
            )

            assert labels.dtype == torch.int64, (
                f"{client_name}/{split_name}: "
                f"expected int64 labels, got {labels.dtype}"
            )

            assert torch.isfinite(images).all(), (
                f"{client_name}/{split_name}: "
                "batch contains non-finite values"
            )

        print(
            f"PASS: {client_name} "
            f"(train={len(train_dataset)}, "
            f"val={len(val_dataset)}, "
            f"test={len(test_dataset)})"
        )


def check_train_sampler_contract() -> None:
    print("\n[4] train RandomSampler / generator contract")

    for client_name in CLIENTS:
        train_loader, _, _, _ = build_loaders_for_client(
            client_name
        )

        assert isinstance(
            train_loader.sampler,
            RandomSampler,
        ), (
            f"{client_name}: train sampler must be "
            f"RandomSampler, got {type(train_loader.sampler)}"
        )

        assert train_loader.sampler.replacement is False, (
            f"{client_name}: train RandomSampler "
            "must use replacement=False"
        )

        assert (
            train_loader.sampler.num_samples
            == len(train_loader.dataset)
        ), (
            f"{client_name}: RandomSampler num_samples "
            "must equal len(train_dataset)"
        )

        assert train_loader.drop_last is False, (
            f"{client_name}: train drop_last must be False"
        )

        assert train_loader.generator is not None, (
            f"{client_name}: train DataLoader must "
            "have a seeded generator"
        )

        assert (
            train_loader.generator.initial_seed()
            == SEED
        ), (
            f"{client_name}: expected generator seed {SEED}, "
            f"got {train_loader.generator.initial_seed()}"
        )

        print(
            f"PASS: {client_name} "
            f"(RandomSampler, replacement=False, "
            f"num_samples={train_loader.sampler.num_samples}, "
            f"seed={train_loader.generator.initial_seed()})"
        )


def check_eval_samplers() -> None:
    print("\n[5] validation/test SequentialSampler contract")

    for client_name in CLIENTS:
        _, val_loader, test_loader, _ = (
            build_loaders_for_client(client_name)
        )

        assert isinstance(
            val_loader.sampler,
            SequentialSampler,
        ), (
            f"{client_name}: val sampler must be "
            f"SequentialSampler, got {type(val_loader.sampler)}"
        )

        assert isinstance(
            test_loader.sampler,
            SequentialSampler,
        ), (
            f"{client_name}: test sampler must be "
            f"SequentialSampler, got {type(test_loader.sampler)}"
        )

        assert val_loader.drop_last is False, (
            f"{client_name}: val drop_last must be False"
        )

        assert test_loader.drop_last is False, (
            f"{client_name}: test drop_last must be False"
        )

        print(
            f"PASS: {client_name} "
            "(val/test SequentialSampler)"
        )


def check_sampler_reproducibility() -> None:
    print("\n[6] train sampler reproducibility with identical seed")

    for client_name in CLIENTS:
        loader_a, _, _, _ = build_loaders_for_client(
            client_name
        )

        loader_b, _, _, _ = build_loaders_for_client(
            client_name
        )

        indices_a = list(
            iter(loader_a.sampler)
        )[:20]

        indices_b = list(
            iter(loader_b.sampler)
        )[:20]

        assert indices_a == indices_b, (
            f"{client_name}: first 20 train sampler indices "
            f"differ for identical seed {SEED}.\n"
            f"A={indices_a}\n"
            f"B={indices_b}"
        )

        print(
            f"PASS: {client_name} "
            "(first 20 indices reproducible)"
        )


def check_cross_client_isolation() -> None:
    print("\n[7] cross-client isolation")

    manifests: Dict[str, pd.DataFrame] = {}

    for client_name in CLIENTS:
        manifests[client_name] = (
            check_manifest_exists_and_schema(
                client_name,
                manifest_path_for(client_name),
            )
        )

    for i, client_a in enumerate(CLIENTS):
        a = manifests[client_a]

        assert a["image_id"].is_unique, (
            f"{client_a}: duplicate image_id values found."
        )

        for client_b in CLIENTS[i + 1 :]:
            b = manifests[client_b]

            image_overlap = set(
                a["image_id"]
            ).intersection(
                b["image_id"]
            )

            patient_overlap = set(
                a["patient_id"]
            ).intersection(
                b["patient_id"]
            )

            path_overlap = set(
                a["cached_image_path"]
            ).intersection(
                b["cached_image_path"]
            )

            assert not image_overlap, (
                f"Cross-client image_id overlap: "
                f"{client_a} vs {client_b}: "
                f"{list(sorted(image_overlap))[:10]}"
            )

            assert not patient_overlap, (
                f"Cross-client patient_id overlap: "
                f"{client_a} vs {client_b}: "
                f"{list(sorted(patient_overlap))[:10]}"
            )

            assert not path_overlap, (
                f"Cross-client cached_image_path overlap: "
                f"{client_a} vs {client_b}: "
                f"{list(sorted(path_overlap))[:10]}"
            )

    print(
        "PASS: image_id/patient_id/"
        "cached_image_path sets are isolated"
    )


def main() -> None:
    print("=" * 72)
    print(
        "DATALOADER INTEGRATION TEST — UPDATED CONTRACT"
    )
    print("=" * 72)

    check_class_weights()
    check_actual_client_class_weights()
    check_dataset_round_trip()
    check_train_sampler_contract()
    check_eval_samplers()
    check_sampler_reproducibility()
    check_cross_client_isolation()

    print("\n" + "=" * 72)
    print(
        "ALL DATALOADER INTEGRATION TESTS PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()