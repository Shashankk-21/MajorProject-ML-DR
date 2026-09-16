from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

from preprocessing.dataset import DRDataset, get_dataloaders
from preprocessing.augmentation import get_eval_transforms, get_train_transforms


DATA_ROOT = Path(r"C:\MajorProject - Datasets")
BATCH_SIZE = 4
NUM_WORKERS = 0
SEED = 42


MANIFESTS = {
    "APTOS": DATA_ROOT.parent / "dummy"  # replaced below
}


def project_manifest(client: str) -> Path:
    """Return the generated final manifest for a client."""
    return Path("data") / "splits" / f"{client}_manifest.csv"


def check_dataset_item(dataset: DRDataset, name: str) -> None:
    """Verify one actual cached image can travel through DRDataset."""
    tensor, label = dataset[0]

    assert isinstance(tensor, torch.Tensor), (
        f"{name}: expected torch.Tensor, got {type(tensor)}"
    )
    assert tensor.shape == (3, 224, 224), (
        f"{name}: unexpected tensor shape {tuple(tensor.shape)}"
    )
    assert tensor.dtype == torch.float32, (
        f"{name}: unexpected dtype {tensor.dtype}"
    )
    assert isinstance(label, int), (
        f"{name}: expected Python int label, got {type(label)}"
    )
    assert 0 <= label <= 4, (
        f"{name}: invalid label {label}"
    )


def check_loader_batch(loader, name: str) -> None:
    """Verify one actual DataLoader batch."""
    images, labels = next(iter(loader))

    assert isinstance(images, torch.Tensor), (
        f"{name}: images are not a torch.Tensor"
    )
    assert isinstance(labels, torch.Tensor), (
        f"{name}: labels are not a torch.Tensor"
    )

    assert images.shape == (BATCH_SIZE, 3, 224, 224), (
        f"{name}: unexpected image batch shape {tuple(images.shape)}"
    )
    assert labels.shape == (BATCH_SIZE,), (
        f"{name}: unexpected label batch shape {tuple(labels.shape)}"
    )

    assert images.dtype == torch.float32, (
        f"{name}: unexpected image dtype {images.dtype}"
    )
    assert labels.dtype in (torch.int64, torch.long), (
        f"{name}: unexpected label dtype {labels.dtype}"
    )

    assert torch.isfinite(images).all(), (
        f"{name}: non-finite values in image tensor"
    )

    assert bool(((labels >= 0) & (labels <= 4)).all()), (
        f"{name}: invalid label found in batch"
    )


def main() -> None:
    print("=" * 70)
    print("DATALOADER INTEGRATION TEST")
    print("=" * 70)

    train_transform = get_train_transforms()
    eval_transform = get_eval_transforms()

    clients = ["APTOS", "IDRiD", "Messidor-2", "EyePACS"]

    for client in clients:
        print("\n" + "=" * 70)
        print(f"CLIENT: {client}")
        print("=" * 70)

        manifest_path = project_manifest(client)

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{client}: final manifest not found:\n{manifest_path}"
            )

        print(f"Manifest: {manifest_path}")

        train_loader, val_loader, test_loader = get_dataloaders(
            client,
            manifest_path=manifest_path,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            train_transform=train_transform,
            eval_transform=eval_transform,
            seed=SEED,
        )

        # --------------------------------------------------------
        # Basic loader lengths
        # --------------------------------------------------------

        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0
        assert len(test_loader.dataset) > 0

        print(
            "Dataset sizes:",
            f"train={len(train_loader.dataset)}",
            f"val={len(val_loader.dataset)}",
            f"test={len(test_loader.dataset)}",
        )

        # --------------------------------------------------------
        # Confirm Dataset object types
        # --------------------------------------------------------

        assert isinstance(train_loader.dataset, DRDataset)
        assert isinstance(val_loader.dataset, DRDataset)
        assert isinstance(test_loader.dataset, DRDataset)

        assert train_loader.dataset.split == "train"
        assert val_loader.dataset.split == "val"
        assert test_loader.dataset.split == "test"

        # --------------------------------------------------------
        # Confirm transforms are attached correctly
        # --------------------------------------------------------

        assert train_loader.dataset.transform is train_transform
        assert val_loader.dataset.transform is eval_transform
        assert test_loader.dataset.transform is eval_transform

        print("Transform assignment: PASS")

        # --------------------------------------------------------
        # Confirm train sampler / val-test sampler behavior
        # --------------------------------------------------------

        assert isinstance(
            train_loader.sampler,
            WeightedRandomSampler,
        ), (
            f"{client}: train loader does not use WeightedRandomSampler"
        )

        assert not isinstance(
            val_loader.sampler,
            WeightedRandomSampler,
        ), (
            f"{client}: validation loader unexpectedly uses WeightedRandomSampler"
        )

        assert not isinstance(
            test_loader.sampler,
            WeightedRandomSampler,
        ), (
            f"{client}: test loader unexpectedly uses WeightedRandomSampler"
        )

        assert train_loader.sampler.num_samples == len(
            train_loader.dataset
        ), (
            f"{client}: sampler num_samples does not equal train dataset length"
        )

        print("Sampler configuration: PASS")

        # --------------------------------------------------------
        # Read one actual item from each split
        # --------------------------------------------------------

        check_dataset_item(train_loader.dataset, f"{client} train")
        check_dataset_item(val_loader.dataset, f"{client} val")
        check_dataset_item(test_loader.dataset, f"{client} test")

        print("DRDataset item round-trip: PASS")

        # --------------------------------------------------------
        # Read one actual batch from each loader
        # --------------------------------------------------------

        check_loader_batch(train_loader, f"{client} train loader")
        check_loader_batch(val_loader, f"{client} val loader")
        check_loader_batch(test_loader, f"{client} test loader")

        print("DataLoader batch round-trip: PASS")

        # --------------------------------------------------------
        # Confirm expected default split sizes
        # --------------------------------------------------------

        expected = {
    "APTOS": (2930, 366, 366),
    "IDRiD": (330, 83, 103),
    "Messidor-2": (1396, 174, 174),
    "EyePACS": (26, 26, 52),
        }

        expected_train, expected_val, expected_test = expected[client]

        assert len(train_loader.dataset) == expected_train, (
            f"{client}: unexpected train size"
        )
        assert len(val_loader.dataset) == expected_val, (
            f"{client}: unexpected val size"
        )
        assert len(test_loader.dataset) == expected_test, (
            f"{client}: unexpected test size"
        )

        print("Expected split sizes: PASS")
        print(f"{client}: PASS")

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print("ALL DATALOADER INTEGRATION TESTS PASSED")


if __name__ == "__main__":
    main()