"""
preprocessing/smote_features.py

STUB ONLY -- per Design Doc Part 1, Step 8 ("Feature-space SMOTE: Deferred
... only build this if per-class recall is still poor after steps 1-7")
and Part 3 ("smote_features.py -- stub only, build later if needed"), this
file intentionally contains ONLY the two finalized function signatures
below, each raising NotImplementedError. Do not implement the real logic
here unless explicitly requested later -- this is a project instruction,
not an oversight.

---------------------------------------------------------------------------
WHY THIS STAYS A STUB FOR NOW
---------------------------------------------------------------------------
Every other file in this pipeline (image_processing.py, augmentation.py,
dataset.py, build_dataset.py) operates on RAW OR CACHED IMAGES and has
everything it needs to run today. This file is structurally different: it
operates on FEATURE VECTORS PRODUCED BY A TRAINED MODEL
(extract_features(model, ...)) -- and no model exists yet. CoAtNet-1,
timm.create_model(...), and all training-loop code are explicitly out of
scope for the preprocessing stage of this project (see this project's
Section 12 constraints). So beyond the "deferred until needed" design
decision, this file is also chronologically blocked: it cannot be
meaningfully implemented before a client has a trained model checkpoint to
extract features from in the first place.

Part 1, Step 8's stated trigger for actually building this out: only if
per-class recall for the priority classes (Mild=1, Moderate=2) is still
inadequate AFTER steps 1-7 have already been tried and evaluated --
WeightedRandomSampler (dataset.py), the full augmentation stack
(augmentation.py), and MixUp/CutMix (training-loop, not yet written). In
other words, this is meant to be a targeted, evidence-based intervention
requested later, not a default part of the pipeline every client runs
through.

---------------------------------------------------------------------------
SKETCH OF THE EVENTUAL REAL IMPLEMENTATION (for whoever picks this up)
---------------------------------------------------------------------------
Not implemented below -- this is documentation only, so the eventual
implementation doesn't have to be re-derived from scratch:

- extract_features(): forward-pass a client's TRAIN-split dataloader
  through `model` up to its penultimate (pre-classification-head) pooled
  feature layer, in eval() mode, under `torch.no_grad()`, moving batches
  to `device`; collect and concatenate each batch's feature vectors and
  labels; return as numpy arrays (features: shape (N, feature_dim),
  labels: shape (N,)) since imbalanced-learn's SMOTE operates on numpy,
  not torch tensors.

- apply_smote(): fit imbalanced-learn's SMOTE (or a variant better suited
  to a 5-class, feature-space setting -- e.g. BorderlineSMOTE, or
  SMOTE with `sampling_strategy` targeted specifically at classes 1/2)
  on (features, labels), and return ONLY the newly synthesized
  minority-class vectors -- per the finalized signature
  "-> synthetic_features, synthetic_labels" -- leaving it to the caller
  to decide how to combine them with the real features (e.g. for a
  separate classifier-head fine-tuning pass).

- Per-client isolation still applies here, same as everywhere else in
  this project: extract_features()/apply_smote() would be called once per
  client, on that client's own train-split features only. Never pool
  feature vectors or synthetic samples across clients.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

__all__ = ["extract_features", "apply_smote"]


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device],
) -> Tuple[np.ndarray, np.ndarray]:
    """STUB -- not implemented. See module docstring.

    Intended eventual behavior: run `dataloader` through `model` in eval
    mode to collect penultimate-layer feature vectors and their true
    labels, for later use by apply_smote(). Deferred per Design Doc Part
    1, Step 8, and structurally blocked until a trained model exists
    (model/training code is out of scope for this preprocessing stage).

    Args:
        model: A trained model (e.g. CoAtNet-1), in eval mode.
        dataloader: A single client's TRAIN-split DataLoader (e.g. from
            dataset.get_dataloaders()) -- never a pooled, cross-client
            loader.
        device: The device to run inference on.

    Returns:
        (features, labels): features of shape (N, feature_dim), labels of
        shape (N,), both as numpy arrays.

    Raises:
        NotImplementedError: always, until this is explicitly requested.
    """
    raise NotImplementedError(
        "extract_features() is an intentional stub (Design Doc Part 1, Step 8: "
        "feature-space SMOTE is deferred until per-class recall for Mild/Moderate "
        "is still inadequate after WeightedRandomSampler + augmentation + MixUp/CutMix "
        "have been tried). It is also structurally blocked until a trained model "
        "checkpoint exists. Implement only when explicitly requested."
    )


def apply_smote(
    features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """STUB -- not implemented. See module docstring.

    Intended eventual behavior: fit a feature-space SMOTE variant (e.g.
    imbalanced-learn's SMOTE/BorderlineSMOTE) on (features, labels) for a
    single client's train split, and return only the newly synthesized
    minority-class samples -- per the finalized signature, not the
    originals concatenated in. Deferred per Design Doc Part 1, Step 8.

    Args:
        features: A single client's train-split feature vectors, shape
            (N, feature_dim) -- typically extract_features()'s output.
        labels: The corresponding true labels, shape (N,).

    Returns:
        (synthetic_features, synthetic_labels): the newly generated
        samples only, same feature_dim as the input.

    Raises:
        NotImplementedError: always, until this is explicitly requested.
    """
    raise NotImplementedError(
        "apply_smote() is an intentional stub (Design Doc Part 1, Step 8: "
        "feature-space SMOTE is deferred until per-class recall for Mild/Moderate "
        "is still inadequate after WeightedRandomSampler + augmentation + MixUp/CutMix "
        "have been tried). Implement only when explicitly requested -- likely via "
        "imbalanced-learn's SMOTE, applied per-client, never pooled across clients."
    )
