"""
Print raw vs final (capped) class weights for the four clients and the pooled
set, from their manifests' train split. Usage (repo root):

    python print_class_weight_table.py --manifest-root <folder with {client}_manifest.csv>
        [--cap 10.0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing.dataset import DEFAULT_CLASS_WEIGHT_CAP, NUM_DR_GRADES, compute_class_weights  # noqa: E402

CLIENTS = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")
GRADES = ("No DR", "Mild", "Moderate", "Severe", "Prolif")


def show(title: str, labels: np.ndarray, cap: float) -> None:
    counts = np.bincount(labels, minlength=NUM_DR_GRADES)
    n = int(counts.sum())
    final = compute_class_weights(labels, cap=cap)
    print(f"\n{title}: N = {n}")
    print(f"  {'class':<9}{'n_c':>8}{'p_c':>9}{'raw N/(K n_c)':>16}{'final':>10}  note")
    for c in range(NUM_DR_GRADES):
        raw = "absent" if counts[c] == 0 else f"{n / (NUM_DR_GRADES * counts[c]):.4f}"
        why = "absent -> 0" if counts[c] == 0 else ("capped" if final[c] >= cap else "")
        print(f"  {c} {GRADES[c]:<7}{counts[c]:>8}{counts[c] / n:>9.4f}{raw:>16}{final[c]:>10.4f}  {why}")
    present = counts > 0
    mean = float((counts[present] / n * final[present]).sum())
    base = n / (NUM_DR_GRADES * counts[present]) * (NUM_DR_GRADES / present.sum())  # uncapped, mean 1
    engaged = bool(base.max() > cap)
    print(f"  mean under own distribution = {mean:.9f}   max = {final.max():.4f}   cap {cap:g} engaged: {'YES' if engaged else 'no'}")
    if not present.all():
        print("  (absent class: final = N/(K_present n_c), i.e. raw x K/K_present; absent weight = 0, excluded from the mean)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-root", required=True)
    ap.add_argument("--cap", type=float, default=DEFAULT_CLASS_WEIGHT_CAP)
    args = ap.parse_args()
    pooled = []
    for name in CLIENTS:
        df = pd.read_csv(Path(args.manifest_root) / f"{name}_manifest.csv")
        labels = df.loc[df["split"] == "train", "label"].to_numpy().astype(np.int64)
        pooled.append(labels)
        show(name, labels, args.cap)
    show("POOLED train (Experiment 1)", np.concatenate(pooled), args.cap)


if __name__ == "__main__":
    main()
