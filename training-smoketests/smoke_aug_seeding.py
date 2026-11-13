import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image

import training.train_centralized as tc
from preprocessing.dataset import FINAL_MANIFEST_COLUMNS, get_dataloaders

CLIENTS = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")


def make_clients(root, n_train=6, n_val=2, n_test=2):
    rng = np.random.default_rng(0)
    paths, k = {}, 0
    for name in CLIENTS:
        rows = []
        for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
            for j in range(n):
                p = root / f"{name}_{split}_{j}.png"
                Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(p)
                row = {c: "na" for c in FINAL_MANIFEST_COLUMNS}
                row.update(image_id=f"{name}_{split}_{j}", cached_image_path=str(p), label=(k + j) % 5, split=split)
                rows.append(row)
            k += 1
        mp = root / f"{name}.csv"
        pd.DataFrame(rows, columns=list(FINAL_MANIFEST_COLUMNS)).to_csv(mp, index=False)
        paths[name] = mp
    return paths


def first(loader, k=3):
    out = []
    for i, (x, y) in enumerate(loader):
        out.append((x.clone(), y.clone()))
        if i + 1 == k:
            break
    return out


def same(a, b):
    return all(torch.equal(p[0], q[0]) and torch.equal(p[1], q[1]) for p, q in zip(a, b))


def main():
    results = []

    def check(name, fn):
        try:
            fn()
            results.append((name, "PASS", ""))
        except Exception as e:
            results.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:300]}"))

    manifest = Path("data") / "splits" / "IDRiD_manifest.csv"
    paths = make_clients(Path(tempfile.mkdtemp(prefix="aug_smoke_")))

    def gd(workers, seed):
        return get_dataloaders("IDRiD", manifest_path=manifest, batch_size=4, num_workers=workers, seed=seed)

    def pooled(workers, seed):
        return tc.build_pooled_dataloaders(paths, batch_size=4, num_workers=workers, seed=seed)

    def wiring():
        tr, va, te = gd(2, 1)
        assert tr.worker_init_fn is not None and va.worker_init_fn is None and te.worker_init_fn is None
        ptr, pva, pte = pooled(2, 1)
        assert ptr.worker_init_fn is not None and pva.worker_init_fn is None and pte.worker_init_fn is None

    def repro(build, workers):
        a, b, c = first(build(workers, 5)[0]), first(build(workers, 5)[0]), first(build(workers, 6)[0])
        assert same(a, b), "same seed did not reproduce the first batches"
        assert not same(a, c), "different seeds produced identical batches"

    check("A1 train loaders get worker_init_fn, val/test do not (get_dataloaders and pooled)", wiring)
    check("A2 get_dataloaders, 2 workers: same seed reproduces, different seed differs", lambda: repro(gd, 2))
    check("A3 get_dataloaders, 0 workers: same seed reproduces, different seed differs", lambda: repro(gd, 0))
    check("A4 pooled loader, 2 workers: same seed reproduces, different seed differs", lambda: repro(pooled, 2))
    check("A5 pooled loader, 0 workers: same seed reproduces, different seed differs", lambda: repro(pooled, 0))

    print()
    for n, s, msg in results:
        print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))
    raise SystemExit(any(s == "FAIL" for _, s, _ in results))


if __name__ == "__main__":
    main()