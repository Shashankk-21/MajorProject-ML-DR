import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.augmentation import get_train_transforms
from preprocessing.dataset import make_augmentation_worker_init_fn


class SameImage(Dataset):
    def __init__(self, n, transform):
        self.n, self.transform = n, transform
        self.img = np.random.default_rng(1).integers(0, 255, (224, 224, 3), dtype=np.uint8)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        out = self.transform(image=self.img)["image"]
        w = torch.utils.data.get_worker_info()
        return torch.as_tensor(np.asarray(out)), (w.id if w else -1)


def build(seed):
    return DataLoader(SameImage(24, get_train_transforms()), batch_size=1, shuffle=False, num_workers=2,
                      generator=torch.Generator().manual_seed(seed),
                      worker_init_fn=make_augmentation_worker_init_fn())


def collect(dl):
    by = {}
    for x, w in dl:
        by.setdefault(int(w[0]), []).append(x[0].clone())
    return by


if __name__ == "__main__":
    dl = build(0)
    e1, e2 = collect(dl), collect(dl)
    ids = sorted(e1)
    a, b = e1[ids[0]], e1[ids[1]]
    print("workers:", ids, "samples per worker:", {k: len(v) for k, v in e1.items()})
    print("1) k-th output identical between worker 0 and worker 1:", [torch.equal(a[k], b[k]) for k in range(min(len(a), len(b)))])
    print("2) worker 0 consecutive outputs differ:", [not torch.equal(a[k], a[k + 1]) for k in range(len(a) - 1)])
    print("3) worker 0 epoch 2 identical to epoch 1:", [torch.equal(e1[ids[0]][k], e2[ids[0]][k]) for k in range(len(a))])
    f1 = collect(build(0))
    print("4) a second loader with the same seed reproduces epoch 1 exactly:", all(torch.equal(e1[w][k], f1[w][k]) for w in ids for k in range(len(e1[w]))))