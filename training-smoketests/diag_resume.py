import sys, tempfile, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image

import training.train_centralized as tc
from preprocessing.dataset import FINAL_MANIFEST_COLUMNS
from preprocessing.augmentation import get_eval_transforms, get_train_transforms

torch.set_num_threads(4)
TMP = Path(tempfile.mkdtemp(prefix="central_diag_"))
CLIENTS = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")

_orig = tc.create_coatnet_lora_model
tc.create_coatnet_lora_model = lambda **kw: _orig(pretrained=False, **{k: v for k, v in kw.items() if k != "pretrained"})
tc.get_train_transforms = get_eval_transforms


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


PATHS = make_clients(TMP)
KW = dict(lora_rank=2, lora_alpha=4, physical_batch_size=4, accumulation_steps=2, effective_batch_size=8,
          num_workers=0, warmup_epochs=0.5, patience=5, seed=7)


class Kill(BaseException):
    pass


def loader_images(seed, transform_fn):
    tc.get_train_transforms = transform_fn
    np.random.seed(0); random.seed(0); torch.manual_seed(0)
    tl, _, _ = tc.build_pooled_dataloaders(PATHS, batch_size=4, num_workers=0, seed=seed)
    out = torch.cat([b[0] for b in tl])
    tc.get_train_transforms = get_eval_transforms
    return out


print("D1 eval transforms: loader images identical across two builds:",
      torch.equal(loader_images(3, get_eval_transforms), loader_images(3, get_eval_transforms)))
print("D1 real train transforms: loader images identical across two builds:",
      torch.equal(loader_images(3, get_train_transforms), loader_images(3, get_train_transforms)))


def traced(d, max_epochs, kill_at=None):
    snaps = []
    orig = tc.save_checkpoint

    def spy(path, **kw):
        if Path(path).name == "latest_centralized.pt":
            snaps.append(torch.cat([p.detach().flatten().clone() for p in kw["model"].parameters() if p.requires_grad]))
        return orig(path, **kw)

    def cb(epoch, metrics):
        if kill_at is not None and epoch == kill_at:
            raise Kill()

    tc.save_checkpoint = spy
    try:
        try:
            tc.run_centralized_training(client_manifest_paths=PATHS, checkpoint_dir=str(d), max_epochs=max_epochs,
                                        on_epoch_end=cb, **KW)
        except Kill:
            pass
    finally:
        tc.save_checkpoint = orig
    return snaps


S = traced(TMP / "straight", 4)
S2 = traced(TMP / "straight2", 2)
K = traced(TMP / "resume", 4, kill_at=2)
R = traced(TMP / "resume", 4)
d = lambda x, y: f"{(x - y).abs().max().item():.2e}"
print("snapshots: straight", len(S), "| straight again", len(S2), "| killed run", len(K), "| resumed run", len(R))
print("D2 run-to-run (two straight runs, same seed), epochs 1-2:", [d(S[i], S2[i]) for i in range(2)])
print("D3 killed run vs straight, epochs 1-2:", [d(K[i], S[i]) for i in range(2)])
print("D4 resumed run vs straight, epochs 3-4:", [d(R[i], S[i + 2]) for i in range(2)])