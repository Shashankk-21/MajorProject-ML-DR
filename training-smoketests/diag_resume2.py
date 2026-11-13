import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image

import training.train_centralized as tc
from preprocessing.dataset import FINAL_MANIFEST_COLUMNS
from preprocessing.augmentation import get_eval_transforms

torch.set_num_threads(4)
TMP = Path(tempfile.mkdtemp(prefix="central_diag2_"))
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


class Rec:
    def __init__(self, dl, log):
        self.dl, self.log = dl, log

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        order = []
        for batch in self.dl:
            order.append(batch[1].tolist())
            yield batch
        self.log.append(order)


class Trace:
    def __init__(self):
        self.params, self.orders, self.at_save2, self.after_load = {}, [], None, None


def snap(model, optimizer, scheduler, gens):
    o = optimizer.state_dict()
    return {
        "lr": [g["lr"] for g in o["param_groups"]],
        "opt_state": {k: {n: (v.clone() if torch.is_tensor(v) else v) for n, v in s.items()} for k, s in o["state"].items()},
        "sched": scheduler.state_dict() if scheduler is not None else None,
        "gens": {n: g.get_state().clone() for n, g in (gens or {}).items()},
        "rng": torch.get_rng_state().clone(),
        "params": torch.cat([p.detach().flatten().clone() for p in model.parameters() if p.requires_grad]),
    }


def same(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def run(d, tr, max_epochs=4, kill_at=None):
    o_save, o_load, o_rte = tc.save_checkpoint, tc.load_checkpoint, tc.run_training_epoch

    def save(path, **kw):
        if Path(path).name == "latest_centralized.pt":
            ep = kw["extra"]["epoch"]
            tr.params[ep] = torch.cat([p.detach().flatten().clone() for p in kw["model"].parameters() if p.requires_grad])
            if ep == 2:
                tr.at_save2 = snap(kw["model"], kw["optimizer"], kw.get("scheduler"), kw.get("extra_generators"))
        return o_save(path, **kw)

    def load(path, **kw):
        r = o_load(path, **kw)
        tr.after_load = snap(kw["model"], kw["optimizer"], kw.get("scheduler"), kw.get("extra_generators"))
        return r

    def rte(*a, **kw):
        a = list(a)
        if len(a) > 1:
            a[1] = Rec(a[1], tr.orders)
        else:
            kw["dataloader"] = Rec(kw["dataloader"], tr.orders)
        return o_rte(*a, **kw)

    def cb(epoch, metrics):
        if kill_at is not None and epoch == kill_at:
            raise Kill()

    tc.save_checkpoint, tc.load_checkpoint, tc.run_training_epoch = save, load, rte
    try:
        try:
            tc.run_centralized_training(client_manifest_paths=PATHS, checkpoint_dir=str(d), max_epochs=max_epochs,
                                        on_epoch_end=cb, **KW)
        except Kill:
            pass
    finally:
        tc.save_checkpoint, tc.load_checkpoint, tc.run_training_epoch = o_save, o_load, o_rte


A, B, C = Trace(), Trace(), Trace()
run(TMP / "a", A)
run(TMP / "b", B)
run(TMP / "c", C, kill_at=2)
run(TMP / "c", C)

d = lambda x, y: f"{(x - y).abs().max().item():.2e}"
print("R1 run-to-run (two straight 4-epoch runs), weight diff at epochs 2,3,4:", [d(A.params[e], B.params[e]) for e in (2, 3, 4)])
print("R2 shuffle orders identical across the two straight runs:", A.orders == B.orders)
print("R3 resumed vs straight weight diff at epochs 3,4:", [d(C.params[e], A.params[e]) for e in (3, 4)])
print("R4 shuffle order, resumed vs straight, epochs 3 and 4 identical:", [C.orders[i] == A.orders[i] for i in (2, 3)])
for key in ("lr", "opt_state", "sched", "gens", "rng", "params"):
    print(f"R5 {key:9} straight-at-epoch-2 == restored-after-load:", same(A.at_save2[key], C.after_load[key]))