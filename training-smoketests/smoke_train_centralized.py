import sys, ast, random, shutil, tempfile
from pathlib import Path

import inspect, math
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import RandomSampler

import training.train_centralized as tc
from preprocessing.dataset import FINAL_MANIFEST_COLUMNS, compute_class_weights
from preprocessing.augmentation import get_eval_transforms, get_train_transforms

torch.set_num_threads(4)
TMP = Path(tempfile.mkdtemp(prefix="central_smoke_"))
RESULTS = []
CLIENTS = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")

_orig_create = tc.create_coatnet_lora_model
tc.create_coatnet_lora_model = lambda **kw: (torch.manual_seed(1234), _orig_create(pretrained=False, **{k: v for k, v in kw.items() if k != "pretrained"}))[1]
tc.get_train_transforms = get_eval_transforms


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, "PASS", ""))
        except Exception as e:
            RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:300]}"))
        return fn
    return deco


def raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise RuntimeError(f"expected {exc}, nothing raised")


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


def variant(name, mutate):
    d = TMP / name
    d.mkdir(exist_ok=True)
    out = {}
    for c, p in PATHS.items():
        df = mutate(c, pd.read_csv(p))
        out[c] = d / f"{c}.csv"
        df.to_csv(out[c], index=False)
    return out


PATHS = make_clients(TMP)
KW = dict(lora_rank=2, lora_alpha=4, physical_batch_size=4, accumulation_steps=2, effective_batch_size=8,
          num_workers=0, warmup_epochs=0.5, patience=5, seed=7)
S4_DIR = TMP / "s4"


def run(d, max_epochs=2, paths=None, **over):
    return tc.run_centralized_training(
        client_manifest_paths=paths or PATHS, checkpoint_dir=str(d), max_epochs=max_epochs, **{**KW, **over})


def train_labels():
    return np.concatenate([pd.read_csv(p).query("split == 'train'")["label"].to_numpy() for p in PATHS.values()])


@check("S1 interface: Mixup and WeightedRandomSampler gone, mixup_fn never passed, class_weights passed")
def _():
    tree = ast.parse(Path(tc.__file__).read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    kws = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords}
    assert not ({"Mixup", "WeightedRandomSampler"} & imported), imported
    assert "mixup_fn" not in kws and "class_weights" in kws, kws


@check("S2 compute_class_weights: expected weight 1, inverse-frequency order, zero-count class gets 0, bad labels rejected")
def _():
    labels = np.array([0] * 70 + [1] * 10 + [2] * 10 + [3] * 5 + [4] * 5)
    w = compute_class_weights(labels)
    p = np.bincount(labels, minlength=5) / len(labels)
    assert w.shape == (5,) and abs((p * w).sum() - 1) < 1e-9 and w[0] < w[1] < w[3]
    w2 = compute_class_weights(np.array([0, 0, 1, 2, 3]))
    assert w2[4] == 0 and abs((np.bincount([0, 0, 1, 2, 3], minlength=5) / 5 * w2).sum() - 1) < 1e-9
    raises(ValueError, lambda: compute_class_weights(np.array([0, 5])))
    raises(ValueError, lambda: compute_class_weights(np.array([], dtype=int)))


@check("S3 pooled loaders: shuffled (no weighted sampler), seeded generator, pooled sizes, reproducible order")
def _():
    tl, vl, _t = tc.build_pooled_dataloaders(PATHS, batch_size=4, num_workers=0, seed=3)
    assert isinstance(tl.sampler, RandomSampler) and tl.generator is not None
    assert len(tl.dataset) == 24 and len(vl.dataset) == 8
    order = lambda dl: [b[1].tolist() for b in dl]
    tl2, _v, _t2 = tc.build_pooled_dataloaders(PATHS, batch_size=4, num_workers=0, seed=3)
    assert order(tl) == order(tl2)


@check("S4 end-to-end 2 epochs: result, checkpoints, run_config, class weights, step cadence, best-before-latest order")
def _():
    seen, saves, steps = {}, [], {"n": 0}
    orig_rte, orig_save = tc.run_training_epoch, tc.save_checkpoint

    def spy_rte(*a, **kw):
        opt = a[2] if len(a) > 2 else kw["optimizer"]
        if not getattr(opt, "_counted", False):
            orig_step = opt.step

            def counted(*aa, **kk):
                steps["n"] += 1
                return orig_step(*aa, **kk)
            opt.step = counted
            opt._counted = True
        seen.update(kw)
        return orig_rte(*a, **kw)

    tc.run_training_epoch = spy_rte
    tc.save_checkpoint = lambda path, **kw: (saves.append(Path(path).name), orig_save(path, **kw))[1]
    try:
        res = run(S4_DIR)
    finally:
        tc.run_training_epoch, tc.save_checkpoint = orig_rte, orig_save
    assert {"best_val_macro_f1", "final_epoch", "best_checkpoint_path"} <= set(res) and res["final_epoch"] == 2
    assert (S4_DIR / "latest_centralized.pt").exists() and Path(res["best_checkpoint_path"]).exists()
    ck = torch.load(str(S4_DIR / "latest_centralized.pt"), weights_only=False)
    rc = ck["run_config"]
    need = {"lora_rank", "lora_alpha", "img_size", "num_classes", "physical_batch_size", "accumulation_steps",
            "effective_batch_size", "lr_lora", "lr_head", "weight_decay", "warmup_epochs", "warmup_start_factor",
            "max_epochs", "patience", "label_smoothing", "seed", "pooled_train_class_counts", "manifest_sha256"}
    assert ck["schema_version"] == 2 and rc["experiment_id"] == "exp1" and need <= set(rc), need - set(rc)
    assert "mixup_fn" not in seen and seen["accumulation_steps"] == 2 and seen["scheduler"] is not None
    want = torch.as_tensor(compute_class_weights(train_labels()), dtype=torch.float64)
    assert seen["class_weights"].dtype == torch.float32
    assert torch.allclose(seen["class_weights"].cpu().double(), want, atol=1e-6)
    assert steps["n"] == 6, steps["n"]
    assert "best_centralized.pt" in saves, saves
    for i, n in enumerate(saves):
        if n == "best_centralized.pt":
            assert saves[i + 1] == "latest_centralized.pt", saves


@check("S5 resume equivalence: killed after epoch 2 of 4, restart == straight 4-epoch run (weights, metrics)")
def _():
    class Kill(BaseException):
        pass

    def killer(at):
        def cb(epoch, metrics):
            if epoch == at:
                raise Kill()
        return cb

    da, db = TMP / "straight", TMP / "resumed"
    ra = run(da, max_epochs=4)
    raises(Kill, lambda: run(db, max_epochs=4, on_epoch_end=killer(2)))
    rb = run(db, max_epochs=4)

    sa = torch.load(
        str(da / "latest_centralized.pt"),
        weights_only=False
    )["model_state_dict"]

    sb = torch.load(
        str(db / "latest_centralized.pt"),
        weights_only=False
    )["model_state_dict"]

    diff = max((sa[k] - sb[k]).abs().max().item() for k in sa)

    print(
        f"   max weight diff {diff:.2e}; "
        f"best F1 straight {ra['best_val_macro_f1']:.4f} "
        f"resumed {rb['best_val_macro_f1']:.4f}"
    )

    assert sa.keys() == sb.keys() and diff < 1e-6
    assert (
        ra["final_epoch"] == rb["final_epoch"] == 4
        and abs(ra["best_val_macro_f1"] - rb["best_val_macro_f1"]) < 1e-9
    )

@check("S6 resume guards: finished run trains nothing; exhausted patience trains nothing; changed hyperparameter rejected")
def _():
    d = TMP / "s6"
    shutil.copytree(S4_DIR, d)
    calls = []
    cb = lambda e, m: calls.append(e)
    res = run(d, on_epoch_end=cb)
    assert calls == [] and res["final_epoch"] == 2, (calls, res)
    p = d / "latest_centralized.pt"
    ck = torch.load(str(p), weights_only=False)
    ck["extra"]["epoch"] = 1
    ck["extra"]["epochs_without_improvement"] = KW["patience"]
    torch.save(ck, str(p))
    res = run(d, on_epoch_end=cb)
    assert calls == [] and res["final_epoch"] == 1 and res["best_val_macro_f1"] == ck["extra"]["best_val_macro_f1"]
    raises(ValueError, lambda: run(d, lr_lora=5e-4))


@check("S7 a changed manifest (same shapes, one label flipped) is rejected on resume")
def _():
    d = TMP / "s7"
    shutil.copytree(S4_DIR, d)

    def flip(c, df):
        if c == "APTOS":
            df.loc[0, "label"] = 1
        return df

    raises(ValueError, lambda: run(d, paths=variant("alt", flip)))


@check("S8 img_size other than 224 is rejected before any work")
def _():
    raises(ValueError, lambda: run(TMP / "s8", img_size=384))


@check("S9 real training augmentation is reproducible with the same seed (num_workers=0)")
def _():
    tc.get_train_transforms = get_train_transforms
    try:
        a = next(iter(tc.build_pooled_dataloaders(PATHS, batch_size=4, num_workers=0, seed=3)[0]))[0]
        b = next(iter(tc.build_pooled_dataloaders(PATHS, batch_size=4, num_workers=0, seed=3)[0]))[0]
        c = next(iter(tc.build_pooled_dataloaders(PATHS, batch_size=4, num_workers=0, seed=4)[0]))[0]
    finally:
        tc.get_train_transforms = get_eval_transforms
    assert torch.equal(a, b) and not torch.equal(a, c)

@check("S10 effective batch is enforced: 4 x 3 != 8 raises before any checkpoint is written")
def _():
    d = TMP / "s10"
    raises(ValueError, lambda: run(d, physical_batch_size=4, accumulation_steps=3))
    assert not (d / "latest_centralized.pt").exists()


@check("S11 exactly the four locked clients are required (missing or extra client rejected)")
def _():
    raises(ValueError, lambda: run(TMP / "s11a", paths={k: PATHS[k] for k in list(PATHS)[:3]}))
    raises(ValueError, lambda: run(TMP / "s11b", paths={**PATHS, "Extra": PATHS["APTOS"]}))


@check("S12 a pooled train class that is completely absent is rejected (not just warned)")
def _():
    def drop4(c, df):
        df.loc[(df["split"] == "train") & (df["label"] == 4), "label"] = 3
        return df

    raises(ValueError, lambda: run(TMP / "s12", paths=variant("no4", drop4)))


@check("S13 a failing on_epoch_end callback does not stop training or prevent checkpoints")
def _():
    calls = []

    def bad(epoch, metrics):
        calls.append(epoch)
        raise RuntimeError("logging backend down")

    d = TMP / "s13"
    res = run(d, on_epoch_end=bad)
    assert calls == [1, 2] and res["final_epoch"] == 2 and (d / "latest_centralized.pt").exists()


@check("S14 epoch metrics: LRs follow the real schedule, the val confusion matrix reproduces val_macro_f1, checkpoint extra is untouched")
def _():
    got = []
    d = TMP / "s14"
    run(d, max_epochs=3, on_epoch_end=lambda e, m: got.append((e, dict(m))))
    assert [e for e, _ in got] == [1, 2, 3]
    wsf = inspect.signature(tc.run_centralized_training).parameters["warmup_start_factor"].default
    warm = max(round(KW["warmup_epochs"] * 3), 1)
    opt = AdamW([{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-4},
                 {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3}])
    sch = SequentialLR(opt, [LinearLR(opt, start_factor=wsf, end_factor=1.0, total_iters=warm),
                             CosineAnnealingLR(opt, T_max=max(9 - warm, 1))], milestones=[warm])
    for e, m in got:
        assert {"epoch", "train_loss", "val_loss", "val_macro_f1", "lr_lora", "lr_head", "val_confusion_matrix"} <= set(m)
        for _ in range(3):
            opt.step(); sch.step()
        assert abs(m["lr_lora"] - opt.param_groups[0]["lr"]) < 1e-12, (e, m["lr_lora"])
        assert abs(m["lr_head"] - opt.param_groups[1]["lr"]) < 1e-12, (e, m["lr_head"])
        cm = m["val_confusion_matrix"]
        assert len(cm) == 25 and all(type(v) is int and v >= 0 for v in cm) and sum(cm) == 8
        a = np.array(cm).reshape(5, 5)
        f1s = []
        for c in range(5):
            tp = a[c, c]; fp = a[:, c].sum() - tp; fn = a[c, :].sum() - tp
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        assert abs(np.mean(f1s) - m["val_macro_f1"]) < 1e-9, (e, np.mean(f1s), m["val_macro_f1"])
    ck = torch.load(str(d / "latest_centralized.pt"), weights_only=False)
    assert set(ck["extra"]) == {"epoch", "best_val_macro_f1", "epochs_without_improvement"}, set(ck["extra"])


@check("S15 a failure building the confusion matrix is logged and skipped; training and checkpoints continue")
def _():
    orig, got = tc._confusion_matrix_flat, []

    def boom(*a, **k):
        raise RuntimeError("cm broke")

    tc._confusion_matrix_flat = boom
    try:
        d = TMP / "s15"
        res = run(d, max_epochs=2, on_epoch_end=lambda e, m: got.append(dict(m)))
    finally:
        tc._confusion_matrix_flat = orig
    assert res["final_epoch"] == 2 and len(got) == 2
    assert all("val_confusion_matrix" not in m and "lr_lora" in m for m in got)
    assert (d / "latest_centralized.pt").exists() and (d / "best_centralized.pt").exists()


@check("S16 patched run still resumes exactly: kill after epoch 1 of 3, restart == straight run")
def _():
    class Kill(BaseException):
        pass

    def killer(e, m):
        if e == 1:
            raise Kill()

    da, db = TMP / "s16a", TMP / "s16b"
    run(da, max_epochs=3)
    raises(Kill, lambda: run(db, max_epochs=3, on_epoch_end=killer))
    run(db, max_epochs=3)
    sa = torch.load(str(da / "latest_centralized.pt"), weights_only=False)["model_state_dict"]
    sb = torch.load(str(db / "latest_centralized.pt"), weights_only=False)["model_state_dict"]
    assert max((sa[k] - sb[k]).abs().max().item() for k in sa) < 1e-6

print()
for n, s, msg in RESULTS:
    print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))
raise SystemExit(any(s == "FAIL" for _, s, _ in RESULTS))