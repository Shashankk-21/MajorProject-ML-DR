import sys, json, tempfile, time

from collections import OrderedDict

from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import numpy as np

import pandas as pd

import torch

from PIL import Image

from peft import get_peft_model_state_dict

from sklearn.metrics import f1_score



import federated.client as fc

from preprocessing.dataset import FINAL_MANIFEST_COLUMNS

from preprocessing.augmentation import get_eval_transforms

from utils.dp_checks import DPGradientCheckError

from utils.privacy_calibration import calibrate_client_noise_multipliers, save_calibration



torch.set_num_threads(4)

TMP = Path(tempfile.mkdtemp(prefix="client_smoke_"))

RESULTS = []

N_TRAIN, N_VAL = 12, 5



_orig_create = fc.create_coatnet_lora_model

fc.create_coatnet_lora_model = lambda **kw: (torch.manual_seed(1234), _orig_create(**{**kw, "pretrained": False}))[1]

_orig_gdl = fc.get_dataloaders

fc.get_dataloaders = lambda client_name, **kw: _orig_gdl(

    client_name, train_transform=get_eval_transforms(), eval_transform=get_eval_transforms(), **kw)





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





def make_manifest(root, name):

    rng = np.random.default_rng(0)

    rows = []

    (root / "img").mkdir(exist_ok=True)

    for split, n in (("train", N_TRAIN), ("val", N_VAL), ("test", 5)):

        for j in range(n):

            p = root / "img" / f"{name}_{split}_{j}.png"

            Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(p)

            row = {c: "na" for c in FINAL_MANIFEST_COLUMNS}

            row.update(image_id=f"{name}_{split}_{j}", cached_image_path=str(p), label=j % 5, split=split)

            rows.append(row)

    pd.DataFrame(rows, columns=list(FINAL_MANIFEST_COLUMNS)).to_csv(root / f"{name}_manifest.csv", index=False)





ROOT = TMP / "manifests"

ROOT.mkdir()

make_manifest(ROOT, "IDRiD")





def cfg_base(**over):

    cfg = {

        "experiment-id": "smoke", "algorithm": "fedavg", "dp": False, "run-seed": 7, "num-rounds": 2,

        "lora-rank": 2, "lora-alpha": 4, "img-size": 224, "num-classes": 5, "pretrained": False,

        "local-epochs": 2, "physical-batch-size": 4, "accumulation-steps": 2, "effective-batch-size": 8,

        "label-smoothing": 0.1, "weight-decay": 0.01, "num-workers": 0, "manifest-root": str(ROOT),

        "round": 1, "lr-lora": 1e-2, "lr-head": 1e-2,

    }

    cfg.update(over)

    return cfg





def make_global(rank=2):

    m = fc.create_coatnet_lora_model(lora_rank=rank, lora_alpha=2 * rank, num_classes=5, img_size=224, pretrained=False)

    return {k: v.detach().clone() for k, v in get_peft_model_state_dict(m).items()}





GLOBAL = make_global()





def rt(cfg, state=None, partition=1, acc=None):

    return fc.train_client_round(partition_id=partition, cfg=cfg,

                                 global_state=GLOBAL if state is None else state, accountant_states=acc)





def dist(a, b):

    return sum(((a[k] - b[k]) ** 2).sum().item() for k in b)





def make_cal(path, rounds, local, eps=5.0, accountant="rdp"):

    res = calibrate_client_noise_multipliers(

        target_epsilon=eps, client_train_sizes={"IDRiD": N_TRAIN}, accountant=accountant,

        logical_batch_size=64, epochs=rounds * local)

    save_calibration(str(path), target_epsilon=eps, clients=res, accountant=accountant, logical_batch_size=64,

                     epochs=rounds * local, target_delta=1e-5, epsilon_tolerance=0.01)

    return res["IDRiD"]





CAL, CAL2 = TMP / "cal.json", TMP / "cal2.json"

ENTRY = make_cal(CAL, rounds=2, local=1)

make_cal(CAL2, rounds=1, local=2)





def cfg_dp(**over):

    cfg = cfg_base(dp=True, **{

        "local-epochs": 1, "calibration-path": str(CAL), "max-grad-norm": 1.0, "delta": 1e-5,

        "grad-sample-mode": "functorch", "dp-logical-batch-size": 64, "accountant": "rdp",

        "target-epsilon": 5.0, "effective-batch-size": 64})

    cfg.update(over)

    return cfg





R1 = {}





@check("C1 client_name_for and derive_seed")

def _():

    assert [fc.client_name_for(i) for i in range(4)] == ["APTOS", "IDRiD", "Messidor-2", "EyePACS"]

    for bad in (4, -1, "1", 1.0, True, None):

        raises(ValueError, lambda b=bad: fc.client_name_for(b))

    s = fc.derive_seed(7, 1, 1)

    assert s == fc.derive_seed(7, 1, 1) and 0 <= s < 2 ** 32

    seeds = {fc.derive_seed(r, t, p) for r in (1, 2) for t in range(1, 11) for p in range(4)}

    assert len(seeds) == 80, "seed collision across runs, rounds or clients"

    raises(ValueError, lambda: fc.derive_seed(7.5, 1, 1))





@check("C2 config validation rejects 13 bad configs before any work")

def _():

    bad = [

        cfg_base(algorithm="fedprox"),

        cfg_base(algorithm="fedprox", **{"proximal-mu": 0.0}),

        cfg_base(**{"proximal-mu": 0.01}),

        cfg_base(algorithm="sgd"),

        cfg_base(**{"img-size": 384}),

        cfg_base(**{"num-classes": 4}),

        cfg_base(**{"effective-batch-size": 16}),

        cfg_base(**{"label-smoothing": 1.0}),

        cfg_base(**{"lora-rank": True}),

        cfg_base(**{"lr-lora": 0.0}),

        cfg_base(dp=True),

        {k: v for k, v in cfg_base().items() if k != "manifest-root"},

        cfg_base(round=0),

    ]

    for c in bad:

        raises(ValueError, lambda c=c: rt(c))





@check("C3 non-DP round: keys, shapes, dtype, weights moved, metrics, no accountant state")

def _():

    t0 = time.time()

    r = rt(cfg_base())

    print(f"   {time.time() - t0:.1f}s  metrics {r['metrics']}")

    assert set(r["state"]) == set(GLOBAL) and len(GLOBAL) == 66

    assert all(r["state"][k].dtype == torch.float32 and r["state"][k].device.type == "cpu" for k in GLOBAL)

    assert all(tuple(r["state"][k].shape) == tuple(GLOBAL[k].shape) for k in GLOBAL)

    assert any(not torch.equal(r["state"][k], GLOBAL[k]) for k in GLOBAL if "lora_B" in k)

    assert any(not torch.equal(r["state"][k], GLOBAL[k]) for k in GLOBAL if "head" in k)

    assert r["num_examples"] == N_TRAIN and r["accountant_state"] is None

    assert {"train-loss", "local-epochs"} <= set(r["metrics"]) and r["metrics"]["train-loss"] > 0





@check("C4 same inputs give identical results; C5 a different round gives a different result")

def _():

    a, b = rt(cfg_base()), rt(cfg_base())

    assert max((a["state"][k] - b["state"][k]).abs().max().item() for k in GLOBAL) < 1e-6

    c = rt(cfg_base(round=2))

    print(f"   distance round1 vs round2 states: {dist(a['state'], c['state']):.3e}")

    assert dist(a["state"], c["state"]) > 0





@check("C6 FedProx (mu=100) ends closer to the global weights than FedAvg")

def _():

    avg = rt(cfg_base())

    prox = rt(cfg_base(algorithm="fedprox", **{"proximal-mu": 100.0}))

    da, dp_ = dist(avg["state"], GLOBAL), dist(prox["state"], GLOBAL)

    print(f"   distance to global: fedavg {da:.4e}  fedprox {dp_:.4e}")

    assert dp_ < da





@check("C7 strict global-state load: wrong rank, missing head, extra key all rejected")

def _():

    raises(ValueError, lambda: rt(cfg_base(), state=make_global(rank=4)))

    raises(ValueError, lambda: rt(cfg_base(), state={k: v for k, v in GLOBAL.items() if "head" not in k}))

    raises(ValueError, lambda: rt(cfg_base(), state={**GLOBAL, "bogus.weight": torch.zeros(1)}))





@check("C8 isolation: only the client's own manifest is read (others absent); another client's is refused")

def _():

    assert sorted(p.name for p in ROOT.glob("*_manifest.csv")) == ["IDRiD_manifest.csv"]

    raises(FileNotFoundError, lambda: rt(cfg_base(), partition=0))





@check("C9 evaluation: 25-int confusion matrix, recalls, macro-F1 consistent, deterministic")

def _():

    e = fc.evaluate_client_round(partition_id=1, cfg=cfg_base(), global_state=GLOBAL)

    m = e["metrics"]

    cm = np.array(m["confusion-matrix"]).reshape(5, 5)

    assert e["num_examples"] == N_VAL == int(cm.sum()) and all(isinstance(v, int) for v in m["confusion-matrix"])

    for c in range(5):

        want = cm[c, c] / cm[c].sum() if cm[c].sum() else 0.0

        assert abs(m[f"recall-class-{c}"] - want) < 1e-12

    y_true = np.concatenate([np.full(int(cm[c].sum()), c) for c in range(5)])

    y_pred = np.concatenate([np.repeat(np.arange(5), cm[c]) for c in range(5)])

    assert abs(m["val-macro-f1"] - f1_score(y_true, y_pred, average="macro", labels=np.arange(5), zero_division=0)) < 1e-9

    assert {"partition-id", "val-loss"} <= set(m)

    assert fc.evaluate_client_round(partition_id=1, cfg=cfg_base(), global_state=GLOBAL)["metrics"] == m





@check("C10 DP rounds 1 and 2: accountant state carried forward, steps and epsilon accounted")

def _():

    r1 = rt(cfg_dp())

    R1.update(r1)

    m1, st1 = r1["metrics"], r1["accountant_state"]

    print("   round 1:", {k: m1[k] for k in ("epsilon", "steps-this-round", "steps-total", "noise-multiplier", "sample-rate")})

    assert st1 is not None and sum(h[2] for h in st1["history"]) == ENTRY["num_batches"] == m1["steps-total"] == m1["steps-this-round"]

    assert 0 < m1["epsilon"] < 5.0

    r2 = rt(cfg_dp(round=2), state=r1["state"], acc={"1": st1})

    m2 = r2["metrics"]

    print("   round 2:", {k: m2[k] for k in ("epsilon", "steps-this-round", "steps-total")})

    assert m2["steps-total"] == 2 * ENTRY["num_batches"] and m2["steps-this-round"] == ENTRY["num_batches"]

    assert m2["epsilon"] > m1["epsilon"] and 4.98 <= m2["epsilon"] <= 5.0 + 1e-6

    assert all(bool(torch.isfinite(v).all()) for v in r2["state"].values())





@check("C11 DP guards: lost state in round 2, overspent budget, wrong epsilon, wrong accountant, hooks mode")
def _():
    r1 = R1 or rt(cfg_dp())
    st1 = r1["accountant_state"]
    raises((ValueError, fc.PrivacyBudgetError), lambda: rt(cfg_dp(round=2), state=r1["state"]))
    over = json.loads(json.dumps(st1))
    over["history"][0][2] = ENTRY["steps"]
    raises(fc.PrivacyBudgetError, lambda: rt(cfg_dp(round=2), state=r1["state"], acc={"1": over}))
    raises(ValueError, lambda: rt(cfg_dp(**{"target-epsilon": 1.0})))
    raises(ValueError, lambda: rt(cfg_dp(accountant="prv")))
    raises(ValueError, lambda: rt(cfg_dp(**{"grad-sample-mode": "hooks"})))



@check("C12 FedProx under DP: the proximal pull is present (2 local steps, mu=100, same seeds)")

def _():

    kw = {"num-rounds": 1, "local-epochs": 2, "calibration-path": str(CAL2)}

    avg = rt(cfg_dp(**kw))

    prox = rt(cfg_dp(algorithm="fedprox", **{**kw, "proximal-mu": 100.0}))

    da, dp_ = dist(avg["state"], GLOBAL), dist(prox["state"], GLOBAL)

    print(f"   distance to global: fedavg+DP {da:.4e}  fedprox+DP {dp_:.4e}")

    assert dp_ < da





@check("C13 Flower handlers (flwr 1.30 here): reply layout, record round trips, accountant state for round 2")

def _():

    assert fc.app is not None and hasattr(fc, "train") and hasattr(fc, "evaluate")

    from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict



    def request(cfg, state=None):

        arrays = ArrayRecord(OrderedDict((k, v) for k, v in (GLOBAL if state is None else state).items()))

        return Message(content=RecordDict({"arrays": arrays, "config": ConfigRecord(cfg)}), dst_node_id=1, message_type="train")



    ctx = Context(run_id=1, node_id=1, node_config={"partition-id": 1}, state=RecordDict(), run_config={})

    rep = fc._train_handler(request(cfg_base()), ctx).content

    assert "arrays" in rep and "metrics" in rep and "accountant" not in rep

    mm = rep["metrics"]

    assert mm["num-examples"] == N_TRAIN and mm["partition-id"] == 1 and mm["train-loss"] > 0

    assert set(rep["arrays"].to_torch_state_dict()) == set(GLOBAL)

    ev = fc._evaluate_handler(request(cfg_base()), ctx).content["metrics"]

    assert len(ev["confusion-matrix"]) == 25 and sum(ev["confusion-matrix"]) == N_VAL

    d1 = fc._train_handler(request(cfg_dp()), ctx).content

    assert "accountant" in d1 and d1["accountant"]["partition-id"] == 1

    st = json.loads(d1["accountant"]["accountant"])

    assert sum(h[2] for h in st["history"]) == ENTRY["num_batches"]

    state2 = d1["arrays"].to_torch_state_dict()

    d2 = fc._train_handler(request(cfg_dp(round=2, **{"accountant-states": json.dumps({"1": st})}), state2), ctx).content

    assert d2["metrics"]["steps-total"] == 2 * ENTRY["num_batches"]





@check("C14 a round outside the configured run is rejected (num-rounds is 2)")
def _():
    raises(ValueError, lambda: rt(cfg_base(round=3)))
    raises(ValueError, lambda: rt(cfg_dp(round=3)))



print()

for n, s, msg in RESULTS:

    print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))

raise SystemExit(any(s == "FAIL" for _, s, _ in RESULTS))