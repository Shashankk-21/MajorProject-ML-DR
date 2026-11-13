import sys, json, tempfile, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score

import federated.client as fc
import federated.server as sv
from flwr.app import Context, MetricRecord, RecordDict
from preprocessing.dataset import FINAL_MANIFEST_COLUMNS
from preprocessing.augmentation import get_eval_transforms
from utils.privacy_calibration import load_calibration

torch.set_num_threads(4)
TMP = Path(tempfile.mkdtemp(prefix="server_smoke_"))
CKPT = TMP / "ckpt"
CAL = TMP / "cal_dp.json"
RESULTS = []
N_TRAIN, N_VAL = 12, 5
NAMES = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")

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


class Crash(BaseException):
    pass


ROOT = TMP / "manifests"
(ROOT / "img").mkdir(parents=True)
for ci, name in enumerate(NAMES):
    rng = np.random.default_rng(ci)
    rows = []
    for split, n in (("train", N_TRAIN), ("val", N_VAL), ("test", 5)):
        for j in range(n):
            p = ROOT / "img" / f"{name}_{split}_{j}.png"
            Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(p)
            row = {c: "na" for c in FINAL_MANIFEST_COLUMNS}
            row.update(image_id=f"{name}_{split}_{j}", cached_image_path=str(p), label=j % 5, split=split)
            rows.append(row)
    pd.DataFrame(rows, columns=list(FINAL_MANIFEST_COLUMNS)).to_csv(ROOT / f"{name}_manifest.csv", index=False)


def scfg(eid, **over):
    cfg = {
        "experiment-id": eid, "algorithm": "fedavg", "dp": False, "run-seed": 7, "num-rounds": 2,
        "local-epochs": 1, "lora-rank": 2, "lora-alpha": 4, "img-size": 224, "num-classes": 5,
        "physical-batch-size": 4, "accumulation-steps": 2, "effective-batch-size": 8,
        "label-smoothing": 0.1, "weight-decay": 0.01, "num-workers": 0, "manifest-root": str(ROOT),
        "lr-lora-peak": 1e-2, "lr-head-peak": 1e-2, "pretrained": False,
        "checkpoint-root": str(CKPT), "timeout": 600.0,
    }
    cfg.update(over)
    return cfg


def dcfg(eid, **over):
    cfg = scfg(eid, algorithm="fedprox", dp=True, **{
        "proximal-mu": 0.01, "calibration-path": str(CAL), "accountant": "rdp", "target-epsilon": 5.0,
        "delta": 1e-5, "max-grad-norm": 1.0, "grad-sample-mode": "functorch",
        "dp-logical-batch-size": 64, "effective-batch-size": 64})
    cfg.update(over)
    return cfg


class FakeGrid:
    NODES = {101: 0, 102: 1, 103: 2, 104: 3}

    def __init__(self, crash_train_call=None, drop_partition=None):
        self.crash_train_call, self.drop_partition = crash_train_call, drop_partition
        self.calls = 0
        self.train_calls = 0
        self.train_cfgs, self.train_out, self.eval_cms = [], [], []

    def get_node_ids(self):
        return list(self.NODES)

    def send_and_receive(self, messages, timeout=None, **kw):
        messages = list(messages)
        self.calls += 1
        kind = messages[0].metadata.message_type if messages else None
        if kind == "train":
            self.train_calls += 1
            if self.crash_train_call == self.train_calls:
                raise Crash()
        replies, rec = [], {}
        for msg in messages:
            node = msg.metadata.dst_node_id
            pid = self.NODES[node]
            ctx = Context(run_id=1, node_id=node, node_config={"partition-id": pid}, state=RecordDict(), run_config={})
            cfg = dict(msg.content["config"])
            if kind == "train":
                self.train_cfgs.append((self.train_calls, pid, cfg))
                rep = fc._train_handler(msg, ctx)
                c = rep.content
                rec[pid] = {"n": int(c["metrics"]["num-examples"]),
                            "state": {k: v.clone() for k, v in c["arrays"].to_torch_state_dict().items()}}
                if pid != self.drop_partition:
                    replies.append(rep)
            else:
                rep = fc._evaluate_handler(msg, ctx)
                rec[pid] = [int(v) for v in rep.content["metrics"]["confusion-matrix"]]
                replies.append(rep)
        (self.train_out if kind == "train" else self.eval_cms).append(rec)
        return replies


def by_round(g):
    out = {}
    for n, pid, cfg in g.train_cfgs:
        out.setdefault(n, {})[pid] = cfg
    return out


def load(p):
    return torch.load(str(p), weights_only=False)


def rounds(eid):
    return sorted((CKPT / eid).glob("round_*.pt"))


def maxdiff(a, b):
    return max((a[k].double() - b[k].double()).abs().max().item() for k in a)


def pooled_ref(cms):
    cm = np.sum([np.array(c).reshape(5, 5) for c in cms], axis=0)
    yt = np.repeat(np.arange(5), cm.sum(axis=1))
    yp = np.concatenate([np.repeat(np.arange(5), cm[c]) for c in range(5)])
    return cm, f1_score(yt, yp, average="macro", labels=np.arange(5), zero_division=0)


@check("V1 round_lr: warmup 0.5, peak at round 2, 0.1 at round 10, never rises")
def _():
    peak = 1e-3
    lr = [sv.round_lr(r, 10, peak) for r in range(1, 11)]
    assert abs(lr[0] - 0.5 * peak) < 1e-12 and abs(lr[1] - peak) < 1e-12 and abs(lr[9] - 0.1 * peak) < 1e-12
    assert all(lr[i + 1] <= lr[i] + 1e-15 for i in range(1, 9))
    for n in (1, 2, 3):
        assert all(sv.round_lr(r, n, peak) > 0 for r in range(1, n + 1))


@check("V2 pooled metrics match sklearn; the result is a valid Flower MetricRecord; bad inputs rejected")
def _():
    rng = np.random.default_rng(3)
    per, cms = {}, []
    for pid in range(4):
        cm = rng.integers(1, 12, (5, 5))
        if pid == 1:
            cm[2] = 0
        cms.append(cm.flatten().tolist())
        per[pid] = {"confusion-matrix": cm.flatten().tolist(), "num-examples": int(cm.sum()), "val-loss": float(rng.random())}
    m = sv.pooled_eval_metrics(per)
    cm, want = pooled_ref(cms)
    assert abs(m["macro-f1-pooled"] - want) < 1e-9
    for c in range(5):
        assert abs(m[f"recall-pooled-class-{c}"] - (cm[c, c] / cm[c].sum() if cm[c].sum() else 0.0)) < 1e-12
    assert list(m["confusion-matrix-pooled"]) == cm.flatten().tolist()
    client_f1 = [pooled_ref([c])[1] for c in cms]
    assert abs(m["macro-f1-client-mean"] - float(np.mean(client_f1))) < 1e-9
    MetricRecord(m)
    raises(Exception, lambda: sv.pooled_eval_metrics({0: per[0]}))
    bad = {**per, 0: {**per[0], "num-examples": per[0]["num-examples"] + 1}}
    raises(Exception, lambda: sv.pooled_eval_metrics(bad))


@check("V3 server config validation: good configs pass, 12 bad configs are rejected")
def _():
    sv.validate_server_config(scfg("v3a"))
    sv.validate_server_config(dcfg("v3b"))
    drop = lambda cfg, k: {a: b for a, b in cfg.items() if a != k}
    bad = [
        scfg("v3", **{"proximal-mu": 0.01}),
        scfg("v3", algorithm="fedprox"),
        scfg("../x"),
        scfg("v3", **{"num-rounds": 0}),
        scfg("v3", **{"img-size": 384}),
        scfg("v3", algorithm="sgd"),
        drop(scfg("v3"), "lr-lora-peak"),
        dcfg("v3", **{"grad-sample-mode": "hooks"}),
        dcfg("v3", **{"effective-batch-size": 32}),
        dcfg("v3", **{"dp-logical-batch-size": 32, "effective-batch-size": 32}),
        drop(dcfg("v3"), "calibration-path"),
        drop(dcfg("v3"), "max-grad-norm"),
    ]
    for c in bad:
        raises(Exception, lambda c=c: sv.validate_server_config(c))


@check("V4 FedAvg 2 rounds on the real Flower loop: files, client contract, aggregation, pooled metrics")
def _():
    cfg = scfg("fedavg-a")
    g = FakeGrid()
    out = sv.run_experiment(cfg, g)
    d = CKPT / "fedavg-a"
    names = sorted(p.name for p in d.iterdir())
    print("   files:", names)
    assert {"latest.pt", "best.pt", "done.json"} <= set(names) and len(rounds("fedavg-a")) == 2
    by = by_round(g)
    assert sorted(by) == [1, 2] and all(len(v) == 4 for v in by.values())
    for r in (1, 2):
        for c in by[r].values():
            assert c["round"] == r and c["num-rounds"] == 2 and c["algorithm"] == "fedavg" and "proximal-mu" not in c
            assert not json.loads(c.get("accountant-states") or "{}")
            assert abs(c["lr-lora"] - sv.round_lr(r, 2, 1e-2)) < 1e-12 and abs(c["lr-head"] - sv.round_lr(r, 2, 1e-2)) < 1e-12
    assert abs(by[1][0]["lr-lora"] - 0.5e-2) < 1e-12
    for r, path in zip((1, 2), rounds("fedavg-a")):
        ck = load(path)
        recs = g.train_out[r - 1]
        tot = sum(v["n"] for v in recs.values())
        exp = {k: sum(v["n"] * v["state"][k].double() for v in recs.values()) / tot for k in recs[0]["state"]}
        got = ck["model_state_dict"]
        assert set(got) == set(exp) and max((exp[k] - got[k].double()).abs().max().item() for k in exp) < 1e-5
        assert ck["round_offset"] == r and ck["opacus_accountant_states"] is None and ck["extra"]["round"] == r
    ex = load(d / "latest.pt")["extra"]
    cm, want = pooled_ref(g.eval_cms[1].values())
    assert ex["selection-metric-name"] == "macro-f1-pooled" and abs(ex["eval"]["macro-f1-pooled"] - want) < 1e-9
    assert load(d / "best.pt")["round_offset"] == ex["best-round"] == out.get("best-round", ex["best-round"])
    assert json.loads((d / "done.json").read_text())


@check("V5 crash in round 2 then resume: true round 2 with the TOTAL num-rounds; result equals the straight run; done run is a no-op")
def _():
    cfg = scfg("fedavg-b")
    raises(Crash, lambda: sv.run_experiment(cfg, FakeGrid(crash_train_call=2)))
    d = CKPT / "fedavg-b"
    assert load(d / "latest.pt")["round_offset"] == 1 and not (d / "done.json").exists()
    g2 = FakeGrid()
    sv.run_experiment(cfg, g2)
    assert g2.train_calls == 1
    for c in by_round(g2)[1].values():
        assert c["round"] == 2 and c["num-rounds"] == 2 and abs(c["lr-lora"] - sv.round_lr(2, 2, 1e-2)) < 1e-12
    a, b = load(d / "latest.pt"), load(CKPT / "fedavg-a" / "latest.pt")
    diff = maxdiff(a["model_state_dict"], b["model_state_dict"])
    print(f"   resumed vs straight max weight diff {diff:.2e}")
    print("V5 diagnostic:", {
        "max_weight_diff": diff,
        "round_offset": a["round_offset"],
        "done_exists": (d / "done.json").exists(),
        "weight_check": diff < 1e-6,
    })
    r1a = load(CKPT / "fedavg-a" / "round_001.pt")["model_state_dict"]
    r1b = load(d / "round_001.pt")["model_state_dict"]

    print(
        "V5 round-1 max weight diff:",
        f"{maxdiff(r1a, r1b):.2e}",
    )

    largest = sorted(
        (
            float(
                (
                    a["model_state_dict"][k].double()
                    - b["model_state_dict"][k].double()
                ).abs().max().item()
            ),
            k,
        )
        for k in a["model_state_dict"]
    )
    print("V5 largest differing tensors:", largest[-8:])
    assert diff < 1e-6 and a["round_offset"] == 2 and (d / "done.json").exists()
    g3 = FakeGrid()
    sv.run_experiment(cfg, g3)
    assert g3.calls == 0


@check("V6 a dropped client reply aborts the round and writes no checkpoint")
def _():
    g = FakeGrid(drop_partition=2)
    raises(Exception, lambda: sv.run_experiment(scfg("drop-a"), g))
    assert not (CKPT / "drop-a" / "latest.pt").exists()


@check("V7 resume with a changed rank, local epochs, seed or LR is refused before any client call")
def _():
    cfg = scfg("refuse-a")
    raises(Crash, lambda: sv.run_experiment(cfg, FakeGrid(crash_train_call=2)))
    for over in ({"lora-rank": 4, "lora-alpha": 8}, {"local-epochs": 2}, {"run-seed": 8}, {"lr-lora-peak": 5e-3}):
        g = FakeGrid()
        raises(Exception, lambda o=over: sv.run_experiment({**cfg, **o}, g))
        assert g.calls == 0, over


@check("V8 FedProx + DP: mu sent every round, no accountant states in round 1, all four in round 2, calibration created and reused")
def _():
    cfg = dcfg("dp-a")
    g = FakeGrid()
    sv.run_experiment(cfg, g)
    by = by_round(g)
    cal = load_calibration(str(CAL))
    assert cal["accountant"] == "rdp" and all(cal["clients"][n]["dataset_size"] == N_TRAIN for n in NAMES)
    for r in (1, 2):
        for c in by[r].values():
            assert c["proximal-mu"] == 0.01 and c["algorithm"] == "fedprox" and c["accountant"] == "rdp"
            assert c["grad-sample-mode"] == "functorch" and c["dp-logical-batch-size"] == 64 and c["target-epsilon"] == 5.0
    assert all(not json.loads(c.get("accountant-states") or "{}") for c in by[1].values())
    st2 = json.loads(by[2][0]["accountant-states"])
    assert sorted(st2) == ["0", "1", "2", "3"] and all(sum(h[2] for h in s["history"]) == 1 for s in st2.values())
    last = load(rounds("dp-a")[-1])
    acc = last["opacus_accountant_states"]
    assert sorted(acc) == ["0", "1", "2", "3"] and all(sum(h[2] for h in s["history"]) == 2 for s in acc.values())
    print("   privacy:", last["extra"]["privacy"])
    assert last["extra"]["privacy"] is not None


@check("V9 DP crash in round 2 then resume: equals the straight run, accountant steps are 2 (not double counted)")
def _():
    cfg = dcfg("dp-b")
    raises(Crash, lambda: sv.run_experiment(cfg, FakeGrid(crash_train_call=2)))
    sv.run_experiment(cfg, FakeGrid())
    a, b = load(CKPT / "dp-b" / "latest.pt"), load(CKPT / "dp-a" / "latest.pt")
    diff = maxdiff(a["model_state_dict"], b["model_state_dict"])
    print(f"   resumed vs straight max weight diff {diff:.2e}")
    assert diff < 1e-6
    assert all(sum(h[2] for h in s["history"]) == 2 for s in a["opacus_accountant_states"].values())


@check("V18 per-client confusion matrices reach the callback and the round checkpoint, equal the client replies, sum to pooled")
def _():
    eid, grid, seen = "cm-client", FakeGrid(), []
    sv.run_experiment(scfg(eid), grid, on_round_end=lambda r, t, e, p: seen.append((r, e)))
    assert [r for r, _ in seen] == [1, 2]
    for i, (r, ev) in enumerate(seen):
        tot = np.zeros(25, dtype=np.int64)
        for pid in range(4):
            got = ev[f"confusion-matrix-client-{pid}"]
            assert got == grid.eval_cms[i][pid], (r, pid)
            tot += np.array(got)
        assert tot.tolist() == ev["confusion-matrix-pooled"]
    ev2 = load(CKPT / eid / "round_002.pt")["extra"]["eval"]
    assert all(f"confusion-matrix-client-{p}" in ev2 for p in range(4))


@check("V19 run hook through the real ServerApp main: callback gets both rounds, run id is in every checkpoint, finish(summary, None)")
def _():
    eid, seen = "hook-ok", {"rounds": [], "finish": None, "cfg": None}

    def hook(cfg):
        seen["cfg"] = cfg
        return (lambda r, t, e, p: seen["rounds"].append(r)), "wb-run-123", lambda s, e: seen.update(finish=(s, e))

    sv.set_run_hook(hook)
    try:
        sv.main(FakeGrid(), Context(run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config=scfg(eid)))
    finally:
        sv.set_run_hook(None)
    assert seen["rounds"] == [1, 2]
    assert seen["finish"][1] is None and seen["finish"][0]["done"] is True
    assert seen["cfg"]["experiment-id"] == eid and seen["cfg"]["warmup-rounds"] == 1
    for p in rounds(eid) + [CKPT / eid / "latest.pt", CKPT / eid / "best.pt"]:
        assert load(p)["wandb_run_id"] == "wb-run-123", p


@check("V20 a hook named by the 'run-hook' run-config key is used; the key never enters the checkpoint run_config")
def _():
    eid, seen = "hook-cfgkey", {}
    mod = types.ModuleType("dr_hook_cfg_mod")
    mod.make = lambda cfg: (None, "from-cfg", lambda s, e: seen.update(ok=(s is not None and e is None)))
    sys.modules["dr_hook_cfg_mod"] = mod
    try:
        sv.main(FakeGrid(), Context(
            run_id=1,
            node_id=0,
            node_config={},
            state=RecordDict(),
            run_config=scfg(eid, **{"run-hook": "dr_hook_cfg_mod:make"}),
        ))
    finally:
        sys.modules.pop("dr_hook_cfg_mod", None)
    ck = load(CKPT / eid / "latest.pt")
    assert seen.get("ok") and ck["wandb_run_id"] == "from-cfg" and "run-hook" not in ck["run_config"]


@check("V21 broken or malformed hook stops the experiment before any round; an invalid config never reaches the hook")
def _():
    called = []

    def bad_hook(cfg):
        called.append("bad")
        raise RuntimeError("wandb down")

    try:
        sv.set_run_hook(bad_hook)
        grid = FakeGrid()
        raises(Exception, lambda: sv.run_experiment_tracked(scfg("hook-broken"), grid))
        assert called == ["bad"] and grid.calls == 0 and not rounds("hook-broken")
        sv.set_run_hook(lambda cfg: ("not callable", 5, None))
        grid = FakeGrid()
        raises(Exception, lambda: sv.run_experiment_tracked(scfg("hook-malformed"), grid))
        assert grid.calls == 0 and not rounds("hook-malformed")
        called.clear()
        sv.set_run_hook(lambda cfg: called.append("never") or (None, None, None))
        raises(ValueError, lambda: sv.run_experiment_tracked(scfg("hook-invalid", **{"lora-rank": 0}), FakeGrid()))
        assert called == []
    finally:
        sv.set_run_hook(None)


@check("V22 crash during the run: finish gets (None, error) and the crash propagates; a failing finish is swallowed")
def _():
    seen = {}
    try:
        sv.set_run_hook(lambda cfg: (None, "id-x", lambda s, e: seen.update(s=s, e=e)))
        raises(Crash, lambda: sv.run_experiment_tracked(scfg("hook-crash"), FakeGrid(crash_train_call=1)))
        assert seen["s"] is None and isinstance(seen["e"], Crash)

        def bad_finish(s, e):
            raise RuntimeError("finish broke")

        sv.set_run_hook(lambda cfg: (None, "id-y", bad_finish))
        out = sv.run_experiment_tracked(scfg("hook-badfinish"), FakeGrid())
        assert out["done"] is True
    finally:
        sv.set_run_hook(None)


@check("V23 the server records the LRs it broadcast: the round checkpoint's train report equals the config sent to the clients")
def _():
    eid, grid = "lr-recorded", FakeGrid()
    sv.run_experiment(scfg(eid), grid)
    sent = by_round(grid)
    for r in (1, 2):
        tr = load(CKPT / eid / f"round_{r:03d}.pt")["extra"]["train"]
        assert tr["lr-lora"] == sent[r][0]["lr-lora"] and tr["lr-head"] == sent[r][0]["lr-head"], r
        assert abs(tr["lr-lora"] - sv.round_lr(r, 2, 1e-2)) < 1e-12


print()
for n, s, msg in RESULTS:
    print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))
raise SystemExit(any(s == "FAIL" for _, s, _ in RESULTS))