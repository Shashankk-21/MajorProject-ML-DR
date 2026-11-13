import ast
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score

import utils.experiment_tracker as et

TMP = Path(tempfile.mkdtemp(prefix="tracker_smoke_"))
RESULTS = []
NAMES = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")
for _k in ("WANDB_API_KEY", "WANDB_MODE", "WANDB_ENTITY"):
    os.environ.pop(_k, None)


class Skip(Exception):
    pass


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, "PASS", ""))
        except Skip as e:
            RESULTS.append((name, "SKIP", str(e)))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:300]}"))
        return fn
    return deco


def raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise RuntimeError(f"expected {exc}, nothing raised")


class FakeRun:
    def __init__(self, kw, log_fails=False):
        self.kw, self.id = kw, kw.get("id") or "fake0001"
        self.logs, self.summary, self.exit, self.finished = [], {}, None, 0
        self.log_fails = log_fails

    def log(self, data, step=None):
        if self.log_fails:
            raise RuntimeError("log backend down")
        self.logs.append((step, dict(data)))

    def finish(self, exit_code=None):
        self.exit, self.finished = exit_code, self.finished + 1


class FakeWandb:
    class Table:
        def __init__(self, columns=None, data=None):
            self.columns, self.data = columns, data

    def __init__(self, fail_online=False, fail_offline=False, log_fails=False, table_fails=False):
        self.fail_online, self.fail_offline, self.log_fails = fail_online, fail_offline, log_fails
        self.inits, self.runs = [], []
        if table_fails:
            class Bad:
                def __init__(self, *a, **k):
                    raise RuntimeError("table broke")
            self.Table = Bad

    def init(self, **kw):
        self.inits.append(dict(kw))
        if kw.get("mode") == "offline":
            if self.fail_offline:
                raise RuntimeError("offline down")
        elif self.fail_online:
            raise RuntimeError("no network")
        run = FakeRun(kw, self.log_fails)
        self.runs.append(run)
        return run


def labels_from(cm):
    m = np.asarray(cm).reshape(5, 5)
    yt, yp = [], []
    for i in range(5):
        for j in range(5):
            yt += [i] * int(m[i, j])
            yp += [j] * int(m[i, j])
    return np.array(yt), np.array(yp)


def rand_cm(rng, sparse=0.4):
    cm = rng.integers(0, 7, size=(5, 5)) * (rng.random((5, 5)) > sparse)
    cm[rng.integers(0, 5), rng.integers(0, 5)] += 1
    return cm.flatten().tolist()


CM = [3, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 1]
EV = {"macro-f1-pooled": 0.5, "macro-f1-client-mean": 0.4, "val-loss-pooled": 1.2, "num-examples": 20,
      "confusion-matrix-pooled": CM}
for _p in range(4):
    EV[f"confusion-matrix-client-{_p}"] = CM
    EV[f"macro-f1-client-{_p}"] = 0.3 + _p / 100
    EV[f"val-loss-client-{_p}"] = 1.0 + _p / 10
    for _c in range(5):
        EV[f"recall-client-{_p}-class-{_c}"] = 0.5
for _c in range(5):
    EV[f"recall-pooled-class-{_c}"] = 0.5
TR = {"aggregated": {"train-loss": 0.9, "local-epochs": 1.0, "partition-id": 1.5},
      "per-client": {p: {"train-loss": 0.8 + p / 10, "num-examples": 10 + p} for p in range(4)},
      "lr-lora": 1e-4, "lr-head": 1e-3}
PRIV = {k: {p: float(p + 1) for p in range(4)}
        for k in ("epsilon", "noise-multiplier", "sample-rate", "steps-this-round", "steps-total")}
PRIV["max-epsilon"] = 4.0


@check("T1 f1 / recall / QWK equal sklearn on 300 random matrices (empty classes included); degenerate QWK is 0.0")
def _():
    rng = np.random.default_rng(0)
    for _ in range(300):
        cm = rand_cm(rng)
        yt, yp = labels_from(cm)
        assert np.allclose(et.f1_per_class(cm), f1_score(yt, yp, labels=range(5), average=None, zero_division=0))
        assert np.allclose(et.recall_per_class(cm), recall_score(yt, yp, labels=range(5), average=None, zero_division=0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = cohen_kappa_score(yt, yp, labels=list(range(5)), weights="quadratic")
        q = et.quadratic_weighted_kappa(cm)
        assert (abs(q - k) < 1e-9) if np.isfinite(k) else q == 0.0, (cm, q, k)
    one = [0] * 25
    one[0] = 5
    assert et.quadratic_weighted_kappa(one) == 0.0 and et.quadratic_weighted_kappa([0] * 25) == 0.0
    perfect = [0] * 25
    for i in range(5):
        perfect[i * 6] = 3
    assert abs(et.quadratic_weighted_kappa(perfect) - 1.0) < 1e-12 and et.f1_per_class(perfect) == [1.0] * 5


@check("T2 bad matrices are rejected: wrong length, 2-D, negative, fractional, NaN, bool, strings")
def _():
    for bad in ([0] * 24, [0] * 26, [[0] * 5] * 5, [-1] + [0] * 24, [0.5] + [0] * 24, [float("nan")] + [0] * 24,
                [True] * 25, ["a"] * 25, None):
        for fn in (et.f1_per_class, et.recall_per_class, et.quadratic_weighted_kappa):
            raises((ValueError, TypeError), lambda: fn(bad))
    assert et.f1_per_class([float(v) for v in CM]) == et.f1_per_class(CM)  # whole-number floats are fine


@check("T3 round_metrics: exact key set for a full DP payload, values equal the references, partition-id never logged")
def _():
    d = et.round_metrics(TR, EV, PRIV, NAMES)
    want = {"val/macro-f1-pooled", "val/macro-f1-client-mean", "val/loss-pooled", "val/qwk-pooled",
            "train/loss", "train/lr-lora", "train/lr-head", "privacy/max-epsilon"}
    for c in range(5):
        want |= {f"val/f1-pooled/class-{c}", f"val/recall-pooled/class-{c}"}
    for n in NAMES:
        want |= {f"val/macro-f1/{n}", f"val/loss/{n}", f"val/qwk/{n}", f"train/loss/{n}"}
        for c in range(5):
            want |= {f"val/f1/{n}/class-{c}", f"val/recall/{n}/class-{c}"}
        for k in ("epsilon", "noise-multiplier", "sample-rate", "steps-this-round", "steps-total"):
            want.add(f"privacy/{k}/{n}")
    assert set(d) == want, (sorted(set(d) - want), sorted(want - set(d)))
    yt, yp = labels_from(CM)
    assert np.allclose([d[f"val/f1-pooled/class-{c}"] for c in range(5)], f1_score(yt, yp, labels=range(5), average=None, zero_division=0))
    assert abs(d["val/qwk-pooled"] - cohen_kappa_score(yt, yp, weights="quadratic")) < 1e-9
    assert d["val/macro-f1-pooled"] == 0.5 and d["val/loss/IDRiD"] == 1.1 and d["train/loss"] == 0.9
    assert d["train/lr-lora"] == 1e-4 and d["train/lr-head"] == 1e-3
    assert d["privacy/epsilon/EyePACS"] == 4.0 and d["privacy/max-epsilon"] == 4.0
    assert all(isinstance(v, float) for v in d.values()) and not any("partition" in k for k in d)
    assert not any(k.startswith("privacy/") for k in et.round_metrics(TR, EV, None, NAMES))
    assert "val/f1/client0/class-0" in et.round_metrics(TR, EV, None, None)  # no names: client<pid>


@check("T4 round_metrics fallbacks: missing or bad matrix, missing aggregated loss, empty payloads never raise")
def _():
    ev = {k: v for k, v in EV.items() if not k.startswith("confusion-matrix")}
    d = et.round_metrics(TR, ev, None, NAMES)
    assert "val/qwk-pooled" not in d and "val/f1-pooled/class-0" not in d
    assert d["val/recall-pooled/class-0"] == 0.5 and d["val/recall/APTOS/class-3"] == 0.5
    ev2 = dict(EV, **{"confusion-matrix-pooled": [1, 2, 3], "confusion-matrix-client-0": "garbage"})
    d2 = et.round_metrics(TR, ev2, None, NAMES)
    assert "val/qwk-pooled" not in d2 and "val/qwk/APTOS" not in d2 and "val/qwk/IDRiD" in d2
    assert d2["val/recall/APTOS/class-0"] == 0.5
    tr2 = {"aggregated": {}, "per-client": TR["per-client"]}
    want = sum((0.8 + p / 10) * (10 + p) for p in range(4)) / sum(10 + p for p in range(4))
    assert abs(et.round_metrics(tr2, EV, None, NAMES)["train/loss"] - want) < 1e-12
    for args in ((None, None, None), ({}, {}, {}), (TR, None, None), (None, EV, None)):
        assert isinstance(et.round_metrics(*args, NAMES), dict)


@check("T5 epoch_metrics: Experiment 1 keys reuse the pooled names; missing or bad matrix is skipped quietly")
def _():
    m = {"epoch": 3, "train_loss": 1.0, "val_loss": 1.1, "val_macro_f1": 0.4, "lr_lora": 1e-4, "lr_head": 1e-3,
         "val_confusion_matrix": CM}
    d = et.epoch_metrics(m)
    assert d["train/loss"] == 1.0 and d["val/loss-pooled"] == 1.1 and d["val/macro-f1-pooled"] == 0.4
    assert d["train/lr-lora"] == 1e-4 and d["train/lr-head"] == 1e-3 and "val/qwk-pooled" in d
    assert all(f"val/recall-pooled/class-{c}" in d and f"val/f1-pooled/class-{c}" in d for c in range(5))
    assert "epoch" not in d and not any("confusion" in k for k in d)
    no_cm = {k: v for k, v in m.items() if k != "val_confusion_matrix"}
    assert "val/qwk-pooled" not in et.epoch_metrics(no_cm) and et.epoch_metrics(None) == {}
    assert "val/qwk-pooled" not in et.epoch_metrics(dict(m, val_confusion_matrix=[1, 2]))


@check("T6 start / log / finish on a fake wandb: init kwargs, steps, status and exit codes, finish is idempotent")
def _():
    fw = FakeWandb()
    cfg = {"experiment-id": "exp4-fedprox-eps50", "algorithm": "fedprox", "dp": True, "diagnostic": True,
           "num-rounds": 10, "lora-rank": 8, "nested": {"a": 1}, "wandb-api-key": "SECRET", "auth-token": "T"}
    tr = et.ExperimentTracker(wandb_module=fw, client_names=NAMES)
    rid = tr.start(cfg)
    kw = fw.inits[0]
    assert len(fw.inits) == 1 and kw["project"] == et.PROJECT and kw["entity"] == et.DEFAULT_ENTITY
    assert kw["name"] == "exp4-fedprox-eps50" and kw["id"] == rid and len(rid) == 8 and "resume" not in kw
    assert kw["tags"] == ["fedprox", "dp", "diagnostic"] and "mode" not in kw
    assert kw["config"]["lora-rank"] == 8 and kw["config"]["nested"] == "{'a': 1}"
    assert "wandb-api-key" not in kw["config"] and "auth-token" not in kw["config"]
    assert tr.start(cfg) == rid and len(fw.inits) == 1  # second start ignored
    tr.log_round(3, TR, EV, PRIV)
    run = fw.runs[0]
    assert run.logs[0][0] == 3 and "val/macro-f1-pooled" in run.logs[0][1]
    tr.finish({"done": True, "rounds-run": 10, "wandb-run-id": rid, "extra": {"x": 1}}, None)
    assert run.summary["status"] == "finished" and run.summary["result/rounds-run"] == 10 and run.exit == 0
    assert "result/wandb-run-id" not in run.summary and "result/extra" not in run.summary
    tr.finish(None, RuntimeError("late"))
    assert run.finished == 1 and run.summary["status"] == "finished"


@check("T7 finish(None, error) marks the run failed with the error text; finish() with no arguments is a clean finish")
def _():
    fw = FakeWandb()
    tr = et.ExperimentTracker(wandb_module=fw)
    tr.start({"experiment-id": "exp2-fedavg"})
    tr.finish(None, ValueError("x" * 5000))
    run = fw.runs[0]
    assert run.summary["status"] == "failed" and run.summary["error"].startswith("ValueError: xxx")
    assert len(run.summary["error"]) <= 2000 and run.exit == 1
    fw2 = FakeWandb()
    t2 = et.ExperimentTracker(wandb_module=fw2)
    t2.start({"experiment-id": "exp1-centralized"})
    t2.finish()
    assert fw2.runs[0].summary["status"] == "finished" and fw2.runs[0].exit == 0
    fw3 = FakeWandb()
    t3 = et.ExperimentTracker(wandb_module=fw3)
    t3.start({"experiment-id": "exp2-fedavg"})
    t3.finish(None, None)
    assert fw3.runs[0].summary["status"] == "failed" and fw3.runs[0].exit == 1


@check("T8 resume: a stored run id is resumed with resume='must' and returned unchanged")
def _():
    fw = FakeWandb()
    tr = et.ExperimentTracker(wandb_module=fw)
    rid = tr.start({"experiment-id": "exp2-fedavg"}, resume_run_id="abc12345")
    assert rid == "abc12345" and fw.inits[0]["id"] == "abc12345" and fw.inits[0]["resume"] == "must"
    fw2 = FakeWandb()
    t2 = et.ExperimentTracker(wandb_module=fw2)
    assert t2.start({"experiment-id": "exp2-fedavg"}, resume_run_id="") != "" and "resume" not in fw2.inits[0]


@check("T9 online failure retries offline (resume 'allow'); both failing raises TrackerStartError; the retry keeps the same id")
def _():
    logging.disable(logging.CRITICAL)
    try:
        fw = FakeWandb(fail_online=True)
        tr = et.ExperimentTracker(wandb_module=fw)
        rid = tr.start({"experiment-id": "exp2-fedavg"})
        assert tr.mode == "offline" and [i.get("mode") for i in fw.inits] == [None, "offline"]
        assert fw.inits[0]["id"] == fw.inits[1]["id"] == rid
        fw2 = FakeWandb(fail_online=True)
        t2 = et.ExperimentTracker(wandb_module=fw2)
        t2.start({"experiment-id": "exp2-fedavg"}, resume_run_id="abc12345")
        assert fw2.inits[0]["resume"] == "must" and fw2.inits[1]["resume"] == "allow" and fw2.inits[1]["mode"] == "offline"
        t3 = et.ExperimentTracker(wandb_module=FakeWandb(fail_online=True, fail_offline=True))
        raises(et.TrackerStartError, lambda: t3.start({"experiment-id": "exp2-fedavg"}))
    finally:
        logging.disable(logging.NOTSET)


@check("T10 no wandb (ImportError) or WANDB_MODE=disabled: no-op that still returns an 8-hex id; logging and finish never raise")
def _():
    logging.disable(logging.CRITICAL)
    prev = sys.modules.get("wandb", "absent")
    try:
        sys.modules["wandb"] = None  # makes `import wandb` raise ImportError
        tr = et.ExperimentTracker()
        rid = tr.start({"experiment-id": "exp2-fedavg"})
        assert tr.mode == "noop" and len(rid) == 8 and int(rid, 16) >= 0
        tr.log_round(1, TR, EV, PRIV)
        tr.log_epoch(1, {"train_loss": 1.0})
        tr.finish({"done": True}, None)
        sys.modules["wandb"] = FakeWandb()
        os.environ["WANDB_MODE"] = "disabled"
        t2 = et.ExperimentTracker()
        assert len(t2.start({"experiment-id": "exp2-fedavg"})) == 8 and t2.mode == "noop"
    finally:
        os.environ.pop("WANDB_MODE", None)
        if prev == "absent":
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = prev
        logging.disable(logging.NOTSET)


@check("T11 logging failures are swallowed: run.log raising, table building raising (scalars still logged)")
def _():
    logging.disable(logging.CRITICAL)
    try:
        fw = FakeWandb(log_fails=True)
        tr = et.ExperimentTracker(wandb_module=fw, client_names=NAMES)
        tr.start({"experiment-id": "exp2-fedavg", "num-rounds": 2})
        tr.log_round(1, TR, EV, None)
        tr.log_round(2, TR, EV, None)
        tr.log_epoch(1, {"train_loss": 1.0})
        tr.finish({"done": True}, None)
        fw2 = FakeWandb(table_fails=True)
        t2 = et.ExperimentTracker(wandb_module=fw2, client_names=NAMES)
        t2.start({"experiment-id": "exp2-fedavg", "num-rounds": 2})
        t2.log_round(2, TR, EV, None)  # last round: tables due but broken
        d = fw2.runs[0].logs[0][1]
        assert "val/macro-f1-pooled" in d and not any("confusion-matrix" in k for k in d)
    finally:
        logging.disable(logging.NOTSET)


@check("T12 confusion-matrix tables: only every N rounds and on the last round; correct rows; N <= 0 disables; no num-rounds")
def _():
    fw = FakeWandb()
    tr = et.ExperimentTracker(wandb_module=fw, client_names=NAMES, cm_every=5)
    tr.start({"experiment-id": "exp2-fedavg", "num-rounds": 7})
    for r in range(1, 8):
        tr.log_round(r, TR, EV, None)
    logs = dict(fw.runs[0].logs)
    due = [r for r in range(1, 8) if "val/confusion-matrix-pooled" in logs[r]]
    assert due == [5, 7], due
    t = logs[5]["val/confusion-matrix-pooled"]
    assert t.columns == ["true", "pred", "count"] and len(t.data) == 25
    assert [row[2] for row in t.data] == CM and t.data[7] == ["1", "2", 1]
    assert all(f"val/confusion-matrix/{n}" in logs[5] for n in NAMES)
    for every, cfg in ((0, {"num-rounds": 7}), (-1, {"num-rounds": 7})):
        f2 = FakeWandb()
        t2 = et.ExperimentTracker(wandb_module=f2, client_names=NAMES, cm_every=every)
        t2.start({"experiment-id": "exp2-fedavg", **cfg})
        for r in range(1, 8):
            t2.log_round(r, TR, EV, None)
        assert not any("confusion-matrix" in k for _s, d in f2.runs[0].logs for k in d)
    f3 = FakeWandb()
    t3 = et.ExperimentTracker(wandb_module=f3, client_names=NAMES, cm_every=3)
    t3.start({"experiment-id": "exp2-fedavg"})
    for r in range(1, 7):
        t3.log_round(r, TR, EV, None)
    assert [s for s, d in f3.runs[0].logs if "val/confusion-matrix-pooled" in d] == [3, 6]


@check("T13 API key: environment wins, key file is the fallback and is only set in this process, the key never reaches init or logs")
def _():
    kf = TMP / "key.txt"
    kf.write_text("FILEKEY-123456\n", encoding="utf-8")
    assert et.resolve_api_key({"WANDB_API_KEY": " ENVKEY "}, kf) == ("ENVKEY", "environment")
    assert et.resolve_api_key({}, kf) == ("FILEKEY-123456", "file")
    assert et.resolve_api_key({}, TMP / "missing.txt") == (None, "none")
    (TMP / "empty.txt").write_text("  \n", encoding="utf-8")
    assert et.resolve_api_key({}, TMP / "empty.txt") == (None, "none")

    records = []

    class H(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    h = H()
    logging.getLogger().addHandler(h)
    old = logging.getLogger().level
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        fw = FakeWandb()
        tr = et.ExperimentTracker(wandb_module=fw, api_key_file=kf)
        tr.start({"experiment-id": "exp2-fedavg", "wandb-api-key": "FILEKEY-123456", "secret": "FILEKEY-123456"})
        tr.finish({"done": True}, None)
        assert os.environ.get("WANDB_API_KEY") == "FILEKEY-123456"
        assert "FILEKEY-123456" not in str(fw.inits) and not any("FILEKEY-123456" in r for r in records), records
    finally:
        os.environ.pop("WANDB_API_KEY", None)
        logging.getLogger().removeHandler(h)
        logging.getLogger().setLevel(old)


@check("T14 sanitize_config and run_name: secrets dropped, non-scalars stringified, bad experiment-id rejected")
def _():
    d = et.sanitize_config({"a": 1, "b": None, "c": [1, 2], "api-key": "x", "Token": "y", "my-secret": "z",
                            "password": "p", "credential-file": "q", "checkpoint-root": "/data/x"})
    assert d == {"a": 1, "b": None, "c": "[1, 2]", "checkpoint-root": "/data/x"}
    assert et.run_name({"experiment-id": " exp2-fedavg "}) == "exp2-fedavg"
    for bad in ({}, {"experiment-id": ""}, {"experiment-id": 5}, {"experiment-id": "   "}):
        raises(ValueError, lambda: et.run_name(bad))


@check("T15 make_hook contract: (callable, str, callable) from a validated-style cfg; logs through; failure marks the run failed")
def _():
    logging.disable(logging.CRITICAL)
    try:
        fw = FakeWandb()
        cfg = {"experiment-id": "exp2-fedavg", "algorithm": "fedavg", "dp": False, "num-rounds": 2, "wandb-cm-every": 1}
        cb, rid, finish = et.make_hook(cfg, wandb_module=fw)
        assert callable(cb) and isinstance(rid, str) and callable(finish)
        cb(1, TR, EV, None)
        assert fw.runs[0].logs[0][0] == 1 and "val/confusion-matrix-pooled" in fw.runs[0].logs[0][1]  # every=1
        finish(None, RuntimeError("boom"))
        assert fw.runs[0].summary["status"] == "failed"
        try:
            import federated.server as sv
        except Exception:
            raise Skip("federated.server not importable here; resume-id lookup not tested (V26 covers it)")
        orig = sv.read_resume_info
        sv.read_resume_info = lambda c: {"wandb_run_id": "stored123"}
        try:
            fw2 = FakeWandb()
            _cb, rid2, _f = et.make_hook(cfg, wandb_module=fw2)
            assert rid2 == "stored123" and fw2.inits[0]["resume"] == "must"
            sv.read_resume_info = lambda c: (_ for _ in ()).throw(RuntimeError("unreadable"))
            fw3 = FakeWandb()
            _cb, rid3, _f = et.make_hook(cfg, wandb_module=fw3)
            assert rid3 != "stored123" and "resume" not in fw3.inits[0]
        finally:
            sv.read_resume_info = orig
    finally:
        logging.disable(logging.NOTSET)


@check("T16 Experiment 1: epoch callback logs per epoch with step = epoch; start_centralized resumes the id stored in latest_centralized.pt")
def _():
    fw = FakeWandb()
    d0 = TMP / "exp1_fresh"
    d0.mkdir()
    tracker, rid = et.start_centralized({"experiment-id": "exp1-centralized", "algorithm": "centralized"}, d0, wandb_module=fw)
    assert et.peek_centralized_run_id(d0) is None and "resume" not in fw.inits[0]
    cb = et.make_epoch_callback(tracker)
    cb(1, {"train_loss": 1.0, "val_loss": 1.1, "val_macro_f1": 0.2, "lr_lora": 1e-5, "lr_head": 1e-4, "val_confusion_matrix": CM})
    cb(2, {"train_loss": 0.9})
    assert [s for s, _ in fw.runs[0].logs] == [1, 2] and "val/qwk-pooled" in fw.runs[0].logs[0][1]
    tracker.finish({"final_epoch": 2}, None)
    assert fw.runs[0].summary["status"] == "finished"
    try:
        import torch
    except ImportError:
        raise Skip("torch not installed here; the latest_centralized.pt peek runs on your machine")
    d1 = TMP / "exp1_resume"
    d1.mkdir()
    torch.save({"wandb_run_id": "stored456", "schema_version": 2}, str(d1 / et.LATEST_CENTRALIZED))
    assert et.peek_centralized_run_id(d1) == "stored456"
    fw2 = FakeWandb()
    _t, rid2 = et.start_centralized({"experiment-id": "exp1-centralized"}, d1, wandb_module=fw2)
    assert rid2 == "stored456" and fw2.inits[0]["resume"] == "must"
    torch.save({"schema_version": 2}, str(d1 / et.LATEST_CENTRALIZED))
    assert et.peek_centralized_run_id(d1) is None
    (d1 / et.LATEST_CENTRALIZED).write_bytes(b"not a checkpoint")
    logging.disable(logging.CRITICAL)
    try:
        assert et.peek_centralized_run_id(d1) is None
    finally:
        logging.disable(logging.NOTSET)


@check("T17 module hygiene: wandb, torch and federated are not imported at module level (circular import and lazy-load rules)")
def _():
    tree = ast.parse(Path(et.__file__).read_text(encoding="utf-8"))
    top = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            top |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            top.add(n.module.split(".")[0])
    assert not ({"wandb", "torch", "federated"} & top), top
    assert {"f1_per_class", "recall_per_class", "quadratic_weighted_kappa", "make_hook", "make_epoch_callback",
            "start_centralized", "peek_centralized_run_id", "ExperimentTracker"} <= set(et.__all__)


print()
for n, s, msg in RESULTS:
    print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))
print(f"\n{sum(s == 'PASS' for _n, s, _m in RESULTS)}/{len(RESULTS)} passed, {sum(s == 'SKIP' for _n, s, _m in RESULTS)} skipped")
raise SystemExit(any(s == "FAIL" for _n, s, _m in RESULTS))