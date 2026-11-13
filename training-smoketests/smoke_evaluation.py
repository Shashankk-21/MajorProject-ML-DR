import contextlib
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import stringzilla  # noqa: F401
except Exception:  # DLL blocked by Windows Application Control, or missing: test-only stand-in
    from types import ModuleType

    _sz = ModuleType("stringzilla")

    def _translate(text, table, start=0, end=None, *, inplace=False):
        src = np.frombuffer(text, dtype=np.uint8)
        out = np.frombuffer(table, dtype=np.uint8)[src]
        if not inplace:
            return out.tobytes()
        src[...] = out

    _sz.translate, _sz.Str, _sz.File = _translate, str, str
    sys.modules["stringzilla"] = _sz

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import torch
except ImportError:
    torch = None
try:
    import sklearn.metrics as skm
except ImportError:
    skm = None

from evaluation import bootstrap as B
from evaluation import evaluate as EV
from evaluation import metrics as M

TESTS = []
CL = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")


class Skip(Exception):
    pass


def test(name):
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


def raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        if contains is not None and contains not in str(e):
            raise RuntimeError(f"error did not mention {contains!r}: {str(e)[:300]}")
        return
    raise RuntimeError(f"expected {exc}, nothing raised")


def need(*mods):
    for m in mods:
        if m is None:
            raise Skip("a needed library is not installed here (torch, pandas or scikit-learn)")


def run_capture(fn, *a, **k):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fn(*a, **k)
    return rc, out.getvalue(), err.getvalue()


_STATE = {}


def tmp():
    if "tmp" not in _STATE:
        _STATE["tmp"] = Path(tempfile.mkdtemp(prefix="evaluation_smoke_"))
    return _STATE["tmp"]


def fresh(name):
    d = tmp() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def rand_case(rng, n=None):
    n = n or int(rng.integers(1, 400))
    allowed = rng.choice(5, size=int(rng.integers(1, 6)), replace=False)
    y = rng.choice(allowed, size=n)
    pred = np.where(rng.random(n) < 0.6, y, rng.integers(0, 5, n))
    probs = np.round(rng.random((n, 5)), 1)  # heavy ties on purpose
    probs[np.arange(n), y] += rng.random(n) * 0.6
    return y.astype(np.int64), pred.astype(np.int64), probs


# ------------------------------------------------------------------ metrics.py
@test("E1 confusion, accuracy and per-class precision/recall equal sklearn on 400 random cases (absent classes, ties); F1 equals sklearn where support > 0 and is NaN where it is 0")
def _():
    need(skm)
    rng = np.random.default_rng(0)
    for _ in range(400):
        y, p, _pr = rand_case(rng)
        assert np.array_equal(M.confusion(y, p), skm.confusion_matrix(y, p, labels=list(range(5))))
        assert abs(M.accuracy(y, p) - skm.accuracy_score(y, p)) < 1e-12
        rep = M.per_class_report(y, p)
        assert np.array_equal(rep["support"], np.bincount(y, minlength=5))
        try:
            pr_, rc_, f1_, _s = skm.precision_recall_fscore_support(y, p, labels=list(range(5)), zero_division=np.nan)
        except ValueError:
            raise Skip("this scikit-learn has no zero_division=nan")
        assert np.allclose(rep["precision"], pr_, equal_nan=True) and np.allclose(rep["recall"], rc_, equal_nan=True)
        sup = rep["support"] > 0
        assert np.allclose(rep["f1"][sup], f1_[sup]) and np.isnan(rep["f1"][~sup]).all()
        present = list(np.flatnonzero(sup))
        want = skm.f1_score(y, p, labels=present, average="macro", zero_division=0)
        assert abs(M.macro_f1(y, p) - want) < 1e-12


@test("E2 quadratic weighted kappa equals sklearn (NaN where sklearn is NaN or nothing to compare); macro_over_defined ignores NaN")
def _():
    need(skm)
    rng = np.random.default_rng(1)
    for _ in range(400):
        y, p, _pr = rand_case(rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = skm.cohen_kappa_score(y, p, labels=list(range(5)), weights="quadratic")
        q = M.qwk(y, p)
        assert (math.isnan(q) and not np.isfinite(k)) or abs(q - k) < 1e-10, (q, k)
    assert math.isnan(M.qwk([], [])) and math.isnan(M.qwk([2, 2, 2], [2, 2, 2]))
    assert abs(M.qwk([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) - 1.0) < 1e-12
    assert M.macro_over_defined([0.5, np.nan, 1.0]) == 0.75 and math.isnan(M.macro_over_defined([np.nan, np.nan]))
    assert math.isnan(M.macro_over_defined([]))


@test("E3 one-vs-rest AUC equals sklearn per class (heavy ties), NaN without positives or negatives, macro over the defined classes")
def _():
    need(skm)
    rng = np.random.default_rng(2)
    for _ in range(300):
        y, _p, pr = rand_case(rng)
        per, macro = M.auc_ovr(y, pr)
        want = np.full(5, np.nan)
        for c in range(5):
            pos = y == c
            if 0 < pos.sum() < y.size:
                want[c] = skm.roc_auc_score(pos, pr[:, c])
        assert np.allclose(per, want, equal_nan=True, atol=1e-12), (per, want)
        assert (math.isnan(macro) and np.isnan(want).all()) or abs(macro - np.nanmean(want)) < 1e-12
    raises(ValueError, lambda: M.auc_ovr([0, 1], np.zeros(2)))
    raises(ValueError, lambda: M.auc_ovr([0, 1], np.array([[np.inf, 0.0], [0.0, 1.0]])))
    raises(ValueError, lambda: M.auc_ovr([0, 1, 1], np.zeros((2, 5))))
    raises(ValueError, lambda: M.auc_ovr([0, 7], np.zeros((2, 5))))


@test("E4 softmax is stable and sums to 1; bad labels and shapes are rejected; summarize returns every headline metric")
def _():
    z = np.array([[1000.0, 1001.0, 999.0, 1000.0, 1000.0], [-1000.0, 0.0, 0.0, 0.0, 0.0]])
    s = M.softmax(z)
    assert np.isfinite(s).all() and np.allclose(s.sum(axis=1), 1.0) and s[0].argmax() == 1
    for bad in ([True, False], [[0, 1]], [0.5, 1.0], [-1, 0], [0, 5], ["a", "b"], [float("nan"), 0.0]):
        raises(ValueError, lambda bad=bad: M.confusion(bad, bad))
    raises(ValueError, lambda: M.confusion([0, 1], [0]))
    y, p, pr = rand_case(np.random.default_rng(3), n=60)
    s = M.summarize(y, p, pr)
    assert {"n", "accuracy", "macro_f1", "qwk", "macro_auc", "auc", "precision", "recall", "f1", "support", "confusion"} <= set(s)
    assert s["n"] == 60 and s["confusion"].shape == (5, 5) and s["support"].sum() == 60
    fns = M.default_metric_fns()
    assert set(fns) == {"macro_f1", "qwk", "macro_auc"} and abs(fns["qwk"](y, p, pr) - M.qwk(y, p)) < 1e-15


# ------------------------------------------------------------------ bootstrap.py
@test("B1 make_units: patients where a client has repeats, images otherwise, client-prefixed, missing ids are their own unit, bad shapes rejected")
def _():
    clients = ["A", "A", "A", "B", "B", "C", "C"]
    images = ["a1", "a2", "a3", "b1", "b2", "c1", "c2"]
    pats = ["p1", "p1", "p2", "p9", "p9", "q1", "q2"]
    u, kinds = B.make_units(clients, images, pats)
    assert kinds == {"A": "patients", "B": "patients", "C": "images"}
    assert u[0] == u[1] != u[2] and u[3] == u[4] and u[5] != u[6] and u[0].startswith("A|pat:") and u[5].startswith("C|img:")
    u2, k2 = B.make_units(["A", "B"], ["x", "y"], ["same", "same"])  # same id in two clients never collides
    assert u2[0] != u2[1] and k2 == {"A": "images", "B": "images"}
    u3, k3 = B.make_units(["A"] * 4, ["a1", "a2", "a3", "a4"], [None, float("nan"), "", "p1"])
    assert k3 == {"A": "images"} and len(set(u3)) == 4
    u4, k4 = B.make_units(["A"] * 3, ["a1", "a2", "a3"], ["p1", "p1", None])
    assert k4 == {"A": "patients"} and u4[0] == u4[1] != u4[2]
    u5, k5 = B.make_units(["A", "A"], ["a1", "a2"])
    assert k5 == {"A": "images"} and u5[0] != u5[1]
    raises(ValueError, lambda: B.make_units(["A"], ["a", "b"]))
    raises(ValueError, lambda: B.make_units(["A"], ["a"], ["p", "q"]))


@test("B2 bootstrap_ci: deterministic per seed, estimate == full-data metric, equals an independent cluster-bootstrap re-derivation, honest width for iid rows")
def _():
    rng = np.random.default_rng(4)
    y, p, pr = rand_case(rng, n=120)
    units = np.array([f"u{i // 3}" for i in range(120)])
    fns = M.default_metric_fns()
    a = B.bootstrap_ci(y, p, pr, units, fns, n=60, seed=7)
    b = B.bootstrap_ci(y, p, pr, units, fns, n=60, seed=7)
    c = B.bootstrap_ci(y, p, pr, units, fns, n=60, seed=8)
    assert a == b and a != c
    for name, fn in fns.items():
        assert a[name]["n_resamples"] == 60 and abs(a[name]["estimate"] - fn(y, p, pr)) < 1e-15 or math.isnan(a[name]["estimate"])
    u, codes = np.unique(units, return_inverse=True)
    g = np.random.default_rng(7)
    vals = []
    for _ in range(60):
        counts = np.bincount(g.integers(0, len(u), size=len(u)), minlength=len(u))
        idx = np.repeat(np.arange(120), counts[codes])
        vals.append(M.macro_f1(y[idx], p[idx]))
    lo, hi = np.percentile(np.array(vals)[np.isfinite(vals)], [2.5, 97.5])
    assert abs(a["macro_f1"]["low"] - lo) < 1e-12 and abs(a["macro_f1"]["high"] - hi) < 1e-12
    # iid rows, accuracy: width ~ 2 * 1.96 * sqrt(p(1-p)/n)
    n = 400
    yy = np.zeros(n, dtype=np.int64)
    pp = (np.random.default_rng(5).random(n) < 0.7).astype(np.int64) * 0  # placeholder
    correct = np.random.default_rng(5).random(n) < 0.7
    acc = {"acc": lambda t, q, r: float(np.mean(t == q))}
    res = B.bootstrap_ci(yy, np.where(correct, 0, 1), np.zeros((n, 5)), np.arange(n), acc, n=2000, seed=1)["acc"]
    width, ph = res["high"] - res["low"], float(correct.mean())
    want = 2 * 1.96 * math.sqrt(ph * (1 - ph) / n)
    assert 0.8 * want < width < 1.2 * want, (width, want)


@test("B3 clustering by patient widens the interval; undefined resamples are dropped (n_valid); all-undefined gives NaN; bad arguments rejected")
def _():
    n = 200
    rng = np.random.default_rng(6)
    base = rng.random(50) < 0.6
    correct = np.repeat(base, 4)  # four identical rows per patient
    y = np.zeros(n, dtype=np.int64)
    pred = np.where(correct, 0, 1)
    acc = {"acc": lambda t, q, r: float(np.mean(t == q))}
    img = B.bootstrap_ci(y, pred, np.zeros((n, 5)), np.arange(n), acc, n=1500, seed=3)["acc"]
    pat = B.bootstrap_ci(y, pred, np.zeros((n, 5)), np.repeat(np.arange(50), 4), acc, n=1500, seed=3)["acc"]
    assert (pat["high"] - pat["low"]) > 1.5 * (img["high"] - img["low"])
    yy = np.array([0] * 57 + [1] * 3)
    only1 = {"has1": lambda t, q, r: 1.0 if (t == 1).any() else float("nan")}
    r = B.bootstrap_ci(yy, yy, np.zeros((60, 5)), np.arange(60), only1, n=300, seed=1)["has1"]
    assert 0 < r["n_valid"] < 300 and r["low"] == 1.0 == r["high"]
    allnan = {"x": lambda t, q, r: float("nan")}
    r = B.bootstrap_ci(yy, yy, np.zeros((60, 5)), np.arange(60), allnan, n=20, seed=1)["x"]
    assert r["n_valid"] == 0 and math.isnan(r["low"]) and math.isnan(r["high"])
    z = np.zeros((4, 5))
    for kw in ({"n": 0}, {"n": True}, {"n": 2.5}, {"alpha": 0.0}, {"alpha": 1.0}):
        raises(ValueError, lambda kw=kw: B.bootstrap_ci([0, 1, 0, 1], [0, 1, 0, 1], z, np.arange(4), acc, **kw))
    raises(ValueError, lambda: B.bootstrap_ci([0, 1], [0, 1], z[:2], np.arange(3), acc))
    raises(ValueError, lambda: B.bootstrap_ci([], [], np.zeros((0, 5)), [], acc))
    raises(ValueError, lambda: B.bootstrap_ci([0, 1], [0, 1], z[:2], np.arange(2), {}))
    raises(ValueError, lambda: B.bootstrap_ci([0, 1], [0], z[:2], np.arange(2), acc))


def _pair_frames(seed=0, n=240):
    need(pd)
    rng = np.random.default_rng(seed)
    clients = np.array(["APTOS"] * 80 + ["EyePACS"] * 160)
    ids = np.array([f"img{i}" for i in range(n)])
    y = rng.choice(5, size=n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    pa = np.where(rng.random(n) < 0.85, y, rng.integers(0, 5, n))
    pb = np.where(rng.random(n) < 0.60, y, rng.integers(0, 5, n))

    def frame(pred):
        probs = np.full((n, 5), 0.05)
        probs[np.arange(n), pred] = 0.8
        d = {"client": clients, "image_id": ids, "patient_id": [f"p{i // 2}" for i in range(n)], "label": y, "pred": pred}
        d.update({f"prob_{c}": probs[:, c] for c in range(5)})
        return pd.DataFrame(d)

    return frame(pa), frame(pb), np.array([f"{c}|pat:p{i // 2}" for i, c in enumerate(clients)])


@test("B4 paired bootstrap: identical files give diff 0 and p 1, a better A gives a positive interval, row order does not matter, units array == mapping, mismatches rejected")
def _():
    a, b, units = _pair_frames()
    f = M.default_metric_fns()["macro_f1"]
    same = B.paired_bootstrap_diff(a, a, units, f, n=100, seed=1)
    assert same["diff"] == 0 and same["low"] == 0 == same["high"] and same["p_two_sided"] == 1.0
    r = B.paired_bootstrap_diff(a, b, units, f, n=300, seed=1)
    assert r["diff"] > 0.1 and r["low"] > 0 and r["p_two_sided"] < 0.05 and r["n_rows"] == 240 and r["n_units"] == 120
    assert abs(r["diff"] - (r["metric_a"] - r["metric_b"])) < 1e-15
    shuffled = b.sample(frac=1.0, random_state=0).reset_index(drop=True)
    r2 = B.paired_bootstrap_diff(a, shuffled, units, f, n=300, seed=1)
    assert r2 == r  # aligned on (client, image_id), A's order, same draws
    mapping = {(c, i): u for c, i, u in zip(a["client"], a["image_id"], units)}
    assert B.paired_bootstrap_diff(a, b, mapping, f, n=300, seed=1) == r
    assert B.paired_bootstrap_diff(a, b, units, f, n=300, seed=2) != r
    raises(ValueError, lambda: B.paired_bootstrap_diff(a, b.iloc[:-1], units, f, n=10), contains="different test images")
    bad = b.copy()
    bad.loc[0, "label"] = (bad.loc[0, "label"] + 1) % 5
    raises(ValueError, lambda: B.paired_bootstrap_diff(a, bad, units, f, n=10), contains="disagree")
    dup = pd.concat([a, a.iloc[:1]], ignore_index=True)
    raises(ValueError, lambda: B.paired_bootstrap_diff(dup, dup, np.append(units, units[0]), f, n=10), contains="duplicate")
    raises(ValueError, lambda: B.paired_bootstrap_diff(a, b, {("APTOS", "nope"): "u"}, f, n=10))
    raises(ValueError, lambda: B.paired_bootstrap_diff(a, b, units[:10], f, n=10))
    probs_only = {"client": a["client"].to_numpy(), "image_id": a["image_id"].to_numpy(), "label": a["label"].to_numpy(),
                  "pred": a["pred"].to_numpy(), "probs": a[[f"prob_{c}" for c in range(5)]].to_numpy()}
    r3 = B.paired_bootstrap_diff(probs_only, b, units, M.default_metric_fns()["macro_auc"], n=20, seed=1)
    assert r3["n_rows"] == 240 and r3 == B.paired_bootstrap_diff(a, b, units, M.default_metric_fns()["macro_auc"], n=20, seed=1)


@test("B5 paired_from_csv and the command line: units from patient ids, one client or POOLED, JSON output, bad metric or client rejected")
def _():
    a, b, _u = _pair_frames(seed=1)
    d = fresh("b5")
    a.to_csv(d / "a.csv", index=False)
    b.to_csv(d / "b.csv", index=False)
    r = B.paired_from_csv(str(d / "a.csv"), str(d / "b.csv"), "qwk", "POOLED", n=40, seed=1)
    assert r["metric"] == "qwk" and r["client"] == "POOLED" and r["unit_kinds"] == {"APTOS": "patients", "EyePACS": "patients"}
    r1 = B.paired_from_csv(str(d / "a.csv"), str(d / "b.csv"), "macro_f1", "APTOS", n=40, seed=1)
    assert r1["n_rows"] == 80 and list(r1["unit_kinds"]) == ["APTOS"]
    raises(ValueError, lambda: B.paired_from_csv(str(d / "a.csv"), str(d / "b.csv"), "bogus"))
    raises(ValueError, lambda: B.paired_from_csv(str(d / "a.csv"), str(d / "b.csv"), "qwk", "Nowhere"))
    rc, out, err = run_capture(B.main, ["--a", str(d / "a.csv"), "--b", str(d / "b.csv"), "--metric", "macro_auc", "--n", "30"])
    assert rc == 0 and json.loads(out)["metric"] == "macro_auc" and "test-set sampling only" in err


# ------------------------------------------------------------------ evaluate.py (pure parts)
@test("V1 experiment ids and checkpoint paths: all eight valid ids, malformed or inconsistent ids rejected, best.pt vs best_centralized.pt")
def _():
    ok = {"exp1-centralized": (1, "centralized", False, None), "exp2-fedavg": (2, "fedavg", False, None), "exp3-fedprox": (3, "fedprox", False, None),
          "exp4-fedprox-eps50": (4, "fedprox", True, 50.0), "exp5-fedprox-eps10": (5, "fedprox", True, 10.0), "exp6-fedprox-eps5": (6, "fedprox", True, 5.0),
          "exp7-fedprox-eps1": (7, "fedprox", True, 1.0), "exp8-fedprox-eps0.5": (8, "fedprox", True, 0.5)}
    for i, (n, alg, dp, eps) in ok.items():
        assert EV.parse_experiment_id(i) == {"n": n, "algorithm": alg, "dp": dp, "target_epsilon": eps}, i
    for bad in ("", None, "exp9-fedavg", "exp0-fedavg", "exp2-centralized", "exp1-fedavg", "exp4-fedprox", "exp2-fedavg-eps5", "exp4-fedavg-eps5",
                "exp4-fedprox-eps0", "exp4-fedprox-eps-1", "Exp2-fedavg", "exp2-fedavg ", "exp3-fedprox-eps"):
        raises(ValueError, lambda bad=bad: EV.parse_experiment_id(bad))
    assert EV.checkpoint_path("/r", "exp1-centralized") == Path("/r/exp1-centralized/best_centralized.pt")
    assert EV.checkpoint_path("/r", "exp4-fedprox-eps50") == Path("/r/exp4-fedprox-eps50/best.pt")


@test("V2 model parameters come from the checkpoint's OWN run_config: FL (hyphenated) and Experiment 1 (underscored); pretrained=False, bad values and wrong class count refused")
def _():
    fl = {"lora-rank": 8, "lora-alpha": 16, "img-size": 224, "num-classes": 5, "pretrained": True}
    ce = {"lora_rank": 4, "lora_alpha": 8, "img_size": 224, "num_classes": 5}
    assert EV.model_params_from_run_config(fl) == {"lora_rank": 8, "lora_alpha": 16, "img_size": 224, "num_classes": 5, "style": "fl"}
    assert EV.model_params_from_run_config(ce)["style"] == "centralized" and EV.model_params_from_run_config(ce)["lora_rank"] == 4
    assert EV.model_params_from_run_config({k: v for k, v in fl.items() if k != "pretrained"})["style"] == "fl"
    raises(ValueError, lambda: EV.model_params_from_run_config({**fl, "pretrained": False}), contains="pretrained")
    for key, val in (("lora-rank", 0), ("lora-rank", True), ("lora-rank", 8.0), ("img-size", "224"), ("lora-alpha", None)):
        raises(ValueError, lambda key=key, val=val: EV.model_params_from_run_config({**fl, key: val}))
    raises(ValueError, lambda: EV.model_params_from_run_config({**fl, "num-classes": 3}), contains="num_classes")
    raises(ValueError, lambda: EV.model_params_from_run_config({"lora-rank": 8}), contains="neither")
    raises(ValueError, lambda: EV.model_params_from_run_config({}))


@test("V3 the stored selection is read from INSIDE the checkpoint: FL round + selection-metric, Experiment 1 epoch + best_val_macro_f1; missing or non-finite refused")
def _():
    fl = {"round": 7, "selection-metric": 0.412, "selection-metric-name": "macro-f1-pooled", "best-round": 7, "best-selection-metric": 0.412}
    s = EV.stored_selection(fl, "fl")
    assert (s["kind"], s["number"], s["val_macro_f1"], s["metric"]) == ("round", 7, 0.412, "macro-f1-pooled") and s["recorded_best_round"] == 7
    s = EV.stored_selection({"epoch": 3, "best_val_macro_f1": 0.5, "epochs_without_improvement": 0}, "centralized")
    assert (s["kind"], s["number"], s["val_macro_f1"]) == ("epoch", 3, 0.5)
    assert EV.stored_selection({"round": 2, "selection-metric": 0.3}, "fl")["metric"] == "macro-f1-pooled"
    for extra, style in (({}, "fl"), ({"round": 1}, "fl"), ({"epoch": 1}, "centralized"), ({"round": 1, "selection-metric": float("nan")}, "fl"),
                         ({"round": 1, "selection-metric": float("inf")}, "fl"), ({"round": "x", "selection-metric": 0.1}, "fl"),
                         ({"epoch": 1, "best_val_macro_f1": None}, "centralized")):
        raises(ValueError, lambda extra=extra, style=style: EV.stored_selection(extra, style))


@test("V4 test-once guard: first run free, second refused (AlreadyEvaluated), --force allowed, counted and remembered; the done file is atomic JSON")
def _():
    d = fresh("v4") / "exp2-fedavg"
    assert EV.check_guard(d, False) == (0, []) and EV.check_guard(d, True) == (0, []) and EV.read_done(d) is None
    rec = EV.write_done(d, exp_id="exp2-fedavg", ckpt_path=Path("/c/best.pt"), md5="abc", force_count=0, previous=[])
    assert rec["force_count"] == 0 and rec["version"] == EV.EVAL_VERSION and rec["checkpoint_md5"] == "abc" and rec["timestamp_utc"].endswith("+00:00")
    assert json.loads((d / EV.DONE_NAME).read_text(encoding="utf-8")) == rec and not list(d.glob("*.tmp"))
    raises(EV.AlreadyEvaluated, lambda: EV.check_guard(d, False), contains="refusing")
    assert issubclass(EV.AlreadyEvaluated, RuntimeError)
    count, prev = EV.check_guard(d, True)
    assert count == 1 and len(prev) == 1 and prev[0]["checkpoint_md5"] == "abc"
    EV.write_done(d, exp_id="exp2-fedavg", ckpt_path=Path("/c/best.pt"), md5="abc", force_count=count, previous=prev)
    count2, prev2 = EV.check_guard(d, True)
    assert count2 == 2 and len(prev2) == 2


@test("V5 manifest check: match, mismatch (per client) and unavailable are told apart; both hash key spellings accepted; sha256 is of the raw bytes")
def _():
    d = fresh("v5")
    paths = {}
    for c in CL:
        p = d / f"{c}_manifest.csv"
        p.write_bytes(f"image_id,label\n{c},1\n".encode())
        paths[c] = p
    sha = {c: hashlib.sha256(p.read_bytes()).hexdigest() for c, p in paths.items()}
    assert all(EV._file_sha256(p) == sha[c] for c, p in paths.items())
    assert EV.check_manifests(paths, {"manifest-sha256": sha})["status"] == "match"
    assert EV.check_manifests(paths, {"manifest_sha256": sha})["status"] == "match"
    r = EV.check_manifests(paths, {"manifest_sha256": {**sha, "IDRiD": "0" * 64}})
    assert r["status"] == "mismatch" and [c for c, v in r["per_client"].items() if not v["match"]] == ["IDRiD"]
    assert EV.check_manifests(paths, {"manifest_sha256": {c: sha[c] for c in CL[:3]}})["status"] == "mismatch"
    assert EV.check_manifests(paths, {})["status"] == "unavailable" and EV.check_manifests(paths, {"manifest_sha256": "x"})["status"] == "unavailable"
    assert EV.file_md5(paths["APTOS"]) == hashlib.md5(paths["APTOS"].read_bytes()).hexdigest()
    assert tuple(EV.manifest_paths("/m")) == EV.client_names() and EV.manifest_paths("/m")["APTOS"] == Path("/m/APTOS_manifest.csv")


def make_pred(seed=0, absent=None):
    need(pd)
    rng = np.random.default_rng(seed)
    rows = []
    for client, n, mode in (("APTOS", 50, "unique"), ("IDRiD", 30, "unique"), ("Messidor-2", 40, "pairs"), ("EyePACS", 120, "pairs")):
        for i in range(n):
            label = int(rng.choice(5, p=[0.45, 0.15, 0.2, 0.1, 0.1]))
            if absent and client in absent and label == absent[client]:
                label = 0
            logits = rng.normal(size=5)
            logits[label] += 1.2
            probs = M.softmax(logits[None, :])[0]
            row = {"client": client, "image_id": f"{client}-i{i}", "patient_id": f"{client}-p{i}" if mode == "unique" else f"{client}-p{i // 2}",
                   "label": label, "pred": int(logits.argmax())}
            row.update({f"logit_{c}": logits[c] for c in range(5)})
            row.update({f"prob_{c}": probs[c] for c in range(5)})
            rows.append(row)
    return pd.DataFrame(rows)


@test("V6 compute_results: per-client and pooled metrics equal the pure functions, share of the pooled set, resampling unit per client, unweighted client mean, NaN for an absent class")
def _():
    pred = make_pred(absent={"IDRiD": 3})
    res = EV.compute_results(pred, n_boot=25)
    assert res["clients"] == list(CL) and res["test_images"] == {"APTOS": 50, "IDRiD": 30, "Messidor-2": 40, "EyePACS": 120, "POOLED": 240}
    assert abs(res["test_share"]["EyePACS"] - 0.5) < 1e-12 and sum(res["test_share"].values()) == 1.0
    assert res["unit_kind"]["APTOS"] == "images" and res["unit_kind"]["IDRiD"] == "images"
    assert res["unit_kind"]["Messidor-2"] == "patients" and res["unit_kind"]["EyePACS"] == "patients"
    y, p = pred["label"].to_numpy(), pred["pred"].to_numpy()
    assert abs(res["metrics"]["POOLED"]["macro_f1"] - M.macro_f1(y, p)) < 1e-15 and abs(res["metrics"]["POOLED"]["qwk"] - M.qwk(y, p)) < 1e-15
    assert res["metrics"]["POOLED"]["n"] == 240
    mean = np.mean([res["metrics"][c]["macro_f1"] for c in CL])
    assert abs(res["client_mean_macro_f1"] - mean) < 1e-12
    m = res["metrics"]["IDRiD"]
    assert m["support"][3] == 0 and math.isnan(m["recall"][3]) and math.isnan(m["f1"][3]) and math.isnan(m["auc"][3])
    assert abs(m["macro_f1"] - np.nanmean(m["f1"])) < 1e-12
    b = res["bootstrap"]["EyePACS"]["macro_f1"]
    assert b["low"] <= b["estimate"] + 0.2 and b["n_resamples"] == 25 and res["bootstrap_settings"] == {"n_resamples": 25, "seed": 42, "alpha": 0.05, "interval": "percentile"}


def _results_for(pred, privacy=None):
    ev = EV.compute_results(pred, n_boot=15)
    return {"experiment_id": "exp4-fedprox-eps50", "version": EV.EVAL_VERSION, "dp": privacy is not None, "algorithm": "fedprox",
            "checkpoint": {"path": "/c/best.pt", "md5": "m" * 32, "stored": {"kind": "round", "number": 7, "val_macro_f1": 0.4123, "metric": "macro-f1-pooled"},
                           "model_params": {"lora_rank": 8}, "run_config": {}, "wandb_run_id": None},
            "manifest_check": {"status": "match"}, "evaluation": ev, "privacy": privacy,
            "guard": {"force_count": 0, "previous_runs": []}, "notes": []}


@test("V7 summary.md and the output files: per-client headline with Mild/Moderate recall and support, pooled-share statement, n/a for undefined, privacy table, JSON without NaN, no temp files")
def _():
    need(pd)
    pred = make_pred(absent={"IDRiD": 3})
    priv = {"best_round": {"round": 7, "epsilon": {"APTOS": 3.0, "IDRiD": 4.0, "Messidor-2": 2.5, "EyePACS": 5.0}, "max_epsilon": 5.0},
            "final_round": {"round": 10, "epsilon": {"APTOS": 4.0, "IDRiD": 5.0, "Messidor-2": 3.5, "EyePACS": 6.0}, "max_epsilon": 6.0},
            "target_epsilon": 50.0, "delta": 1e-5, "accountant": "rdp", "problems": []}
    results = _results_for(pred, priv)
    md = EV.render_summary(results)
    for needle in ("## Headline: per client", "Mild recall", "Moderate recall", "dominated by EyePACS", "50.0%", "n/a", "NOT training randomness",
                   "zero_division=0", "## Privacy cost (DP run)", "| max | 5.000 | 6.000 |", "Best round: 7; final round: 10", "Manifest hashes vs the checkpoint's run_config: match"):
        assert needle in md, needle
    assert "## Privacy cost" not in EV.render_summary(_results_for(pred, None))
    out = fresh("v7") / "exp4-fedprox-eps50"
    EV.write_outputs(out, pred, results)
    assert sorted(p.name for p in out.iterdir()) == ["per_client.csv", "predictions.csv", "results.json", "summary.md"]
    txt = (out / "results.json").read_text(encoding="utf-8")
    assert "NaN" not in txt and "Infinity" not in txt and json.loads(txt)["experiment_id"] == "exp4-fedprox-eps50"
    back = pd.read_csv(out / "predictions.csv")
    assert list(back.columns) == list(pred.columns) and np.allclose(back[[f"prob_{c}" for c in range(5)]], pred[[f"prob_{c}" for c in range(5)]], atol=1e-8)
    pc = pd.read_csv(out / "per_client.csv")
    assert list(pc["client"]) == list(CL) + ["POOLED", "CLIENT_MEAN"] and pc.loc[pc.client == "IDRiD", "support_c3"].iloc[0] == 0
    assert pd.isna(pc.loc[pc.client == "IDRiD", "recall_c3"].iloc[0])
    assert abs(pc.loc[pc.client == "EyePACS", "share_of_pooled"].iloc[0] - 0.5) < 1e-9


@test("V8 privacy is read from the stored round checkpoints: best round from best.pt's extra, final round from latest.pt (else the highest round file), pid -> client name")
def _():
    need(torch)
    d = fresh("v8")

    def priv(shift):
        return {"epsilon": {0: 3.0 + shift, 1: 4.0 + shift, 2: 2.0 + shift, 3: 5.0 + shift}, "noise-multiplier": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
                "steps-total": {0: 10, 1: 20, 2: 30, 3: 40}, "max-epsilon": 5.0 + shift}

    best_extra = {"round": 7, "privacy": priv(0.0)}
    torch.save({"extra": {"round": 10, "privacy": priv(1.0)}}, d / "latest.pt")
    rc = {"target-epsilon": 50, "delta": 1e-5, "accountant": "rdp"}
    r = EV.read_privacy(d, best_extra, rc)
    assert r["best_round"]["round"] == 7 and r["best_round"]["epsilon"] == {"APTOS": 3.0, "IDRiD": 4.0, "Messidor-2": 2.0, "EyePACS": 5.0}
    assert r["final_round"]["round"] == 10 and r["final_round"]["epsilon"]["EyePACS"] == 6.0 and r["final_round"]["max_epsilon"] == 6.0
    assert r["best_round"]["noise_multiplier"]["IDRiD"] == 2.0 and r["best_round"]["steps_total"]["EyePACS"] == 40.0
    assert r["target_epsilon"] == 50.0 and r["delta"] == 1e-5 and r["accountant"] == "rdp" and r["problems"] == []
    (d / "latest.pt").unlink()
    torch.save({"extra": {"round": 8, "privacy": priv(0.5)}}, d / "round_008.pt")
    torch.save({"extra": {"round": 9, "privacy": priv(0.8)}}, d / "round_009.pt")
    assert EV.read_privacy(d, best_extra, rc)["final_round"]["round"] == 9
    for f in d.glob("round_*.pt"):
        f.unlink()
    r = EV.read_privacy(d, {"round": 7}, rc)
    assert r["best_round"] is None and r["final_round"] is None and len(r["problems"]) == 2


# ------------------------------------------------------------------ end to end on CPU (real model code, tiny synthetic data)
def make_manifests():
    if "mroot" in _STATE:
        return _STATE["mroot"]
    from PIL import Image
    from preprocessing.dataset import FINAL_MANIFEST_COLUMNS
    root = tmp() / "data"
    root.mkdir()
    rng = np.random.default_rng(0)
    for k, name in enumerate(CL):
        rows = []
        for split, n in (("train", 10), ("val", 4), ("test", 4)):
            for j in range(n):
                p = root / f"{name}_{split}_{j}.png"
                Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(p)
                row = {c: "na" for c in FINAL_MANIFEST_COLUMNS}
                row.update(image_id=f"{name}_{split}_{j}", cached_image_path=str(p), label=(k + j) % 5, split=split, patient_id=f"{name}-p{j // 2}")
                rows.append(row)
        pd.DataFrame(rows, columns=list(FINAL_MANIFEST_COLUMNS)).to_csv(root / f"{name}_manifest.csv", index=False)
    _STATE["mroot"] = root
    return root


def patch_models():
    if _STATE.get("patched"):
        return
    import models.coatnet_lora as cl
    import preprocessing.augmentation as aug
    import training.train_centralized as tc

    orig = cl.create_coatnet_lora_model

    def fake(**kw):
        kw = {k: v for k, v in kw.items() if k != "pretrained"}
        torch.manual_seed(1234)
        return orig(pretrained=False, **kw)

    cl.create_coatnet_lora_model = fake
    tc.create_coatnet_lora_model = fake
    tc.get_train_transforms = aug.get_eval_transforms
    _STATE["patched"] = True


def exp1_checkpoints():
    """A real (tiny) Experiment 1 run: returns (checkpoint root, manifest root, trainer result)."""
    if "exp1" in _STATE:
        return _STATE["exp1"]
    need(torch, pd)
    patch_models()
    import training.train_centralized as tc
    mroot = make_manifests()
    ckroot = fresh("ck")
    paths = {c: mroot / f"{c}_manifest.csv" for c in CL}
    res = tc.run_centralized_training(client_manifest_paths=paths, checkpoint_dir=str(ckroot / "exp1-centralized"), lora_rank=2, lora_alpha=4,
                                      physical_batch_size=4, accumulation_steps=2, effective_batch_size=8, num_workers=0, warmup_epochs=0.5,
                                      patience=5, seed=7, max_epochs=1)
    _STATE["exp1"] = (ckroot, mroot, res)
    return _STATE["exp1"]


def ck_copy(name):
    ckroot, mroot, res = exp1_checkpoints()
    dst = tmp() / name
    shutil.copytree(ckroot, dst)
    return dst, mroot, res


def run_eval(exp_id, ckroot, mroot, out, *extra):
    argv = ["--experiment-id", exp_id, "--checkpoint-root", str(ckroot), "--manifest-root", str(mroot), "--out", str(out),
            "--device", "cpu", "--batch-size", "4", "--num-workers", "1", "--bootstrap-n", "20", *extra]
    return run_capture(EV.main, argv)


@test("X1 REAL Experiment 1 evaluation on CPU: 16 test images, outputs equal an independent sklearn recomputation, stored selection and manifest hashes match the trainer's, guard refuses a second run, --force counts")
def _():
    need(torch, pd, skm)
    ckroot, mroot, tres = ck_copy("ck_x1")
    out = fresh("x1out")
    rc, _o, err = run_eval("exp1-centralized", ckroot, mroot, out)
    assert rc == 0, err
    d = out / "exp1-centralized"
    pred = pd.read_csv(d / "predictions.csv")
    assert len(pred) == 16 and sorted(pred["client"].unique()) == sorted(CL) and (pred.groupby("client").size() == 4).all()
    assert all(pred["image_id"].str.contains("_test_"))
    lg = pred[[f"logit_{c}" for c in range(5)]].to_numpy()
    pr = pred[[f"prob_{c}" for c in range(5)]].to_numpy()
    assert np.array_equal(lg.argmax(axis=1), pred["pred"].to_numpy()) and np.allclose(pr.sum(axis=1), 1.0, atol=1e-6)
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    st = res["checkpoint"]["stored"]
    assert st["kind"] == "epoch" and st["number"] == 1 and abs(st["val_macro_f1"] - tres["best_val_macro_f1"]) < 1e-12
    assert res["manifest_check"]["status"] == "match" and res["privacy"] is None and res["checkpoint"]["model_params"]["lora_rank"] == 2
    assert res["checkpoint"]["md5"] == hashlib.md5((ckroot / "exp1-centralized" / "best_centralized.pt").read_bytes()).hexdigest()
    y, p = pred["label"].to_numpy(), pred["pred"].to_numpy()
    pc = pd.read_csv(d / "per_client.csv").set_index("client")
    present = [c for c in range(5) if (y == c).any()]
    assert abs(pc.loc["POOLED", "macro_f1"] - skm.f1_score(y, p, labels=present, average="macro", zero_division=0)) < 1e-5
    for c in CL:
        m = (pred["client"] == c).to_numpy()
        pres = [k for k in range(5) if (y[m] == k).any()]
        assert abs(pc.loc[c, "macro_f1"] - skm.f1_score(y[m], p[m], labels=pres, average="macro", zero_division=0)) < 1e-5
    gfile = ckroot / "exp1-centralized" / EV.DONE_NAME  # the guard sits next to the checkpoint, not in --out
    assert not (d / EV.DONE_NAME).exists()
    done = json.loads(gfile.read_text(encoding="utf-8"))
    assert done["force_count"] == 0 and done["checkpoint_md5"] == res["checkpoint"]["md5"]
    rc, _o, err = run_eval("exp1-centralized", ckroot, mroot, out)
    assert rc == 3 and "already evaluated" in err and json.loads(gfile.read_text(encoding="utf-8")) == done
    rc, _o, err = run_eval("exp1-centralized", ckroot, mroot, fresh("x1other"))  # a different --out must not bypass the guard
    assert rc == 3 and not (tmp() / "x1other" / "exp1-centralized" / "predictions.csv").exists()
    rc, _o, err = run_eval("exp1-centralized", ckroot, mroot, out, "--force")
    assert rc == 0, err
    done2 = json.loads(gfile.read_text(encoding="utf-8"))
    assert done2["force_count"] == 1 and len(done2["previous_runs"]) == 1 and done2["checkpoint_md5"] == done["checkpoint_md5"]
    again = pd.read_csv(d / "predictions.csv")
    assert np.allclose(again[[f"logit_{c}" for c in range(5)]], lg, atol=1e-6) and np.array_equal(again["pred"], pred["pred"])


@test("X2 REAL FL-style DP evaluation on CPU: checkpoint saved the way the server saves it, hyphenated run_config, privacy at the best and final round, wrong experiment id and dp mismatch refused")
def _():
    need(torch, pd)
    ckroot, mroot, _res = ck_copy("ck_x2")
    from utils.checkpoint import save_checkpoint
    model, _info = EV.load_model(ckroot / "exp1-centralized" / "best_centralized.pt", torch.device("cpu"))
    sha = {c: EV._file_sha256(mroot / f"{c}_manifest.csv") for c in CL}

    def rc_for(exp_id, dp):
        return {"experiment-id": exp_id, "algorithm": "fedprox" if dp else "fedavg", "dp": dp, "lora-rank": 2, "lora-alpha": 4, "img-size": 224,
                "num-classes": 5, "pretrained": True, "target-epsilon": 50.0, "delta": 1e-5, "accountant": "rdp", "manifest-sha256": sha}

    def priv(shift):
        return {"epsilon": {0: 3.0 + shift, 1: 4.0 + shift, 2: 2.0 + shift, 3: 5.0 + shift}, "noise-multiplier": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
                "steps-total": {0: 10, 1: 20, 2: 30, 3: 40}, "max-epsilon": 5.0 + shift}

    def save(exp_id, dp, rnd, name):
        d = ckroot / exp_id
        d.mkdir(exist_ok=True)
        extra = {"round": rnd, "selection-metric": 0.42, "selection-metric-name": "macro-f1-pooled", "best-round": 7, "best-selection-metric": 0.42,
                 "privacy": priv(0.0 if rnd == 7 else 1.0)}
        save_checkpoint(d / name, model=model, optimizer=None, round_offset=rnd, opacus_accountant_states=None, run_config=rc_for(exp_id, dp),
                        wandb_run_id="x", extra=extra)

    save("exp4-fedprox-eps50", True, 7, "best.pt")
    save("exp4-fedprox-eps50", True, 10, "latest.pt")
    out = fresh("x2out")
    rc, _o, err = run_eval("exp4-fedprox-eps50", ckroot, mroot, out)
    assert rc == 0, err
    res = json.loads((out / "exp4-fedprox-eps50" / "results.json").read_text(encoding="utf-8"))
    p = res["privacy"]
    assert p["best_round"]["round"] == 7 and p["final_round"]["round"] == 10 and p["target_epsilon"] == 50.0
    assert p["best_round"]["epsilon"]["EyePACS"] == 5.0 and p["final_round"]["epsilon"]["EyePACS"] == 6.0 and p["problems"] == []
    assert res["checkpoint"]["stored"]["kind"] == "round" and res["checkpoint"]["stored"]["number"] == 7 and res["checkpoint"]["model_params"]["style"] == "fl"
    assert "Privacy cost" in (out / "exp4-fedprox-eps50" / "summary.md").read_text(encoding="utf-8")
    shutil.copytree(ckroot / "exp4-fedprox-eps50", ckroot / "exp5-fedprox-eps10", ignore=shutil.ignore_patterns(EV.DONE_NAME))
    rc, _o, err = run_eval("exp5-fedprox-eps10", ckroot, mroot, out)
    assert rc == 2 and "belongs to" in err and not (ckroot / "exp5-fedprox-eps10" / EV.DONE_NAME).exists()
    save("exp2-fedavg", True, 7, "best.pt")  # run_config says dp=True for a non-DP experiment id
    rc, _o, err = run_eval("exp2-fedavg", ckroot, mroot, out)
    assert rc == 2 and "dp=" in err


@test("X3 REAL refusals on CPU: a changed manifest, a missing checkpoint or manifest, a malformed id; --skip-manifest-check overrides only the hash check; nothing is marked as evaluated on failure")
def _():
    need(torch, pd)
    ckroot, mroot, _res = ck_copy("ck_x3")
    bad = fresh("x3data")
    for f in mroot.glob("*_manifest.csv"):
        shutil.copy(f, bad / f.name)
    with open(bad / "IDRiD_manifest.csv", "ab") as fh:
        fh.write(b"\n")
    out = fresh("x3out")
    rc, _o, err = run_eval("exp1-centralized", ckroot, bad, out)
    assert rc == 2 and "differ" in err and "IDRiD" in err and not (ckroot / "exp1-centralized" / EV.DONE_NAME).exists()
    rc, _o, err = run_eval("exp1-centralized", ckroot, bad, out, "--skip-manifest-check")
    assert rc == 0, err
    assert json.loads((out / "exp1-centralized" / "results.json").read_text(encoding="utf-8"))["manifest_check"]["status"] == "mismatch"
    rc, _o, err = run_eval("exp3-fedprox", ckroot, mroot, out)
    assert rc == 2 and "not found" in err
    rc, _o, err = run_eval("exp9-nonsense", ckroot, mroot, out)
    assert rc == 2
    empty = fresh("x3empty")
    rc, _o, err = run_eval("exp1-centralized", ckroot, empty, fresh("x3out2"))
    assert rc == 2 and "manifests not found" in err
    raises(SystemExit, lambda: run_capture(EV.parse_args, ["--experiment-id", "exp1-centralized", "--checkpoint-root", "c", "--manifest-root", "m", "--out", "o", "--batch-size", "0"]))


def main():
    results = []
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    for name, fn in TESTS:
        if only and not any(name.startswith(o + " ") for o in only):
            continue
        try:
            fn()
            results.append((name, "PASS", ""))
        except Skip as e:
            results.append((name, "SKIP", str(e)))
        except BaseException as e:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc().strip().splitlines()
            results.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:300]}\n      " + "\n      ".join(tb[-4:])))
    print()
    for n, s, msg in results:
        print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))
    print(f"\n{sum(s == 'PASS' for _n, s, _m in results)}/{len(results)} passed, {sum(s == 'SKIP' for _n, s, _m in results)} skipped")
    raise SystemExit(any(s == "FAIL" for _n, s, _m in results))


if __name__ == "__main__":
    main()
