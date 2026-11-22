import contextlib
import copy
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SZ = ROOT / "notebooks" / "stepzero"
for _p in (str(SZ), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

import stepzero_lib as Z

try:
    import torch
except ImportError:
    torch = None

TESTS = []
STEP = {"lora-rank": "2", "lora-alpha": "4", "max-grad-norm": "1.0", "num-workers": "1"}


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


def need_torch():
    if torch is None:
        raise Skip("torch not installed here")


def run_capture(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = fn(*a, **k)
    return res, buf.getvalue()


_STATE = {}


def tmp():
    if "tmp" not in _STATE:
        _STATE["tmp"] = Path(tempfile.mkdtemp(prefix="stepzero_smoke_"))
    return _STATE["tmp"]


def config_dir(name, **subs):
    """Temp copy of the real configs/*.yaml; step-zero values filled (or as given) and a small worker count."""
    d = tmp() / name
    d.mkdir()
    for f in (ROOT / "configs").glob("*.yaml"):
        shutil.copy(f, d)
    text = (d / "base.yaml").read_text(encoding="utf-8")
    for k, v in {**STEP, **subs}.items():
        text, n = re.subn(rf"^{re.escape(k)}:.*$", f"{k}: {v}", text, flags=re.M)
        assert n == 1, k
    (d / "base.yaml").write_text(text, encoding="utf-8")
    return d


def make_manifests():
    """Tiny synthetic clients (10 train / 4 val / 4 test images each, all five grades) as {client}_manifest.csv."""
    if "manifests" in _STATE:
        return _STATE["manifests"]
    import pandas as pd
    from PIL import Image
    from configs import loader as ld
    from preprocessing.dataset import FINAL_MANIFEST_COLUMNS

    root = tmp() / "data"
    root.mkdir()
    rng = np.random.default_rng(0)
    for k, name in enumerate(ld.CLIENTS):
        rows = []
        for split, n in (("train", 10), ("val", 4), ("test", 4)):
            for j in range(n):
                p = root / f"{name}_{split}_{j}.png"
                Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(p)
                row = {c: "na" for c in FINAL_MANIFEST_COLUMNS}
                row.update(image_id=f"{name}_{split}_{j}", cached_image_path=str(p), label=(k + j) % 5, split=split)
                rows.append(row)
        pd.DataFrame(rows, columns=list(FINAL_MANIFEST_COLUMNS)).to_csv(root / f"{name}_manifest.csv", index=False)
    _STATE["manifests"] = root
    return root


def patch_models():
    """Real model code, but no pretrained download: same trick as smoke_train_centralized."""
    if _STATE.get("patched"):
        return
    import models.coatnet_lora as cl
    import training.train_centralized as tc

    orig = cl.create_coatnet_lora_model

    def fake(**kw):
        kw = {k: v for k, v in kw.items() if k != "pretrained"}
        torch.manual_seed(1234)
        return orig(pretrained=False, **kw)

    cl.create_coatnet_lora_model = fake
    tc.create_coatnet_lora_model = fake
    _STATE["patched"] = True


# ------------------------------------------------------------------ stepzero_lib
@test("Z1 select_rank: tie set, lowest rank wins, inclusive boundary, QWK manual-review flag, bad input rejected")
def _():
    base = {4: (0.800, 0.70), 8: (0.810, 0.72), 16: (0.795, 0.71), 32: (0.700, 0.60)}
    mk = lambda d: {r: {"val_macro_f1": f, "qwk": q} for r, (f, q) in d.items()}  # noqa: E731
    s = Z.select_rank(mk(base))
    assert s["tie_set"] == [4, 8] and s["chosen"] == 4 and s["best_f1"] == 0.81 and s["best_qwk"] == 0.72
    assert s["manual_review"] is False  # 0.70 is exactly 0.02 below 0.72: not MORE than 0.02
    assert set(s) == {"chosen", "tie_set", "best_f1", "best_qwk", "manual_review", "reason"}
    s = Z.select_rank(mk({**base, 4: (0.800, 0.69)}))
    assert s["chosen"] == 4 and s["manual_review"] is True and "MANUAL REVIEW" in s["reason"]
    assert Z.select_rank(mk({4: (0.5, 0.5), 8: (0.6, 0.5)}))["chosen"] == 8
    assert Z.select_rank(mk({4: (0.70, 0.5), 8: (0.705, 0.5), 16: (0.71, 0.5), 32: (0.709, 0.5)}))["chosen"] == 4
    assert Z.select_rank(mk({16: (0.7, 0.5)}))["chosen"] == 16
    assert Z.select_rank(mk(base), tie=0.0)["chosen"] == 8
    for bad in ({}, None, {0: {"val_macro_f1": 1, "qwk": 1}}, {True: {"val_macro_f1": 1, "qwk": 1}}, {4.0: {"val_macro_f1": 1, "qwk": 1}},
                {4: {"val_macro_f1": float("nan"), "qwk": 1}}, {4: {"val_macro_f1": 1}}, {4: {"val_macro_f1": "x", "qwk": 1}}):
        raises(ValueError, lambda bad=bad: Z.select_rank(bad))
    raises(ValueError, lambda: Z.select_rank(mk(base), tie=-0.1))


@test("Z2 percentile_summary, fraction_clipped, geometric_mean, recommend_max_grad_norm, phase_ratio, round_sig")
def _():
    rng = np.random.default_rng(0)
    a = rng.lognormal(0, 1, size=997)
    s = Z.percentile_summary(a)
    assert s["n"] == 997 and all(abs(s[k] - np.percentile(a, q)) < 1e-12 for k, q in (("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)))
    assert s["max"] == a.max()
    for bad in ([], [1.0, float("nan")], [float("inf")]):
        raises(ValueError, lambda bad=bad: Z.percentile_summary(bad))
    assert Z.fraction_clipped([1, 1, 2], 1.0) == 1 / 3  # strictly greater
    assert Z.fraction_clipped(a, 1.0) == float(np.mean(a > 1.0))
    for c in (0, -1, float("nan")):
        raises(ValueError, lambda c=c: Z.fraction_clipped([1.0], c))
    assert abs(Z.geometric_mean([1, 4]) - 2.0) < 1e-12 and abs(Z.geometric_mean([2, 2, 2]) - 2.0) < 1e-12
    for bad in ([], [1, 0], [1, -1], [1, float("nan")]):
        raises(ValueError, lambda bad=bad: Z.geometric_mean(bad))
    r = Z.recommend_max_grad_norm({"a": 1.0, "b": 4.0, "c": 2.0})
    assert abs(r["value"] - 2.0) < 1e-12 and r["ratio"] == 4.0 and r["flag"] is True
    assert Z.recommend_max_grad_norm({"a": 1.0, "b": 3.0})["flag"] is False  # exactly 3x is not above 3x
    assert Z.recommend_max_grad_norm({"a": 1.0, "b": 3.01})["flag"] is True
    assert abs(Z.recommend_max_grad_norm({"a": 5.0})["value"] - 5.0) < 1e-12
    raises(ValueError, lambda: Z.recommend_max_grad_norm({}))
    assert Z.phase_ratio(2.0, 6.0)["ratio"] == 3.0 and Z.phase_ratio(2.0, 6.0)["flag"] is False and Z.phase_ratio(7.0, 2.0)["flag"] is True
    assert Z.round_sig(0.123456) == 0.123 and Z.round_sig(1234.5) == 1230.0 and Z.round_sig(0.000123456, 2) == 0.00012
    raises(ValueError, lambda: Z.round_sig(0.0))


@test("Z3 flat_per_sample_norms == sqrt(sum over parameters of each sample's squared L2), numpy and torch inputs")
def _():
    rng = np.random.default_rng(1)
    grads = [rng.normal(size=(7, 3, 4)), rng.normal(size=(7, 5)), rng.normal(size=(7,)), rng.normal(size=(7, 2, 2, 2))]
    want = np.array([np.sqrt(sum(float((g[i] ** 2).sum()) for g in grads)) for i in range(7)])
    got = Z.flat_per_sample_norms(grads)
    assert got.shape == (7,) and got.dtype == np.float64 and np.allclose(got, want, rtol=1e-12)
    if torch is not None:
        tg = [torch.as_tensor(g, dtype=torch.float32) for g in grads]
        assert np.allclose(Z.flat_per_sample_norms(tg), want, rtol=1e-5)
    raises(ValueError, lambda: Z.flat_per_sample_norms([]))
    raises(ValueError, lambda: Z.flat_per_sample_norms([grads[0], grads[1][:5]]))
    raises(ValueError, lambda: Z.flat_per_sample_norms([np.float64(1.0)]))
    assert np.isinf(Z.flat_per_sample_norms([np.array([[np.inf]])])[0])  # callers check finiteness


@test("Z4 timing math: non-DP epoch seconds; expected DP micro-batches equal the exact binomial expectation (scipy)")
def _():
    assert Z.epoch_seconds_nondp(70960, 32, 0.5) == math.ceil(70960 / 32) * 0.5 and Z.epoch_seconds_nondp(32, 32, 1.0) == 1.0
    for bad in ((0, 32, 1.0), (10, 0, 1.0), (10, 32, -1.0), (10, 32, float("nan"))):
        raises(ValueError, lambda bad=bad: Z.epoch_seconds_nondp(*bad))
    assert Z.expected_micro_batches_dp(50, 64, 32) == 2.0 and Z.expected_micro_batches_dp(64, 64, 16) == 4.0  # one logical batch
    try:
        from scipy.stats import binom
    except ImportError:
        raise Skip("scipy missing")
    for n in (330, 1396, 2930, 70960):
        n_logical = math.ceil(n / 64)
        q = 1.0 / n_logical
        k = np.arange(n + 1)
        pmf = binom.pmf(k, n, q)
        for phys in (16, 32, 48, 64):
            want = n_logical * float((pmf * np.ceil(k / phys)).sum())
            got = Z.expected_micro_batches_dp(n, 64, phys)
            assert abs(got - want) / want < 1e-6, (n, phys, got, want)
    assert Z.expected_micro_batches_dp(70960, 64, 16) > Z.expected_micro_batches_dp(70960, 64, 64)
    assert abs(Z.epoch_seconds_dp(330, 64, 32, 0.5) - 0.5 * Z.expected_micro_batches_dp(330, 64, 32)) < 1e-12
    raises(ValueError, lambda: Z.expected_micro_batches_dp(0, 64, 32))


@test("Z5 helpers: largest_fitting, fmt_hms, format_table, atomic_write_json keeps the old file when a write fails")
def _():
    assert Z.largest_fitting({16: "ok", 32: "ok", 48: "OOM", 64: "OOM"}) == 32 and Z.largest_fitting({16: "OOM"}) is None
    assert Z.largest_fitting({16: "ok", 64: "ok", 32: "OOM"}) == 64 and Z.largest_fitting({}) is None
    assert Z.fmt_hms(3725) == "1:02:05" and Z.fmt_hms(59.6) == "0:01:00" and Z.fmt_hms(0) == "0:00:00"
    t = Z.format_table(["a", "bb"], [[1, 22], [333, 4]]).splitlines()
    assert len(t) == 4 and set(t[1].replace(" ", "")) == {"-"} and len({len(x) for x in t}) == 1
    d = tmp() / "io"
    assert Z.read_json(d / "none.json") is None
    Z.atomic_write_json(d / "x.json", {"b": 1, "a": [1, 2]})
    assert Z.read_json(d / "x.json") == {"a": [1, 2], "b": 1}
    raises(TypeError, lambda: Z.atomic_write_json(d / "x.json", {"bad": {1, 2}}))
    assert Z.read_json(d / "x.json") == {"a": [1, 2], "b": 1} and not list(d.glob("*.tmp"))


# ------------------------------------------------------------------ timing_probe (no GPU needed)
@test("T1 timing_probe pure parts: arguments, timestamp wrapper, OOM detection, train sizes, report tables")
def _():
    import timing_probe as tp
    a = tp.parse_args(["--manifest-root", "m", "--checkpoint-root", "c"])
    assert a.batch_sizes == [16, 32, 48, 64] and a.steps == 20 and a.warmup_steps == 3 and a.lora_rank == 8 and a.lora_alpha == 16
    assert tp.parse_args(["--manifest-root", "m", "--checkpoint-root", "c", "--lora-rank", "4"]).lora_alpha == 8
    for bad in (["--steps", "0"], ["--batch-sizes", "0"], ["--lora-rank", "0"], ["--warmup-steps", "-1"]):
        raises(SystemExit, lambda bad=bad: tp.parse_args(["--manifest-root", "m", "--checkpoint-root", "c", *bad]))
    old_manifest = os.environ.pop("DR_MANIFEST_ROOT", None)
    try:
        raises(SystemExit, lambda: tp.parse_args(["--checkpoint-root", "c"]))
    finally:
        if old_manifest is not None:
            os.environ["DR_MANIFEST_ROOT"] = old_manifest
    stamps, syncs = [], []
    got = list(tp._stamped(iter(range(100)), stamps, 5, lambda: syncs.append(1)))
    assert got == [0, 1, 2, 3, 4] and len(stamps) == 5 and len(syncs) == 5 and stamps == sorted(stamps)
    assert tp._is_oom(RuntimeError("CUDA out of memory. Tried to allocate")) and not tp._is_oom(ValueError("out of memory"))
    assert not tp._is_oom(RuntimeError("shape mismatch"))
    import pandas as pd
    mp = {}
    for name, n in (("APTOS", 5), ("EyePACS", 7)):
        p = tmp() / f"tp_{name}.csv"
        pd.DataFrame({"split": ["train"] * n + ["val"] * 2 + ["test"] * 3}).to_csv(p, index=False)
        mp[name] = p
    assert tp.train_sizes(mp) == {"APTOS": 5, "EyePACS": 7}
    clients = ["APTOS", "IDRiD", "Messidor-2", "EyePACS"]
    sizes = {"APTOS": 2930, "IDRiD": 330, "Messidor-2": 1396, "EyePACS": 70960}
    store = {"results": {
        "nondp-32": {"mode": "nondp", "physical_batch": 32, "status": "ok", "seconds_per_step": 0.5, "peak_mem_gib": 5.0, "accumulation_steps": 2},
        "nondp-48": {"mode": "nondp", "physical_batch": 48, "status": "ok", "seconds_per_step": 0.7, "peak_mem_gib": 7.0, "accumulation_steps": 2},
        "nondp-64": {"mode": "nondp", "physical_batch": 64, "status": "OOM"},
        "dp-16": {"mode": "dp", "physical_batch": 16, "status": "ok", "seconds_per_step": 0.4, "peak_mem_gib": 4.0, "accumulation_steps": 1},
        "dp-32": {"mode": "dp", "physical_batch": 32, "status": "OOM"}}}
    _r, out = run_capture(tp.report, store, sizes, clients, 3, 64, "Fake GPU, 16.0 GiB")
    assert "Fake GPU" in out and Z.fmt_hms(math.ceil(70960 / 32) * 0.5) in out and "real config cannot use 48" in out
    assert "nondp: 48" in out and "dp: 16" in out and "pooled 75616" in out
    expect_dp = Z.fmt_hms(3 * Z.epoch_seconds_dp(70960, 64, 16, 0.4))
    assert expect_dp in out


@test("T2 timing_probe refuses to run without a GPU (exit code 2, clear message)")
def _():
    need_torch()
    if torch.cuda.is_available():
        raise Skip("a GPU is present: run it for real on the pod")
    import timing_probe as tp
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = tp.main(["--manifest-root", "m", "--checkpoint-root", str(tmp() / "tpout")])
    assert rc == 2 and "GPU" in err.getvalue()


@test("T3 timing_probe: only DP physical sizes that split the logical batch evenly are real tests (Opacus splits 64 into equal chunks)")
def _():
    import timing_probe as tp
    assert [b for b in (16, 32, 48, 64) if tp.dp_size_is_representative(64, b)] == [16, 32, 64]
    assert tp.dp_size_is_representative(64, 8) and not tp.dp_size_is_representative(64, 24) and not tp.dp_size_is_representative(64, 48)
    raises(ValueError, lambda: tp.dp_size_is_representative(64, 0))
    assert [len(x) for x in np.array_split(np.arange(64), math.ceil(64 / 48))] == [32, 32]  # what Opacus really does at max 48


@test("G5 grad_norm_probe notes: pooled is described honestly (up to --batches per client, IDRiD contributes fewer samples)")
def _():
    import grad_norm_probe as gp
    rng = np.random.default_rng(4)
    cn = {c: _norms(rng, 1.0) for c in ("APTOS", "IDRiD", "Messidor-2", "EyePACS")}
    store = {"phases": {"epoch1": {"clients": {c: {"norms": v} for c, v in cn.items()}}}}
    _r, out = run_capture(gp.report, store)
    flat = " ".join(out.split())
    assert "IDRiD" in flat and "contributes fewer samples" in flat and "equal batches per client" not in flat


# ------------------------------------------------------------------ rank_sweep
@test("R1 rank_sweep best_epoch: highest val macro-F1, earliest on ties, records without a QWK are ignored")
def _():
    import rank_sweep as rs
    st = {"epochs": {"1": {"val_macro_f1": 0.3, "qwk": 0.2}, "2": {"val_macro_f1": 0.5, "qwk": 0.4},
                     "3": {"val_macro_f1": 0.5, "qwk": 0.45}, "4": {"val_macro_f1": 0.9, "qwk": None}}}
    b = rs.best_epoch(st)
    assert b["epoch"] == 2 and b["val_macro_f1"] == 0.5 and b["qwk"] == 0.4
    assert rs.best_epoch({"epochs": {}}) is None and rs.best_epoch({"epochs": {"1": {"val_macro_f1": None, "qwk": 0.1}}}) is None


@test("R2 rank_sweep build_kwargs: Experiment 1 kwargs with this rank/alpha and the schedule proxy, nothing else changed")
def _():
    import rank_sweep as rs
    from configs import loader as ld
    d = config_dir("r2", **{"lora-rank": "null", "lora-alpha": "null"})
    cfg = ld.load_experiment_config(1, allow_unset=True, manifest_root="/m", checkpoint_root="/c", env={}, config_dir=d)
    assert cfg["lora-rank"] is None
    kw = rs.build_kwargs(cfg, 16, Path("/c/stepzero/rank16"))
    full = ld.to_centralized_kwargs({**cfg, "lora-rank": 16, "lora-alpha": 32})
    assert set(kw) == set(full)
    assert kw["lora_rank"] == 16 and kw["lora_alpha"] == 32
    assert kw["max_epochs"] == 4 and kw["warmup_epochs"] == 0.2 and kw["patience"] == 99
    assert str(kw["checkpoint_dir"]).replace("\\", "/").endswith("stepzero/rank16")
    for k in set(full) - {"lora_rank", "lora_alpha", "checkpoint_dir", "max_epochs", "warmup_epochs", "patience"}:
        assert kw[k] == full[k], k
    p = rs._params_of(kw)
    assert "client_manifest_paths" not in p and "checkpoint_dir" not in p and p["lora_rank"] == 16


@test("R6 rank_sweep: always the decided seed 42, and a lost epoch record (trainer best above recorded best) is refused")
def _():
    import rank_sweep as rs
    from configs import loader as ld
    assert rs.SEED == 42
    d = config_dir("r6", **{"lora-rank": "null", "lora-alpha": "null"})
    cfg = ld.load_experiment_config(1, allow_unset=True, manifest_root="/m", checkpoint_root="/c", env={}, config_dir=d)
    assert rs.build_kwargs(dict(cfg, **{"run-seed": 7}), 4, Path("/c/x"))["seed"] == 42
    st = {"epochs": {"1": {"val_macro_f1": 0.3, "qwk": 0.2}, "2": {"val_macro_f1": 0.5, "qwk": 0.4}}, "result": {"best_val_macro_f1": 0.5}}
    args = (4, Path("s.json"), Path("rank4"))
    rs._check_best_recorded(st, *args)
    st["result"]["best_val_macro_f1"] = 0.62
    raises(SystemExit, lambda: rs._check_best_recorded(st, *args), contains="record was lost")
    for res in (None, {}, {"best_val_macro_f1": None}):
        st["result"] = res
        rs._check_best_recorded(st, *args)
    rs._check_best_recorded({"epochs": {}, "result": {"best_val_macro_f1": 0.9}}, *args)


@test("R3 rank_sweep print_report on stored states: tables, selection, the two lines to paste, partial-sweep and manual-review warnings")
def _():
    import rank_sweep as rs

    def state(rank, f1s, qwks):
        return {"rank": rank, "epochs": {str(i + 1): {"train_loss": 1.0, "val_loss": 1.1, "val_macro_f1": f, "qwk": q, "lr_lora": 1e-4,
                                                       "lr_head": 1e-3, "seconds": 100.0 + i, "includes_setup": i == 0}
                                         for i, (f, q) in enumerate(zip(f1s, qwks))}}

    full = {4: state(4, [0.30, 0.50, 0.55, 0.54], [0.4, 0.5, 0.60, 0.6]), 8: state(8, [0.3, 0.52, 0.595, 0.55], [0.4, 0.5, 0.69, 0.6]),
            16: state(16, [0.3, 0.5, 0.60, 0.58], [0.4, 0.5, 0.70, 0.69]), 32: state(32, [0.3, 0.4, 0.5, 0.5], [0.3, 0.4, 0.5, 0.5])}
    d = tmp() / "r3"
    sel, out = run_capture(rs.print_report, full, d)
    assert sel["chosen"] == 8 and "lora-rank: 8" in out and "lora-alpha: 16" in out and "WARNING" not in out
    assert "MANUAL REVIEW" not in out and "*" in out
    assert Z.read_json(d / "rank_sweep_summary.json")["selection"]["chosen"] == 8
    full[8] = state(8, [0.3, 0.52, 0.595, 0.55], [0.4, 0.5, 0.60, 0.60])
    full[16] = state(16, [0.3, 0.5, 0.60, 0.58], [0.4, 0.5, 0.90, 0.70])
    sel, out = run_capture(rs.print_report, full, tmp() / "r3b")
    assert sel["manual_review"] and "MANUAL REVIEW NEEDED" in out
    sel, out = run_capture(rs.print_report, {4: full[4], 8: full[8]}, tmp() / "r3c")
    assert "partial sweep" in out


@test("R4 rank_sweep refuses to reuse a stored rank state written with different settings (before any training)")
def _():
    need_torch()
    import rank_sweep as rs
    from configs import loader as ld
    d = config_dir("r4")
    cfg = ld.load_experiment_config(1, allow_unset=True, manifest_root="/m", checkpoint_root="/c", env={}, config_dir=d)
    out = tmp() / "r4out"
    out.mkdir()
    kw = rs.build_kwargs(cfg, 2, out / "rank2")
    Z.atomic_write_json(out / "rank2.json", {"rank": 2, "alpha": 4, "params": {**rs._params_of(kw), "seed": 7}, "epochs": {}, "result": None})
    raises(SystemExit, lambda: rs.run_rank(2, cfg, out, False), contains="different settings")


@test("R5 REAL TRAINING on CPU: 2 ranks x 2 epochs on tiny synthetic clients; records with QWK, summary, skip when finished, refuse on changed settings")
def _():
    need_torch()
    import rank_sweep as rs
    patch_models()
    mroot, cdir, out = make_manifests(), config_dir("r5"), tmp() / "r5out"
    old = rs.MAX_EPOCHS
    rs.MAX_EPOCHS = 2
    try:
        argv = ["--manifest-root", str(mroot), "--checkpoint-root", str(out), "--ranks", "2", "4", "--config-dir", str(cdir)]
        rc, text = run_capture(rs.main, argv)
        sd = out / "stepzero"
        assert rc == 0 and (sd / "rank2.json").is_file() and (sd / "rank4.json").is_file() and (sd / "rank_sweep_summary.json").is_file()
        for r in (2, 4):
            st = Z.read_json(sd / f"rank{r}.json")
            assert st["alpha"] == 2 * r and sorted(st["epochs"]) == ["1", "2"] and st["result"]["final_epoch"] == 2
            for rec in st["epochs"].values():
                assert rec["qwk"] is not None and -1.0 <= rec["qwk"] <= 1.0 and rec["val_macro_f1"] is not None and rec["seconds"] > 0
                assert rec["lr_lora"] >= 0 and rec["lr_head"] >= 0 and rec["train_loss"] > 0  # cosine ends at exactly 0 after the last step
            assert st["epochs"]["1"]["lr_lora"] > 0 and st["epochs"]["1"]["lr_head"] > 0
            assert st["epochs"]["1"]["includes_setup"] is True and st["epochs"]["2"]["includes_setup"] is False
            assert (sd / f"rank{r}" / "latest_centralized.pt").is_file()
        assert "lora-rank:" in text and "lora-alpha:" in text and "partial sweep" in text
        rc, text = run_capture(rs.main, argv)
        assert rc == 0 and text.count("already finished") == 2
        old_w = rs.WARMUP_EPOCHS
        rs.WARMUP_EPOCHS = 0.4
        try:
            raises(SystemExit, lambda: run_capture(rs.main, argv), contains="different settings")
        finally:
            rs.WARMUP_EPOCHS = old_w
    finally:
        rs.MAX_EPOCHS = old


# ------------------------------------------------------------------ grad_norm_probe
@test("G1 grad_norm_probe refuses (exit code 2, nothing written) while lora-rank or lora-alpha is null")
def _():
    import grad_norm_probe as gp
    for sub in ({"lora-rank": "null"}, {"lora-alpha": "null"}):
        d = config_dir("g1", **sub) if "g1" not in _STATE else config_dir("g1b", **sub)
        err = io.StringIO()
        out = tmp() / f"g1out{len(list(tmp().iterdir()))}"
        with contextlib.redirect_stderr(err):
            rc = gp.main(["--manifest-root", "/m", "--checkpoint-root", str(out), "--config-dir", str(d)])
        assert rc == 2 and "rank_sweep" in err.getvalue() and not (out / "stepzero").exists()
        _STATE["g1"] = True
    raises(SystemExit, lambda: gp.parse_args(["--manifest-root", "m", "--checkpoint-root", "c", "--batches", "0"]))
    raises(SystemExit, lambda: gp.parse_args(["--manifest-root", "m", "--checkpoint-root", "c", "--batch-size", "1"]))
    a = gp.parse_args(["--manifest-root", "m", "--checkpoint-root", "c"])
    assert a.batches == 20 and a.phases == ["init", "epoch1"] and a.batch_size is None and a.force is False


def _norms(rng, scale):
    return (rng.lognormal(0, 0.3, size=40) * scale).tolist()


@test("G2 grad_norm_probe summarize_phase and report: percentiles, clipped fractions, geometric-mean recommendation, 3x flags, the line to paste")
def _():
    import grad_norm_probe as gp
    rng = np.random.default_rng(3)
    cn = {"APTOS": _norms(rng, 1.0), "IDRiD": _norms(rng, 2.0), "Messidor-2": _norms(rng, 1.5), "EyePACS": _norms(rng, 0.8)}
    ph = gp.summarize_phase(cn)
    meds = {c: float(np.percentile(v, 50)) for c, v in cn.items()}
    assert abs(ph["recommendation"]["value"] - float(np.exp(np.mean(np.log(list(meds.values())))))) < 1e-9
    assert ph["recommendation"]["value_rounded"] == Z.round_sig(ph["recommendation"]["value"])
    assert set(ph["summary"]) == set(cn) | {"POOLED"} and set(ph["clipped"]) == set(cn) | {"POOLED"}
    assert ph["summary"]["POOLED"]["n"] == 160 and ph["summary"]["APTOS"]["p50"] == meds["APTOS"]
    for c in cn:
        for v in ph["candidates"]:
            assert abs(ph["clipped"][c][str(v)] - float(np.mean(np.array(cn[c]) > v))) < 1e-12
    assert {0.5, 1.0, 2.0, 5.0} <= set(ph["candidates"]) and ph["recommendation"]["flag"] is False
    skew = dict(cn, IDRiD=_norms(rng, 9.0))
    assert gp.summarize_phase(skew)["recommendation"]["flag"] is True
    store = {"phases": {"init": {"clients": {c: {"norms": v} for c, v in cn.items()}},
                        "epoch1": {"clients": {c: {"norms": [x * 4 for x in v]} for c, v in cn.items()}}}}
    _r, out = run_capture(gp.report, store)
    primary = gp.summarize_phase({c: [x * 4 for x in v] for c, v in cn.items()})["recommendation"]["value_rounded"]
    assert f"max-grad-norm: {primary:g}" in out and "Phase comparison" in out and "FLAG: phases differ by more than 3x" in out
    assert "PROXY" in out and "Phase 'init'" in out and "Phase 'epoch1'" in out
    _r, out = run_capture(gp.report, {"phases": {"init": store["phases"]["init"]}})
    assert "max-grad-norm:" not in out and "not measured" in out


@test("G3 REAL per-sample norms on CPU: probe numbers == naive single-sample autograd norms (flat L2, weighted loss), two batches, no carry-over")
def _():
    need_torch()
    import copy as _copy
    import torch.nn as nn

    import grad_norm_probe as gp
    from configs import loader as ld
    from models.coatnet_lora import create_coatnet_lora_model, freeze_bn_eval
    from preprocessing.dataset import compute_class_weights, get_dataloaders
    from training.local_trainer import weighted_label_smoothed_ce
    patch_models()
    mroot, cdir = make_manifests(), config_dir("g3")
    cfg = ld.load_experiment_config(1, manifest_root=str(mroot), checkpoint_root=str(tmp() / "g3out"), env={}, config_dir=cdir)
    manifests = ld.to_centralized_kwargs(cfg)["client_manifest_paths"]
    import models.coatnet_lora as cl
    model = cl.create_coatnet_lora_model(lora_rank=2, lora_alpha=4, num_classes=5, img_size=224, pretrained=False)
    ref = _copy.deepcopy(model)
    for m in model.modules():
        if (isinstance(m, nn.Dropout) and m.p > 0) or getattr(m, "drop_prob", 0.0) > 0:
            raise Skip("the model has stochastic layers; a single-sample reference cannot match the batch")
    device = torch.device("cpu")
    for client in ("IDRiD", "EyePACS"):
        m2 = _copy.deepcopy(ref)
        got = gp.measure_clients(m2, device=device, cfg=cfg, manifests=manifests, clients=[client], batches=2, batch_size=4,
                                 grad_sample_mode="functorch", seed=42)[client]
        assert len(got) == 8 and np.isfinite(got).all() and min(got) > 0
        gp._seed_everything(42)
        loader = get_dataloaders(client, manifest_path=manifests[client], batch_size=4, num_workers=int(cfg["num-workers"]), seed=42)[0]
        w = torch.as_tensor(compute_class_weights(loader.dataset.data["label"].to_numpy()), dtype=torch.float32)
        r = _copy.deepcopy(ref)
        r.train()
        freeze_bn_eval(r)
        want, it = [], iter(loader)
        for _ in range(2):
            images, targets = next(it)
            for i in range(images.shape[0]):
                r.zero_grad()
                loss = weighted_label_smoothed_ce(r(images[i:i + 1]), targets[i:i + 1], w, label_smoothing=float(cfg["label-smoothing"]))
                loss.backward()
                want.append(math.sqrt(sum(float((p.grad.double() ** 2).sum()) for p in r.parameters() if p.requires_grad and p.grad is not None)))
        assert np.allclose(got, want, rtol=2e-3, atol=1e-7), (client, got, want)


@test("G4 REAL probe run on CPU: both phases, four clients, stored norms, the line to paste, reuse on rerun, refuses changed settings")
def _():
    need_torch()
    import grad_norm_probe as gp
    patch_models()
    mroot, cdir, out = make_manifests(), config_dir("g4"), tmp() / "g4out"
    argv = ["--manifest-root", str(mroot), "--checkpoint-root", str(out), "--batches", "1", "--batch-size", "4", "--config-dir", str(cdir)]
    rc, text = run_capture(gp.main, argv)
    assert rc == 0
    store = Z.read_json(out / "stepzero" / "grad_norm_probe.json")
    assert set(store["phases"]) == {"init", "epoch1"} and store["params"]["lora_rank"] == 2 and store["params"]["batch_size"] == 4
    for ph in ("init", "epoch1"):
        cl = store["phases"][ph]["clients"]
        assert set(cl) == {"APTOS", "IDRiD", "Messidor-2", "EyePACS"}
        for c, d in cl.items():
            assert len(d["norms"]) == 4 and np.isfinite(d["norms"]).all() and min(d["norms"]) > 0, (ph, c)
    assert (out / "stepzero" / "probe_epoch1" / "best_centralized.pt").is_file()
    assert re.search(r"max-grad-norm: [0-9.e+-]+", text) and "Phase comparison" in text and "gate passed" in text
    rc, text = run_capture(gp.main, argv)
    assert rc == 0 and text.count("stored measurements reused") == 2
    raises(SystemExit, lambda: run_capture(gp.main, argv[:argv.index("--batches")] + ["--batches", "2"] + argv[argv.index("--batches") + 2:]),
           contains="different settings")


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