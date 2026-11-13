import sys, math, json, os, inspect, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.accountants import create_accountant
from opacus.data_loader import DPDataLoader

import utils.privacy_calibration as pc

B = 64
TMP = Path(tempfile.mkdtemp(prefix="calib_smoke_"))
RESULTS = []
SIZES = {"aptos": 2930, "idrid": 330, "messidor2": 1396, "eyepacs": 70960}
EPS = (50.0, 10.0, 5.0, 1.0, 0.5)
DELTA, EPOCHS, TOL = 1e-5, 30, 0.01


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, "PASS", ""))
        except Exception as e:
            RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:300]}"))
        return fn
    return deco


def raises(excs, fn):
    try:
        fn()
    except excs:
        return
    raise RuntimeError(f"expected {excs}, nothing raised")


def calib(eps, acc="rdp", sizes=SIZES, **kw):
    return pc.calibrate_client_noise_multipliers(
        target_epsilon=eps, client_train_sizes=sizes, accountant=acc, logical_batch_size=B, **kw)


def save_kw(res, eps=5.0):
    return dict(target_epsilon=eps, clients=res, accountant="rdp", logical_batch_size=B,
                epochs=EPOCHS, target_delta=DELTA, epsilon_tolerance=TOL)


def eps_of(acc, sigma, q, steps):
    a = create_accountant(mechanism=acc)
    a.history = [(sigma, q, steps)]
    return a.get_epsilon(delta=DELTA)


def dp_loader(n, bs=B):
    ds = TensorDataset(torch.zeros(n, 1), torch.zeros(n, dtype=torch.long))
    return DPDataLoader.from_data_loader(DataLoader(ds, batch_size=bs, drop_last=False))


@check("P1 interface: no physical_batch_size parameter, accountant and logical_batch_size required, new names exported")
def _():
    for fn in (pc.sample_rate_for, pc.calibrate_client_noise_multipliers, pc.calibrate_and_save_all):
        assert "physical_batch_size" not in inspect.signature(fn).parameters, fn.__name__
    raises(TypeError, lambda: pc.calibrate_client_noise_multipliers(
        target_epsilon=5.0, client_train_sizes=SIZES, logical_batch_size=B))
    raises(TypeError, lambda: pc.calibrate_client_noise_multipliers(
        target_epsilon=5.0, client_train_sizes=SIZES, accountant="rdp"))
    for name in ("sample_rate_for", "num_batches_for", "check_loader_matches_calibration",
                 "save_calibration", "load_calibration", "CalibrationError"):
        assert name in pc.__all__, name


@check("P2 num_batches_for / sample_rate_for equal what Opacus's DP loader really uses (all 4 clients)")
def _():
    for name, n in SIZES.items():
        dp = dp_loader(n)
        print(f"   {name:10} len(dp)={len(dp):5} dp.sample_rate={dp.sample_rate:.6f} 64/N={B / n:.6f}")
        assert pc.num_batches_for(n, B) == len(dp) == math.ceil(n / B)
        assert abs(pc.sample_rate_for(n, B) - dp.sample_rate) < 1e-12


@check("P3 result shape: exact keys, steps == 30*num_batches, sample_rate == 1/num_batches")
def _():
    res = calib(5.0)
    assert set(res) == set(SIZES)
    for c, r in res.items():
        assert set(r) == {"noise_multiplier", "sample_rate", "num_batches", "steps", "dataset_size"}, set(r)
        assert r["dataset_size"] == SIZES[c] and r["num_batches"] == math.ceil(SIZES[c] / B)
        assert r["steps"] == EPOCHS * r["num_batches"] and abs(r["sample_rate"] - 1 / r["num_batches"]) < 1e-15


@check("P4 deterministic: two identical calibrations are identical")
def _():
    assert calib(5.0) == calib(5.0)


@check("P5 sanity: smaller epsilon needs more noise; IDRiD needs more noise than EyePACS")
def _():
    prev = None
    for e in EPS:
        s = {c: r["noise_multiplier"] for c, r in calib(e).items()}
        print(f"   eps={e:5}: " + "  ".join(f"{c} {v:.3f}" for c, v in s.items()))
        assert s["idrid"] > s["eyepacs"]
        if prev:
            assert all(s[c] > prev[c] for c in s)
        prev = s


@check("P6 every multiplier hits its target: eps in [target - 0.01, target] with the REAL loader's q and steps")
def _():
    for e in EPS:
        for c, r in calib(e).items():
            dp = dp_loader(SIZES[c])
            actual = eps_of("rdp", r["noise_multiplier"], dp.sample_rate, EPOCHS * len(dp))
            assert e - TOL - 1e-6 <= actual <= e + 1e-9, f"{c} eps={e}: achieved {actual:.4f}"
    print("   all 20 (epsilon x client) combinations land within 0.01 below target")


@check("P7 check_loader_matches_calibration: accepts the right loader; rejects batch-32, and same-batch-count different-size loaders")
def _():
    res = calib(5.0)
    for c in SIZES:
        pc.check_loader_matches_calibration(dp_loader(SIZES[c]), res[c])
        raises(pc.CalibrationError, lambda c=c: pc.check_loader_matches_calibration(dp_loader(SIZES[c], bs=32), res[c]))
    d350 = dp_loader(350)
    assert len(d350) == 6 and abs(d350.sample_rate - 1 / 6) < 1e-12
    raises(pc.CalibrationError, lambda: pc.check_loader_matches_calibration(d350, res["idrid"]))


@check("P8 save/load round trip records accountant, batch size and per-client entries; schema v1 rejected")
def _():
    res = calib(5.0)
    p = TMP / "eps5.json"
    pc.save_calibration(str(p), **save_kw(res))
    raw = json.loads(p.read_text())
    assert raw["schema_version"] == 2 and raw["accountant"] == "rdp" and raw["logical_batch_size"] == B
    assert "client_noise_multipliers" not in raw and set(raw["clients"]) == set(SIZES)
    back = pc.load_calibration(str(p))
    assert back["accountant"] == "rdp" and back["clients"] == res
    raw["schema_version"] = 1
    p.write_text(json.dumps(raw))
    raises(pc.CalibrationError, lambda: pc.load_calibration(str(p)))


@check("P9 load_calibration validates content: missing/extra keys, wrong values, inconsistent steps/rate/batch size all rejected")
def _():
    p = TMP / "p9.json"
    pc.save_calibration(str(p), **save_kw(calib(5.0)))
    base = json.loads(p.read_text())

    def variant(mut):
        raw = json.loads(json.dumps(base))
        mut(raw)
        q = TMP / "p9v.json"
        q.write_text(json.dumps(raw))
        return str(q)

    cases = [
        lambda r: r.pop("accountant"),
        lambda r: r.update(bogus=1),
        lambda r: r["clients"]["idrid"].pop("steps"),
        lambda r: r["clients"]["idrid"].update(extra=1),
        lambda r: r["clients"]["idrid"].update(steps=r["clients"]["idrid"]["steps"] + 1),
        lambda r: r["clients"]["idrid"].update(sample_rate=0.5),
        lambda r: r["clients"]["idrid"].update(noise_multiplier=float("nan")),
        lambda r: r.update(target_epsilon=-1),
        lambda r: r.update(logical_batch_size=32),
    ]
    for i, mut in enumerate(cases):
        try:
            pc.load_calibration(variant(mut))
        except pc.CalibrationError:
            continue
        raise RuntimeError(f"corrupted variant {i} was accepted")


@check("P10 metadata: caller metadata kept, opacus_version recorded and not overridable")
def _():
    p = TMP / "p10.json"
    pc.save_calibration(str(p), **save_kw(calib(5.0)), metadata={"note": "x", "opacus_version": "fake"})
    m = pc.load_calibration(str(p))["metadata"]
    assert m["note"] == "x" and isinstance(m["opacus_version"], str) and m["opacus_version"] != "fake"


@check("P11 atomic JSON write: a failed save keeps the old file and leaves no temp file")
def _():
    d = TMP / "p11"; d.mkdir()
    p = d / "c.json"
    kw = save_kw(calib(5.0))
    pc.save_calibration(str(p), **kw)
    before = p.read_bytes()
    orig = json.dump
    json.dump = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated kill"))
    try:
        raises(RuntimeError, lambda: pc.save_calibration(str(p), **kw))
    finally:
        json.dump = orig
    assert p.read_bytes() == before and [f.name for f in d.iterdir()] == ["c.json"]


@check("P12 calibrate_and_save_all writes 5 files, each recording the accountant, returns all 20 values")
def _():
    d = TMP / "p12"
    out = pc.calibrate_and_save_all(str(d), client_train_sizes=SIZES, accountant="rdp", logical_batch_size=B)
    files = sorted(d.glob("epsilon_*.json"))
    assert len(files) == 5 and len(out) == 5, files
    for f in files:
        assert pc.load_calibration(str(f))["accountant"] == "rdp"


@check("P13 input validation: bad sizes, epsilon, delta, epochs, tolerance, batch size, accountant, empty clients")
def _():
    bad = (ValueError, pc.CalibrationError)
    kw = dict(target_epsilon=5.0, client_train_sizes=SIZES, accountant="rdp", logical_batch_size=B)
    for sizes in ({"x": 0}, {"x": -5}, {"x": 330.0}, {"x": True}, {}):
        raises(bad, lambda s=sizes: calib(5.0, sizes=s))
    for eps in (0.0, -1.0, float("nan"), float("inf")):
        raises(bad, lambda e=eps: calib(e))
    for override in ({"logical_batch_size": 0}, {"logical_batch_size": 64.0}, {"accountant": ""},
                     {"target_delta": 0.0}, {"target_delta": 1.0}, {"epochs": 0}, {"epsilon_tolerance": 0.0}):
        raises(bad, lambda o=override: pc.calibrate_client_noise_multipliers(**{**kw, **o}))


@check("P14 the accountant read from the calibration file builds an engine with the SAME mechanism")
def _():
    p = TMP / "p14.json"
    pc.save_calibration(str(p), **save_kw(calib(5.0)))
    cal = pc.load_calibration(str(p))
    assert PrivacyEngine(accountant=cal["accountant"]).accountant.mechanism() == cal["accountant"] == "rdp"
    print("   PrivacyEngine() with no argument uses:", PrivacyEngine().accountant.mechanism())


@check("P15 IDRiD regression: N=330 -> 6 batches, q=1/6; neither 32/330 nor 64/330 is accepted as equivalent")
def _():
    assert pc.num_batches_for(330, B) == 6 and pc.sample_rate_for(330, B) == 1 / 6
    assert abs(32 / 330 - 1 / 6) > 0.05 and abs(64 / 330 - 1 / 6) > 0.02
    assert abs(dp_loader(330, bs=32).sample_rate - pc.sample_rate_for(330, B)) > 0.05


@check("P16 [info] rdp vs prv: noise needed, and eps the OTHER accountant reports for the same noise (eps 5)")
def _():
    for c in ("idrid", "messidor2"):
        sub = {c: SIZES[c]}
        r, p = calib(5.0, "rdp", sub)[c], calib(5.0, "prv", sub)[c]
        other = eps_of("prv", r["noise_multiplier"], r["sample_rate"], r["steps"])
        print(f"   {c:10} sigma rdp {r['noise_multiplier']:.4f}  sigma prv {p['noise_multiplier']:.4f}  "
              f"eps if the rdp sigma is judged by prv: {other:.3f}")


print()
for n, s, msg in RESULTS:
    print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))
raise SystemExit(any(s == "FAIL" for _, s, _ in RESULTS))