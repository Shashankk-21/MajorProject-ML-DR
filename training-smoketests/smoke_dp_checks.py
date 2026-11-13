import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import opacus

from models.coatnet_lora import create_coatnet_lora_model
import utils.dp_checks as dc

DEV = torch.device("cpu")
RESULTS = []
CALLS = {"n": 0}


def factory():
    CALLS["n"] += 1
    return create_coatnet_lora_model(lora_rank=2, lora_alpha=4, pretrained=False)


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


@check("G1 public names exist; DPGradientCheckError is a RuntimeError")
def _():
    for n in ("verify_per_sample_gradients", "ensure_per_sample_gradients_verified",
              "clear_verification_cache", "DPGradientCheckError"):
        assert hasattr(dc, n), n
    assert issubclass(dc.DPGradientCheckError, RuntimeError)


@check("G2 real model, functorch mode: passes, full summary, 66 tensors x 4 samples compared")
def _():
    t0 = time.time()
    s = dc.verify_per_sample_gradients(factory, device=DEV)
    print(f"   {time.time() - t0:.1f}s  summary: {s}")
    need = {"worst_rel_err", "n_tensors", "n_comparisons", "grad_sample_mode", "device_type",
            "torch_version", "opacus_version", "rel_tol", "batch_size"}
    assert need <= set(s), need - set(s)
    assert s["worst_rel_err"] < 1e-3 and s["n_tensors"] == 66 and s["n_comparisons"] == 66 * 4
    assert s["grad_sample_mode"] == "functorch" and s["device_type"] == "cpu"
    assert s["torch_version"] == torch.__version__ and s["opacus_version"] == opacus.__version__


@check("G3 the gate catches the known Opacus bug: hooks mode must FAIL and name the qkv layers")
def _():
    try:
        dc.verify_per_sample_gradients(factory, device=DEV, grad_sample_mode="hooks")
    except dc.DPGradientCheckError as e:
        print("   message:", str(e)[:300].replace("\n", " | "))
        assert "qkv" in str(e), "error message does not name the failing layers"
        return
    raise RuntimeError("hooks mode PASSED the gate: it cannot detect the known bug")


@check("G4 cache: one run per key, summary copy is safe, a failure is cached and re-raised, keys are independent")
def _():
    dc.clear_verification_cache()
    CALLS["n"] = 0
    s1 = dc.ensure_per_sample_gradients_verified(factory, device=DEV)
    n1 = CALLS["n"]
    assert n1 > 0
    s2 = dc.ensure_per_sample_gradients_verified(factory, device=DEV)
    assert CALLS["n"] == n1 and s1 == s2, "second call should not re-run the check"
    s2["worst_rel_err"] = 99.0
    assert dc.ensure_per_sample_gradients_verified(factory, device=DEV)["worst_rel_err"] != 99.0
    CALLS["n"] = 0
    raises(dc.DPGradientCheckError, lambda: dc.ensure_per_sample_gradients_verified(factory, device=DEV, grad_sample_mode="hooks"))
    n2 = CALLS["n"]
    assert n2 > 0
    raises(dc.DPGradientCheckError, lambda: dc.ensure_per_sample_gradients_verified(factory, device=DEV, grad_sample_mode="hooks"))
    assert CALLS["n"] == n2, "a cached failure should be replayed, not re-run"
    CALLS["n"] = 0
    dc.ensure_per_sample_gradients_verified(factory, device=DEV)
    assert CALLS["n"] == 0, "the functorch result must stay cached after a hooks failure"
    dc.clear_verification_cache()


@check("G5 the global torch RNG state is unchanged by the gate")
def _():
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    dc.verify_per_sample_gradients(factory, device=DEV)
    assert torch.equal(torch.get_rng_state(), before), "gate disturbed the global RNG"


@check("G6 argument validation: batch_size 1, rel_tol <= 0, unknown grad_sample_mode are rejected")
def _():
    raises(Exception, lambda: dc.verify_per_sample_gradients(factory, device=DEV, batch_size=1))
    raises(Exception, lambda: dc.verify_per_sample_gradients(factory, device=DEV, rel_tol=0.0))
    raises(Exception, lambda: dc.verify_per_sample_gradients(factory, device=DEV, grad_sample_mode="nonsense"))


@check("G7 non-vacuous: a model without LoRA layers is rejected, not passed")
def _():
    def plain():
        return nn.Sequential(nn.Conv2d(3, 4, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5))
    raises(Exception, lambda: dc.verify_per_sample_gradients(plain, device=DEV))


@check("G8 [pod only] TF32 flags are restored after the check on CUDA")
def _():
    if not torch.cuda.is_available():
        print("   skipped: no CUDA here")
        return
    flags = lambda: (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    before = flags()
    dc.verify_per_sample_gradients(factory, device=torch.device("cuda"))
    assert flags() == before


print()
for n, s, msg in RESULTS:
    print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))
raise SystemExit(any(s == "FAIL" for _, s, _ in RESULTS))