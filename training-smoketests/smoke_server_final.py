"""
Smoke tests for the tracker-support patches (put next to smoke_serverV2.py in
tests/; run from the repo root's environment: python tests/smoke_patchesV1.py).

Covers: per-client confusion matrices in pooled_eval_metrics, the run hook
(run_experiment_tracked), and train_centralized._confusion_matrix_flat.
Does NOT replace smoke_serverV2.py or the train_centralized tests: re-run
those too, unchanged.
"""
import ast
import logging
import os
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import federated.server as sv

logging.disable(logging.CRITICAL)  # the failure paths log tracebacks on purpose
RESULTS = []


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, "PASS", ""))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:300]}"))
        return fn
    return deco


def raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise RuntimeError(f"expected {exc}, nothing raised")


# ---------------------------------------------------------------- helpers
def _per_client(seed=0):
    rng = np.random.default_rng(seed)
    per, cms = {}, {}
    for pid in range(4):
        cm = rng.integers(0, 20, size=(5, 5))
        cm[0, 0] += 1  # never empty
        cms[pid] = cm
        per[pid] = {"confusion-matrix": cm.flatten().tolist(), "num-examples": int(cm.sum()),
                    "val-loss": float(rng.random())}
    return per, cms


class FakeRunner:
    calls = []
    mode = "ok"

    def __init__(self, cfg, *, on_round_end=None, wandb_run_id=None):
        FakeRunner.calls.append({"cfg": dict(cfg), "on_round_end": on_round_end, "wandb_run_id": wandb_run_id})

    def run(self, grid):
        if FakeRunner.mode == "fail":
            raise RuntimeError("boom")
        if FakeRunner.mode == "interrupt":
            raise KeyboardInterrupt()
        return {"done": True, "grid": grid}


def _reset():
    FakeRunner.calls = []
    FakeRunner.mode = "ok"
    sv._ROUND_END_CALLBACK = None
    sv._RUN_HOOK = None
    os.environ.pop(sv.RUN_HOOK_ENV, None)
    sv.ExperimentRunner = FakeRunner


_REAL_RUNNER = sv.ExperimentRunner
CFG = {"experiment-id": "exp4-fedprox-eps50", "algorithm": "fedprox"}


# ------------------------------------------------------------------ tests
@check("P1 pooled_eval_metrics: per-client confusion matrices present, ints, sum to num-examples and to pooled")
def _():
    per, cms = _per_client()
    m = sv.pooled_eval_metrics(per)
    total = np.zeros((5, 5), dtype=np.int64)
    for pid in range(4):
        got = m[f"confusion-matrix-client-{pid}"]
        assert len(got) == 25 and all(type(v) is int for v in got)
        assert got == cms[pid].flatten().tolist()
        assert sum(got) == per[pid]["num-examples"]
        total += np.asarray(got).reshape(5, 5)
    assert total.flatten().tolist() == m["confusion-matrix-pooled"]
    # the old keys are untouched
    for key in ("macro-f1-pooled", "macro-f1-client-mean", "recall-pooled-class-1", "recall-client-3-class-2",
                "val-loss-pooled", "num-examples"):
        assert key in m, key
    try:  # still a valid Flower MetricRecord (needs flwr)
        from flwr.app import MetricRecord
        MetricRecord(m)
    except ImportError:
        pass


@check("P2 pooled_eval_metrics still rejects bad input")
def _():
    per, _c = _per_client()
    raises(Exception, lambda: sv.pooled_eval_metrics({0: per[0]}))
    bad = dict(per)
    bad[1] = {**per[1], "num-examples": per[1]["num-examples"] + 1}
    raises(Exception, lambda: sv.pooled_eval_metrics(bad))


@check("P3 hook: callback, run id and finish are wired; summary returned; finish gets (summary, None)")
def _():
    _reset()
    seen = {}

    def cb(*a):
        return None

    def finish(summary, error):
        seen["finish"] = (summary, error)

    def hook(cfg):
        seen["cfg"] = cfg
        return cb, "run123", finish

    out = sv.run_experiment_tracked(CFG, "GRID", hook=hook)
    call = FakeRunner.calls[0]
    assert call["on_round_end"] is cb and call["wandb_run_id"] == "run123"
    assert seen["cfg"] == CFG and seen["cfg"] is not CFG  # the hook gets a copy
    assert out["done"] and seen["finish"] == (out, None)


@check("P4 no hook: the registered round-end callback is used, run id is None")
def _():
    _reset()
    sentinel = lambda *a: None  # noqa: E731
    sv.set_round_end_callback(sentinel)
    sv.run_experiment_tracked(CFG, "G")
    call = FakeRunner.calls[0]
    assert call["on_round_end"] is sentinel and call["wandb_run_id"] is None


@check("P5 hook that raises: the run still happens, with the default callback and no run id")
def _():
    _reset()
    sentinel = lambda *a: None  # noqa: E731
    sv.set_round_end_callback(sentinel)

    def hook(cfg):
        raise RuntimeError("wandb down")

    out = sv.run_experiment_tracked(CFG, "G", hook=hook)
    call = FakeRunner.calls[0]
    assert out["done"] and call["on_round_end"] is sentinel and call["wandb_run_id"] is None


@check("P6 malformed hook results (non-str id, non-callable callback or finish, wrong arity) never stop the run")
def _():
    for bad in ((None, 123, None), ("not callable", "id", None), (None, "id", "not callable"), (None, "id"), None):
        _reset()
        out = sv.run_experiment_tracked(CFG, "G", hook=lambda cfg, bad=bad: bad)
        call = FakeRunner.calls[0]
        assert out["done"] and call["wandb_run_id"] is None, bad


@check("P7 hook returning only a run id (callback None) keeps the registered callback")
def _():
    _reset()
    sentinel = lambda *a: None  # noqa: E731
    sv.set_round_end_callback(sentinel)
    sv.run_experiment_tracked(CFG, "G", hook=lambda cfg: (None, "abc", None))
    call = FakeRunner.calls[0]
    assert call["on_round_end"] is sentinel and call["wandb_run_id"] == "abc"


@check("P8 run failure: finish gets (None, error), the original error propagates; KeyboardInterrupt too")
def _():
    for mode, exc in (("fail", RuntimeError), ("interrupt", KeyboardInterrupt)):
        _reset()
        FakeRunner.mode = mode
        seen = {}
        raises(exc, lambda: sv.run_experiment_tracked(
            CFG, "G", hook=lambda cfg: (None, "id", lambda s, e: seen.update(s=s, e=e))))
        assert seen["s"] is None and isinstance(seen["e"], exc)


@check("P9 finish that raises is swallowed (success path and failure path)")
def _():
    def finish(s, e):
        raise RuntimeError("finish broke")

    _reset()
    out = sv.run_experiment_tracked(CFG, "G", hook=lambda cfg: (None, "id", finish))
    assert out["done"]
    _reset()
    FakeRunner.mode = "fail"
    raises(RuntimeError, lambda: sv.run_experiment_tracked(CFG, "G", hook=lambda cfg: (None, "id", finish)))


@check("P10 DR_RUN_HOOK: a valid 'module:function' is loaded; a bad one is skipped and the run continues")
def _():
    mod = types.ModuleType("dr_fake_hook_mod")
    mod.make = lambda cfg: (None, "from-env", None)
    mod.not_callable = 5
    sys.modules["dr_fake_hook_mod"] = mod
    try:
        _reset()
        os.environ[sv.RUN_HOOK_ENV] = "dr_fake_hook_mod:make"
        sv.run_experiment_tracked(CFG, "G")
        assert FakeRunner.calls[0]["wandb_run_id"] == "from-env"
        for spec in ("dr_fake_hook_mod:missing", "dr_fake_hook_mod:not_callable", "no_such_module_xyz:f",
                     "no-colon", ":f", "dr_fake_hook_mod:"):
            _reset()
            os.environ[sv.RUN_HOOK_ENV] = spec
            out = sv.run_experiment_tracked(CFG, "G")
            assert out["done"] and FakeRunner.calls[0]["wandb_run_id"] is None, spec
    finally:
        sys.modules.pop("dr_fake_hook_mod", None)


@check("P11 a hook registered with set_run_hook wins over DR_RUN_HOOK")
def _():
    _reset()
    os.environ[sv.RUN_HOOK_ENV] = "no_such_module_xyz:f"
    sv.set_run_hook(lambda cfg: (None, "registered", None))
    sv.run_experiment_tracked(CFG, "G")
    assert FakeRunner.calls[0]["wandb_run_id"] == "registered"


def _load_conf_helper():
    """train_centralized._confusion_matrix_flat; falls back to extracting just
    that function when the module's heavy imports are unavailable."""
    try:
        from training.train_centralized import _confusion_matrix_flat
        return _confusion_matrix_flat, "imported"
    except ImportError:
        src = (ROOT / "training" / "train_centralized.py").read_text(encoding="utf-8")
        node = next(n for n in ast.parse(src).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_confusion_matrix_flat")
        ns = {"np": np, "Any": object, "List": list}
        exec(compile(ast.Module([node], []), "train_centralized.py", "exec"), ns)  # noqa: S102
        return ns["_confusion_matrix_flat"], "extracted"


@check("P12 _confusion_matrix_flat: correct on arrays (and on torch tensors if torch is real), rejects bad input")
def _():
    fn, how = _load_conf_helper()
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 5))
    targets = rng.integers(0, 5, size=200)
    got = fn(logits, targets, 5)
    want = np.zeros((5, 5), dtype=int)
    for t, p in zip(targets, logits.argmax(1)):
        want[t, p] += 1
    assert got == want.flatten().tolist() and all(type(v) is int for v in got) and sum(got) == 200
    try:
        import torch
        if hasattr(torch, "tensor"):
            assert fn(torch.tensor(logits), torch.tensor(targets), 5) == got
    except ImportError:
        pass
    raises(ValueError, lambda: fn(logits, targets[:-1], 5))
    raises(ValueError, lambda: fn(logits, targets + 1, 5))
    raises(ValueError, lambda: fn(logits, targets - 1, 5))
    # a class that never appears in targets or predictions is fine
    one_hot = np.zeros((6, 5)); one_hot[:, 2] = 1.0
    assert fn(one_hot, np.array([2, 2, 2, 2, 2, 1]), 5)[2 * 5 + 2] == 5

sv.ExperimentRunner = _REAL_RUNNER
logging.disable(logging.NOTSET)
width = max(len(n) for n, _s, _d in RESULTS)
for name, status, detail in RESULTS:
    print(f"{status}  {name}" + (f"   [{detail}]" if detail else ""))
failed = [n for n, s, _d in RESULTS if s != "PASS"]
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
sys.exit(1 if failed else 0)
