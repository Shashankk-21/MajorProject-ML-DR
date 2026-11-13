import ast
import inspect
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import preprocessing.dataset as ds
from preprocessing.dataset import DEFAULT_CLASS_WEIGHT_CAP, DatasetError, compute_class_weights

logging.disable(logging.CRITICAL)
RESULTS = []
K = 5


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, "PASS", ""))
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


def labels(counts):
    return np.repeat(np.arange(len(counts)), counts)


def exact_reference(counts, cap, k=K):
    """Independent closed-form solver: the weights with sum_c p_c * min(s * w_c, cap) = 1 over present classes."""
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum()
    present = np.where(counts > 0)[0]
    w = n / (k * counts[present])
    p = counts[present] / n
    out = np.zeros(len(counts))
    order = np.argsort(w)  # ascending: the largest weights are capped first
    for j in range(len(order), -1, -1):  # j classes stay uncapped
        unc, cpd = order[:j], order[j:]
        denom = float((p[unc] * w[unc]).sum())
        if denom == 0:
            continue
        s = (1.0 - cap * float(p[cpd].sum())) / denom
        if s <= 0:
            continue
        ok_unc = j == 0 or s * w[order[j - 1]] <= cap * (1 + 1e-12)
        ok_cpd = j == len(order) or s * w[order[j]] >= cap * (1 - 1e-12)
        if ok_unc and ok_cpd:
            out[present] = np.minimum(s * w, cap)
            return out
    raise RuntimeError("reference solver found no solution")


def plain(counts):
    counts = np.asarray(counts, dtype=np.float64)
    pr = counts > 0
    out = np.zeros(len(counts))
    out[pr] = counts.sum() / (pr.sum() * counts[pr])
    return out


def mean_weight(counts, w):
    counts = np.asarray(counts, dtype=np.float64)
    pr = counts > 0
    return float((counts[pr] / counts.sum() * w[pr]).sum())


def random_counts(rng, absent_prob=0.2):
    c = rng.integers(0, 4000, size=K) * (rng.random(K) > absent_prob)
    c[rng.integers(0, K)] += int(rng.integers(1, 50))
    return c


@check("W1 default cap is pinned at 10.0 (module constant, keyword-only argument default)")
def _():
    assert DEFAULT_CLASS_WEIGHT_CAP == 10.0
    p = inspect.signature(compute_class_weights).parameters
    assert p["cap"].default == DEFAULT_CLASS_WEIGHT_CAP and p["cap"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "DEFAULT_CLASS_WEIGHT_CAP" in ds.__all__
    raises(TypeError, lambda: compute_class_weights(labels([5, 5, 5, 5, 5]), 5, 10.0))


@check("W2 cap not engaging: output equals the plain formula N/(K_present*n_c) bit for bit, absent class = 0")
def _():
    rng = np.random.default_rng(0)
    seen = 0
    for _ in range(500):
        c = rng.integers(20, 400, size=K) * (rng.random(K) > 0.15)
        c[0] += 30
        if plain(c).max() > DEFAULT_CLASS_WEIGHT_CAP:
            continue
        seen += 1
        w = compute_class_weights(labels(c))
        assert np.array_equal(w, plain(c)), c
    assert seen > 100
    c = [70, 10, 10, 5, 5]
    assert np.array_equal(compute_class_weights(labels(c)), plain(c))
    assert np.array_equal(compute_class_weights(labels(c), cap=1e12), plain(c))  # a huge cap never engages
    assert np.array_equal(compute_class_weights(labels([50, 30, 20, 0, 5])), plain([50, 30, 20, 0, 5]))


@check("W3 hand-checked example [200,100,20,7,3]: exact weights, mean 1, max 10")
def _():
    c = [200, 100, 20, 7, 3]
    w = compute_class_weights(labels(c))
    assert np.allclose(w, [0.383333333, 0.766666667, 3.833333333, 10.0, 10.0], atol=1e-6), w
    assert abs(mean_weight(c, w) - 1) < 1e-9 and w.max() <= 10.0


@check("W4 cap engaging: matches the independent closed-form solver, mean 1, max <= cap, on 5000 random count vectors and 4 caps")
def _():
    rng = np.random.default_rng(1)
    engaged = 0
    for _ in range(5000):
        c = random_counts(rng)
        cap = float(rng.choice([2.0, 5.0, 10.0, 25.0]))
        w = compute_class_weights(labels(c), cap=cap)
        pr = c > 0
        assert abs(mean_weight(c, w) - 1) < 1e-9, (c, cap, w)
        assert w.max() <= cap + 1e-12 and np.all(w[pr] > 0) and np.all(w[~pr] == 0) and np.all(np.isfinite(w))
        assert np.allclose(w, exact_reference(c, cap), atol=1e-8, rtol=1e-8), (c, cap, w, exact_reference(c, cap))
        engaged += plain(c).max() > cap
    assert engaged > 500


@check("W5 two classes over the cap at once, and extreme imbalance where an under-cap class also ends up capped")
def _():
    c = [200, 100, 20, 7, 3]  # naive cap-then-renormalize leaves 11.2 and 10.6
    w = compute_class_weights(labels(c))
    assert (w >= 10.0 - 1e-9).sum() == 2 and w.max() <= 10.0
    c = [95000, 4000, 600, 60, 20]
    w = compute_class_weights(labels(c))
    assert abs(mean_weight(c, w) - 1) < 1e-9 and w.max() <= 10.0
    assert (w >= 10.0 - 1e-9).sum() == 4 and 0.5 < w[0] < 0.6  # forced by mean == 1 with a hard cap
    assert np.allclose(w, exact_reference(c, 10.0), atol=1e-8)


@check("W6 absent classes: weight 0 (not the cap), no NaN or inf, mean 1 over the present classes, with and without the cap engaging")
def _():
    for c in ([50, 30, 20, 0, 5], [500, 300, 2, 0, 1], [0, 0, 0, 40, 10], [10, 0, 0, 0, 0], [0, 7, 0, 0, 3]):
        w = compute_class_weights(labels(c))
        assert w.shape == (5,) and np.all(np.isfinite(w)) and np.all(w[np.array(c) == 0] == 0.0), (c, w)
        assert abs(mean_weight(c, w) - 1) < 1e-9 and w.max() <= 10.0
    assert compute_class_weights(labels([0, 0, 40, 0, 0]))[2] == 1.0  # one class present: weight 1


@check("W7 cap = 1.0 gives all ones over the present classes; a very large cap reproduces the plain formula")
def _():
    c = [200, 100, 20, 7, 3]
    assert np.allclose(compute_class_weights(labels(c), cap=1.0), 1.0, atol=1e-9)
    w = compute_class_weights(labels([200, 100, 0, 7, 3]), cap=1.0)
    assert np.allclose(w[[0, 1, 3, 4]], 1.0) and w[2] == 0.0
    for big in (1e3, 1e6, 1e12, 1e300):
        assert np.array_equal(compute_class_weights(labels(c), cap=big), plain(c)), big


@check("W8 invalid cap values and bad labels raise DatasetError")
def _():
    y = labels([10, 10, 10, 10, 10])
    for bad in (0.5, 0, -1, 0.999, float("nan"), float("inf"), float("-inf"), True, False, "10", None, [10]):
        raises(DatasetError, lambda bad=bad: compute_class_weights(y, cap=bad))
    for bad in (np.array([]), np.array([[0, 1], [2, 3]]), np.array([0, 1, 5]), np.array([-1, 0]), np.array([0.5, 1.0]), np.array([0, 1, 7])):
        raises(DatasetError, lambda bad=bad: compute_class_weights(bad))
    assert compute_class_weights(labels([5, 5, 5, 5, 5]), cap=np.float64(10.0)).shape == (5,)
    assert compute_class_weights(labels([5, 5, 5, 5, 5]), cap=10).shape == (5,)  # int cap is fine


@check("W9 ordering and shape: a rarer class never gets a smaller weight; float64, shape (K,), input untouched; other dtypes and num_classes")
def _():
    rng = np.random.default_rng(2)
    for _ in range(2000):
        c = random_counts(rng)
        w = compute_class_weights(labels(c))
        pr = np.where(c > 0)[0]
        for a in pr:
            for b in pr:
                if c[a] < c[b]:
                    assert w[a] >= w[b] - 1e-12, (c, w)
                if c[a] == c[b]:
                    assert abs(w[a] - w[b]) < 1e-12
    y = labels([60, 20, 10, 5, 5])
    y0 = y.copy()
    w = compute_class_weights(y)
    assert w.dtype == np.float64 and w.shape == (5,) and np.array_equal(y, y0)
    assert np.array_equal(compute_class_weights(y.astype(np.int32)), w)
    assert np.array_equal(compute_class_weights(y.astype(np.float64)), w)
    assert np.array_equal(compute_class_weights(list(y)), w)
    w3 = compute_class_weights(labels([100, 20, 2]), 3)
    assert w3.shape == (3,) and abs(mean_weight([100, 20, 2], w3) - 1) < 1e-9 and w3.max() <= 10.0


@check("W10 every call site in the repo uses the default cap (no cap keyword, at most labels and num_classes positionally)")
def _():
    sites = []
    for path in ROOT.rglob("*.py"):
        parts = set(path.relative_to(ROOT).parts)
        if parts & {"training-smoketests", "tests", "archived_destructive", ".git", "__pycache__", "notebooks"}:
            continue
        if path.name in {"dataset.py", "print_class_weight_table.py"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                if name == "compute_class_weights":
                    assert not any(k.arg == "cap" for k in node.keywords), (path, node.lineno)
                    assert len(node.args) <= 2, (path, node.lineno)
                    sites.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    print("   call sites:", ", ".join(sorted(sites)) or "none")
    assert len(sites) >= 2, sites


@check("W11 the training loss reads class_weights[target] only: an absent class's weight (0, 10 or 1e6) changes neither loss nor gradient")
def _():
    import torch
    from training.local_trainer import weighted_label_smoothed_ce
    g = torch.Generator().manual_seed(0)
    logits0 = torch.randn(64, 5, generator=g)
    targets = torch.randint(0, 5, (64,), generator=g)
    targets[targets == 3] = 2  # class 3 never occurs in this batch
    base = torch.as_tensor(compute_class_weights(labels([200, 100, 20, 0, 3])), dtype=torch.float32)
    assert base[3] == 0
    outs = []
    for absent_w in (0.0, 10.0, 1e6):
        w = base.clone()
        w[3] = absent_w
        logits = logits0.clone().requires_grad_(True)
        loss = weighted_label_smoothed_ce(logits, targets, w, label_smoothing=0.1)
        loss.backward()
        outs.append((loss.detach().clone(), logits.grad.clone()))
    for loss, grad in outs[1:]:
        assert torch.equal(loss, outs[0][0]) and torch.equal(grad, outs[0][1])
    w2 = base.clone()
    w2[2] = w2[2] * 2
    assert not torch.equal(weighted_label_smoothed_ce(logits0, targets, w2, 0.1), outs[0][0])  # a present class does matter


@check("W12 loss = mean(class_weights[target] * label-smoothed CE) with the capped weights; differs from nn weight= (why the project avoids it)")
def _():
    import torch
    import torch.nn.functional as F
    from training.local_trainer import weighted_label_smoothed_ce
    g = torch.Generator().manual_seed(1)
    w = torch.as_tensor(compute_class_weights(labels([200, 100, 20, 7, 3])), dtype=torch.float32)
    logits = torch.randn(64, 5, generator=g)
    targets = torch.randint(0, 5, (64,), generator=g)
    got = weighted_label_smoothed_ce(logits, targets, w, 0.1)
    want = (w[targets] * F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.1)).mean()
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.allclose(got, F.cross_entropy(logits, targets, weight=w, label_smoothing=0.1))


print()
for n, s, msg in RESULTS:
    print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))
print(f"\n{sum(s == 'PASS' for _n, s, _m in RESULTS)}/{len(RESULTS)} passed")
raise SystemExit(any(s == "FAIL" for _n, s, _m in RESULTS))
