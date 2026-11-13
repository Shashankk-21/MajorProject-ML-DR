import copy
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

import training.local_trainer as lt

torch.set_num_threads(4)
logging.disable(logging.CRITICAL)
DEV = torch.device("cpu")
RESULTS = []
W = torch.full((5,), 0.05)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 5)
        for p in self.bn.parameters():
            p.requires_grad = False
        self.bn.eval()

    def forward(self, x):
        return self.fc(self.bn(self.conv(x)).mean((2, 3)))


def tiny():
    torch.manual_seed(0)
    return Tiny()


def trainable(m):
    return [p for p in m.parameters() if p.requires_grad]


def data(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 8, 8, generator=g) * 0.1, torch.randint(0, 5, (n,), generator=g)


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, "PASS", ""))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:300]}"))
        return fn
    return deco


def run(model, x, y, bs, accum, mu=0.0, g=None, sched=None):
    """One epoch with lr=0 (weights never move); returns (grads seen at each optimizer step, epoch loss)."""
    dl = DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=False)
    opt = torch.optim.SGD(trainable(model), lr=0.0)
    grads, orig = [], opt.step

    def spy(*a, **k):
        grads.append([p.grad.detach().clone() for p in trainable(model)])
        return orig(*a, **k)

    opt.step = spy
    ret = lt.run_training_epoch(model, dl, opt, DEV, accumulation_steps=accum, class_weights=W, label_smoothing=0.1,
                                global_params=g, proximal_mu=mu, scheduler=sched)
    return grads, ret


def ref_grads(model, x, y, mu=0.0, g=None):
    """Example-weighted mean gradient of the whole (x, y) group, clipped like the trainer, plus one proximal gradient."""
    m = copy.deepcopy(model)
    m.zero_grad()
    loss = lt.weighted_label_smoothed_ce(m(x), y, W, label_smoothing=0.1)
    if g is not None:
        loss = loss + lt._fedprox_proximal_term(m, g, mu)
    loss.backward()
    clip_grad_norm_((p for p in m.parameters() if p.requires_grad), max_norm=1.0)
    return [p.grad.detach().clone() for p in trainable(m)]


def same(a, b, tol=1e-6):
    return all(torch.allclose(u, v, rtol=1e-4, atol=tol) for u, v in zip(a, b))


def total_norm(gs):
    return torch.sqrt(sum((t ** 2).sum() for t in gs)).item()


def prox_setup(m):
    torch.manual_seed(5)
    return [p.detach().clone() + 0.05 * torch.randn_like(p) for p in trainable(m)]


@check("M1 a 32+20 group steps with the EXAMPLE-weighted mean gradient (not the equal-weight mean of the two micro-batches)")
def _():
    x, y = data(52)
    m = tiny()
    grads, _ = run(m, x, y, 32, 2)
    assert len(grads) == 1
    want = ref_grads(tiny(), x, y)
    assert total_norm(want) < 0.9, "precondition: clip must be inactive"
    assert same(grads[0], want)
    a, b = ref_grads(tiny(), x[:32], y[:32]), ref_grads(tiny(), x[32:], y[32:])
    old = [(u + v) / 2 for u, v in zip(a, b)]
    assert not same(grads[0], old, tol=1e-7), "test cannot tell the old equal-weight rule apart"


@check("M2 FedProx adds ONE proximal gradient per update on a short group: step grad == example-weighted task grad + mu*(p - g)")
def _():
    x, y = data(52)
    m = tiny()
    g = prox_setup(m)
    grads, _ = run(m, x, y, 32, 2, mu=0.3, g=g)
    want = ref_grads(tiny(), x, y, mu=0.3, g=g)
    assert total_norm(want) < 0.9, "precondition: clip must be inactive or a doubled proximal term would be hidden"
    assert len(grads) == 1 and same(grads[0], want)
    once = ref_grads(tiny(), x, y, mu=0.0)
    twice = [w + (w - o) for w, o in zip(want, once)]
    assert not same(grads[0], twice, tol=1e-7), "test cannot tell a doubled proximal term apart"


@check("M3 several groups incl. a short tail (n=100, batch 32, accum 2): each update equals its own group's weighted mean (+ one prox)")
def _():
    x, y = data(100)
    for mu in (0.0, 0.3):
        m = tiny()
        g = prox_setup(m) if mu else None
        grads, _ = run(m, x, y, 32, 2, mu=mu, g=g)
        assert len(grads) == 2
        assert total_norm(ref_grads(tiny(), x[:64], y[:64], mu=mu, g=g)) < 0.9
        assert same(grads[0], ref_grads(tiny(), x[:64], y[:64], mu=mu, g=g))
        assert same(grads[1], ref_grads(tiny(), x[64:], y[64:], mu=mu, g=g))


@check("M4 full equal groups still equal the plain mean (n=64, batch 32, accum 2); accum 1 steps once per micro-batch")
def _():
    x, y = data(64)
    grads, _ = run(tiny(), x, y, 32, 2)
    assert len(grads) == 1 and same(grads[0], ref_grads(tiny(), x, y))
    grads, _ = run(tiny(), x, y, 32, 1)
    assert len(grads) == 2
    assert same(grads[0], ref_grads(tiny(), x[:32], y[:32])) and same(grads[1], ref_grads(tiny(), x[32:], y[32:]))


@check("M5 the gradient of one logical batch does not depend on how it is split into micro-batches (13x4, 26x2, 20+20+12, 52x1)")
def _():
    x, y = data(52)
    want = ref_grads(tiny(), x, y)
    for bs, accum in ((13, 4), (26, 2), (20, 3), (52, 1), (7, 8)):
        grads, _ = run(tiny(), x, y, bs, accum)
        assert len(grads) == 1 and same(grads[0], want), (bs, accum)


@check("M6 epoch training loss is the per-example mean: equals the full-batch loss and does not depend on the batch size")
def _():
    x, y = data(52)
    full = lt.weighted_label_smoothed_ce(tiny()(x), y, W, label_smoothing=0.1).item()
    vals = [run(tiny(), x, y, bs, 2)[1] for bs in (32, 52, 13, 7)]
    assert all(abs(v - full) < 1e-6 for v in vals), (vals, full)
    old = (lt.weighted_label_smoothed_ce(tiny()(x[:32]), y[:32], W, label_smoothing=0.1).item()
           + lt.weighted_label_smoothed_ce(tiny()(x[32:]), y[32:], W, label_smoothing=0.1).item()) / 2
    assert abs(old - full) > 1e-7, "test cannot tell the old mean-of-batch-means apart"


@check("M7 validation loss is the per-example mean: equals the full-batch cross-entropy for every batch size; outputs unchanged")
def _():
    x, y = data(52)
    m = tiny()
    m.eval()
    with torch.no_grad():
        out = m(x)
        full = F.cross_entropy(out, y).item()
    for bs in (32, 52, 13, 7, 1):
        v, lg, tg = lt.run_validation_epoch(m, DataLoader(TensorDataset(x, y), batch_size=bs), DEV)
        assert abs(v - full) < 1e-6, (bs, v, full)
        assert lg.shape == (52, 5) and tg.shape == (52,) and torch.allclose(torch.as_tensor(lg), out, atol=1e-6)


@check("M8 step and scheduler counts follow the groups: 52/32/2 -> 1, 100/32/2 -> 2, 40/4/4 -> 3, accum 1 -> one per micro-batch")
def _():
    class Ctr:
        n = 0

        def step(self):
            self.n += 1

    for n, bs, accum, want in ((52, 32, 2, 1), (100, 32, 2, 2), (40, 4, 4, 3), (40, 4, 1, 10), (330, 32, 2, 6)):
        x, y = data(n)
        s = Ctr()
        grads, _ = run(tiny(), x, y, bs, accum, sched=s)
        assert len(grads) == want and s.n == want, (n, bs, accum, len(grads), s.n)


@check("M9 the group-step helper ignores an empty group (no optimizer step) and divides by the group's example count otherwise")
def _():
    m = tiny()
    opt = torch.optim.SGD(trainable(m), lr=0.1)
    called = []
    orig = opt.step
    opt.step = lambda *a, **k: (called.append(1), orig(*a, **k))[1]
    lt._non_dp_group_step(m, opt, None, 0, None, 0.0)
    assert not called
    x, y = data(8)
    (lt.weighted_label_smoothed_ce(m(x), y, W, 0.1) * 8).backward()
    summed = [p.grad.detach().clone() for p in trainable(m)]
    lt._non_dp_group_step(m, opt, None, 8, None, 0.0)
    assert called == [1] and all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in trainable(m))
    assert total_norm(summed) > 0


print()
for n, s, msg in RESULTS:
    print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))
print(f"\n{sum(s == 'PASS' for _n, s, _m in RESULTS)}/{len(RESULTS)} passed")
raise SystemExit(any(s == "FAIL" for _n, s, _m in RESULTS))