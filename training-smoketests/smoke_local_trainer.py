import sys, math, inspect, logging

from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import f1_score

from opacus import PrivacyEngine

from opacus.optimizers import DPOptimizer

from opacus.utils.batch_memory_manager import BatchMemoryManager



import training.local_trainer as lt

from models.coatnet_lora import create_coatnet_lora_model, freeze_bn_eval



run_training_epoch = lt.run_training_epoch

run_validation_epoch = lt.run_validation_epoch

compute_selection_metric = lt.compute_selection_metric

PROX = getattr(lt, "_fedprox_proximal_term", None)



torch.set_num_threads(4)

DEV = torch.device("cpu")

RESULTS = []

ONES = torch.ones(5)
GSM = "functorch"



SIG = inspect.signature(run_training_epoch)

CW_KW = next((k for k in SIG.parameters if "weight" in k.lower() and "global" not in k.lower()), None)

LOSS = getattr(lt, "weighted_label_smoothed_ce", None) or next(

    (getattr(lt, n) for n in dir(lt) if "weighted" in n.lower() and callable(getattr(lt, n))), None)

print("run_training_epoch params:", list(SIG.parameters))

print("class-weights kwarg:", CW_KW, "| loss function:", getattr(LOSS, "__name__", None))

assert CW_KW, "no class-weights parameter found in run_training_epoch; set CW_KW by hand"





def check(name):

    def deco(fn):

        try:

            fn()

            RESULTS.append((name, "PASS", ""))

        except Exception as e:

            RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:300]}"))

        return fn

    return deco





class Ctr:

    def __init__(self):

        self.n = 0



    def step(self):

        self.n += 1





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





def real():

    torch.manual_seed(0)

    return create_coatnet_lora_model(lora_rank=2, lora_alpha=4, pretrained=False)





def loader(n, bs, scale=1.0, seed=0, size=8):

    g = torch.Generator().manual_seed(seed)

    x = torch.randn(n, 3, size, size, generator=g) * scale

    y = torch.randint(0, 5, (n,), generator=g)

    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=False)





def trainable(m):

    return [p for p in m.parameters() if p.requires_grad]





def epoch(m, dl, opt, w=None, **kw):

    kw[CW_KW] = ONES if w is None else w

    return run_training_epoch(m, dl, opt, DEV, **kw)





def make_dp(model, dl, lr, noise=0.0, max_norm=1e3, poisson=False):

    opt = torch.optim.SGD(trainable(model), lr=lr)

    pe = PrivacyEngine(accountant="rdp")

    return pe.make_private(

        module=model, optimizer=opt, data_loader=dl,

        noise_multiplier=noise, max_grad_norm=max_norm, poisson_sampling=poisson,
        grad_sample_mode=GSM,

    )





def spy_steps(opt, norms, m):

    orig = opt.step



    def spy(*a, **k):

        norms.append(torch.sqrt(sum((p.grad ** 2).sum() for p in trainable(m) if p.grad is not None)).item())

        return orig(*a, **k)



    opt.step = spy





# ---------------- interface ----------------



@check("R1 Mixup fully removed (no mixup_fn param, no Mixup / SoftTargetCrossEntropy imported)")

def _():

    assert "mixup_fn" not in SIG.parameters

    assert not hasattr(lt, "Mixup") and not hasattr(lt, "SoftTargetCrossEntropy")





# ---------------- weighted loss ----------------



W = torch.tensor([0.5, 2.0, 1.0, 3.0, 4.0])





@check("W1 loss == (w[y] * CE(reduction=none, smoothing 0.1)).mean()")

def _():

    torch.manual_seed(0)

    lg, y = torch.randn(8, 5), torch.randint(0, 5, (8,))

    want = (W[y] * F.cross_entropy(lg, y, reduction="none", label_smoothing=0.1)).mean()

    got = LOSS(lg, y, W)

    print(f"   got {got.item():.6f} want {want.item():.6f}")

    assert torch.allclose(got, want, atol=1e-6)





@check("W2 no cross-sample coupling: batch loss == mean of single-sample losses")

def _():

    torch.manual_seed(1)

    lg, y = torch.randn(6, 5), torch.tensor([0, 1, 2, 3, 4, 4])

    batch = LOSS(lg, y, W)

    singles = torch.stack([LOSS(lg[i:i + 1], y[i:i + 1], W) for i in range(6)]).mean()

    print(f"   batch {batch.item():.6f} mean-of-singles {singles.item():.6f}")

    assert torch.allclose(batch, singles, atol=1e-6)





@check("W3 smoothing applied once (ones weights == plain smoothed CE), weights linear, zero weight kills a class")

def _():

    torch.manual_seed(2)

    lg, y = torch.randn(8, 5), torch.randint(0, 5, (8,))

    assert torch.allclose(LOSS(lg, y, ONES), F.cross_entropy(lg, y, label_smoothing=0.1), atol=1e-6)

    assert torch.allclose(LOSS(lg, y, 2 * W), 2 * LOSS(lg, y, W), atol=1e-6)

    z = torch.tensor([1., 1., 1., 1., 0.])

    assert LOSS(lg[:3], torch.tensor([4, 4, 4]), z).abs().item() < 1e-9





@check("W4 label_smoothing argument exists and 0.0 gives weighted plain CE")

def _():

    assert "label_smoothing" in inspect.signature(LOSS).parameters

    torch.manual_seed(3)

    lg, y = torch.randn(8, 5), torch.randint(0, 5, (8,))

    want = (W[y] * F.cross_entropy(lg, y, reduction="none")).mean()

    assert torch.allclose(LOSS(lg, y, W, label_smoothing=0.0), want, atol=1e-6)





# ---------------- non-DP path ----------------



@check("L1 cadence: 10 micro-batches, accum 4 -> 3 opt steps, 3 sched steps, clip <= 1.0")

def _():

    m = tiny()

    opt = torch.optim.SGD(trainable(m), lr=0.01)

    norms, sch = [], Ctr()

    spy_steps(opt, norms, m)

    epoch(m, loader(40, 4, scale=50.0), opt, accumulation_steps=4, scheduler=sch)

    print("   steps:", len(norms), "sched:", sch.n, "norms:", [round(x, 3) for x in norms])

    assert len(norms) == 3 and sch.n == 3

    assert all(x <= 1.0 + 1e-4 for x in norms)





@check("L1b [M3] final partial accumulation group is not under-scaled")

def _():

    m = tiny()

    x = torch.randn(1, 3, 8, 8).repeat(12, 1, 1, 1) * 0.1

    y = torch.ones(12, dtype=torch.long)

    dl = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    opt = torch.optim.SGD(trainable(m), lr=0.0)

    norms = []

    spy_steps(opt, norms, m)

    epoch(m, dl, opt, w=torch.full((5,), 0.1), accumulation_steps=2)

    print("   step grad norms (full group, partial group):", [round(n, 5) for n in norms])

    assert len(norms) == 2 and norms[0] < 0.99

    assert abs(norms[1] / norms[0] - 1.0) < 1e-3





@check("L2 frozen BN stays eval and its buffers are untouched through an epoch")

def _():

    m = tiny()

    seen = []

    m.bn.register_forward_hook(lambda mod, i, o: seen.append(mod.training))

    before = {k: v.clone() for k, v in m.bn.state_dict().items()}

    opt = torch.optim.SGD(trainable(m), lr=0.01)

    epoch(m, loader(16, 4), opt, accumulation_steps=2)

    assert seen and not any(seen)

    assert all(torch.equal(before[k], v) for k, v in m.bn.state_dict().items())





@check("L3 proximal term: value, gradient == mu*(p-g), length-mismatch assertion")

def _():

    m = tiny()

    tp = trainable(m)

    g = [p.detach().clone() + torch.randn_like(p) for p in tp]

    mu = 0.3

    val = PROX(m, g, mu)

    exp = mu / 2 * sum(((p.detach() - q) ** 2).sum() for p, q in zip(tp, g))

    assert torch.allclose(val, exp, atol=1e-6)

    val.backward()

    for p, q in zip(tp, g):

        assert torch.allclose(p.grad, mu * (p.detach() - q), atol=1e-6)

    try:

        PROX(m, g[:-1], mu)

    except AssertionError:

        return

    raise RuntimeError("length mismatch not caught")





@check("L4 non-DP FedProx pulls params toward global (mu=5 drifts less than mu=0)")

def _():

    def run(mu):

        m = tiny()

        tp = trainable(m)

        gp = [p.detach().clone() for p in tp]

        opt = torch.optim.SGD(tp, lr=0.1)

        kw = dict(global_params=gp, proximal_mu=mu) if mu > 0 else {}

        for _ in range(5):

            epoch(m, loader(32, 4), opt, accumulation_steps=2, **kw)

        return sum(((p.detach() - q) ** 2).sum().item() for p, q in zip(tp, gp))



    d0, d1 = run(0.0), run(5.0)

    print(f"   drift mu=0: {d0:.4f}   mu=5: {d1:.4f}")

    assert d1 < d0





@check("L5 validation epoch: shapes, eval mode, params untouched, no grads")

def _():

    m = tiny()

    before = [p.detach().clone() for p in m.parameters()]

    loss, lg, tg = run_validation_epoch(m, loader(10, 4), DEV)

    assert lg.shape == (10, 5) and tg.shape == (10,) and not m.training

    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))

    assert all(p.grad is None for p in m.parameters())





@check("L6 [M2] selection metric == macro-F1 with fixed labels 0-4, zero_division=0")

def _():

    cases = [

        ([0, 0, 1, 1, 2, 2, 4, 4], [0, 0, 1, 1, 2, 2, 4, 3]),

        ([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]),

        ([0, 1, 2, 3, 4, 0, 1], [0, 0, 0, 3, 4, 0, 2]),

    ]

    for t, p in cases:

        t, p = np.array(t), np.array(p)

        got = compute_selection_metric(np.eye(5)[p], t)

        want = f1_score(t, p, average="macro", labels=np.arange(5), zero_division=0)

        print(f"   got {got:.4f} want {want:.4f}")

        assert abs(got - want) < 1e-9





@check("L7 class weights act on non-DP training (zero weight on absent class => no update)")

def _():

    m = tiny()

    g = torch.Generator().manual_seed(0)

    x = torch.randn(16, 3, 8, 8, generator=g)

    y = torch.randint(0, 4, (16,), generator=g)

    dl = DataLoader(TensorDataset(x, y), batch_size=4)

    before = [p.detach().clone() for p in trainable(m)]

    opt = torch.optim.SGD(trainable(m), lr=0.1)

    epoch(m, dl, opt, w=torch.tensor([0., 0., 0., 0., 1.]), accumulation_steps=2)

    assert all(torch.equal(a, b) for a, b in zip(before, trainable(m)))

    epoch(m, dl, opt, accumulation_steps=2)

    assert any(not torch.equal(a, b) for a, b in zip(before, trainable(m)))





# ---------------- DP path ----------------



@check("D0 DPOptimizer detected by isinstance, plain SGD is not")

def _():

    _, dp_opt, _ = make_dp(tiny(), loader(8, 8), 0.1)

    assert isinstance(dp_opt, DPOptimizer)

    assert not isinstance(torch.optim.SGD(trainable(tiny()), lr=0.1), DPOptimizer)





@check("D1a [M1] scheduler passed on the DP path raises ValueError")

def _():

    dp_m, dp_opt, dp_dl = make_dp(tiny(), loader(16, 8), 0.01, noise=1.0, poisson=True)

    try:

        epoch(dp_m, dp_dl, dp_opt, accumulation_steps=1, scheduler=Ctr())

    except ValueError:

        return

    raise RuntimeError("no ValueError raised")





def dp_cadence(prox):

    torch.manual_seed(0)

    m = tiny()

    pe = PrivacyEngine(accountant="rdp")

    opt = torch.optim.SGD(trainable(m), lr=0.01)

    dp_m, dp_opt, dp_dl = pe.make_private(

        module=m, optimizer=opt, data_loader=loader(64, 16),

        noise_multiplier=1.0, max_grad_norm=1.0, poisson_sampling=True,
        grad_sample_mode=GSM,

    )

    real_steps, orig = Ctr(), dp_opt.original_optimizer.step



    def spy(*a, **k):

        real_steps.n += 1

        return orig(*a, **k)



    dp_opt.original_optimizer.step = spy

    kw = dict(global_params=[p.detach().clone() for p in trainable(dp_m)], proximal_mu=0.5) if prox else {}

    n_logical = len(dp_dl)

    with BatchMemoryManager(data_loader=dp_dl, max_physical_batch_size=6, optimizer=dp_opt) as bl:

        epoch(dp_m, bl, dp_opt, accumulation_steps=999, **kw)

    acc = sum(h[2] for h in pe.accountant.history)

    print(f"   logical batches {n_logical}, real steps {real_steps.n}, accountant steps {acc}")

    assert real_steps.n == n_logical == acc





@check("D1b DP + BatchMemoryManager: real steps == logical batches == accountant steps")

def _():

    dp_cadence(prox=False)





@check("D4 same with FedProx on: cadence intact, accountant still counts every step")

def _():

    dp_cadence(prox=True)





@check("D5 old 'NOT been verified' FedProx-under-DP warning is gone")

def _():

    class Cap(logging.Handler):

        def __init__(self):

            super().__init__()

            self.msgs = []



        def emit(self, r):

            self.msgs.append(r.getMessage())



    cap = Cap()

    root = logging.getLogger()

    root.addHandler(cap)

    old = root.level

    root.setLevel(logging.WARNING)

    try:

        dp_cadence(prox=True)

    finally:

        root.removeHandler(cap)

        root.setLevel(old)

    bad = [m for m in cap.msgs if "verified" in m.lower()]

    assert not bad, bad[:1]





def prox_shift(build, make_loader, lr, mu, w=None):

    outs = []

    for use_prox in (False, True):

        m = build()

        dp_m, dp_opt, dp_dl = make_dp(m, make_loader(), lr, noise=0.0, max_norm=1e3, poisson=False)

        tp = trainable(dp_m)

        gp = [p.detach().clone() + 1.0 for p in tp]

        kw = dict(global_params=gp, proximal_mu=mu) if use_prox else {}

        epoch(dp_m, dp_dl, dp_opt, w=w, accumulation_steps=1, **kw)

        outs.append([p.detach().clone() for p in tp])

    shifts = torch.cat([(b - a).flatten() for a, b in zip(*outs)])

    expected = lr * mu

    err = (shifts - expected).abs().max().item()

    print(f"   mean shift {shifts.mean().item():.6f}  expected {expected:.6f}  max err {err:.2e}")

    assert err < 1e-5, "FedProx shift under DP is wrong"





@check("D2 [B1] tiny model: DP update includes FedProx shift of exactly lr*mu (class-weighted loss)")

def _():

    prox_shift(tiny, lambda: loader(8, 8), lr=0.1, mu=1.0, w=W)





@check("D3 [B1] real CoAtNet+LoRA (rank 2): same exact-shift test under Opacus")

def _():

    prox_shift(real, lambda: loader(4, 4, size=224), lr=0.01, mu=1.0, w=W)





def grad_match(build, x, y):

    ref = build()

    ref.train()

    freeze_bn_eval(ref)

    LOSS(ref(x), y, W).backward()

    ref_g = [p.grad.clone() for p in ref.parameters() if p.requires_grad]



    dl = DataLoader(TensorDataset(x, y), batch_size=len(x))

    dp_m, dp_opt, _ = make_dp(build(), dl, 0.0, noise=0.0, max_norm=1e6, poisson=False)

    dp_m.train()

    freeze_bn_eval(dp_m)

    LOSS(dp_m(x), y, W).backward()

    assert dp_opt.pre_step()

    dp_g = [p.grad.clone() for p in dp_m.parameters() if p.requires_grad]



    assert len(ref_g) == len(dp_g)

    worst = max(((a - b).abs().max() / (a.abs().max() + 1e-12)).item() for a, b in zip(ref_g, dp_g))

    total = sum(g.abs().sum().item() for g in ref_g)

    print(f"   {len(ref_g)} tensors, worst relative error {worst:.2e}, total |grad| {total:.3e}")

    assert total > 0 and worst < 1e-3





@check("E1t Opacus per-sample grads == autograd for the weighted loss (tiny model)")

def _():

    torch.manual_seed(1)

    grad_match(tiny, torch.randn(6, 3, 8, 8), torch.tensor([0, 1, 2, 3, 4, 4]))





@check("E1 same on the real CoAtNet+LoRA model")

def _():

    torch.manual_seed(1)

    grad_match(real, torch.randn(4, 3, 224, 224), torch.randint(0, 5, (4,)))





@check("E2 real model, Poisson + BatchMemoryManager + noise: runs, accountant steps == logical batches")

def _():

    torch.manual_seed(2)

    m = real()

    opt = torch.optim.AdamW(trainable(m), lr=1e-4)

    pe = PrivacyEngine(accountant="rdp")

    dp_m, dp_opt, dp_dl = pe.make_private(

        module=m, optimizer=opt, data_loader=loader(16, 8, size=224),

        noise_multiplier=1.0, max_grad_norm=1.0, poisson_sampling=True,
        grad_sample_mode=GSM,

    )

    with BatchMemoryManager(data_loader=dp_dl, max_physical_batch_size=3, optimizer=dp_opt) as bl:

        epoch(dp_m, bl, dp_opt, w=W, accumulation_steps=1)

    eps = pe.accountant.get_epsilon(1e-5)

    steps = sum(h[2] for h in pe.accountant.history)

    print(f"   history {pe.accountant.history}, epsilon {eps:.3f}")

    assert math.isfinite(eps) and eps > 0 and steps == len(dp_dl)





@check("I1 [info] sample rate the DP loader really gets at batch 64")

def _():

    for name, n in (("APTOS", 2930), ("IDRiD", 330), ("Messidor-2", 1396), ("EyePACS", 70960)):

        L = math.ceil(n / 64)

        q = 1 / L

        print(f"   {name:10} len={L:5} q=1/len={q:.5f} (64/N={64 / n:.5f}) expected batch={int(n * q)}")





def real_perturbed():
    m = real()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    return m


def persample_check(mode, build, x, y, assert_ok=True):
    ref = build()
    ref.train(); freeze_bn_eval(ref)
    singles = []
    for i in range(len(x)):
        ref.zero_grad(set_to_none=True)
        LOSS(ref(x[i:i + 1]), y[i:i + 1], W).backward()
        singles.append([p.grad.clone() for p in ref.parameters() if p.requires_grad])

    m = build()
    opt = torch.optim.SGD(trainable(m), lr=0.0)
    dl = DataLoader(TensorDataset(x, y), batch_size=len(x))
    dp_m, dp_opt, _ = PrivacyEngine(accountant="rdp").make_private(
        module=m, optimizer=opt, data_loader=dl, noise_multiplier=0.0,
        max_grad_norm=1e6, poisson_sampling=False, grad_sample_mode=mode,
    )
    dp_m.train(); freeze_bn_eval(dp_m)
    LOSS(dp_m(x), y, W).backward()
    gs = [p.grad_sample for p in dp_m.parameters() if p.requires_grad]
    worst, bad, total = 0.0, 0, 0
    for k, g in enumerate(gs):
        for i in range(len(x)):
            ref_g = singles[i][k]
            n = ref_g.norm().item()
            rel = (g[i] - ref_g).norm().item() / n if n > 0 else 0.0
            worst, bad, total = max(worst, rel), bad + (rel > 1e-3), total + 1
    print(f"   {mode}: worst per-sample rel err {worst:.1e}, bad {bad}/{total}")
    if assert_ok:
        assert worst < 1e-3


X4, Y4 = torch.randn(4, 3, 224, 224), torch.randint(0, 5, (4,))


@check("E1p per-sample gradients == single-sample autograd, real model, functorch mode (the gate)")
def _():
    persample_check("functorch", real_perturbed, X4, Y4)


@check("E1p-hooks [info] same check in Opacus default hooks mode (documents the bug, no assertion)")
def _():
    persample_check("hooks", real_perturbed, X4, Y4, assert_ok=False)


print()

for n, s, msg in RESULTS:

    print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))

raise SystemExit(any(s == "FAIL" for _, s, _ in RESULTS))