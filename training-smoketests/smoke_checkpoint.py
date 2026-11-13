import sys, os, time, random, inspect, logging, tempfile

from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR

import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine



from models.coatnet_lora import create_coatnet_lora_model, freeze_bn_eval, split_trainable_parameters

from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint



torch.set_num_threads(4)

TMP = Path(tempfile.mkdtemp(prefix="ckpt_smoke_"))

RESULTS = []





def check(name):

    def deco(fn):

        try:

            fn()

            RESULTS.append((name, "PASS", ""))

        except Exception as e:

            RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:300]}"))

        return fn

    return deco





def must_raise(fn):

    try:

        fn()

    except Exception:

        return

    raise RuntimeError("expected an exception, none raised")





def real(rank=2):

    torch.manual_seed(0)

    return create_coatnet_lora_model(lora_rank=rank, lora_alpha=2 * rank, pretrained=False)





def make_opt(m):

    lora, head = split_trainable_parameters(m)

    return torch.optim.AdamW([{"params": lora, "lr": 1e-4}, {"params": head, "lr": 1e-3}], weight_decay=0.01)





def tvec(m):

    return torch.cat([p.detach().flatten() for p in m.parameters() if p.requires_grad])





def core(m):

    return getattr(m, "_module", m)





def step(m, opt):

    x, y = torch.randn(2, 3, 224, 224), torch.randint(0, 5, (2,))

    m.train(); freeze_bn_eval(m)

    loss = F.cross_entropy(m(x), y)

    opt.zero_grad(); loss.backward(); opt.step()





def dp_stack(noise=0.0, max_norm=1e6):

    m = real()

    opt = make_opt(m)

    g = torch.Generator().manual_seed(0)

    x = torch.randn(2, 3, 224, 224, generator=g)

    y = torch.randint(0, 5, (2,), generator=g)

    dl = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    pe = PrivacyEngine(accountant="rdp")

    dp_m, dp_opt, dp_dl = pe.make_private(

        module=m, optimizer=opt, data_loader=dl,

        noise_multiplier=noise, max_grad_norm=max_norm, poisson_sampling=False,

    )

    return pe, dp_m, dp_opt, dp_dl





def dp_step(dp_m, dp_opt, dl):

    x, y = next(iter(dl))

    dp_m.train(); freeze_bn_eval(dp_m)

    F.cross_entropy(dp_m(x), y).backward()

    dp_opt.step(); dp_opt.zero_grad()





def sched(o):
    return LambdaLR(o, lambda s: min(1.0, (s + 1) / 5))


def raises(exc, fn, needle=None):
    try:
        fn()
    except exc as e:
        if needle:
            assert needle in str(e), f"message lacks {needle!r}: {str(e)[:200]}"
        return
    raise RuntimeError(f"expected {exc.__name__}, nothing raised")


class Cap(logging.Handler):

    def __init__(self):

        super().__init__()

        self.msgs = []



    def emit(self, r):

        self.msgs.append(r.getMessage())





@check("C1 plain round trip: weights, outputs, optimizer state, param-group LRs, metadata")

def _():

    m = real(); opt = make_opt(m)

    torch.manual_seed(5)

    for _ in range(2):

        step(m, opt)

    p = TMP / "c1.pt"

    save_checkpoint(str(p), model=m, optimizer=opt, round_offset=3, wandb_run_id="abc", extra={"epoch": 4})

    m2 = real(); opt2 = make_opt(m2)

    assert not torch.equal(tvec(m), tvec(m2))

    info = load_checkpoint(str(p), model=m2, optimizer=opt2)

    assert torch.equal(tvec(m), tvec(m2))

    x = torch.randn(2, 3, 224, 224)

    m.eval(); m2.eval()

    with torch.no_grad():

        assert torch.allclose(m(x), m2(x), atol=1e-6)

    s1, s2 = opt.state_dict(), opt2.state_dict()

    assert [g["lr"] for g in s1["param_groups"]] == [g["lr"] for g in s2["param_groups"]]

    assert s1["state"].keys() == s2["state"].keys() and len(s1["state"]) > 0

    for k in s1["state"]:

        for n in ("exp_avg", "exp_avg_sq", "step"):

            assert torch.equal(s1["state"][k][n], s2["state"][k][n])

    assert info["round_offset"] == 3 and info["wandb_run_id"] == "abc" and info["extra"] == {"epoch": 4}

    assert info["opacus_accountant_states"] is None





@check("C2 RNG: torch, numpy, python and an extra generator resume exactly")

def _():

    m = real(); opt = make_opt(m)

    g = torch.Generator().manual_seed(11)

    torch.manual_seed(1); np.random.seed(2); random.seed(3)

    p = TMP / "c2.pt"

    save_checkpoint(str(p), model=m, optimizer=opt, extra_generators={"loader": g})

    draw = lambda: (torch.rand(3), np.random.rand(3), [random.random() for _ in range(3)], torch.rand(3, generator=g))

    a = draw()

    torch.manual_seed(100); np.random.seed(100); random.seed(100); g.manual_seed(100)

    load_checkpoint(str(p), model=m, optimizer=opt, extra_generators={"loader": g})

    b = draw()

    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2] and torch.equal(a[3], b[3])





@check("C2b generator missing from checkpoint -> warning, no crash")

def _():

    m = real(); opt = make_opt(m)

    p = TMP / "c2b.pt"

    save_checkpoint(str(p), model=m, optimizer=opt)

    cap = Cap()

    lg = logging.getLogger("utils.checkpoint")

    lg.addHandler(cap)

    old = lg.level

    lg.setLevel(logging.WARNING)

    try:

        load_checkpoint(str(p), model=m, optimizer=opt, extra_generators={"loader": torch.Generator()})

    finally:

        lg.removeHandler(cap)

        lg.setLevel(old)

    assert any("no saved RNG state" in s for s in cap.msgs), cap.msgs





@check("C3 atomic write: failed save (at torch.save and at os.replace) keeps the old file, leaves no temp file")

def _():

    m = real(); opt = make_opt(m)

    d = TMP / "c3"; d.mkdir()

    p = d / "latest.pt"

    save_checkpoint(str(p), model=m, optimizer=opt, extra={"v": 1})

    before = p.read_bytes()



    def attempt():

        try:

            save_checkpoint(str(p), model=m, optimizer=opt, extra={"v": 2})

        except Exception:

            return

        raise AssertionError("save should have failed")



    orig_save, orig_rep = torch.save, os.replace

    try:

        torch.save = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated kill"))

        attempt()

    finally:

        torch.save = orig_save

    assert p.read_bytes() == before and [f.name for f in d.iterdir()] == ["latest.pt"]

    try:

        os.replace = lambda *a, **k: (_ for _ in ()).throw(OSError("simulated"))

        attempt()

    finally:

        os.replace = orig_rep

    assert p.read_bytes() == before and [f.name for f in d.iterdir()] == ["latest.pt"]





@check("C4 find_latest_checkpoint: newest by mtime, ignores .tmp, None when empty or missing")

def _():

    d = TMP / "c4"; d.mkdir()

    assert find_latest_checkpoint(str(d)) is None

    assert find_latest_checkpoint(str(d / "nope")) is None

    a, b, t = d / "round_01.pt", d / "round_02.pt", d / ".latest.pt.xyz.tmp"

    for f in (a, b, t):

        f.write_bytes(b"x")

    now = time.time()

    os.utime(a, (now - 100, now - 100)); os.utime(b, (now - 50, now - 50)); os.utime(t, (now, now))

    assert find_latest_checkpoint(str(d)) == str(b)

    os.utime(a, (now + 10, now + 10))

    assert find_latest_checkpoint(str(d)) == str(a)





@check("C5 wrong schema_version raises ValueError")

def _():

    m = real(); opt = make_opt(m)

    p = TMP / "c5.pt"

    save_checkpoint(str(p), model=m, optimizer=opt)

    ck = torch.load(str(p), weights_only=False)

    ck["schema_version"] = 999

    torch.save(ck, str(p))

    try:

        load_checkpoint(str(p), model=m, optimizer=opt)

    except ValueError:

        return

    raise RuntimeError("not rejected")





@check("C6 [B2] missing or extra adapter keys are rejected, not silently half-loaded")

def _():

    m = real(); opt = make_opt(m)

    p = TMP / "c6.pt"

    save_checkpoint(str(p), model=m, optimizer=opt)

    base = torch.load(str(p), weights_only=False)



    def variant(mutate):

        ck = {**base, "model_state_dict": dict(base["model_state_dict"])}

        mutate(ck["model_state_dict"])

        q = TMP / "c6_v.pt"

        torch.save(ck, str(q))

        m2 = real(); o2 = make_opt(m2)

        load_checkpoint(str(q), model=m2, optimizer=o2)



    must_raise(lambda: variant(lambda sd: [sd.pop(k) for k in [k for k in sd if "head" in k]]))

    must_raise(lambda: variant(lambda sd: sd.update(

        {"base_model.model.stages.2.blocks.0.attn.qkv.lora_A.default.weight_bogus": torch.zeros(1)})))





@check("C7a [B1] save_checkpoint accepts a DP-wrapped (GradSampleModule) model")

def _():

    pe, dp_m, dp_opt, dl = dp_stack()

    save_checkpoint(str(TMP / "c7a.pt"), model=dp_m, optimizer=dp_opt)





@check("C7b [B1] checkpoint from a DP-wrapped model loads into a plain peft model and a fresh DP-wrapped one")

def _():

    pe, dp_m, dp_opt, dl = dp_stack()

    dp_step(dp_m, dp_opt, dl)

    p = TMP / "c7b.pt"

    save_checkpoint(str(p), model=dp_m, optimizer=dp_opt)

    plain = real(); o1 = make_opt(plain)

    load_checkpoint(str(p), model=plain, optimizer=o1)

    assert torch.equal(tvec(plain), tvec(dp_m))

    pe2, dp_m2, dp_opt2, _ = dp_stack()

    load_checkpoint(str(p), model=dp_m2, optimizer=dp_opt2)

    assert torch.equal(tvec(dp_m2), tvec(dp_m))

    assert len(dp_opt2.state_dict()["state"]) == len(dp_opt.state_dict()["state"]) > 0





@check("C8 accountant blob (rdp and prv): survives save/load, epsilon identical, history keeps growing")

def _():

    for mech in ("rdp", "prv"):

        pe = PrivacyEngine(accountant=mech)

        for _ in range(3):

            pe.accountant.step(noise_multiplier=1.0, sample_rate=0.1)

        eps1 = pe.accountant.get_epsilon(delta=1e-5)

        m = real(); opt = make_opt(m)

        p = TMP / f"c8_{mech}.pt"

        save_checkpoint(str(p), model=m, optimizer=opt, opacus_accountant_states={"IDRiD": pe.accountant.state_dict()})

        info = load_checkpoint(str(p), model=m, optimizer=opt)

        pe2 = PrivacyEngine(accountant=mech)

        pe2.accountant.load_state_dict(info["opacus_accountant_states"]["IDRiD"])

        eps2 = pe2.accountant.get_epsilon(delta=1e-5)

        pe2.accountant.step(noise_multiplier=1.0, sample_rate=0.1)

        eps3 = pe2.accountant.get_epsilon(delta=1e-5)

        print(f"   {mech}: eps before {eps1:.4f}, restored {eps2:.4f}, after 1 more step {eps3:.4f}")

        assert abs(eps1 - eps2) < 1e-9 and eps3 > eps2





@check("C9 resume equivalence, non-DP: 2 steps + checkpoint + restart + 2 steps == 4 straight steps")

def _():

    m = real(); opt = make_opt(m)

    torch.manual_seed(7)

    for _ in range(4):

        step(m, opt)

    final = tvec(m)



    m = real(); opt = make_opt(m)

    torch.manual_seed(7)

    for _ in range(2):

        step(m, opt)

    p = TMP / "c9.pt"

    save_checkpoint(str(p), model=m, optimizer=opt)

    m2 = real(); opt2 = make_opt(m2)

    torch.manual_seed(123)

    load_checkpoint(str(p), model=m2, optimizer=opt2)

    for _ in range(2):

        step(m2, opt2)

    diff = (tvec(m2) - final).abs().max().item()

    print(f"   max abs diff: {diff:.3e}")

    assert diff < 1e-7





@check("C10 resume equivalence under DP (noise 1.0): params and accountant epsilon identical after restart")

def _():

    pe, dp_m, dp_opt, dl = dp_stack(noise=1.0, max_norm=1.0)

    torch.manual_seed(7)

    for _ in range(4):

        dp_step(dp_m, dp_opt, dl)

    final, eps_cont = tvec(dp_m), pe.accountant.get_epsilon(delta=1e-5)



    pe, dp_m, dp_opt, dl = dp_stack(noise=1.0, max_norm=1.0)

    torch.manual_seed(7)

    for _ in range(2):

        dp_step(dp_m, dp_opt, dl)

    p = TMP / "c10.pt"

    save_checkpoint(str(p), model=core(dp_m), optimizer=dp_opt, opacus_accountant_states={"c": pe.accountant.state_dict()})



    pe2, dp_m2, dp_opt2, dl2 = dp_stack(noise=1.0, max_norm=1.0)

    torch.manual_seed(123)

    info = load_checkpoint(str(p), model=core(dp_m2), optimizer=dp_opt2)

    pe2.accountant.load_state_dict(info["opacus_accountant_states"]["c"])

    for _ in range(2):

        dp_step(dp_m2, dp_opt2, dl2)

    diff = (tvec(dp_m2) - final).abs().max().item()

    eps_res = pe2.accountant.get_epsilon(delta=1e-5)

    print(f"   max abs diff {diff:.3e}, eps continuous {eps_cont:.4f}, eps resumed {eps_res:.4f}")

    assert diff < 1e-6 and abs(eps_cont - eps_res) < 1e-9





@check("C11 [M2] scheduler state saved and restored (needed for Experiment 1 resume)")

def _():

    assert "scheduler" in inspect.signature(save_checkpoint).parameters, "no scheduler argument"

    mk = lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda s: min(1.0, (s + 1) / 5))

    m = real(); opt = make_opt(m); sch = mk(opt)

    for _ in range(3):

        opt.step(); sch.step()

    p = TMP / "c11.pt"

    save_checkpoint(str(p), model=m, optimizer=opt, scheduler=sch)

    m2 = real(); opt2 = make_opt(m2); sch2 = mk(opt2)

    load_checkpoint(str(p), model=m2, optimizer=opt2, scheduler=sch2)

    assert sch2.last_epoch == sch.last_epoch == 3

    assert [g["lr"] for g in opt2.param_groups] == [g["lr"] for g in opt.param_groups]





@check("C13 [M1, if accepted] save/load work with optimizer=None (FL server has no optimizer)")

def _():

    m = real()

    p = TMP / "c13.pt"

    save_checkpoint(str(p), model=m, optimizer=None, round_offset=2)

    info = load_checkpoint(str(p), model=real(), optimizer=None)

    assert info["round_offset"] == 2





@check("C14 [info] checkpoint size with AdamW state")

def _():

    for r in (4, 32):

        m = real(r); opt = make_opt(m)

        torch.manual_seed(0)

        step(m, opt)

        p = TMP / f"c14_r{r}.pt"

        save_checkpoint(str(p), model=m, optimizer=opt)

        print(f"   rank {r}: {os.path.getsize(p) / 1e6:.2f} MB")





@check("N1 run_config: match passes (float tolerance), mismatches, missing key and None config all raise")
def _():
    cfg = {"experiment_id": "exp5", "lora_rank": 8, "noise_multiplier": 1.0372,
           "sample_rate": 1 / 6, "accountant": "rdp", "grad_sample_mode": "functorch"}
    m = real(); p = TMP / "n1.pt"
    save_checkpoint(str(p), model=m, run_config=cfg)
    info = load_checkpoint(str(p), model=real(), expected_config={**cfg, "noise_multiplier": 1.0372 + 1e-12})
    assert info["run_config"] == cfg
    for bad in ({"lora_rank": 4}, {"accountant": "prv"}, {"grad_sample_mode": "hooks"},
                {"noise_multiplier": 1.05}, {"not_saved_key": 1}):
        raises(ValueError, lambda b=bad: load_checkpoint(str(p), model=real(), expected_config=b))
    p2 = TMP / "n1b.pt"
    save_checkpoint(str(p2), model=m)
    raises(ValueError, lambda: load_checkpoint(str(p2), model=real(), expected_config={"lora_rank": 8}))
    load_checkpoint(str(p2), model=real())


@check("N2 optimizer/scheduler presence rules on load")
def _():
    m = real(); opt = make_opt(m); sch = sched(opt)
    pa, pb = TMP / "n2_none.pt", TMP / "n2_full.pt"
    save_checkpoint(str(pa), model=m)
    save_checkpoint(str(pb), model=m, optimizer=opt, scheduler=sch)
    m2 = real(); o2 = make_opt(m2)
    raises(ValueError, lambda: load_checkpoint(str(pa), model=m2, optimizer=o2), "optimizer")
    raises(ValueError, lambda: load_checkpoint(str(pa), model=real(), scheduler=sched(o2)), "scheduler")
    cap = Cap(); lg = logging.getLogger("utils.checkpoint")
    lg.addHandler(cap); old = lg.level; lg.setLevel(logging.WARNING)
    try:
        load_checkpoint(str(pb), model=real())
    finally:
        lg.removeHandler(cap); lg.setLevel(old)
    assert any("optimizer" in s for s in cap.msgs) and any("scheduler" in s for s in cap.msgs), cap.msgs


@check("N3 a schema_version 1 checkpoint is rejected")
def _():
    p = TMP / "n3.pt"
    save_checkpoint(str(p), model=real())
    ck = torch.load(str(p), weights_only=False); ck["schema_version"] = 1; torch.save(ck, str(p))
    raises(ValueError, lambda: load_checkpoint(str(p), model=real()))


@check("N4 same keys but different LoRA rank is rejected as shape-mismatched")
def _():
    p = TMP / "n4.pt"
    save_checkpoint(str(p), model=real(4))
    raises(ValueError, lambda: load_checkpoint(str(p), model=real(2)), "Shape-mismatched")


@check("N5 a non-peft model gets a clear TypeError")
def _():
    raises(TypeError, lambda: save_checkpoint(str(TMP / "n5.pt"), model=torch.nn.Linear(2, 2)))


@check("N6 [ordering] a rejected load leaves model, optimizer and RNG untouched (expected FAIL until patch)")
def _():
    src = real(); sopt = make_opt(src)
    torch.manual_seed(5); step(src, sopt)
    p = TMP / "n6.pt"
    save_checkpoint(str(p), model=src, optimizer=sopt, run_config={"lora_rank": 2})

    def attempt(**kw):
        m = real(); o = make_opt(m)
        before = tvec(m).clone()
        torch.manual_seed(321)
        r0 = torch.get_rng_state().clone()
        raises(ValueError, lambda: load_checkpoint(str(p), model=m, optimizer=o, **kw))
        assert torch.equal(tvec(m), before), "model weights were modified by a rejected load"
        assert torch.equal(torch.get_rng_state(), r0), "RNG was modified by a rejected load"
        assert len(o.state_dict()["state"]) == 0, "optimizer was modified by a rejected load"

    attempt(expected_config={"lora_rank": 99})
    attempt(scheduler=sched(make_opt(real())))


@check("N7 fsync runs before os.replace")
def _():
    calls = []
    of, orr = os.fsync, os.replace
    os.fsync = lambda fd: (calls.append("fsync"), of(fd))[1]
    os.replace = lambda a, b: (calls.append("replace"), orr(a, b))[1]
    try:
        save_checkpoint(str(TMP / "n7.pt"), model=real())
    finally:
        os.fsync, os.replace = of, orr
    assert calls == ["fsync", "replace"], calls


@check("N8 DP resume passing the Opacus-wrapped model directly, with run_config validation")
def _():
    cfg = {"accountant": "rdp", "noise_multiplier": 1.0, "max_grad_norm": 1.0}
    pe, dp_m, dp_opt, dl = dp_stack(noise=1.0, max_norm=1.0)
    torch.manual_seed(7)
    for _ in range(4):
        dp_step(dp_m, dp_opt, dl)
    final, eps_cont = tvec(dp_m), pe.accountant.get_epsilon(delta=1e-5)

    pe, dp_m, dp_opt, dl = dp_stack(noise=1.0, max_norm=1.0)
    torch.manual_seed(7)
    for _ in range(2):
        dp_step(dp_m, dp_opt, dl)
    p = TMP / "n8.pt"
    save_checkpoint(str(p), model=dp_m, optimizer=dp_opt, run_config=cfg,
                    opacus_accountant_states={"c": pe.accountant.state_dict()})
    pe2, dp_m2, dp_opt2, dl2 = dp_stack(noise=1.0, max_norm=1.0)
    torch.manual_seed(123)
    info = load_checkpoint(str(p), model=dp_m2, optimizer=dp_opt2, expected_config=cfg)
    pe2.accountant.load_state_dict(info["opacus_accountant_states"]["c"])
    for _ in range(2):
        dp_step(dp_m2, dp_opt2, dl2)
    diff = (tvec(dp_m2) - final).abs().max().item()
    eps_res = pe2.accountant.get_epsilon(delta=1e-5)
    print(f"   max abs diff {diff:.3e}, eps continuous {eps_cont:.4f}, eps resumed {eps_res:.4f}")
    assert diff < 1e-6 and abs(eps_cont - eps_res) < 1e-9


print()

for n, s, msg in RESULTS:

    print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))

raise SystemExit(any(s == "FAIL" for _, s, _ in RESULTS))