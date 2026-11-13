import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import timm
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.lora import LoraLayer

from models.coatnet_lora import (
    create_coatnet_lora_model,
    freeze_bn_eval,
    verify_lora_injection,
    split_trainable_parameters,
)

torch.set_num_threads(4)

RESULTS = []
BN = torch.nn.modules.batchnorm._BatchNorm


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, "PASS", ""))
        except Exception as e:
            RESULTS.append(
                (name, "FAIL", f"{type(e).__name__}: {str(e)[:300]}")
            )
        return fn

    return deco


def build(r, **kw):
    return create_coatnet_lora_model(
        lora_rank=r,
        lora_alpha=2 * r,
        pretrained=False,
        **kw,
    )


def trainable(m):
    return {
        n: p.detach().clone()
        for n, p in m.named_parameters()
        if p.requires_grad
    }


# ---------------------------------------------------------------------
# T0 — Inspect actual timm model structure
# ---------------------------------------------------------------------

base = timm.create_model(
    "coatnet_1_rw_224",
    pretrained=False,
    num_classes=5,
)

clf = base.get_classifier()

print("T0 head type:", type(base.head).__name__)
print("T0 classifier type:", type(clf).__name__)
print(
    "T0 direct-child match:",
    [n for n, m in base.named_children() if m is clf],
)
print(
    "T0 qualified name:",
    [n for n, m in base.named_modules() if m is clf],
)
print(
    "T0 top-level children:",
    [n for n, _ in base.named_children()],
)
print(
    "T0 head children:",
    [n for n, _ in base.head.named_children()],
)

del base


# ---------------------------------------------------------------------
# T1 — Correct PEFT modules_to_save target
# ---------------------------------------------------------------------

@check("T1 builds, modules_to_save key")
def _():
    m = build(2)

    key = m.peft_config["default"].modules_to_save

    print("   modules_to_save:", key)

    assert list(key) == ["head.fc"]


# ---------------------------------------------------------------------
# T2 — Verify actual injected LoRA modules
# ---------------------------------------------------------------------

@check("T2 LoRA targets: 32 LoraLayer modules, stages 2-3 only")
def _():
    names = [
        n
        for n, x in build(2).named_modules()
        if isinstance(x, LoraLayer)
    ]

    print("   LoRA module count:", len(names))

    assert len(names) == 32, len(names)

    assert all(
        ".stages.2." in n or ".stages.3." in n
        for n in names
    )

    assert sum(n.endswith(".qkv") for n in names) == 16
    assert sum(n.endswith(".proj") for n in names) == 16


# ---------------------------------------------------------------------
# T3 — Exact trainable parameter count
# ---------------------------------------------------------------------

@check("T3 trainable count == 38592*r + 3845")
def _():
    for r in (2, 4, 8):
        got = sum(
            p.numel()
            for p in build(r).parameters()
            if p.requires_grad
        )

        expected = 38592 * r + 3845

        print(
            f"   r={r}: got {got:,} "
            f"expected {expected:,}"
        )

        assert got == expected


# ---------------------------------------------------------------------
# T4 — Only LoRA + classifier are trainable
# ---------------------------------------------------------------------

@check("T4 only lora_A/B in stages 2-3 + head.fc are trainable")
def _():
    names = list(trainable(build(2)))

    lora = [
        n for n in names
        if "lora_" in n
    ]

    head = [
        n for n in names
        if "lora_" not in n
    ]

    print("   head trainable names:", head)

    assert all(
        ".stages.2." in n or ".stages.3." in n
        for n in lora
    )

    assert len(head) == 2
    assert all("head.fc" in n for n in head)


# ---------------------------------------------------------------------
# T5 — Frozen BN behavior
# ---------------------------------------------------------------------

@check("T5 BN eval survives peft wrap; train() flips, freeze_bn_eval restores")
def _():
    m = build(2)

    bns = [
        x for x in m.modules()
        if isinstance(x, BN)
    ]

    print("   BN modules:", len(bns))

    assert bns
    assert all(not b.training for b in bns)

    assert all(
        not p.requires_grad
        for b in bns
        for p in b.parameters()
    )

    m.train()

    assert all(
        b.training
        for b in bns
    )

    assert freeze_bn_eval(m) == len(bns)

    assert all(
        not b.training
        for b in bns
    )


# ---------------------------------------------------------------------
# T6 — LoRA B starts at zero
# ---------------------------------------------------------------------

@check("T6 fresh model == adapter-disabled model (LoRA B=0)")
def _():
    m = build(4).eval()

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        a = m(x)

        with m.disable_adapter():
            b = m(x)

    print(
        "   max abs diff:",
        (a - b).abs().max().item(),
    )

    assert torch.allclose(
        a,
        b,
        atol=1e-5,
    )


# ---------------------------------------------------------------------
# T7 — Verification must not mutate BN buffers
# ---------------------------------------------------------------------

@check("T7 verify_lora_injection leaves BN buffers untouched")
def _():
    m = build(2)

    snap = lambda: {
        k: v.clone()
        for k, v in m.state_dict().items()
        if "running_" in k or "num_batches" in k
    }

    before = snap()

    verify_lora_injection(
        m,
        lora_rank=2,
    )

    changed = [
        k
        for k in before
        if not torch.equal(
            before[k],
            snap()[k],
        )
    ]

    print(
        "   buffers changed:",
        len(changed),
        changed[:2],
    )

    assert not changed


# ---------------------------------------------------------------------
# T8 — Verification works for higher ranks
# ---------------------------------------------------------------------

@check("T8 verify_lora_injection passes at r=8 and r=32")
def _():
    for r in (8, 32):
        verify_lora_injection(
            build(r),
            lora_rank=r,
        )


# ---------------------------------------------------------------------
# T9 — Backward reaches LoRA B + head
# ---------------------------------------------------------------------

@check("T9 backward: LoRA B and head get nonzero grads")
def _():
    m = build(4)

    m.train()
    freeze_bn_eval(m)

    out = m(
        torch.randn(
            4,
            3,
            224,
            224,
        )
    )

    torch.nn.functional.cross_entropy(
        out,
        torch.tensor([0, 1, 2, 3]),
    ).backward()

    g = {
        n: p.grad
        for n, p in m.named_parameters()
        if p.requires_grad
    }

    assert all(
        v is not None
        for v in g.values()
    )

    assert (
        sum(
            v.abs().sum().item()
            for n, v in g.items()
            if "lora_B" in n
        )
        > 0
    )

    assert (
        sum(
            v.abs().sum().item()
            for n, v in g.items()
            if "head" in n
        )
        > 0
    )


# ---------------------------------------------------------------------
# T10 — Initialization determinism
# ---------------------------------------------------------------------

@check("T10 determinism under manual_seed")
def _():
    torch.manual_seed(7)
    a = trainable(build(4))

    torch.manual_seed(7)
    b = trainable(build(4))

    torch.manual_seed(8)
    c = trainable(build(4))

    assert all(
        torch.equal(a[k], b[k])
        for k in a
    )

    assert any(
        not torch.equal(a[k], c[k])
        for k in a
    )


# ---------------------------------------------------------------------
# T11 — PEFT state-dict roundtrip
# ---------------------------------------------------------------------

@check("T11 peft state-dict roundtrip (FL weight transfer)")
def _():
    torch.manual_seed(1)
    m1 = build(4).eval()

    torch.manual_seed(1)
    m2 = build(4).eval()

    with torch.no_grad():

        for p in (
            trainable(m1).keys()
            and [
                p
                for p in m1.parameters()
                if p.requires_grad
            ]
        ):
            p.add_(
                0.05 * torch.randn_like(p)
            )

        x = torch.randn(
            2,
            3,
            224,
            224,
        )

        assert not torch.allclose(
            m1(x),
            m2(x),
            atol=1e-4,
        )

        sd = get_peft_model_state_dict(m1)

        print(
            "   keys:",
            len(sd),
            "payload MB:",
            round(
                sum(
                    v.numel() * 4
                    for v in sd.values()
                ) / 1e6,
                2,
            ),
        )

        print(
            "   head keys:",
            [
                k
                for k in sd
                if "head" in k
            ],
        )

        assert all(
            "lora_" in k or "head" in k
            for k in sd
        )

        res = set_peft_model_state_dict(
            m2,
            sd,
        )

        print(
            "   unexpected keys:",
            getattr(
                res,
                "unexpected_keys",
                None,
            ),
        )

        assert torch.allclose(
            m1(x),
            m2(x),
            atol=1e-5,
        )


# ---------------------------------------------------------------------
# T12 — Input resolution
# ---------------------------------------------------------------------

@check("T12 img_size 224 and 384 build and forward")
def _():
    m = build(2).eval()

    with torch.no_grad():
        assert m(
            torch.randn(
                2,
                3,
                224,
                224,
            )
        ).shape == (2, 5)

    m = build(
        2,
        img_size=384,
    ).eval()

    with torch.no_grad():
        assert m(
            torch.randn(
                1,
                3,
                384,
                384,
            )
        ).shape == (1, 5)


# ---------------------------------------------------------------------
# T13 — Explicit LoRA/head parameter split
# ---------------------------------------------------------------------

@check("T13 split_trainable_parameters: 64 LoRA + 2 head, disjoint, complete")
def _():
    m = build(4)

    lora, head = split_trainable_parameters(m)

    print(
        "   lora tensors:",
        len(lora),
        "head tensors:",
        len(head),
    )

    assert len(lora) == 64
    assert len(head) == 2

    ids = (
        {id(p) for p in lora}
        | {id(p) for p in head}
    )

    assert len(ids) == (
        len(lora) + len(head)
    )

    assert ids == {
        id(p)
        for p in m.parameters()
        if p.requires_grad
    }


# ---------------------------------------------------------------------
# T14 — Negative tests
# ---------------------------------------------------------------------

@check("T14 verify actually fails when it should (negative tests)")
def _():

    # Wrong rank must be detected
    m = build(4)

    try:
        verify_lora_injection(
            m,
            lora_rank=8,
        )

        raise RuntimeError(
            "wrong-rank not caught"
        )

    except AssertionError:
        pass


    # Accidental frozen-backbone leak must be detected
    m = build(4)

    leak = [
        p
        for n, p in m.named_parameters()
        if "stem" in n
    ][0]

    leak.requires_grad = True

    try:
        verify_lora_injection(
            m,
            lora_rank=4,
        )

        raise RuntimeError(
            "leaked backbone param not caught"
        )

    except AssertionError:
        pass


# ---------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------

print()

for n, s, msg in RESULTS:
    print(
        f"{s:4}  {n}"
        + (f"\n      {msg}" if msg else "")
    )

raise SystemExit(
    any(
        s == "FAIL"
        for _, s, _ in RESULTS
    )
)