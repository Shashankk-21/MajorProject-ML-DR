import sys, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import opacus, peft, timm
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from peft import LoraConfig, get_peft_model
from timm.models.maxxvit import Attention2d

from models.coatnet_lora import create_coatnet_lora_model, freeze_bn_eval

torch.set_num_threads(4)
print("torch", torch.__version__, "| opacus", opacus.__version__, "| peft", peft.__version__, "| timm", timm.__version__)


def perturb_B(m):
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)


def real(perturb=True):
    torch.manual_seed(0)
    m = create_coatnet_lora_model(lora_rank=2, lora_alpha=4, pretrained=False)
    if perturb:
        perturb_B(m)
    return m


class AttnNet(nn.Module):
    def __init__(self, fused):
        super().__init__()
        self.pre = nn.Conv2d(3, 32, 1)
        self.attn = Attention2d(32, dim_head=8)
        self.attn.fused_attn = fused
        self.head = nn.Linear(32, 5)

    def forward(self, x):
        h = self.pre(x)
        return self.head((h + self.attn(h)).mean((2, 3)))


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 8, 1)
        self.conv1 = nn.Conv2d(8, 8, 1)
        self.head = nn.Linear(8, 5)

    def forward(self, x):
        return self.head(F.gelu(self.conv1(F.gelu(self.conv0(x)))).mean((2, 3)))


def mini(kind, fused=True, lora=False):
    torch.manual_seed(0)
    m = AttnNet(fused) if kind == "attn" else ConvNet()
    if lora:
        tm = ["qkv", "proj"] if kind == "attn" else ["conv1"]
        m = get_peft_model(m, LoraConfig(r=2, lora_alpha=4, target_modules=tm, modules_to_save=["head"]))
        perturb_B(m)
    return m


def ref_grads(m, x, y):
    m.train(); freeze_bn_eval(m)
    F.cross_entropy(m(x), y).backward()
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.requires_grad}


def loop_grads(m, x, y):
    acc = None
    for i in range(len(x)):
        m.zero_grad(set_to_none=True)
        m.train(); freeze_bn_eval(m)
        F.cross_entropy(m(x[i:i + 1]), y[i:i + 1]).backward()
        g = {n: p.grad.clone() for n, p in m.named_parameters() if p.requires_grad}
        acc = g if acc is None else {n: acc[n] + g[n] for n in acc}
    return {n: v / len(x) for n, v in acc.items()}


def dp_grads(m, x, y, mode="hooks"):
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.0)
    dl = DataLoader(TensorDataset(x, y), batch_size=len(x))
    pe = PrivacyEngine(accountant="rdp")
    dp_m, dp_opt, _ = pe.make_private(
        module=m, optimizer=opt, data_loader=dl, noise_multiplier=0.0,
        max_grad_norm=1e6, poisson_sampling=False, grad_sample_mode=mode,
    )
    dp_m.train(); freeze_bn_eval(dp_m)
    F.cross_entropy(dp_m(x), y).backward()
    assert dp_opt.pre_step()
    return {n.replace("_module.", "", 1): p.grad.clone() for n, p in dp_m.named_parameters() if p.requires_grad}


def compare(ref, other):
    rows = []
    for n, a in ref.items():
        b = other[n]
        na, nb, nd = a.norm().item(), b.norm().item(), (a - b).norm().item()
        rel = 0.0 if (na == 0 and nb == 0) else (nd / na if na > 0 else float("inf"))
        cos = (a.flatten() @ b.flatten()).item() / (na * nb) if na > 0 and nb > 0 else float("nan")
        rows.append((n, na, nb, rel, cos))
    return rows


def show_tensors(title, rows):
    print(f"\n--- {title}")
    for n, na, nb, rel, cos in rows:
        print(f"  {n[-46:]:46} ref {na:.2e} dp {nb:.2e} rel {rel:.1e} cos {cos:.4f}")


def grp(n):
    kind = "lora_A" if "lora_A" in n else "lora_B" if "lora_B" in n else "head" if "head" in n else "other"
    mod = "qkv" if ".qkv" in n else "proj" if ".proj" in n else ""
    st = re.search(r"stages\.(\d)", n)
    return f"{'s' + st.group(1) + ' ' if st else ''}{mod} {kind}".strip()


def show_groups(title, rows):
    print(f"\n--- {title}")
    groups = {}
    for r in rows:
        groups.setdefault(grp(r[0]), []).append(r)
    for g in sorted(groups):
        rs = groups[g]
        bad = sum(1 for t in rs if t[3] > 1e-3)
        cos = [t[4] for t in rs if t[4] == t[4]]
        print(f"  {g:16} n={len(rs):2}  bad(rel>1e-3)={bad:2}  min cos={min(cos, default=float('nan')):.4f}  max rel={max(t[3] for t in rs):.1e}")


def safe(title, fn, show):
    try:
        show(title, fn())
    except Exception as e:
        print(f"\n--- {title}\n  FAILED: {type(e).__name__}: {str(e)[:200]}")


torch.manual_seed(1)
X, Y = torch.randn(4, 3, 224, 224), torch.randint(0, 5, (4,))
xs, ys = torch.randn(4, 3, 6, 6), torch.randint(0, 5, (4,))

safe("E0 real model: batched autograd vs mean of 4 single-sample autograd (no Opacus)",
     lambda: compare(ref_grads(real(), X, Y), loop_grads(real(), X, Y)), show_groups)
safe("E1 real model: batched autograd vs Opacus hooks",
     lambda: compare(ref_grads(real(), X, Y), dp_grads(real(), X, Y, "hooks")), show_groups)
for mode in ("functorch", "ew"):
    safe(f"E2 real model: batched autograd vs Opacus grad_sample_mode={mode}",
         lambda mode=mode: compare(ref_grads(real(), X, Y), dp_grads(real(), X, Y, mode)), show_groups)
for fused in (True, False):
    safe(f"E3 plain timm Attention2d (fused_attn={fused}) under Opacus hooks, no peft",
         lambda fused=fused: compare(ref_grads(mini("attn", fused), xs, ys), dp_grads(mini("attn", fused), xs, ys)), show_tensors)
for fused in (True, False):
    safe(f"E4 peft LoRA on Attention2d qkv/proj (fused_attn={fused}) under Opacus hooks",
         lambda fused=fused: compare(ref_grads(mini("attn", fused, True), xs, ys), dp_grads(mini("attn", fused, True), xs, ys)), show_tensors)
safe("E5 peft LoRA on a plain 1x1 conv (no attention) under Opacus hooks",
     lambda: compare(ref_grads(mini("conv", lora=True), xs, ys), dp_grads(mini("conv", lora=True), xs, ys)), show_tensors)