import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from timm.layers import LayerNorm2d

from models.coatnet_lora import create_coatnet_lora_model, freeze_bn_eval

torch.set_num_threads(4)


class NormConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 8, 1)
        self.norm = LayerNorm2d(8)
        self.conv1 = nn.Conv2d(8, 8, 1)
        self.head = nn.Linear(8, 5)

    def forward(self, x):
        return self.head(F.gelu(self.conv1(self.norm(self.conv0(x)))).mean((2, 3)))


def mini():
    torch.manual_seed(0)
    return NormConv()


def real():
    torch.manual_seed(0)
    m = create_coatnet_lora_model(lora_rank=2, lora_alpha=4, pretrained=False)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    return m


def make_contig(m):
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d) and any(p.requires_grad for p in mod.parameters(recurse=False)):
            mod.register_forward_pre_hook(lambda mm, a: (a[0].contiguous(),) + tuple(a[1:]))


def grads_ref(m, x, y):
    m.train(); freeze_bn_eval(m)
    F.cross_entropy(m(x), y).backward()
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.requires_grad}


def grads_dp(m, x, y, mode):
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.0)
    dl = DataLoader(TensorDataset(x, y), batch_size=len(x))
    dp_m, dp_opt, _ = PrivacyEngine(accountant="rdp").make_private(
        module=m, optimizer=opt, data_loader=dl, noise_multiplier=0.0,
        max_grad_norm=1e6, poisson_sampling=False, grad_sample_mode=mode,
    )
    dp_m.train(); freeze_bn_eval(dp_m)
    F.cross_entropy(dp_m(x), y).backward()
    assert dp_opt.pre_step()
    return {n.replace("_module.", "", 1): p.grad.clone() for n, p in dp_m.named_parameters() if p.requires_grad}


def report(title, ref, dp):
    rel = {}
    for n, a in ref.items():
        na = a.norm().item()
        rel[n] = (a - dp[n]).norm().item() / na if na > 0 else 0.0
    bad = [n for n, v in rel.items() if v > 1e-3]
    print(f"{title:46} worst rel err {max(rel.values()):.1e}   bad tensors {len(bad)}/{len(rel)}")
    return bad


torch.manual_seed(1)
xs, ys = torch.randn(4, 3, 6, 6), torch.randint(0, 5, (4,))
X, Y = torch.randn(4, 3, 224, 224), torch.randint(0, 5, (4,))

r = grads_ref(mini(), xs, ys)
bad = report("mini (LayerNorm2d -> conv1) hooks", r, grads_dp(mini(), xs, ys, "hooks"))
print("   bad tensors:", bad)
report("mini functorch", r, grads_dp(mini(), xs, ys, "functorch"))
m = mini(); make_contig(m)
report("mini hooks + contiguous pre-hook", r, grads_dp(m, xs, ys, "hooks"))

r = grads_ref(real(), X, Y)
m = real(); make_contig(m)
report("real model hooks + contiguous pre-hook", r, grads_dp(m, X, Y, "hooks"))