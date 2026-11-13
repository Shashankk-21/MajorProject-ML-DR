import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

from models.coatnet_lora import create_coatnet_lora_model, freeze_bn_eval
from training.local_trainer import weighted_label_smoothed_ce

torch.set_num_threads(4)
W = torch.tensor([0.5, 2.0, 1.0, 3.0, 4.0])


def build(perturb):
    torch.manual_seed(0)
    m = create_coatnet_lora_model(lora_rank=2, lora_alpha=4, pretrained=False)
    if perturb:
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    return m


def ref_grads(m, x, y, loss):
    m.train(); freeze_bn_eval(m)
    out = m(x)
    loss(out, y).backward()
    g = {n: p.grad.clone() for n, p in m.named_parameters() if p.requires_grad}
    return g, out.detach().abs().max().item()


def dp_grads(m, x, y, loss):
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.0)
    dl = DataLoader(TensorDataset(x, y), batch_size=len(x))
    pe = PrivacyEngine(accountant="rdp")
    dp_m, dp_opt, _ = pe.make_private(
        module=m, optimizer=opt, data_loader=dl,
        noise_multiplier=0.0, max_grad_norm=1e6, poisson_sampling=False,
    )
    dp_m.train(); freeze_bn_eval(dp_m)
    loss(dp_m(x), y).backward()
    assert dp_opt.pre_step()
    return {n.replace("_module.", "", 1): p.grad.clone() for n, p in dp_m.named_parameters() if p.requires_grad}


def cat(n):
    return "lora_A" if "lora_A" in n else "lora_B" if "lora_B" in n else "head"


torch.manual_seed(1)
x = torch.randn(4, 3, 224, 224)
y = torch.randint(0, 5, (4,))

LOSSES = {
    "weighted": lambda lg, t: weighted_label_smoothed_ce(lg, t, W),
    "plain CE": lambda lg, t: F.cross_entropy(lg, t),
}

for perturb in (False, True):
    for lname, loss in LOSSES.items():
        ref, logit_max = ref_grads(build(perturb), x, y, loss)
        dp = dp_grads(build(perturb), x, y, loss)
        assert ref.keys() == dp.keys(), (set(ref) ^ set(dp))
        rows = []
        for n, a in ref.items():
            b = dp[n]
            na, nb, nd = a.norm().item(), b.norm().item(), (a - b).norm().item()
            rel = 0.0 if (na == 0 and nb == 0) else (nd / na if na > 0 else float("inf"))
            cos = (a.flatten() @ b.flatten()).item() / (na * nb) if na > 0 and nb > 0 else float("nan")
            rows.append((n, na, nb, rel, cos))
        print(f"\n=== lora_B perturbed={perturb} | loss={lname} | logits abs max {logit_max:.2f} ===")
        for c in ("lora_A", "lora_B", "head"):
            r = [t for t in rows if cat(t[0]) == c]
            zero = sum(1 for t in r if t[1] == 0 and t[2] == 0)
            print(f"  {c:7} n={len(r):2}  both-zero={zero:2}  max rel err={max(t[3] for t in r):.2e}  "
                  f"min cos={min((t[4] for t in r if t[4] == t[4]), default=float('nan')):.5f}")
        print("  worst 4 tensors (name | ref norm | dp norm | rel err | cos):")
        for n, na, nb, rel, cos in sorted(rows, key=lambda t: -t[3])[:4]:
            print(f"    {n[-58:]:58} {na:.3e} {nb:.3e} {rel:.2e} {cos:.5f}")