"""Train the verifier: (z_d, z_f_candidate) -> P(correct). Trained on gen_verifier_data output
where labels are STRUCTURAL match (not numerical fit) -> learns true correctness, rejects the
degenerate garbage that games the numerical scorer. Small MLP on latent features + interactions.
Balanced sampling for the ~6% positive rate. Used as a first-stage filter (top-20) before numerical.
Usage: python train_verifier.py <data.npz> <out.pt> [steps]
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

data_path, out_path = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

d = np.load(data_path)
zd = torch.tensor(d['zd'], dtype=torch.float32)
zf = torch.tensor(d['zf'], dtype=torch.float32)
lab = torch.tensor(d['lab'], dtype=torch.float32)
dim = zd.shape[1]
pos_idx = torch.nonzero(lab == 1).squeeze(1)
neg_idx = torch.nonzero(lab == 0).squeeze(1)
print(f'{len(lab)} samples | {len(pos_idx)} pos {len(neg_idx)} neg | dim {dim}', flush=True)


class Verifier(nn.Module):
    def __init__(self, dim, h=512):
        super().__init__()
        # features: z_d, z_f, |z_d-z_f|, z_d*z_f, cos
        self.net = nn.Sequential(
            nn.Linear(dim * 4 + 1, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, 1))

    def forward(self, zd, zf):
        cos = (F.normalize(zd, dim=-1) * F.normalize(zf, dim=-1)).sum(-1, keepdim=True)
        x = torch.cat([zd, zf, (zd - zf).abs(), zd * zf, cos], dim=-1)
        return self.net(x).squeeze(-1)


model = Verifier(dim).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
B = 512
zd, zf, lab = zd.to(DEVICE), zf.to(DEVICE), lab.to(DEVICE)

# held-out split for monitoring
perm = torch.randperm(len(lab), device=DEVICE)
n_val = len(lab) // 10
val_i, tr_i = perm[:n_val], perm[n_val:]
tr_pos = tr_i[lab[tr_i] == 1]; tr_neg = tr_i[lab[tr_i] == 0]

for step in range(1, STEPS + 1):
    # balanced batch: half pos, half neg
    pi = tr_pos[torch.randint(len(tr_pos), (B // 2,), device=DEVICE)]
    ni = tr_neg[torch.randint(len(tr_neg), (B // 2,), device=DEVICE)]
    idx = torch.cat([pi, ni])
    logit = model(zd[idx], zf[idx])
    loss = F.binary_cross_entropy_with_logits(logit, lab[idx])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 1000 == 0:
        with torch.no_grad():
            vl = model(zd[val_i], zf[val_i]); p = torch.sigmoid(vl)
            yv = lab[val_i]
            # rank-based: of val positives, what fraction score above the median negative?
            pos_s = p[yv == 1]; neg_s = p[yv == 0]
            auc = (pos_s.unsqueeze(1) > neg_s.unsqueeze(0)).float().mean().item()
            acc = ((p > 0.5).float() == yv).float().mean().item()
        print(f'step {step} loss={loss.item():.3f} val_acc={acc:.3f} val_auc={auc:.3f}', flush=True)

torch.save({'model': model.state_dict(), 'dim': dim}, out_path)
print(f'DONE -> {out_path}', flush=True)
