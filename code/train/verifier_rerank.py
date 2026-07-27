"""Rerank candidates with the LEARNED verifier: score = verifier(z_d, z_f_reencoded). Test whether
it improves @1 vs numerical ranking on OOD test sets (does the AUC-0.99 verifier transfer?).
Outputs verifier-top-3 per task for manual review.
Usage: python verifier_rerank.py <ckpt> <verifier> <real591> <dump.json> <out.json>
"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from train.train_manifold import Manifold, load_state_compat, build_ast_edges, MAX_VARS
from train.eval_sr import tasks_from_real591
from data.ast_grammar import expr_to_nodes


class Verifier(nn.Module):
    def __init__(self, dim, h=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4 + 1, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, 1))

    def forward(self, zd, zf):
        cos = (F.normalize(zd, dim=-1) * F.normalize(zf, dim=-1)).sum(-1, keepdim=True)
        x = torch.cat([zd, zf, (zd - zf).abs(), zd * zf, cos], dim=-1)
        return self.net(x).squeeze(-1)

DEVICE = 'cuda'
ckpt, vckpt, real_path, dump_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

ck = torch.load(ckpt, map_location=DEVICE)
d = ck['d']
model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                 n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                 dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 80),
                 n_ftokens=ck.get('n_ftokens', 0)).to(DEVICE)
load_state_compat(model, ck['model']); model.eval()
vck = torch.load(vckpt, map_location=DEVICE)
verif = Verifier(vck['dim']).to(DEVICE); verif.load_state_dict(vck['model']); verif.eval()


def encode_formula(expr_str, var_order):
    try:
        r = expr_to_nodes(expr_str, var_order)
        if r is None or not r[0] or len(r[0]) > 200:
            return None
        seq = r[0]; edges = build_ast_edges(seq)
        ei = (torch.tensor(edges).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long))
        with torch.no_grad():
            z = model.formula_enc(torch.tensor(seq, device=DEVICE),
                                  torch.zeros(len(seq), device=DEVICE), ei.to(DEVICE),
                                  torch.zeros(len(seq), dtype=torch.long, device=DEVICE), 1)
        return F.normalize(z, dim=-1)[0]
    except Exception:
        return None


tasks = tasks_from_real591(real_path, 150)
dump = json.load(open(dump_path, encoding='utf-8'))
out = []
for t, dr in zip(tasks, dump):
    target, others, sv = t['target'], t['others'], t['symvals']
    nv = len(others); n = min(128, len(sv[target]))
    pts = np.zeros((1, n, MAX_VARS + 1), np.float32)
    for k, s in enumerate(others):
        pts[0, :, k] = np.asarray(sv[s], np.float32)[:n]
    pts[0, :, MAX_VARS] = np.asarray(sv[target], np.float32)[:n]
    vm = np.zeros((1, MAX_VARS), np.float32); vm[0, :nv] = 1
    pm = np.ones((1, n), np.float32)
    P, V, M = (torch.from_numpy(x).to(DEVICE) for x in (pts, vm, pm))
    with torch.no_grad():
        zd = model.data_enc(P, V, M)
        if isinstance(zd, tuple):
            zd = zd[0]
    zdn = F.normalize(zd, dim=-1)                                     # (1,d)
    vo = others + [target]
    scored = []
    for c in dr['candidates']:
        f = c['formula'].strip()
        if f in ('0', ''):
            continue
        zf = encode_formula(f, vo)
        if zf is None:
            s = -9.0
        else:
            with torch.no_grad():
                s = float(verif(zdn, zf.unsqueeze(0)).item())
        scored.append((s, f))
    scored.sort(key=lambda x: -x[0])
    seen = set(); top = []
    for s, f in scored:
        if f in seen:
            continue
        seen.add(f); top.append(f)
        if len(top) >= 3:
            break
    out.append({'i': dr['i'], 'truth': dr['truth'], 'target': target, 'others': others,
                'verif_top': top})
json.dump(out, open(out_path, 'w'))
print(f'DONE: verifier rerank -> {out_path} ({len(out)} tasks)')
