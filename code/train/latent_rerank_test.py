"""Test the user's idea: rerank flow candidates by LATENT similarity cos(z_d, formula_enc(candidate))
instead of numerical fit. Degenerate garbage games the numerical scorer (fits sampled points) but
re-encodes to a nonsense z_f far from z_d -> latent scorer rejects it. Leverages retr@1 alignment.
Outputs latent-top-3 per task for reference judging vs the numerical order.
Usage: python latent_rerank_test.py <ckpt> <real591> <dump.json> <out.json>
"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F
from train.train_manifold import Manifold, load_state_compat, build_ast_edges, MAX_VARS
from train.eval_sr import tasks_from_real591
from data.ast_grammar import expr_to_nodes

DEVICE = 'cuda'
ckpt, real_path, dump_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

ck = torch.load(ckpt, map_location=DEVICE)
d = ck['d']
model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                 n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                 dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 80),
                 n_ftokens=ck.get('n_ftokens', 0)).to(DEVICE)
load_state_compat(model, ck['model']); model.eval()
fe = model.formula_enc


def encode_formula(expr_str, var_order):
    try:
        r = expr_to_nodes(expr_str, var_order)
    except Exception:
        return None
    if r is None:
        return None
    seq = r[0]
    if not seq or len(seq) > 200:
        return None
    edges = build_ast_edges(seq)
    ei = (torch.tensor(edges).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long))
    try:
        with torch.no_grad():
            z = fe(torch.tensor(seq, device=DEVICE),
                   torch.zeros(len(seq), device=DEVICE),
                   ei.to(DEVICE),
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
    zdn = F.normalize(zd, dim=-1)[0]
    vo = others + [target]
    scored = []
    for c in dr['candidates']:
        f = c['formula'].strip()
        if f in ('0', ''):
            continue
        zf = encode_formula(f, vo)
        s = float((zdn @ zf).item()) if zf is not None else -2.0
        scored.append((s, f))
    scored.sort(key=lambda x: -x[0])
    seen = set(); lat = []
    for s, f in scored:
        if f in seen:
            continue
        seen.add(f); lat.append(f)
        if len(lat) >= 3:
            break
    out.append({'i': dr['i'], 'truth': dr['truth'], 'target': target, 'others': others,
                'latent_top': lat})
json.dump(out, open(out_path, 'w'))
print(f'DONE: latent rerank -> {out_path} ({len(out)} tasks)')
