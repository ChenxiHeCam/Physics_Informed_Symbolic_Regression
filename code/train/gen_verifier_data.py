"""Generate verifier training data: for N cache rows, run the flow to get candidate formulas,
label each by STRUCTURAL match to the true formula (NOT numerical fit — so the verifier learns
true correctness, able to reject degenerate garbage that games the numerical scorer). Save
(z_d, z_f_candidate, label) so a small verifier net can learn P(correct | z_d, z_f).
Usage: python gen_verifier_data.py <ckpt> <cache> <N> <out.npz>
"""
import sys, random
import numpy as np
import torch
import torch.nn.functional as F
import sympy as sp
from train.train_manifold import Manifold, MemmapPairDataset, load_state_compat, build_ast_edges, MAX_VARS
from data.ast_grammar import expr_to_nodes, nodes_to_expr

DEVICE = 'cuda'
ckpt, cache_dir, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

ck = torch.load(ckpt, map_location=DEVICE)
d = ck['d']
model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                 n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                 dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 80),
                 n_ftokens=ck.get('n_ftokens', 0)).to(DEVICE)
load_state_compat(model, ck['model']); model.eval()
ds = MemmapPairDataset(cache_dir, max_rows=0, max_points=400)
exprs = open(f'{cache_dir}/exprs.txt', encoding='utf-8').read().splitlines()


def canon(e):
    try:
        return sp.simplify(sp.expand(e))
    except Exception:
        return None


def same_struct(cand_expr, true_expr, nv):
    # structural equivalence allowing free constants: substitute random values for vars,
    # fit a single scale? -> too slow. Use: simplify(cand/true) is a constant, OR
    # cand - true simplifies to something with no vars. Cheap proxy: compare on random points
    # after zeroing constants is hard; use symbolic ratio constant test.
    try:
        r = sp.simplify(cand_expr - true_expr)
        if r == 0:
            return True
        # allow overall multiplicative/additive constant differences
        syms = list(true_expr.free_symbols)
        if not syms:
            return False
        d1 = sp.diff(cand_expr, syms[0]); d2 = sp.diff(true_expr, syms[0])
        ratio = sp.simplify(d1 / d2)
        return len(ratio.free_symbols) == 0 and ratio != 0
    except Exception:
        return False


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
        return F.normalize(z, dim=-1)[0].cpu().numpy()
    except Exception:
        return None


rows = random.sample(range(len(ds)), min(N, len(ds)))
ZD, ZF, LAB = [], [], []
K = 12
done = 0
for idx in rows:
    b = ds.collate([idx])
    b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
    nv = int(ds.nv[idx])
    vo = [f'x{k}' for k in range(nv)] + ['ytarget']
    te = exprs[idx] if idx < len(exprs) else ''
    try:
        true_expr = nodes_to_expr([int(x) for x in ds.seq[idx, :int(ds.slen[idx])]],
                                  [float(x) for x in ds.consts[idx, :int(ds.slen[idx])]], vo)
    except Exception:
        continue
    with torch.no_grad():
        zd, dtok = model.data_enc(b['points'], b['var_mask'], b['point_mask'], return_tokens=True)
        zdn = F.normalize(zd, dim=-1)[0].cpu().numpy()
        zf_samp = model.flow.sample(dtok, k=K, n_steps=20)[0]
        zd_rep = zd.expand(zf_samp.size(0), -1)
        dtok_rep = dtok.expand(zf_samp.size(0), -1, -1)
        gens = model.decoder.generate(zd_rep, zf_samp, data_tokens=dtok_rep, max_len=70,
                                      greedy=False, temperature=0.9, n_vars=nv + 1)
    seen = set()
    for (seq, consts) in gens:
        try:
            ce = nodes_to_expr([int(x) for x in seq], [float(x) for x in consts], vo)
        except Exception:
            continue
        if ce is None:
            continue
        key = sp.srepr(ce) if hasattr(ce, 'free_symbols') else str(ce)
        if key in seen:
            continue
        seen.add(key)
        zfc = encode_formula(str(ce), vo)
        if zfc is None:
            continue
        lab = 1 if same_struct(ce, true_expr, nv) else 0
        ZD.append(zdn); ZF.append(zfc); LAB.append(lab)
    done += 1
    if done % 200 == 0:
        print(f'{done}/{len(rows)} | samples={len(LAB)} pos={sum(LAB)}', flush=True)

np.savez(out, zd=np.array(ZD, np.float32), zf=np.array(ZF, np.float32), lab=np.array(LAB, np.int8))
print(f'DONE: {len(LAB)} samples ({sum(LAB)} pos) -> {out}', flush=True)
