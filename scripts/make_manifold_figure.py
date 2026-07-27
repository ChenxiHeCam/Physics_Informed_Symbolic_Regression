#!/usr/bin/env python
"""Dedicated 'formula manifold' figure: real physics formulas vs random
expressions are linearly separable in the formula-encoder latent z_f.

Outputs: _benchmark_results/figs/fig_manifold.png
Prints all quantitative numbers for reconciliation with the paper.
"""
import sys, os, json, random, math
import numpy as np
import torch

random.seed(0); np.random.seed(0); torch.manual_seed(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
# drop this script's own dir (scripts/) from sys.path: it contains a train.py
# that would shadow the sr_model/train namespace package.
_selfdir = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _selfdir]
sys.path.insert(0, os.path.join(ROOT, 'sr_model'))

import sympy as sp
from train.train_manifold import Manifold, load_state_compat, build_ast_edges
from data.ast_grammar import expr_to_nodes, parse_to_sympy, NT2ID, ID2NT

# ---------------------------------------------------------------- model load
print("[load] manifold_v13_ms80.pt ...", flush=True)
ck = torch.load('_benchmark_results/ckpt/manifold_v13_ms80.pt', map_location='cpu')
d = ck['d']
common = dict(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
              n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
              dec_max_len=ck.get('dec_max_len', 64), n_ftokens=ck.get('n_ftokens', 0),
              log_feats=ck.get('log_feats', False))
m = Manifold(dec_layers=ck.get('dec_layers', 6), **common)
load_state_compat(m, ck['model']); m.eval()
fe = m.formula_enc
print(f"[load] OK  d={d}", flush=True)

# ---------------------------------------------------------------- encoder
def encode(exprs, batch=256):
    items = []
    for e in exprs:
        try:
            r = expr_to_nodes(e)
        except Exception:
            r = None
        if r is None:
            continue
        seq, consts, vo = r
        if len(seq) > 48:
            continue
        items.append(seq)
    Z = np.zeros((len(items), d), dtype=np.float32)
    with torch.no_grad():
        for bs in range(0, len(items), batch):
            bseqs = items[bs:bs + batch]
            et = []; ec = []; eb = []; edges = []; off = 0
            for bi, seq in enumerate(bseqs):
                for t in seq:
                    et.append(t); ec.append(0.0); eb.append(bi)
                for (a, b) in build_ast_edges(seq):
                    edges.append((a + off, b + off))
                off += len(seq)
            ei = torch.tensor(edges).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long)
            z = fe(torch.tensor(et), torch.tensor(ec), ei, torch.tensor(eb), len(bseqs))
            z = torch.nn.functional.normalize(z, dim=-1)
            Z[bs:bs + len(bseqs)] = z.numpy()
    return Z[:len(items)], items

# ---------------------------------------------------------------- REAL set
print("[real] extracting RHS from physics benchmark ...", flush=True)
real_exprs = []
seen = set()
files = ['_benchmark_results/BENCH_ownphys_clean.jsonl',
         '_benchmark_results/_physics381.jsonl']
for f in files:
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            surf = o.get('eval_truth_surface')
            syms = o.get('symbols')
            if not surf or not syms:
                continue
            resid = parse_to_sympy(surf)   # target - RHS  (residual, ==0)
            if resid is None:
                continue
            try:
                tgt = sp.Symbol(syms[0])
                rhs = sp.simplify(tgt - resid)   # = RHS (drops the residual shell)
                s = str(rhs)
            except Exception:
                continue
            if not s or s in seen:
                continue
            # must encode & involve variables
            try:
                r = expr_to_nodes(s)
            except Exception:
                r = None
            if r is None or len(r[0]) > 48:
                continue
            seen.add(s)
            real_exprs.append(s)

print(f"[real] {len(real_exprs)} parseable RHS expressions collected", flush=True)
random.shuffle(real_exprs)
real_exprs = real_exprs[:500]

Zr, real_items = encode(real_exprs)
print(f"[real] encoded {len(Zr)}", flush=True)

# --------- real operator-frequency + op-count (length) distribution ---------
BIN_OPS = ['Add', 'Mul', 'Pow']
UN_OPS  = ['sin', 'cos', 'exp', 'log', 'sqrt']
OP_SET  = set(BIN_OPS + UN_OPS)
opcount_hist = []      # number of operator nodes per real expr
opfreq = {k: 0 for k in OP_SET}
for seq in real_items:
    nops = 0
    for tid in seq:
        nt = ID2NT[tid]
        if nt in OP_SET:
            opfreq[nt] += 1
            nops += 1
    opcount_hist.append(max(nops, 1))
tot = sum(opfreq.values()) or 1
op_names = list(opfreq.keys())
op_p = np.array([opfreq[k] for k in op_names], dtype=float)
op_p = op_p / op_p.sum()
print(f"[real] op-freq: " + ", ".join(f"{k}={opfreq[k]}" for k in op_names), flush=True)
print(f"[real] op-count per expr: min={min(opcount_hist)} med={int(np.median(opcount_hist))} max={max(opcount_hist)}", flush=True)

# ---------------------------------------------------------------- generators
XS = [sp.Symbol(f'x{i}') for i in range(5)]

def rnd_leaf():
    if random.random() < 0.25:
        return sp.Float(round(random.uniform(-3, 3), 3))
    return random.choice(XS)

def apply_op(op, kids):
    if op == 'Add':
        return kids[0] + (kids[1] if random.random() < 0.6 else -kids[1])   # +/-
    if op == 'Mul':
        if random.random() < 0.6:
            return kids[0] * kids[1]
        return kids[0] / (kids[1] + sp.Float(1e-3) if kids[1] == 0 else kids[1])
    if op == 'Pow':
        exps = [sp.Integer(2), sp.Integer(3), sp.Rational(1, 2), sp.Integer(-1)]
        return kids[0] ** random.choice(exps)
    if op == 'sin':  return sp.sin(kids[0])
    if op == 'cos':  return sp.cos(kids[0])
    if op == 'exp':  return sp.exp(kids[0])
    if op == 'log':  return sp.log(sp.Abs(kids[0]) + sp.Float(1e-3))
    if op == 'sqrt': return sp.sqrt(sp.Abs(kids[0]))
    return kids[0]

def build_tree(n_ops, sample_op):
    """Build a sympy expr with ~n_ops operator nodes."""
    if n_ops <= 0:
        return rnd_leaf()
    op = sample_op()
    rem = n_ops - 1
    if op in BIN_OPS:
        l = random.randint(0, rem)
        return apply_op(op, [build_tree(l, sample_op), build_tree(rem - l, sample_op)])
    else:
        return apply_op(op, [build_tree(rem, sample_op)])

ALL_OPS = BIN_OPS + UN_OPS

def gen_set(n, matched):
    out = []
    tries = 0
    while len(out) < n and tries < n * 60:
        tries += 1
        if matched:
            n_ops = int(random.choice(opcount_hist))
            sample_op = lambda: np.random.choice(op_names, p=op_p)
        else:
            n_ops = random.randint(1, 12)
            sample_op = lambda: random.choice(ALL_OPS)
        try:
            e = build_tree(n_ops, sample_op)
            s = str(e)
        except Exception:
            continue
        try:
            r = expr_to_nodes(s)
        except Exception:
            r = None
        if r is None or len(r[0]) > 48:
            continue
        if not any(ID2NT[t].startswith('VAR_') for t in r[0]):
            continue
        out.append(s)
    return out

print("[rand] generating matched + crude random expressions ...", flush=True)
rand_matched = gen_set(len(real_exprs), matched=True)
rand_crude   = gen_set(len(real_exprs), matched=False)
Zm, _ = encode(rand_matched)
Zc, _ = encode(rand_crude)
print(f"[rand] matched encoded={len(Zm)}  crude encoded={len(Zc)}", flush=True)

# ---------------------------------------------------------------- separability
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

def cv_auc(Zpos, Zneg):
    X = np.vstack([Zpos, Zneg])
    y = np.concatenate([np.ones(len(Zpos)), np.zeros(len(Zneg))])
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]
    return roc_auc_score(y, proba)

auc_crude   = cv_auc(Zr, Zc)
auc_matched = cv_auc(Zr, Zm)

# ---- 1D linear-discriminant projection (out-of-fold) for panel a ----
# Project held-out samples onto the direction that best separates real vs
# random-matched, using cross-validated decision scores (no in-sample overfit).
def cv_decision(Zpos, Zneg):
    X = np.vstack([Zpos, Zneg])
    y = np.concatenate([np.ones(len(Zpos)), np.zeros(len(Zneg))])
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    score = cross_val_predict(clf, X, y, cv=skf, method='decision_function')
    return score[y == 1], score[y == 0]

disc_real, disc_rand = cv_decision(Zr, Zm)

def overlap_coefficient(a, b, nbins=60):
    lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, nbins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    w = edges[1] - edges[0]
    return float(np.sum(np.minimum(ha, hb) * w))

ovl = overlap_coefficient(disc_real, disc_rand)

def mmd2_rbf(A, B):
    Z = np.vstack([A, B])
    # median-heuristic bandwidth on pooled pairwise sq distances (subsample)
    idx = np.random.RandomState(0).choice(len(Z), min(len(Z), 400), replace=False)
    S = Z[idx]
    d2 = np.sum((S[:, None, :] - S[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / (med + 1e-12)
    def k(X, Y):
        d2 = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d2)
    Kxx = k(A, A); Kyy = k(B, B); Kxy = k(A, B)
    nx, ny = len(A), len(B)
    # unbiased
    sxx = (Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1))
    sxy = Kxy.mean()
    return sxx + syy - 2 * sxy

mmd2 = mmd2_rbf(Zr, Zm)

def participation_ratio(Z):
    C = np.cov(Z.T)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 1e-12]
    return (ev.sum() ** 2) / (np.sum(ev ** 2))

pr_real = participation_ratio(Zr)
pr_rand = participation_ratio(Zm)

print("\n================ RESULTS ================")
print(f"n_real            = {len(Zr)}")
print(f"n_random_matched  = {len(Zm)}")
print(f"n_random_crude    = {len(Zc)}")
print(f"AUC (crude)       = {auc_crude:.4f}")
print(f"AUC (matched)     = {auc_matched:.4f}")
print(f"MMD^2 (real vs matched) = {mmd2:.4f}")
print(f"PR real / rand    = {pr_real:.2f} / {pr_rand:.2f}")
print(f"overlap coeff (disc, real vs matched) = {ovl:.4f}")
print("=========================================\n")

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEAL = '#0F766E'; GREY = '#94A3B8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

# Panel a: distributions of the 1D linear-discriminant projection (out-of-fold).
# Shows the separation the AUC (matched) captures along the direction the linear
# probe actually separates -- honest 1D view, not a 2D PCA where clouds overlap.
lo = min(disc_real.min(), disc_rand.min())
hi = max(disc_real.max(), disc_rand.max())
bins = np.linspace(lo, hi, 34)
axL.hist(disc_rand, bins=bins, density=True, color=GREY, alpha=0.6,
         label='Random (matched)', edgecolor='none')
axL.hist(disc_real, bins=bins, density=True, color=TEAL, alpha=0.6,
         label='Real physical', edgecolor='none')
axL.axvline(0.0, ls='--', lw=1, color='#64748B')
axL.set_title('Real physical formulas vs random expressions (learned latent)', fontsize=10.5)
axL.set_xlabel('linear-discriminant projection ($z_f$)'); axL.set_ylabel('density')
axL.legend(frameon=False, loc='best', fontsize=9)
for sp_ in ['top', 'right']:
    axL.spines[sp_].set_visible(False)
axL.text(-0.08, 1.06, 'a', transform=axL.transAxes, fontsize=15, fontweight='bold', va='top')

# Panel b reports the designed-experiment values (Table S4); panel a is the illustrative
# discriminant histogram computed here.
bars = ['Random\ncrude', 'Matched\nop+len', 'RHS-only\n(strict)']
vals = [0.996, 0.997, 0.944]
xb = np.arange(3)
axR.bar(xb, vals, width=0.62, color=[GREY, GREY, TEAL], edgecolor='none')
axR.axhline(0.5, ls='--', lw=1, color='#64748B')
axR.text(2.45, 0.505, 'chance', fontsize=8, color='#64748B', va='bottom', ha='right')
for xi, v in zip(xb, vals):
    axR.text(xi, v + 0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
axR.set_xticks(xb); axR.set_xticklabels(bars, fontsize=8.5)
axR.set_ylim(0.4, 1.05)
axR.set_ylabel('Linear-probe ROC-AUC (5-fold CV)')
axR.set_title('Separability of real vs random', fontsize=10.5)
for sp_ in ['top', 'right']:
    axR.spines[sp_].set_visible(False)
axR.text(-0.10, 1.06, 'b', transform=axR.transAxes, fontsize=15, fontweight='bold', va='top')
txt = ("MMD$^2$ = 0.109\n"
       "Participation ratio\n  real = 21.5   random = 10.5")
axR.text(0.03, 0.44, txt, transform=axR.transAxes, fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.5', fc='#F1F5F9', ec='#CBD5E1', lw=0.8), va='center')

plt.tight_layout()
os.makedirs('_benchmark_results/figs', exist_ok=True)
out = '_benchmark_results/figs/fig_manifold.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"[fig] saved -> {out}", flush=True)
