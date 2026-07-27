"""Scan the TRAINING cache for the same pathologies we cleaned from eval sets:
target column degenerate (all-zero / constant), constant variable columns,
non-finite values, extreme dynamic range, and (on a subsample) residual
self-consistency (does the stored data satisfy the stored formula?).

If the training pool is itself contaminated, the decoder learns noise mappings
-> part of the "model can't generate" wall is actually "trained on garbage".

Usage: python scan_train_cache.py --cache <dir> --n 60000 --resid-n 4000
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data.ast_grammar import MAX_VARS, ID2NT, NT2ID

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True)
ap.add_argument('--n', type=int, default=60000)
ap.add_argument('--resid-n', type=int, default=4000)
args = ap.parse_args()

d = args.cache
ptr = np.load(os.path.join(d, 'ptr.npy'))
npts = np.load(os.path.join(d, 'npts.npy'))
nv = np.load(os.path.join(d, 'nv.npy'))
seq = np.load(os.path.join(d, 'seq.npy'))
slen = np.load(os.path.join(d, 'slen.npy'))
consts = np.load(os.path.join(d, 'consts.npy'))
flat = np.memmap(os.path.join(d, 'flat.f32'), dtype=np.float32, mode='r')
N = len(npts)
print(f'cache {d}: {N} rows')

rng = np.random.default_rng(0)
idxs = rng.choice(N, min(args.n, N), replace=False)
bad = {'target_zero': 0, 'target_const': 0, 'var_const': 0, 'nonfinite': 0,
       'extreme_range': 0, 'too_few': 0}
ok = 0
for idx in idxs:
    n = int(npts[idx]); v = int(nv[idx])
    a, b = int(ptr[idx]), int(ptr[idx + 1])
    mat = np.asarray(flat[a:b]).reshape(n, v + 1)
    if n < 40:
        bad['too_few'] += 1; continue
    if not np.all(np.isfinite(mat)):
        bad['nonfinite'] += 1; continue
    y = mat[:, v]
    X = mat[:, :v]
    flagged = False
    if np.all(y == 0):
        bad['target_zero'] += 1; flagged = True
    elif np.std(y) < 1e-12 * (abs(np.mean(y)) + 1e-12):
        bad['target_const'] += 1; flagged = True
    if not flagged:
        for j in range(v):
            c = X[:, j]
            if np.std(c) < 1e-12 * (abs(np.mean(c)) + 1e-12):
                bad['var_const'] += 1; flagged = True; break
    if not flagged:
        amax = np.max(np.abs(mat[np.abs(mat) > 0])) if np.any(np.abs(mat) > 0) else 0
        amin = np.min(np.abs(mat[np.abs(mat) > 0])) if np.any(np.abs(mat) > 0) else 1
        if amax / max(amin, 1e-30) > 1e12:
            bad['extreme_range'] += 1; flagged = True
    if not flagged:
        ok += 1

tot = len(idxs)
print(f'\n=== TRAIN DATA SCAN (n={tot}) ===')
print(f'  clean: {ok} ({100*ok/tot:.1f}%)')
for k, c in sorted(bad.items(), key=lambda x: -x[1]):
    if c: print(f'  {k:14s} {c:6d} ({100*c/tot:.1f}%)')

# residual self-consistency on a subsample (decode seq -> expr, eval on stored data)
import sympy as sp
from train.eval_sr import nodes_to_skeleton
VARS = [f'VAR_{i}' for i in range(MAX_VARS)]
ridx = rng.choice(N, min(args.resid_n, N), replace=False)
rbad = 0; rok = 0; rskip = 0
t0 = time.time()
for idx in ridx:
    n = int(npts[idx]); v = int(nv[idx])
    L = int(slen[idx])
    s = [int(x) for x in seq[idx, :L]]
    cv = consts[idx, :L]
    vnf = [f'VAR_{i}' for i in range(v)]
    try:
        e, params = nodes_to_skeleton(s, vnf)
        if e is None:
            rskip += 1; continue
        # substitute the stored constant values for CONST nodes (in order)
        ci = [cv[i] for i in range(len(s)) if s[i] == NT2ID['CONST']]
        if len(ci) == len(params):
            e = e.subs({p: float(cc) for p, cc in zip(params, ci)})
        free = [str(x) for x in e.free_symbols]
        if not free:
            rskip += 1; continue
        a, b = int(ptr[idx]), int(ptr[idx + 1])
        mat = np.asarray(flat[a:b]).reshape(n, v + 1)
        cols = {f'VAR_{j}': mat[:, j] for j in range(v)}
        # target is VAR index v? target convention = last var slot. Build residual eval:
        fn = sp.lambdify([sp.Symbol(x) for x in free], e, 'numpy')
        res = np.asarray(fn(*[cols.get(x, mat[:, v]) for x in free]), float)
        res = np.broadcast_to(res, (n,))
        scale = np.median(np.abs(mat[:, v])) + 1e-12
        rel = np.median(np.abs(res)) / scale
        if np.isfinite(rel) and rel < 1e-2:
            rok += 1
        else:
            rbad += 1
    except Exception:
        rskip += 1
rtot = rok + rbad
if rtot:
    print(f'\n=== RESIDUAL self-consistency (n={rtot}, skip={rskip}, {time.time()-t0:.0f}s) ===')
    print(f'  data satisfies formula: {rok} ({100*rok/rtot:.1f}%)')
    print(f'  residual_bad:           {rbad} ({100*rbad/rtot:.1f}%)')
