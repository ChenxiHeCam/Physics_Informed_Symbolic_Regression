"""Full-pass clean-row index for a training cache: classify ALL rows by the
data-only pathology checks (target_zero / target_const / var_const / nonfinite /
extreme_range / too_few) and save a boolean mask of CLEAN rows + per-structure
tags for oversampling.

structure tags (from the decoded seq, for v3 oversampling):
  has_div, has_sqrt, has_trans (exp/log/trig), deep (>=2 nested Div), implicit
  (target var appears >1 time or in a denominator).

Usage: python build_clean_index.py --cache <dir> --out clean_v6.npz
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data.ast_grammar import MAX_VARS, NT2ID, ID2NT

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
d = args.cache
ptr = np.load(os.path.join(d, 'ptr.npy'))
npts = np.load(os.path.join(d, 'npts.npy'))
nv = np.load(os.path.join(d, 'nv.npy'))
seq = np.load(os.path.join(d, 'seq.npy'))
slen = np.load(os.path.join(d, 'slen.npy'))
flat = np.memmap(os.path.join(d, 'flat.f32'), dtype=np.float32, mode='r')
N = len(npts)
print(f'cache {d}: {N} rows')

DIV = NT2ID.get('Div', NT2ID.get('div'))
SQRT = NT2ID.get('sqrt')
TRANS = set(NT2ID[k] for k in ('exp', 'log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh') if k in NT2ID)

clean = np.zeros(N, bool)
tag_div = np.zeros(N, bool); tag_sqrt = np.zeros(N, bool)
tag_trans = np.zeros(N, bool); tag_deep = np.zeros(N, bool); tag_struct = np.zeros(N, bool)
POW = NT2ID.get('Pow'); ADD = NT2ID.get('Add')
t0 = time.time()
for idx in range(N):
    n = int(npts[idx]); v = int(nv[idx])
    if n < 40:
        continue
    a, b = int(ptr[idx]), int(ptr[idx + 1])
    mat = np.asarray(flat[a:b]).reshape(n, v + 1)
    if not np.all(np.isfinite(mat)):
        continue
    y = mat[:, v]
    if np.all(y == 0) or np.std(y) < 1e-12 * (abs(np.mean(y)) + 1e-12):
        continue
    # var_const: a constant input column is ONLY pathological if it is NOT a
    # fixed fundamental constant. The generator fixes constants (k_B,c,hbar,...)
    # to exactly 1.0 -> a constant col == 1.0 is a legit fixed constant and the
    # row is a perfectly usable training example (y + real vars still vary).
    # Only drop if a constant col has a NON-unity value (a real var that
    # pathologically failed to vary). (Earlier over-aggressive rule wrongly
    # killed ~25% of GOOD physics rows that contain fundamental constants.)
    vc = False
    for j in range(v):
        c = mat[:, j]
        if np.std(c) < 1e-9 * (abs(np.mean(c)) + 1e-9) and abs(abs(np.mean(c)) - 1.0) > 1e-3:
            vc = True; break
    if vc:
        continue
    nz = np.abs(mat[np.abs(mat) > 0])
    if nz.size and nz.max() / max(nz.min(), 1e-30) > 1e12:
        continue
    clean[idx] = True
    L = int(slen[idx]); s = seq[idx, :L]
    if TRANS: tag_trans[idx] = np.any(np.isin(s, list(TRANS)))
    # tag_struct = Pow immediately followed by Add in preorder = (sum)^exp, i.e.
    # 1/(B+C), sqrt(B+C), (B+C)^n -> the #1 missing structure from the val diagnosis
    # (no Div/sqrt tokens in this grammar; division/sqrt are encoded via Pow).
    if L >= 2 and POW is not None:
        tag_struct[idx] = bool(np.any((s[:-1] == POW) & (s[1:] == ADD)))
    if idx % 500000 == 0:
        print(f'  {idx}/{N} clean={clean[:idx+1].sum()} ({time.time()-t0:.0f}s)', flush=True)

nc = int(clean.sum())
np.savez(args.out, clean=clean, tag_div=tag_div, tag_sqrt=tag_sqrt,
         tag_trans=tag_trans, tag_deep=tag_deep, tag_struct=tag_struct)
print(f'\n=== clean index -> {args.out} ===')
print(f'  clean rows: {nc}/{N} = {100*nc/N:.1f}%')
print(f'  among clean: struct(Pow+Add)={int((tag_struct&clean).sum())} '
      f'trans={int((tag_trans&clean).sum())}')
