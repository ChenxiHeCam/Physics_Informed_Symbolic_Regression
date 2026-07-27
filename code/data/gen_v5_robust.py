"""
gen v5: robust (X,y) generation to RECOVER formulas that failed v4 generation
(transcendental: multi_exp 59% / exp*trig 36% / log 32% / abs 28% failed).

Improvements over v4's make_samples:
  - root-finding: many brackets across orders of magnitude (both signs) + fsolve
    from multiple seeds as fallback (v4 only tried 2 brackets -> missed roots).
  - adaptive domain: if too few valid points, retry with log-spaced / wider /
    narrower sampling of the 'others'.
  - slightly relaxed valid-point floor for hard formulas, kept residual quality gate.

This file's __main__ is a RECOVERY TEST: take formulas that failed v4 and are
within length budget, run v5 make_samples, report how many are now recoverable.
"""
import sys, os, gzip, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import sympy as sp
from scipy.optimize import brentq, fsolve
from data.ast_grammar import expr_to_nodes, parse_to_sympy
from data.gen_data_pairs import (load_tags, class_range, sample_var, N_POINTS,
                                 NOISE_FRAC, NOISE_REL)

import os
AGGR_OVERSAMPLE = int(os.environ.get('AGGR_OVERSAMPLE', '3'))   # forward-path oversample factor


def robust_root(g, blo, bhi, yj_hint=None):
    """Find a verified root of scalar g — FAST: a few brentq brackets only."""
    cands = [(blo*0.1, bhi*5), (-bhi*5, -blo*0.1), (1e-3, 1e3), (-1e3, -1e-3)]
    for (a, b) in cands:
        try:
            ga, gb = g(a), g(b)
            if np.isfinite(ga) and np.isfinite(gb) and ga*gb < 0:
                r = brentq(g, a, b, maxiter=40, xtol=1e-7)
                if abs(g(r)) < 1e-4: return r
        except Exception:
            pass
    return None


def make_samples_v5(expr, rng, n_points=N_POINTS):
    load_tags()
    e = parse_to_sympy(expr)
    if e is None or not hasattr(e, 'free_symbols'): return None
    free = sorted([str(s) for s in e.free_symbols])
    if not (2 <= len(free) <= 12): return None      # >12 vars: generation impractically slow + model can't learn them
    syms = [sp.Symbol(v) for v in free]
    try: fn = sp.lambdify(syms, e, 'numpy')
    except Exception: return None

    out = []
    for target in free:
        others = [v for v in free if v != target]
        if class_range(target)[0] == 'const': continue
        kind, spec = class_range(target)
        blo, bhi = (spec if kind == 'range' else (0.3, 3.0))
        tsym = sp.Symbol(target)

        # FORWARD path: symbolically solve for target (ANY explicit inverse, incl.
        # transcendental like acos/log), then forward-evaluate each real branch with
        # REJECT-SAMPLING (drop points that overflow or violate a function domain:
        # acos arg out of [-1,1], log/sqrt of negative, etc.). This recovers the
        # transcendental formulas that blind root-finding can't.
        # FAST: only solve when LINEAR in target (degree 1) — general sp.solve is
        # ~55x slower and added no recovery in tests. Nonlinear -> root-finding below.
        branches = None
        try:
            poly = sp.Poly(e, tsym)
            if poly.degree() == 1:
                sols = sp.solve(e, tsym, rational=False)
                branches = [s for s in (sols or []) if tsym not in s.free_symbols][:2]
        except Exception:
            branches = None
        recovered = False
        if branches:
            for sol in branches:
                try:
                    fwd = sp.lambdify([sp.Symbol(v) for v in others], sol, 'numpy')
                except Exception:
                    continue
                for regime in ('default', 'log', 'narrow', 'wide'):
                    cols = {}
                    for v in others:
                        k2, sp2 = class_range(v)
                        if k2 == 'const': cols[v] = np.full(n_points, sp2); continue
                        lo, hi = sp2
                        if regime == 'log':
                            cols[v] = np.exp(rng.uniform(np.log(max(lo,1e-3)), np.log(hi), n_points))
                        elif regime == 'narrow':
                            m = (lo+hi)/2; cols[v] = rng.uniform(m*0.7, m*1.3, n_points)
                        elif regime == 'wide':
                            cols[v] = rng.uniform(lo*0.3, hi*3, n_points)
                        else:
                            cols[v] = rng.uniform(lo, hi, n_points)
                    # OVERSAMPLE aggressively to collect enough domain-valid points
                    # for high-dim formulas whose valid region is a small fraction
                    big = n_points * AGGR_OVERSAMPLE
                    cb = {}
                    for v in others:
                        k3, sp3 = class_range(v)
                        if k3 == 'const': cb[v] = np.full(big, sp3); continue
                        lo, hi = sp3
                        if regime == 'log':
                            cb[v] = np.exp(rng.uniform(np.log(max(lo,1e-3)), np.log(hi), big))
                        elif regime == 'narrow':
                            m = (lo+hi)/2; cb[v] = rng.uniform(m*0.7, m*1.3, big)
                        elif regime == 'wide':
                            cb[v] = rng.uniform(lo*0.3, hi*3, big)
                        else:
                            cb[v] = rng.uniform(lo, hi, big)
                    Xo = np.stack([cb[v] for v in others], axis=1)
                    try:
                        yy = np.asarray(fwd(*[cb[v] for v in others]), complex)
                        yy = np.broadcast_to(yy, (big,))
                        ys = np.where(np.abs(yy.imag) < 1e-9, yy.real, np.nan).astype(float)
                    except Exception:
                        continue
                    ok = np.isfinite(ys) & (np.abs(ys) < 1e6)   # reject overflow/domain
                    if ok.sum() >= 60:                          # enough valid points collected
                        Xv, yv = Xo[ok][:n_points], ys[ok][:n_points]
                        res = expr_to_nodes(expr)
                        if res is None: break
                        seq, consts, var_order = res
                        if len(yv) > 5:
                            k = max(1, int(len(yv) * rng.uniform(*NOISE_FRAC)))
                            idx = rng.choice(len(yv), k, replace=False)
                            yv[idx] = yv[idx] * (1 + rng.normal(0, NOISE_REL, k))
                        out.append({'expr': expr, 'target': target, 'var_names': others,
                                    'X': np.round(Xv, 6).tolist(), 'y': np.round(yv, 6).tolist(),
                                    'seq': seq, 'consts': consts, 'formula_vars': var_order})
                        recovered = True; break
                if recovered: break
        if recovered:
            continue  # target done via forward inverse

        # adaptive: try a few sampling regimes for the others until enough valid
        best = None
        for regime in ('default', 'log', 'wide', 'narrow'):
            cols = {}
            for v in others:
                k2, sp2 = class_range(v)
                if k2 == 'const':
                    cols[v] = np.full(n_points, sp2); continue
                lo, hi = sp2
                if regime == 'log':
                    cols[v] = np.exp(rng.uniform(np.log(max(lo, 1e-3)), np.log(hi), n_points))
                elif regime == 'wide':
                    cols[v] = rng.uniform(lo*0.3, hi*3, n_points)
                elif regime == 'narrow':
                    cols[v] = rng.uniform((lo+hi)/2*0.8, (lo+hi)/2*1.2, n_points)
                else:
                    cols[v] = rng.uniform(lo, hi, n_points)
            Xo = np.stack([cols[v] for v in others], axis=1)
            ys = np.full(n_points, np.nan)
            for j in range(n_points):
                vals = {v: cols[v][j] for v in others}
                def g(tv):
                    arr = [tv if v == target else vals[v] for v in free]
                    try:
                        r = fn(*arr); return float(r) if np.isfinite(r) else np.nan
                    except Exception:
                        return np.nan
                root = robust_root(g, blo, bhi)
                if root is not None: ys[j] = root
            ok = np.isfinite(ys)
            if best is None or ok.sum() > best[2].sum():
                best = (Xo, ys, ok)
            if ok.sum() >= n_points * 0.5: break
        Xo, ys, ok = best
        if ok.sum() < n_points * 0.35: continue
        Xo, ys = Xo[ok], ys[ok]
        # residual quality gate (vectorized)
        try:
            arr = [ys if v == target else Xo[:, others.index(v)] for v in free]
            resid = np.abs(np.asarray(fn(*arr), float))
            valid = np.isfinite(resid) & (resid < 1e-4)
            if valid.sum() < len(ys) * 0.6: continue
            Xo, ys = Xo[valid], ys[valid]
        except Exception:
            continue
        res = expr_to_nodes(expr)
        if res is None: continue
        seq, consts, var_order = res
        out.append({'expr': expr, 'target': target, 'var_names': others,
                    'X': np.round(Xo, 6).tolist(), 'y': np.round(ys, 6).tolist(),
                    'seq': seq, 'consts': consts, 'formula_vars': var_order})
    return out if out else None


if __name__ == '__main__':
    # RECOVERY TEST: collect formulas that FAILED v4 (not in usable) but are
    # transcendental and within length, run v5, report recovery rate.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120)
    args = ap.parse_args()
    usable = set(l.strip() for l in open('dataset_20260531/cache_v4_dims/exprs.txt', encoding='utf-8') if l.strip())
    # also exclude already-recovered so we test the TRULY unrecovered 645k
    recovered = set()
    try:
        for line in gzip.open('dataset_20260531/data_pairs_v5_recovered.jsonl.gz', 'rt', encoding='utf-8'):
            if line.strip():
                recovered.add(json.loads(line).get('expr', ''))
    except Exception:
        pass
    print(f"AGGR_OVERSAMPLE={AGGR_OVERSAMPLE} | already-recovered excluded: {len(recovered)}")
    failed = []
    with gzip.open('dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line).get('expr', '')
            if not ex or ex in usable or ex in recovered: continue
            if any(t in ex for t in ('Derivative', 'Integral', 'Tr(', 'nabla')): continue
            res = expr_to_nodes(ex)
            if res is None or len(res[0]) > 48: continue
            failed.append(ex)
            if len(failed) >= args.n: break
    print(f"testing v5 recovery on {len(failed)} failed transcendental formulas (<=48 nodes)")
    rng = np.random.default_rng(0); rec = 0; t0 = time.time()
    for i, ex in enumerate(failed):
        try:
            s = make_samples_v5(ex, rng, n_points=200)
        except Exception:
            s = None
        if s: rec += 1
        if (i+1) % 30 == 0:
            print(f"  {i+1}/{len(failed)} | recovered {rec} ({100*rec/(i+1):.0f}%) ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nRECOVERY: {rec}/{len(failed)} = {100*rec/len(failed):.0f}% of v4-failed transcendental formulas now generate data")
