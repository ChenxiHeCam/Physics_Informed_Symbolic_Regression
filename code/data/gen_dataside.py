"""Data-side generation for the NEW dataset (master 5.62M + aug 28M): ONE target per
formula (the natural OUTPUT), FORWARD evaluation (user insight: formulas are lhs-rhs
residuals / definitions, so we compute forward, NOT reverse root-find). Three cases:
  (1) some var is LINEAR in the residual (coeff indep of it) -> solve algebraically once,
      vectorized forward eval (hint/output var first, else any linear var).  [~93%, fast]
  (2) hint (output) NOT in the expr = pure DEFINITION `y = def(inputs)` -> eval directly,
      target = external hint, X = all free vars.  [fast, also fixes a correctness bug where
      these were being brentq-solved for an INPUT var -> wrong formula]
  (3) truly implicit (no linear var, hint in expr) -> per-point brentq.  [~7%, rare, slow]
Reuses gen_data_pairs helpers (semantic sampling, valid-domain gate, degenerate drop, noise).
master: hint = variables[0].sym (output). aug: hint = source formula's output (aug_source_id).
Streams master+aug, sharded gzip, resumable by shard.
"""
import sys, os, gzip, json, time, argparse, glob, warnings
warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))            # for `data.dimensions`
import numpy as np
import sympy as sp
from scipy.optimize import brentq
import gen_data_pairs as G
from ast_grammar import expr_to_nodes, parse_to_sympy


def _sample(cols_vars, trig_syms, n_points, rng):
    cols = {v: G.sample_var(v, n_points, rng) for v in cols_vars}
    for v in cols_vars:
        if v in trig_syms:
            k = rng.uniform(1.0, 2.5) if G._JITTER else 2.0
            cols[v] = rng.uniform(0.0, 2.0 * np.pi * k, n_points)
    return cols


def _finish(Xo, ys, fn, free, target, others, expr, n_points, rng, verify):
    """Shared validity gates + degenerate drop + noise + AST -> pair dict (or None).
    verify=True: re-check residual e~0 (solved targets). verify=False: direct-eval defn."""
    ok = np.isfinite(ys)
    if ok.sum() < n_points * 0.4 or ok.sum() < 40:
        return None
    Xo, ys = Xo[ok], ys[ok]
    if verify:
        try:
            arr = [ys if v == target else Xo[:, others.index(v)] for v in free]
            resid = np.abs(np.asarray(fn(*arr), dtype=float))
            good = np.isfinite(resid) & (resid < 1e-4)
            if good.sum() < len(ys) * 0.6:
                return None
            Xo, ys = Xo[good], ys[good]
        except Exception:
            return None
    if len(ys) < 40:
        return None
    if np.all(ys == 0) or np.std(ys) < 1e-9 * (abs(np.mean(ys)) + 1e-9):
        return None
    # constant-column drop, but EXCLUDE physical constants (c, hbar, k_B, ...): they are
    # SUPPOSED to be constant, so a fixed column is legitimate, not degenerate. Only a real
    # (non-const) variable coming out constant signals a sampling degeneracy worth dropping.
    # (was dropping ~33% of clean lhs-rhs formulas that merely contain a physical constant.)
    if any(G.class_range(others[k])[0] != 'const'
           and np.std(Xo[:, k]) < 1e-9 * (abs(np.mean(Xo[:, k])) + 1e-9)
           for k in range(Xo.shape[1])):
        return None
    if len(ys) > 5 and rng.random() < 0.5:
        ys = ys * (1 + rng.normal(0, rng.uniform(0.0, G.NOISE_REL), len(ys)))
    res = expr_to_nodes(expr)
    if res is None:
        return None
    seq, consts, var_order = res
    return {'expr': expr, 'target': target, 'var_names': others,
            'X': np.round(Xo, 6).tolist(), 'y': np.round(ys, 6).tolist(),
            'seq': seq, 'consts': consts, 'formula_vars': var_order}


def make_one(expr, rng, hint=None):
    G.load_tags()
    # Store a LARGE point pool (~700) per formula: training subsamples a different subset
    # each epoch, so more stored points = more augmentation diversity. (Encoder-size jitter
    # now happens at train time via subsampling, not here.)
    n_points = int(rng.integers(600, 901))
    # equation form "lhs = rhs" -> residual "lhs - (rhs)" so it parses/solves like the rest.
    if '=' in expr:
        parts = expr.split('=')
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            expr = f"({parts[0]}) - ({parts[1]})"
    e = parse_to_sympy(expr)
    if e is None or not hasattr(e, 'free_symbols'):
        return None
    free = sorted([str(s) for s in e.free_symbols])
    if not (2 <= len(free) <= 12):
        return None
    syms = [sp.Symbol(v) for v in free]
    trig_syms = G._trig_arg_symbols(e)
    try:
        fn = sp.lambdify(syms, e, 'numpy')
    except Exception:
        return None

    # (1) FORWARD: find a target that is LINEAR in the residual (hint/output first, then any).
    cand = ([hint] if hint in free else []) + [v for v in free if v != hint]
    for target in cand:
        if G.class_range(target)[0] == 'const':
            continue
        tsym = sp.Symbol(target)
        try:
            dedt = sp.diff(e, tsym)
        except Exception:
            continue
        if dedt == 0 or tsym in dedt.free_symbols:      # not linear in target -> skip
            continue
        others = [v for v in free if v != target]
        cols = _sample(others, trig_syms, n_points, rng)
        try:
            # e is linear in target: e = dedt*target + rest, rest = e|_{target=0}.
            # target = -rest/dedt. Use subs (O(size)) NOT expand (exponential blowup on
            # complex exprs -> was the 1-4s/formula bottleneck).
            solfn = sp.lambdify([sp.Symbol(v) for v in others],
                                -e.subs(tsym, 0) / dedt, 'numpy')
            yv = np.asarray(solfn(*[cols[v] for v in others]), float)
            ys = np.broadcast_to(yv, (n_points,)).astype(float).copy()
        except Exception:
            continue
        Xo = np.stack([cols[v] for v in others], axis=1)
        pair = _finish(Xo, ys, fn, free, target, others, expr, n_points, rng, verify=True)
        if pair is not None:
            return pair

    # (2) DEFINITION form: output (hint) not in expr -> y = def(inputs) directly.
    if hint is not None and hint not in free:
        cols = _sample(free, trig_syms, n_points, rng)
        try:
            yv = np.asarray(fn(*[cols[v] for v in free]), float)
            ys = np.broadcast_to(yv, (n_points,)).astype(float).copy()
            Xo = np.stack([cols[v] for v in free], axis=1)
            pair = _finish(Xo, ys, fn, free, hint, free, expr, n_points, rng, verify=False)
            if pair is not None:
                return pair
        except Exception:
            pass

    # (3) IMPLICIT non-linear (no linear var, output in expr) -> per-point brentq. Rare.
    for target in cand:
        if G.class_range(target)[0] == 'const':
            continue
        others = [v for v in free if v != target]
        cols = _sample(others, trig_syms, n_points, rng)
        Xo = np.stack([cols[v] for v in others], axis=1)
        ys = np.full(n_points, np.nan)
        kind, spec = G.class_range(target)
        blo, bhi = (spec if kind == 'range' else G._DEFAULT_RANGE)
        for j in range(n_points):
            vals = {v: cols[v][j] for v in others}
            def g(tv):
                arr = [tv if v == target else vals[v] for v in free]
                try:
                    r = fn(*arr); return float(r) if np.isfinite(r) else np.nan
                except Exception:
                    return np.nan
            try:
                root = None
                for (a, b) in [(blo*0.1, bhi*5), (-bhi*5, bhi*5)]:
                    ga, gb = g(a), g(b)
                    if np.isfinite(ga) and np.isfinite(gb) and ga*gb < 0:
                        root = brentq(g, a, b, maxiter=40, xtol=1e-7); break
                if root is not None and abs(g(root)) < 1e-4:
                    ys[j] = root
            except Exception:
                pass
        pair = _finish(Xo, ys, fn, free, target, others, expr, n_points, rng, verify=True)
        if pair is not None:
            return pair
    return None


def _worker(t):
    expr, hint, uid, seed = t
    rng = np.random.default_rng(seed)
    try:
        s = make_one(expr, rng, hint)
    except Exception:
        return None
    if s is None:
        return None
    s['uid'] = uid
    return json.dumps(s)          # serialize in the WORKER (parallel) — the single main
                                  # process doing json.dumps of 1500-float dicts was the
                                  # throughput cap (workers starved at ~190/s). Main now
                                  # just writes the ready string.


def gen_tasks(master, aug_glob, limit):
    id2out = {}; n = 0
    with open(master, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:                                          # ANY malformed record -> skip
                r = json.loads(line)
                vs = r.get('variables') or []             # variables: list of dicts OR of
                out = (vs[0].get('sym') if isinstance(vs[0], dict) else   # bare name strings
                       (vs[0] if isinstance(vs[0], str) else None)) if vs else None
                if r.get('id'): id2out[r['id']] = out
                ex = r.get('expr'); uid = r.get('id', '')
            except Exception:
                continue
            if ex:
                yield (ex, out, uid, 42 + n); n += 1
                if limit and n >= limit: return
    for fp in sorted(glob.glob(aug_glob)):
        with open(fp, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line)
                    ex = r.get('new_expr')
                    hint = id2out.get(r.get('aug_source_id'))
                    uid = r.get('aug_source_id', '')
                except Exception:
                    continue
                if not ex: continue
                yield (ex, hint, uid, 42 + n); n += 1
                if limit and n >= limit: return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--master', default='data/master_nodes.jsonl')
    ap.add_argument('--aug-glob', default='data/data/augmented/round1_ds_dedup_*.jsonl')
    ap.add_argument('--out-prefix', default='data/dataside/pairs')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--timeout', type=float, default=10.0)
    ap.add_argument('--shard', type=int, default=500000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start-shard', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout

    t0 = time.time(); done = nok = nsamp = nto = 0
    skip = args.start_shard * args.shard
    shard = args.start_shard; w = None
    def open_shard(k): return gzip.open(f"{args.out_prefix}_{k:04d}.jsonl.gz", 'wt',
                                        encoding='utf-8', compresslevel=4)  # 4 ~5x faster than 9
    BATCH = 4000; REFRESH = 200000
    ex = ProcessPoolExecutor(max_workers=args.workers); since = 0
    buf = []
    gen = gen_tasks(args.master, args.aug_glob, args.limit)
    try:
        for i, task in enumerate(gen):
            if i < skip: continue
            buf.append(task)
            if len(buf) < BATCH: continue
            done, nok, nsamp, nto, shard, w, since = _flush(
                ex, buf, args, done, nok, nsamp, nto, shard, w, open_shard, since, t0)
            buf = []
            if since >= REFRESH:
                ex.shutdown(wait=False, cancel_futures=True)
                ex = ProcessPoolExecutor(max_workers=args.workers); since = 0
        if buf:
            done, nok, nsamp, nto, shard, w, since = _flush(
                ex, buf, args, done, nok, nsamp, nto, shard, w, open_shard, since, t0)
    finally:
        if w: w.close()
        ex.shutdown(wait=False, cancel_futures=True)
    open(f"{args.out_prefix}_ALL.done", 'w').close()
    print(f"\nDONE: processed {done} | usable {nok} | {nsamp} pairs | timeout {nto} | "
          f"{time.time()-t0:.0f}s", flush=True)


def _flush(ex, buf, args, done, nok, nsamp, nto, shard, w, open_shard, since, t0):
    from concurrent.futures import TimeoutError as FTimeout
    if w is None: w = open_shard(shard)
    futs = [ex.submit(_worker, t) for t in buf]
    for fut in futs:
        try: res = fut.result(timeout=args.timeout)
        except (FTimeout, Exception): res = None; nto += 1
        done += 1
        if res:
            nok += 1
            w.write(res + '\n'); nsamp += 1          # res is already a JSON string
            if nsamp % args.shard == 0:
                w.close(); shard += 1; w = open_shard(shard)
    since += len(buf)
    print(f"  {done} done | {nok} ok | {nsamp} pairs | {nto} to | "
          f"{done/(time.time()-t0):.0f}/s | {time.time()-t0:.0f}s", flush=True)
    return done, nok, nsamp, nto, shard, w, since


if __name__ == '__main__':
    main()
