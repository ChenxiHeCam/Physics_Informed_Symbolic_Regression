"""NUMERIC/SYMBOLIC-OPERATOR recovery track (runs in PARALLEL with the algebraic
gen_dataside on a separate machine). Targets formulas the restricted parse_to_sympy
REJECTS but that contain Integral/Derivative/Sum/Limit — i.e. integro-differential
physics. Strategy: parse with evaluate=False (avoids sympify's auto-simplify blowup),
then .doit() under a hard timeout to get a CLOSED-FORM algebraic expression; if it
closes, hand the algebraic form to gen_dataside.make_one for the normal forward-eval
point-cloud generation. Formulas whose .doit() times out or stays operator-valued are
skipped (no numerical quad here — kept robust/fast; those are genuinely hard).

Robustness: signal.alarm guards most slow .doit()s; any C-level hang is caught by the
pool fut.result timeout + periodic pool refresh (recreates hung workers).
Run: python gen_numeric.py --master .. --aug-glob '..' --out-prefix .. --workers 10
"""
import sys, os, gzip, json, time, argparse, glob, re, signal, warnings
warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import gen_dataside as GD                       # reuse make_one (forward eval + gates)

_OP = re.compile(r'Integral|Derivative|Sum\(|Limit')       # candidate operator formulas


class _TO(Exception):
    pass


def _alarm(s, f):
    raise _TO()


def doit_algebraic(expr_str, timeout=4):
    """parse (no auto-eval) then .doit() under an alarm timeout. Return an algebraic
    expression STRING if it closes to something with no Integral/Derivative/Sum/Limit,
    else None."""
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
    except Exception:
        pass
    try:
        e = parse_expr(expr_str, evaluate=False)
        d = e.doit()
        try: signal.alarm(0)
        except Exception: pass
        if d.has(sp.Integral, sp.Derivative, sp.Sum, sp.Limit):
            return None
        if not hasattr(d, 'free_symbols') or not (2 <= len(d.free_symbols) <= 12):
            return None
        return str(d)
    except (_TO, Exception):
        return None
    finally:
        try: signal.alarm(0)
        except Exception: pass


def _worker(t):
    expr, hint, uid, seed = t
    alg = doit_algebraic(expr)
    if alg is None:
        return None
    rng = np.random.default_rng(seed)
    try:
        s = GD.make_one(alg, rng, hint)
    except Exception:
        return None
    if s is None:
        return None
    s['uid'] = uid
    s['orig_expr'] = expr            # keep the integro-differential source form too
    return json.dumps(s)


def gen_tasks(master, aug_glob, limit):
    """Yield (expr, hint, uid, seed) for operator-formulas (Integral/Derivative/Sum/Limit).
    The fast _OP regex is the filter: these are exactly what the restricted algebraic parser
    rejects, so there is no overlap with gen_dataside's output (and no slow per-formula
    parse_to_sympy scan of all 33.6M)."""
    id2out = {}; n = 0
    with open(master, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                r = json.loads(line)
                vs = r.get('variables') or []
                out = (vs[0].get('sym') if isinstance(vs[0], dict) else
                       (vs[0] if isinstance(vs[0], str) else None)) if vs else None
                if r.get('id'): id2out[r['id']] = out
                ex = r.get('expr'); uid = r.get('id', '')
            except Exception:
                continue
            if ex and _OP.search(ex):
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
                if ex and _OP.search(ex):
                    yield (ex, hint, uid, 42 + n); n += 1
                    if limit and n >= limit: return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--master', default='data/master_nodes.jsonl')
    ap.add_argument('--aug-glob', default='data/data/augmented/round1_ds_dedup_*.jsonl')
    ap.add_argument('--out-prefix', default='data/dataside_num/pairs')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--timeout', type=float, default=8.0)
    ap.add_argument('--shard', type=int, default=200000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start-shard', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout

    t0 = time.time(); done = nok = nsamp = nto = cand = 0
    skip = args.start_shard * args.shard
    shard = args.start_shard; w = None
    def open_shard(k): return gzip.open(f"{args.out_prefix}_{k:04d}.jsonl.gz", 'wt',
                                        encoding='utf-8', compresslevel=4)
    BATCH = 2000; REFRESH = 20000            # frequent refresh recovers any hung worker
    ex = ProcessPoolExecutor(max_workers=args.workers); since = 0
    buf = []
    gen = gen_tasks(args.master, args.aug_glob, args.limit)
    try:
        for i, task in enumerate(gen):
            cand = i + 1
            if i < skip: continue
            buf.append(task)
            if len(buf) < BATCH: continue
            futs = [ex.submit(_worker, t) for t in buf]
            for fut in futs:
                try: res = fut.result(timeout=args.timeout)
                except (FTimeout, Exception): res = None; nto += 1
                done += 1
                if res:
                    nok += 1
                    if w is None: w = open_shard(shard)
                    w.write(res + '\n'); nsamp += 1
                    if nsamp % args.shard == 0:
                        w.close(); shard += 1; w = open_shard(shard)
            since += len(buf); buf = []
            print(f"  cand {cand} | {done} done | {nok} recovered | {nsamp} pairs | "
                  f"{nto} to | {done/(time.time()-t0):.0f}/s | {time.time()-t0:.0f}s", flush=True)
            if since >= REFRESH:
                ex.shutdown(wait=False, cancel_futures=True)
                ex = ProcessPoolExecutor(max_workers=args.workers); since = 0
        if buf:
            futs = [ex.submit(_worker, t) for t in buf]
            for fut in futs:
                try: res = fut.result(timeout=args.timeout)
                except (FTimeout, Exception): res = None; nto += 1
                done += 1
                if res:
                    nok += 1
                    if w is None: w = open_shard(shard)
                    w.write(res + '\n'); nsamp += 1
    finally:
        if w: w.close()
        ex.shutdown(wait=False, cancel_futures=True)
    open(f"{args.out_prefix}_ALL.done", 'w').close()
    print(f"\nDONE: {cand} candidates | {done} attempted | {nok} recovered | "
          f"{nsamp} pairs | {nto} to | {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
