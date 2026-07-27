"""
Full re-generation of training data — RICH + PHYSICAL.

  PHYSICAL : regime-aware sampling (wide log-space for dimensional vars so
             dimensionless groups hν/kT, v/c sweep their transition; relativistic/
             quantum/thermo regimes are all covered), domain-valid points only.
  RICH     : per-target (every non-constant variable becomes the dependent var)
             x N_REP independent datasets per target (different draws/regimes).

Runs make_samples (which already does regime sampling + per-target) N_REP times
per formula. Robust pooling: fresh mp.Pool per chunk + terminate() (no orphan
workers), single-thread BLAS, Idle OS priority (set by launcher) -> never lags.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, gzip, json, time, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np

NODES = 'dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz'
N_REP = int(os.environ.get('N_REP', '3'))   # datasets per (formula, target)


def _pool_init():
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[v] = '1'


def _worker(t):
    from data.gen_data_pairs import make_samples
    expr, uid = t
    out = []
    for rep in range(N_REP):
        rng = np.random.default_rng(hash((uid, rep)) & 0x7fffffff)
        # VARIABLE data amount per dataset so z_d is diverse + robust to #points
        npts = int(rng.integers(64, 512))
        try:
            s = make_samples(expr, rng, n_points=npts)
        except Exception:
            s = None
        if s:
            for p in s: p['uid'] = uid
            out.extend(s)
    return out if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='dataset_20260531/data_pairs_regen.jsonl.gz')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--timeout', type=float, default=15.0)
    ap.add_argument('--max', type=int, default=0)
    args = ap.parse_args()
    import multiprocessing as mp

    # only regenerate the 777k formulas KNOWN to produce data (usable + recovered);
    # skip the 645k that fail anyway (saves hours of wasted attempts)
    known = set(l.strip() for l in open('dataset_20260531/cache_v4_dims/exprs.txt', encoding='utf-8') if l.strip())
    try:
        for line in gzip.open('dataset_20260531/data_pairs_v5_recovered.jsonl.gz', 'rt', encoding='utf-8'):
            if line.strip(): known.add(json.loads(line).get('expr', ''))
    except Exception:
        pass
    print(f"known-generatable formulas: {len(known)}", flush=True)
    tasks = []; seen = set()
    with gzip.open(NODES, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line); ex = r.get('expr', '')
            if not ex or ex in seen or ex not in known: continue
            seen.add(ex)
            tasks.append((ex, r.get('id', '')))
            if args.max and len(tasks) >= args.max: break
    print(f"formulas to regenerate: {len(tasks)} | N_REP={N_REP}", flush=True)

    t0 = time.time(); nok = nsamp = done = nto = 0
    BATCH = 4000
    with gzip.open(args.out, 'wt', encoding='utf-8') as w:
        for bs in range(0, len(tasks), BATCH):
            pool = mp.Pool(args.workers, initializer=_pool_init)
            try:
                ars = [pool.apply_async(_worker, (t,)) for t in tasks[bs:bs+BATCH]]
                for ar in ars:
                    try: res = ar.get(timeout=args.timeout)
                    except Exception: res = None; nto += 1
                    done += 1
                    if res:
                        nok += 1
                        for p in res: w.write(json.dumps(p)+'\n'); nsamp += 1
            finally:
                pool.terminate(); pool.join()
            w.flush()
            print(f"  {done}/{len(tasks)} | ok {nok} | {nsamp} samples "
                  f"({nsamp/max(nok,1):.1f}/f) | to {nto} | {done/(time.time()-t0):.0f}/s | "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"\nDONE: {done} | {nok} usable | {nsamp} samples | {time.time()-t0:.0f}s -> {args.out}")


if __name__ == '__main__':
    main()
