"""
Full v5 generation: run robust make_samples_v5 on ALL formulas that FAILED v4
(not in the usable set), recovering whatever can now produce clean (X,y) data
(~18% — mostly the rare transcendental types under-represented in training).
Output appends to a new jsonl.gz to be merged into the training cache.
"""
import os
# MUST be set before numpy/scipy import (incl. in spawned workers re-importing this
# module) so BLAS stays single-threaded and workers don't each grab all cores.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, gzip, json, time, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np

NODES = 'dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz'
USABLE = 'dataset_20260531/cache_v4_dims/exprs.txt'


def _pool_init():
    # force single-threaded BLAS inside every worker (else each worker grabs all
    # cores via numpy/scipy threads and saturates the machine)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[v] = '1'


def _worker(t):
    from data.gen_v5_robust import make_samples_v5
    expr, uid, seed = t
    rng = np.random.default_rng(seed)
    try:
        s = make_samples_v5(expr, rng, n_points=300)
    except Exception:
        return None
    if not s: return None
    for p in s: p['uid'] = uid
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='dataset_20260531/data_pairs_v5_recovered.jsonl.gz')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--timeout', type=float, default=20.0)
    ap.add_argument('--max-nodes', type=int, default=0)  # 0 = all failed
    ap.add_argument('--append', action='store_true')  # append to existing out
    args = ap.parse_args()
    import multiprocessing as mp

    usable = set(l.strip() for l in open(USABLE, encoding='utf-8') if l.strip())
    print(f"usable set: {len(usable)}")
    tasks = []
    with gzip.open(NODES, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            r = json.loads(line); ex = r.get('expr', '')
            if not ex or ex in usable: continue          # only FAILED formulas
            # skip PDE/tensor (unrepresentable in our algebraic grammar) — fast filter
            if any(t in ex for t in ('Derivative', 'Integral', 'Tr(', 'nabla', 'grad(', 'div(')):
                continue
            tasks.append((ex, r.get('id', ''), 1000 + len(tasks)))
            if args.max_nodes and len(tasks) >= args.max_nodes: break
    print(f"failed formulas to retry: {len(tasks)}")

    t0 = time.time(); nok = nsamp = done = nto = 0
    BATCH = 4000   # large -> infrequent pool re-creation -> no periodic CPU spikes
    mode = 'at' if args.append else 'wt'
    with gzip.open(args.out, mode, encoding='utf-8') as w:
        for bs in range(0, len(tasks), BATCH):
            chunk = tasks[bs:bs+BATCH]
            # fresh Pool per chunk; terminate() FORCIBLY kills workers (incl. those
            # hung in a slow sympy call) so no orphan workers accumulate -> no lag.
            pool = mp.Pool(args.workers, initializer=_pool_init)
            try:
                ars = [pool.apply_async(_worker, (t,)) for t in chunk]
                for ar in ars:
                    try: res = ar.get(timeout=args.timeout)
                    except Exception: res = None; nto += 1
                    done += 1
                    if res:
                        nok += 1
                        for p in res: w.write(json.dumps(p)+'\n'); nsamp += 1
            finally:
                pool.terminate(); pool.join()      # hard-kill all workers in this chunk
            w.flush()
            if (bs//BATCH) % 3 == 0:
                print(f"  {done}/{len(tasks)} | recovered {nok} ({100*nok/max(done,1):.0f}%) | "
                      f"{nsamp} samples | to {nto} | {done/(time.time()-t0):.0f}/s | "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"\nDONE: {done} retried | {nok} recovered ({100*nok/max(done,1):.0f}%) | "
          f"{nsamp} samples | {time.time()-t0:.0f}s -> {args.out}")


if __name__ == '__main__':
    main()
