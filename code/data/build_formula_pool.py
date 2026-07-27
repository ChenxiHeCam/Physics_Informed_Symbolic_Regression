"""
Build the FORMULA-ONLY pool for z_f autoencoding training.

Every parseable formula in the unified node set (INCLUDING the ~hundreds-of-k
with no (X,y) data: PDEs, long, unrecoverable) -> (seq, consts) arrays. Used by
train_manifold.py --fpool: a formula-autoencoding aux (z_f -> decode -> formula)
so z_f covers the FULL pool, not just formulas that happen to have data.

Output dir: seq.npy (int16, pad -1), consts.npy (f32), slen.npy (int16), exprs.txt
Same layout MemmapPairDataset/FormulaPoolDataset expect.
"""
import os, sys, gzip, json, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from data.ast_grammar import expr_to_nodes, MAX_VARS

NODES = 'dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='dataset_20260531/formula_pool_v6')
    ap.add_argument('--max-seq', type=int, default=128)
    ap.add_argument('--nodes', default=NODES)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ms = args.max_seq

    seq_rows, const_rows, slen_l = [], [], []
    exprs_f = open(os.path.join(args.out, 'exprs.txt'), 'w', encoding='utf-8')
    seen = set()
    tot = parsed = kept = too_long = too_vars = unparse = 0
    import time; t0 = time.time()
    for l in gzip.open(args.nodes, 'rt', encoding='utf-8'):
        if not l.strip():
            continue
        e = json.loads(l).get('expr', '')
        if not e or e in seen:
            continue
        seen.add(e); tot += 1
        res = expr_to_nodes(e)
        if res is None:
            unparse += 1; continue
        seq, consts, _vn = res
        parsed += 1
        if len(seq) > ms:
            too_long += 1; continue
        srow = np.full(ms, -1, dtype=np.int16)
        crow = np.zeros(ms, dtype=np.float32)
        L = len(seq)
        srow[:L] = np.asarray(seq, dtype=np.int16)
        cv = np.nan_to_num(np.asarray(consts, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        crow[:L] = np.clip(cv, -1e6, 1e6).astype(np.float32)
        seq_rows.append(srow); const_rows.append(crow); slen_l.append(L)
        exprs_f.write(e + '\n'); kept += 1
        if tot % 100000 == 0:
            print(f"  scanned {tot} | kept {kept} | {tot/(time.time()-t0):.0f}/s", flush=True)
    exprs_f.close()
    np.save(os.path.join(args.out, 'seq.npy'), np.stack(seq_rows))
    np.save(os.path.join(args.out, 'consts.npy'), np.stack(const_rows))
    np.save(os.path.join(args.out, 'slen.npy'), np.asarray(slen_l, dtype=np.int16))
    print(f"\nunique exprs scanned: {tot}")
    print(f"  parseable: {parsed} | unparseable: {unparse}")
    print(f"  kept (<= {ms} tok): {kept} | dropped too-long: {too_long}")
    print(f"-> {args.out}  ({kept} formulas)")


if __name__ == '__main__':
    main()
