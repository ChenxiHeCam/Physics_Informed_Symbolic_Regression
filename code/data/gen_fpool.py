"""Build the formula-only pool for --fpool: run expr_to_nodes over ALL master+aug formulas
(incl the ones whose DATA was dropped), keep the AST-parseable ones, write {seq,consts}
jsonl. This feeds train_manifold's formula-only autoencoding loss (--w-fae) so the formula
encoder+decoder learn from far more formula structures than the 6.5M with-data cache.
LaTeX/integral/non-standard formulas that don't parse are auto-excluded (expr_to_nodes None).
Usage: python gen_fpool.py <master.jsonl> "<aug_glob>" <out.jsonl> [workers]
"""
import sys, os, json, glob
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))
from ast_grammar import expr_to_nodes
from multiprocessing import Pool


def proc(expr):
    if not expr:
        return None
    if '=' in expr:
        p = expr.split('=')
        expr = f'({p[0]})-({p[1]})' if len(p) == 2 and p[0].strip() and p[1].strip() else expr
    try:
        r = expr_to_nodes(expr)
    except Exception:
        return None
    if r is None:
        return None
    seq, consts, var_order = r
    if not (2 <= len(seq) <= 200):
        return None
    return json.dumps({'seq': seq, 'consts': consts})


def formulas(master, aug_glob):
    with open(master, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try: yield json.loads(line).get('expr')
            except Exception: continue
    for fp in sorted(glob.glob(aug_glob)):
        with open(fp, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try: yield json.loads(line).get('new_expr')
                except Exception: continue


if __name__ == '__main__':
    master, aug_glob, out = sys.argv[1], sys.argv[2], sys.argv[3]
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    n = ok = 0
    with Pool(nw) as pool, open(out, 'w', encoding='utf-8') as w:
        for res in pool.imap_unordered(proc, formulas(master, aug_glob), chunksize=256):
            n += 1
            if res:
                w.write(res + '\n'); ok += 1
            if n % 1000000 == 0:
                print(f'  {n} processed | {ok} valid ({ok*100//n}%)', flush=True)
    print(f'DONE: {n} processed | {ok} AST-valid formulas -> {out}', flush=True)
