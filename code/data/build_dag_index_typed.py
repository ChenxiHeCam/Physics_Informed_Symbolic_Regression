"""Build a TYPED + DIRECTED per-row DAG-neighbor index (for TransE-style relational
loss, not the old collapse-everything-together InfoNCE).

For each cache row we store ONE directed, typed edge:
  nbr[i]  : a DAG-related row (or -1)
  rel[i]  : relation-type id (0..K-1), or -1 (none)
  dir[i]  : +1 if row i is the edge SOURCE (z[i] + R[rel] ~= z[nbr]),
            -1 if row i is the DESTINATION (z[i] - R[rel] ~= z[nbr]).

The TransE loss then constrains z[i] + dir[i]*R[rel[i]] ~= z[nbr[i]], so related
formulas stay NEAR but separated by a learned, navigable per-relation offset —
instead of being collapsed onto each other (which hurts discriminability).

Output: <cache>/neighbors_typed.npz  {nbr int32[N], rel int8[N], dir int8[N], relations [str]}
"""
import os, sys, gzip, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np

NODES = 'dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz'
EDGES = 'dataset_20260531/unified_edges.jsonl.gz'

STRUCTURAL = sorted({
    'derivation', 'special_case', 'limiting_case', 'generalization',
    'equation_of_motion', 'constitutive_relation', 'transform_pair',
    'conjugate_pair', 'applied_operator', 'operator_application',
    'solves', 'conservation_symmetry', 'generating_function', 'same_framework',
})
REL2ID = {r: i for i, r in enumerate(STRUCTURAL)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--min-conf', type=float, default=0.5)
    args = ap.parse_args()

    exprs = open(os.path.join(args.cache, 'exprs.txt'), encoding='utf-8').read().splitlines()
    N = len(exprs)
    expr2rows = {}
    for i, e in enumerate(exprs):
        expr2rows.setdefault(e, []).append(i)
    cache_exprs = set(expr2rows)
    print(f"cache rows: {N} | unique exprs: {len(cache_exprs)} | relations: {len(STRUCTURAL)}")

    uid2expr = {}
    for l in gzip.open(NODES, 'rt', encoding='utf-8'):
        if not l.strip():
            continue
        r = json.loads(l); e = r.get('expr', '')
        if e in cache_exprs:
            uid2expr[r.get('id', '')] = e
    print(f"uids mapped to cache exprs: {len(uid2expr)}")

    # directed typed adjacency at EXPR level: expr -> list of (other_expr, rel_id, dir)
    adj = {}
    kept = 0
    for l in gzip.open(EDGES, 'rt', encoding='utf-8'):
        if not l.strip():
            continue
        r = json.loads(l)
        rel = r.get('relation')
        if rel not in REL2ID:
            continue
        if r.get('confidence', 1.0) < args.min_conf:
            continue
        es = uid2expr.get(r.get('src')); ed = uid2expr.get(r.get('dst'))
        if es is None or ed is None or es == ed:
            continue
        rid = REL2ID[rel]
        adj.setdefault(es, []).append((ed, rid, +1))   # es is source: z_es + R ~= z_ed
        adj.setdefault(ed, []).append((es, rid, -1))   # ed is dest:   z_ed - R ~= z_es
        kept += 1
    print(f"typed structural edges within cache: {kept} | exprs with >=1 edge: {len(adj)}")

    nbr = np.full(N, -1, dtype=np.int32)
    rel = np.full(N, -1, dtype=np.int8)
    dir_ = np.zeros(N, dtype=np.int8)
    nfilled = 0
    for i, e in enumerate(exprs):
        edges = adj.get(e)
        if not edges:
            continue
        edges = sorted(edges)                          # deterministic
        oe, rid, dd = edges[hash(('n', e)) % len(edges)]
        rows = expr2rows.get(oe)
        if not rows:
            continue
        nbr[i] = rows[hash(('r', e, i)) % len(rows)]
        rel[i] = rid; dir_[i] = dd
        nfilled += 1
    out = os.path.join(args.cache, 'neighbors_typed.npz')
    np.savez(out, nbr=nbr, rel=rel, dir=dir_, relations=np.array(STRUCTURAL))
    print(f"rows with a typed DAG edge: {nfilled}/{N} ({100*nfilled/N:.1f}%) -> {out}")
    # relation histogram
    import collections
    h = collections.Counter(int(x) for x in rel if x >= 0)
    print("per-relation row counts:", {STRUCTURAL[k]: v for k, v in sorted(h.items())})


if __name__ == '__main__':
    main()
