"""
Build a per-row DAG-neighbor index for a memmap cache.

The knowledge graph has 4.9M physics edges between formulas (derivation,
special_case, limiting_case, generalization, ...). Formulas linked by these
SHOULD sit near each other in z_f space — so the flow/retrieval can hop between
physically related forms. This precomputes, for every cache row, ONE related row
(same cache, a formula DAG-linked to this row's formula), or -1 if none.

Output: <cache>/neighbors.npy  int32[N]  (row -> related row, or -1)
Consumed by train_manifold.py --dag (a formula-side contrastive aux loss that
pulls DAG-related z_f together; off by default, for the retrain ablation).

Deterministic (no RNG): neighbor chosen by a stable hash so resume is exact.
"""
import os, sys, gzip, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np

NODES = 'dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz'
EDGES = 'dataset_20260531/unified_edges.jsonl.gz'

# relations that imply the two formulas should be NEAR in z_f (structural /
# physical kinship useful for SR). same_framework/often_used_together are weaker
# (domain co-occurrence) — included but the strong ones dominate by count.
STRUCTURAL = {
    'derivation', 'special_case', 'limiting_case', 'generalization',
    'equation_of_motion', 'constitutive_relation', 'transform_pair',
    'conjugate_pair', 'applied_operator', 'operator_application',
    'solves', 'conservation_symmetry', 'generating_function',
    'same_framework',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--min-conf', type=float, default=0.5)
    args = ap.parse_args()

    # 1) cache row -> expr ; expr -> list of rows
    exprs = open(os.path.join(args.cache, 'exprs.txt'), encoding='utf-8').read().splitlines()
    N = len(exprs)
    expr2rows = {}
    for i, e in enumerate(exprs):
        expr2rows.setdefault(e, []).append(i)
    cache_exprs = set(expr2rows)
    print(f"cache rows: {N} | unique exprs: {len(cache_exprs)}")

    # 2) uid <-> expr (only for exprs present in the cache)
    uid2expr = {}
    for l in gzip.open(NODES, 'rt', encoding='utf-8'):
        if not l.strip():
            continue
        r = json.loads(l); e = r.get('expr', '')
        if e in cache_exprs:
            uid2expr[r.get('id', '')] = e
    print(f"uids mapped to cache exprs: {len(uid2expr)}")

    # 3) adjacency at EXPR level (both endpoints in cache, structural, confident)
    adj = {}   # expr -> set(related expr)
    kept = 0
    for l in gzip.open(EDGES, 'rt', encoding='utf-8'):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get('relation') not in STRUCTURAL:
            continue
        if r.get('confidence', 1.0) < args.min_conf:
            continue
        es = uid2expr.get(r.get('src')); ed = uid2expr.get(r.get('dst'))
        if es is None or ed is None or es == ed:
            continue
        adj.setdefault(es, set()).add(ed)
        adj.setdefault(ed, set()).add(es)   # undirected for nearness
        kept += 1
    print(f"structural edges within cache: {kept} | exprs with >=1 neighbor: {len(adj)}")

    # 4) per-row neighbor row (deterministic pick via stable hash)
    neighbors = np.full(N, -1, dtype=np.int32)
    nfilled = 0
    for i, e in enumerate(exprs):
        rel = adj.get(e)
        if not rel:
            continue
        rel = sorted(rel)
        ne = rel[hash(('n', e)) % len(rel)]          # pick a related expr
        rows = expr2rows.get(ne)
        if not rows:
            continue
        neighbors[i] = rows[hash(('r', e, i)) % len(rows)]
        nfilled += 1
    out = os.path.join(args.cache, 'neighbors.npy')
    np.save(out, neighbors)
    print(f"rows with a DAG neighbor: {nfilled}/{N} ({100*nfilled/N:.1f}%) -> {out}")


if __name__ == '__main__':
    main()
