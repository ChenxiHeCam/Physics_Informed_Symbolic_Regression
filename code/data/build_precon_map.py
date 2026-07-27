"""
Recover per-formula preconditions (physical allowed-domains) by propagating the
~1849 seed preconditions down the derivation DAG via ancestor_seed_ids.

  seed precondition_expr  (e.g. 'N>0', '0<=k<N', 'v**2<1')   [1849 seeds]
  FINAL v3 node.ancestor_seed_ids -> AND of those seeds' preconditions
  -> expr -> precondition map  (used for sampling-domain constraints + z_f signal)

Output: dataset_20260531/expr_precon.json   { expr : "precon1 & precon2 ..." }
"""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

SEEDS = glob.glob('dataset_20260531/seeds_*.jsonl')
FINAL = 'data/_FINAL_CLEAN_20260528/FINAL_nodes_v3.jsonl'
OUT = 'dataset_20260531/expr_precon.json'


def main():
    # 1) seed_id -> precondition_expr (only real constraints, skip empty/True/equations
    #    that are just definitions)
    sid2pre = {}
    for fn in SEEDS:
        for line in open(fn, encoding='utf-8'):
            if not line.strip():
                continue
            r = json.loads(line)
            pe = r.get('precondition_expr', '')
            sid = r.get('id')
            if not sid or not pe:
                continue
            pe = str(pe).strip()
            # keep inequality/domain constraints (the useful sampling bounds);
            # skip trivial 'True'/'None' and pure definitional equalities with no <,>
            if pe in ('', 'None', 'True'):
                continue
            if ('<' in pe or '>' in pe or '!=' in pe):
                sid2pre[str(sid)] = pe
    print(f"seeds with usable (inequality) precon: {len(sid2pre)}")

    # 2) propagate to FINAL v3 nodes via ancestor_seed_ids -> expr -> AND of precons
    expr_precon = {}
    n = 0; hit = 0
    if not os.path.exists(FINAL):
        print(f"FINAL v3 not found at {FINAL}"); return
    with open(FINAL, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            r = json.loads(line)
            ex = r.get('expr', '')
            anc = r.get('ancestor_seed_ids') or []
            if not ex or not anc:
                continue
            precs = []
            seen = set()
            for sid in anc:
                p = sid2pre.get(str(sid))
                if p and p not in seen:
                    precs.append(p); seen.add(p)
            if precs:
                expr_precon[ex] = ' & '.join(precs)
                hit += 1
    print(f"FINAL v3 nodes scanned: {n} | with propagated precon: {hit}")
    json.dump(expr_precon, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f"expr->precon map: {len(expr_precon)} formulas -> {OUT}")
    # show a few
    for i, (e, p) in enumerate(expr_precon.items()):
        if i >= 8: break
        print(f"   {p[:40]:42} | {e[:55]}")


if __name__ == '__main__':
    main()
