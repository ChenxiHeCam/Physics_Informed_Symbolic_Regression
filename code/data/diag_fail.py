"""Diagnose WHY the 82% of failed transcendental formulas can't generate data."""
import sys, os, gzip, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, sympy as sp
from data.ast_grammar import expr_to_nodes, parse_to_sympy
from data.gen_data_pairs import load_tags, class_range
from data.gen_v5_robust import make_samples_v5

load_tags()
usable = set(l.strip() for l in open('dataset_20260531/cache_v4_dims/exprs.txt', encoding='utf-8') if l.strip())
failed = []
with gzip.open('dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        ex = json.loads(line).get('expr', '')
        if not ex or ex in usable: continue
        ne = ex.count('exp'); trig = ('sin' in ex or 'cos' in ex)
        if not (ne >= 2 or (ne >= 1 and trig) or 'log' in ex or 'Abs' in ex): continue
        res = expr_to_nodes(ex)
        if res is None or len(res[0]) > 48: continue
        failed.append(ex)
        if len(failed) >= 120: break

rng = np.random.default_rng(0); shown = 0
for ex in failed:
    s = None
    try: s = make_samples_v5(ex, rng, n_points=150)
    except Exception: s = None
    if s: continue              # recovered, skip
    # diagnose this failing one
    e = parse_to_sympy(ex)
    if e is None: continue
    free = sorted(str(x) for x in e.free_symbols)
    nv = len(free)
    # how many free vars are 'const' (fixed -> can't be target/sampled)
    nconst = sum(1 for v in free if class_range(v)[0] == 'const')
    # can sympy solve for the first non-const var?
    tgt = next((v for v in free if class_range(v)[0] != 'const'), None)
    solvable = '?'
    if tgt:
        try:
            sols = sp.solve(e, sp.Symbol(tgt), rational=False)
            solvable = f'{len(sols)} sol' if sols else 'NO_SOL'
        except Exception as ex2:
            solvable = 'SOLVE_ERR'
    # valid fraction: sample others in default ranges, plug forward residual
    print(f"nv={nv} nconst={nconst} solve({tgt})={solvable} | {ex[:75]}")
    shown += 1
    if shown >= 25: break
print(f"\n(showed {shown} failing formulas)")
