"""Randomly sample FAILED formulas (not in usable set) and categorize them, with
random examples per category — to see the full breakdown beyond high-dim transcendental."""
import sys, os, gzip, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from data.ast_grammar import expr_to_nodes, parse_to_sympy

usable = set(l.strip() for l in open('dataset_20260531/cache_v4_dims/exprs.txt', encoding='utf-8') if l.strip())
rng = np.random.default_rng(7)
# reservoir-sample failed formulas across the whole file
failed = []
with gzip.open('dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        ex = json.loads(line).get('expr', '')
        if not ex or ex in usable: continue
        if len(failed) < 4000: failed.append(ex)
        elif rng.random() < 0.02:
            failed[rng.integers(0, 4000)] = ex
print(f"sampled {len(failed)} failed formulas\n")

cats = {'junk_nv0': [], 'len>48': [], 'highdim_transc': [], 'lowdim_transc': [],
        'plain_multivar': [], 'other': []}
for ex in failed:
    e = parse_to_sympy(ex)
    if e is None or not hasattr(e, 'free_symbols'):
        cats['other'].append(ex); continue
    nv = len(e.free_symbols)
    res = expr_to_nodes(ex); sl = len(res[0]) if res else 999
    transc = any(t in ex for t in ('exp', 'log', 'sin', 'cos', 'tan', 'Abs', 'sqrt'))
    if nv == 0:
        cats['junk_nv0'].append(ex)
    elif sl > 48:
        cats['len>48'].append(ex)
    elif transc and nv >= 8:
        cats['highdim_transc'].append(ex)
    elif transc:
        cats['lowdim_transc'].append(ex)
    elif nv >= 2:
        cats['plain_multivar'].append(ex)
    else:
        cats['other'].append(ex)

tot = len(failed)
print("category          count   pct   examples")
for c, lst in sorted(cats.items(), key=lambda x: -len(x[1])):
    print(f"\n=== {c}: {len(lst)} ({100*len(lst)/tot:.0f}%) ===")
    idx = rng.choice(len(lst), min(4, len(lst)), replace=False) if lst else []
    for i in idx:
        print(f"   {lst[i][:90]}")
