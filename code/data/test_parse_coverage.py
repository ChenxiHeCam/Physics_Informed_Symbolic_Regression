"""
Test AST-grammar parse coverage on the real 1.7M formula dataset.
Categorize failures so we know what to fix.
"""
import sys, os, gzip, json, re, warnings
from collections import Counter
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from ast_grammar import expr_to_nodes, parse_to_sympy

NODES = 'dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz'
if not os.path.exists(NODES):
    NODES = 'dataset_20260531/unified_nodes.jsonl.gz'

N_TEST = int(os.environ.get('N_TEST', '50000'))

def classify_fail(expr):
    """Why did parsing fail? Return a category tag."""
    s = expr
    if '=> eliminate' in s: return 'combo_eliminate'
    if any(t in s for t in ['nabla','∇','partial','d_dx','d_dt','grad','div','curl']):
        return 'differential_operator'
    if any(t in s.lower() for t in ['constant','const','where','for all','exists','such that',
                                     'probability','approximately','proportional']):
        return 'natural_language'
    if '~' in s or '≈' in s or '∝' in s: return 'approx_symbol'
    if '\\' in s: return 'latex'
    if 'int_' in s or '∫' in s or 'sum_' in s or '∑' in s: return 'integral_sum'
    if '[' in s and ']' in s: return 'bracket_index'
    if re.search(r'\b(erf|erfc|gamma|Gamma|zeta|besselj|bessel)\b', s): return 'special_function'
    if '{' in s or '}' in s: return 'brace'
    # try sympy to get the actual error
    try:
        sp.sympify(re.sub(r'=\s*0\s*$','',s.split('=')[0] if '=' in s else s))
        return 'parsed_by_sympy_not_grammar'  # sympy ok but our grammar rejects
    except Exception as e:
        return f'sympy_err:{type(e).__name__}'

ok = 0
fail = 0
fail_cats = Counter()
fail_samples = {}
n = 0
with gzip.open(NODES, 'rt', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        expr = r.get('expr', '')
        if not expr: continue
        n += 1
        res = expr_to_nodes(expr)
        if res is not None:
            ok += 1
        else:
            fail += 1
            cat = classify_fail(expr)
            fail_cats[cat] += 1
            if cat not in fail_samples:
                fail_samples[cat] = []
            if len(fail_samples[cat]) < 4:
                fail_samples[cat].append(expr[:75])
        if n >= N_TEST: break

print(f"Tested {n} real formulas")
print(f"  parse OK: {ok} ({100*ok/n:.1f}%)")
print(f"  FAIL:     {fail} ({100*fail/n:.1f}%)\n")
print("Failure breakdown (sorted):")
for cat, cnt in fail_cats.most_common():
    print(f"  {cat:35s} {cnt:6d} ({100*cnt/n:.1f}%)")
    for s in fail_samples.get(cat, [])[:3]:
        print(f"        e.g. {s}")
