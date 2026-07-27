"""Parse the combined-equation set 'eq1=0 + eq2=0 => eliminate v' surfaces into a real combined residual."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sympy as sp
from data.ast_grammar import parse_to_sympy

def parse_combo(surface):
    if '=>' not in surface or 'eliminate' not in surface:
        return None
    body, elim = surface.rsplit('=>', 1)
    var = elim.strip().replace('eliminate', '').strip()
    # split the two "expr = 0" clauses on "= 0"
    parts = [p.strip().lstrip('+').strip() for p in body.split('= 0') if p.strip().lstrip('+').strip()]
    if len(parts) < 2:
        return None
    r1 = parse_to_sympy(parts[0]); r2 = parse_to_sympy(parts[1])
    if r1 is None or r2 is None:
        return None
    v = sp.Symbol(var)
    try:
        sols = sp.solve(sp.Eq(r1, 0), v)
        if not sols:
            sols = sp.solve(sp.Eq(r2, 0), v)
            if not sols: return None
            combined = r1.subs(v, sols[0])
        else:
            combined = r2.subs(v, sols[0])
        combined = sp.simplify(combined)
        if combined == 0 or v in combined.free_symbols:
            return None
        return combined
    except Exception:
        return None

if __name__ == '__main__':
    import json
    C = 'data/combo.jsonl'
    rows = [json.loads(l) for l in open(C, encoding='utf-8') if l.strip()]
    ok = 0; wp = 0
    for r in rows[:8]:
        s = r.get('eval_truth_surface', ''); c = parse_combo(s)
        print('surface:', s[:75])
        print('  combined:', str(c)[:75] if c is not None else 'FAIL')
        if c is not None:
            ok += 1
            tv = r.get('train_values', {})
            need = {str(x) for x in c.free_symbols}
            have = set(tv)
            print('  well-posed:', need.issubset(have), '| need-have:', need - have)
    # full pass
    ok = sum(1 for r in rows if parse_combo(r.get('eval_truth_surface','')) is not None)
    print(f'\nfull: {ok}/{len(rows)} parse+eliminate OK')
