import sys, os, json
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))
import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef
from itertools import islice
import gen_dataside as GD

try:
    from sympy.parsing.latex import parse_latex
    HAVE = True
except Exception as e:
    HAVE = False
    print('parse_latex import FAILED:', type(e).__name__, str(e)[:80])

rng = np.random.default_rng(5)
fail = latex_ok = algebraic = 0
if HAVE:
    with open('data/data/augmented/round1_ds_dedup_0050.jsonl') as f:
        for l in islice(f, 0, 2000):
            e = json.loads(l).get('new_expr')
            if not e or GD.make_one(e, rng, None):
                continue
            fail += 1
            try:
                ex = parse_latex(e)
                latex_ok += 1
                if ex.atoms(AppliedUndef) or ex.has(sp.I) or ex.has(sp.Integral, sp.Derivative):
                    continue
                fs = getattr(ex, 'free_symbols', set())
                if 2 <= len(fs) <= 12:
                    algebraic += 1
            except Exception:
                pass
    print(f'failing {fail}: parse_latex解析成功 {latex_ok}, 其中纯代数(可点云) {algebraic} = {algebraic*100//max(fail,1)}%')
