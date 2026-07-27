"""Equivalence-rearrangement generator for flow-target augmentation.

A physical law R(vars)=0 has many algebraically-equivalent residual forms (solve for
a different variable, multiply through by a variable). The formula encoder maps each
to a DIFFERENT z_f (~0.6 cosine apart), so augmenting the flow's target with these
forms turns each law's single z_f target into a spread cloud -> the flow hits one
more often -> higher ceiling. The frozen decoder decodes whichever form the sampled
z_f lands on, and the root-match judge accepts any of them.

This module: gen_equiv_forms(expr_str, var_names) -> list of equivalent residual
strings (parseable, within length). Used both to prototype/measure and to build the
augmented training index.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sympy as sp
from data.ast_grammar import parse_to_sympy, expr_to_nodes


def _ok_form(expr, var_order, max_len):
    """Parseable into the grammar AND within token length."""
    try:
        s = str(expr)
        res = expr_to_nodes(s, var_order=var_order)
        if res is None:
            return None
        if len(res[0]) > max_len:
            return None
        return s
    except Exception:
        return None


def gen_equiv_forms(expr_str, var_names, max_len=128, max_forms=4):
    """Return up to max_forms equivalent residual strings (excluding the original).
    Strategy: MULTIPLY/DIVIDE the residual through by variables (pure AST, no sp.solve
    -> instant, never hangs). R*v and R/v share the same zero manifold on the physical
    (positive) domain, and the root-match judge accepts them. These realize exactly the
    user's examples: R*r^2 -> 'F r^2 = kQq', R/k -> 'F/k = Qq/r^2'. Skips forms that
    blow the token length or don't parse into the grammar."""
    te = parse_to_sympy(expr_str)
    if te is None:
        return []
    forms = []
    seen = {str(sp.expand(te)) if sp.count_ops(te) < 40 else str(te)}
    syms = sorted([str(s) for s in te.free_symbols])
    # multiply/divide by each variable (and a couple of squares) -> different ASTs
    factors = []
    for v in syms:
        V = sp.Symbol(v)
        factors += [V, 1 / V]
    for v in syms[:2]:
        V = sp.Symbol(v)
        factors += [V ** 2]
    for fac in factors:
        try:
            resid = sp.together(sp.expand(te * fac))
            if resid == 0 or resid.free_symbols != te.free_symbols:
                continue
            key = str(resid)
            if key in seen:
                continue
            s = _ok_form(resid, var_names, max_len)
            if s is None:
                continue
            seen.add(key); forms.append(s)
            if len(forms) >= max_forms:
                break
        except Exception:
            continue
    return forms


if __name__ == '__main__':
    import time, random, numpy as np, torch
    # measure on a sample of cache formulas: yield rate, count, cost, z_f spread
    from train.train_manifold import Manifold, build_ast_edges
    exprs = [l.strip() for l in open('dataset_20260531/cache_v6/exprs.txt', encoding='utf-8')]
    random.seed(0); sample = random.sample(exprs, 80)

    ck = torch.load('sr_model/ckpt/manifold_v6_fixed.pt', map_location='cpu'); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64))
    model.load_state_dict(ck['model']); model.eval()

    def zf(s, vnf):
        res = expr_to_nodes(s, var_order=vnf)
        if res is None: return None
        seq = res[0]; ee = build_ast_edges(seq)
        ei = torch.tensor(ee).t().contiguous() if ee else torch.zeros(2, 0, dtype=torch.long)
        with torch.no_grad():
            return model.formula_enc(torch.tensor(seq), torch.zeros(len(seq)), ei,
                                     torch.zeros(len(seq), dtype=torch.long), 1)[0].numpy()

    def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    n_with = 0; total_forms = 0; spreads = []; t0 = time.time(); n_done = 0
    for s in sample:
        te = parse_to_sympy(s)
        if te is None: continue
        vn = sorted([str(x) for x in te.free_symbols])
        if len(vn) < 2: continue
        n_done += 1
        forms = gen_equiv_forms(s, vn, max_len=ck.get('dec_max_len', 128))
        if forms:
            n_with += 1; total_forms += len(forms)
            z0 = zf(s, vn)
            for f in forms:
                zfi = zf(f, vn)
                if z0 is not None and zfi is not None:
                    spreads.append(cos(z0, zfi))
    dt = time.time() - t0
    print(f"sampled {n_done} multi-var formulas")
    print(f"  {n_with} ({100*n_with/max(n_done,1):.0f}%) yielded >=1 equiv form")
    print(f"  avg equiv forms / yielding formula: {total_forms/max(n_with,1):.2f}")
    print(f"  z_f cos(original, rearrangement): mean={np.mean(spreads):.3f} (lower=more spread/useful)")
    print(f"  cost: {dt:.1f}s / {n_done} formulas = {1000*dt/max(n_done,1):.0f} ms/formula")
