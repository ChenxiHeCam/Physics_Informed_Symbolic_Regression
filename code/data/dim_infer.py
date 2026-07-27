"""Infer SI dimensions for ALL variables of a formula from dimensional consistency,
anchored on tag-known vars + fundamental constants. Fills the dims the tag-db misses
so the cache `dims` field is fully populated (dimensional info IN the training data).

Per base dim independently: each var has an unknown exponent; constraints from the AST
(Add summands equal; Mul dims add; Pow(const) scales; exp/log/trig args dimensionless)
form a linear system solved by least squares with anchors substituted.
"""
import sympy as sp
import numpy as np

FUND = {
    'c': (0,1,-1,0,0,0), 'hbar': (1,2,-1,0,0,0), 'h': (1,2,-1,0,0,0),
    'k_B': (1,2,-2,-1,0,0), 'k': (1,2,-2,-1,0,0), 'G': (-1,3,-2,0,0,0),
    'e': (0,0,1,0,1,0), 'epsilon_0': (-1,-3,4,0,2,0), 'mu_0': (1,1,-2,0,-2,0),
    'N_A': (0,0,0,0,0,-1), 'pi': (0,0,0,0,0,0), 'R_gas': (1,2,-2,-1,0,-1),
}
_FUNCS = (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan,
          sp.sinh, sp.cosh, sp.tanh, sp.erf)


def infer_dims(expr_str, known):
    """known: {var: 6-tuple anchors}. Returns {var: 6-list} for all free vars, or None."""
    if isinstance(expr_str, str):
        try:
            from data.ast_grammar import parse_to_sympy
            e = parse_to_sympy(expr_str)
        except Exception:
            try: e = sp.sympify(expr_str)
            except Exception: return None
    else:
        e = expr_str
    if e is None or not hasattr(e, 'free_symbols'):
        return None
    free = sorted(str(s) for s in e.free_symbols)
    if not free:
        return None
    idx = {v: i for i, v in enumerate(free)}
    nV = len(free)
    constraints = []
    def dim_of(node):
        if node.is_Symbol:
            v = np.zeros(nV); v[idx[str(node)]] = 1.0; return v
        if node.is_Number:
            return np.zeros(nV)
        if node.is_Add:
            ch = [dim_of(a) for a in node.args]
            for k in range(1, len(ch)):
                constraints.append(ch[0] - ch[k])
            return ch[0]
        if node.is_Mul:
            tot = np.zeros(nV)
            for a in node.args: tot = tot + dim_of(a)
            return tot
        if node.is_Pow:
            b, p = node.args
            if p.is_Number:
                return float(p) * dim_of(b)
            constraints.append(dim_of(b)); return np.zeros(nV)
        if isinstance(node, _FUNCS):
            for a in node.args: constraints.append(dim_of(a))
            return np.zeros(nV)
        tot = np.zeros(nV)
        for a in node.args: tot = tot + dim_of(a)
        return tot
    try:
        dim_of(e)
    except Exception:
        return None
    if not constraints:
        return None
    A = np.array(constraints)
    out = {}
    for bd in range(6):
        anchored = {idx[v]: known[v][bd] for v in free if v in known}
        for v in free:
            if v in FUND and idx[v] not in anchored: anchored[idx[v]] = FUND[v][bd]
        if not anchored:
            anchored[idx[free[0]]] = 0.0
        free_idx = [i for i in range(nV) if i not in anchored]
        if not free_idx:
            x = np.zeros(nV)
            for i, val in anchored.items(): x[i] = val
        else:
            Af = A[:, free_idx]
            rhs = -A[:, list(anchored)] @ np.array([anchored[i] for i in anchored], float)
            try:
                sol, *_ = np.linalg.lstsq(Af, rhs, rcond=None)
            except Exception:
                return None
            x = np.zeros(nV)
            for i, val in anchored.items(): x[i] = val
            for j, i in enumerate(free_idx): x[i] = sol[j]
        for v in free: out.setdefault(v, [0]*6)[bd] = int(round(float(x[idx[v]])))
    return out
