"""
Dimensional analysis over a formula AST: propagate SI dimensions through the
expression given each variable's dimension, and report whether the formula is
dimensionally CONSISTENT.

This is the basis for "dimensions are LEARNED, not just input":
  - a dimensional-consistency LOSS uses formula_dim(...).consistent
  - the key physics rule: arguments of exp/log/sin/cos/... MUST be dimensionless,
    which is exactly why physical formulas form dimensionless groups (hν/kT, v/c).

dim vector = (M, L, T, Theta, I, N)  (6 SI base dims; luminous dropped).
"""
import sympy as sp
import numpy as np

DIMLEN = 6
ZERO = (0.0,) * DIMLEN

_TRANSCENDENTAL = (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh,
                   sp.tanh, sp.asin, sp.acos, sp.atan, sp.sign, sp.Abs)


class DimResult:
    __slots__ = ('dim', 'consistent')
    def __init__(self, dim, consistent):
        self.dim = dim; self.consistent = consistent


def _add(a, b): return tuple(x + y for x, y in zip(a, b))
def _scale(a, k): return tuple(x * k for x, y in zip(a, [0]*DIMLEN))
def _eq(a, b, tol=1e-6): return all(abs(x - y) < tol for x, y in zip(a, b))
def _is_zero(a, tol=1e-6): return all(abs(x) < tol for x in a)


def formula_dim(expr, var_dims):
    """expr: sympy; var_dims: {symbol_name: 6-tuple}. Returns DimResult."""
    ok = [True]

    def walk(e):
        if e.is_Symbol:
            return var_dims.get(str(e), ZERO)         # unknown -> treat dimensionless
        if e.is_Number or e is sp.pi or e is sp.E:
            return ZERO                                # pure numbers are dimensionless
        if e.is_Add:
            ds = [walk(a) for a in e.args]
            base = ds[0]
            for d in ds[1:]:
                if not _eq(d, base):
                    ok[0] = False                      # adding unlike dimensions
            return base
        if e.is_Mul:
            tot = ZERO
            for a in e.args:
                tot = _add(tot, walk(a))               # dims add under multiplication
            return tot
        if e.is_Pow:
            base, exp = e.args
            bd = walk(base)
            if exp.is_Number:
                k = float(exp)
                return tuple(x * k for x in bd)        # dim scales by exponent
            else:
                # variable exponent -> base must be dimensionless to be consistent
                ed = walk(exp)
                if not _is_zero(bd):
                    ok[0] = False
                if not _is_zero(ed):
                    ok[0] = False
                return ZERO
        if isinstance(e, _TRANSCENDENTAL) or (hasattr(e, 'func') and e.func in _TRANSCENDENTAL):
            for a in e.args:
                ad = walk(a)
                if not _is_zero(ad):
                    ok[0] = False                      # transcendental arg must be dimensionless!
            return ZERO
        # unknown function: walk args, return dimensionless (best effort)
        for a in getattr(e, 'args', []):
            walk(a)
        return ZERO

    try:
        d = walk(expr)
    except Exception:
        return DimResult(ZERO, False)
    return DimResult(d, ok[0])


if __name__ == '__main__':
    # base dims (M,L,T,Theta,I,N)
    M=(1,0,0,0,0,0); L=(0,1,0,0,0,0); T=(0,0,1,0,0,0); Th=(0,0,0,1,0,0)
    E=(1,2,-2,0,0,0)  # energy
    tests = [
        ('E=mc^2', 'E - m*c**2', {'E':E,'m':M,'c':(0,1,-1,0,0,0)}),
        ('Planck arg', 'h*nu/(k_B*Tt)', {'h':(1,2,-1,0,0,0),'nu':(0,0,-1,0,0,0),'k_B':(1,2,-2,-1,0,0),'Tt':Th}),
        ('relativistic', '1/sqrt(1 - v**2/c**2)', {'v':(0,1,-1,0,0,0),'c':(0,1,-1,0,0,0)}),
        ('BAD: add L+T', 'x + t', {'x':L,'t':T}),
        ('BAD: exp(mass)', 'exp(m)', {'m':M}),
        ('Boltzmann exp(-E/kT)', 'exp(-E/(k_B*Tt))', {'E':E,'k_B':(1,2,-2,-1,0,0),'Tt':Th}),
    ]
    for name, ex, vd in tests:
        e = sp.sympify(ex, locals={s: sp.Symbol(s) for s in vd})
        r = formula_dim(e, vd)
        print(f"{name:24} consistent={r.consistent}  dim={tuple(round(x,2) for x in r.dim)}")
