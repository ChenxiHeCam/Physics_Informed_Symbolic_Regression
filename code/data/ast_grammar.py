"""
AST grammar for physics formulas.

Converts between:
  expr string ("F = m*a")  <->  sympy AST  <->  flat prefix node sequence

The node sequence is what the tree decoder produces, one node at a time,
top-down (prefix / pre-order). Grammar constraints during decoding guarantee
syntactically valid output (each operator consumes a fixed arity of children).

Vocabulary (node types):
  - BINOPS  : Add Mul Pow  (sympy n-ary flattened to binary at encode time)
  - UNOPS   : Neg sin cos tan exp log sqrt tanh sinh cosh Abs asin acos atan
  - LEAF    : VAR_i (variable slot), CONST (numeric, value carried separately),
              SYMCONST (pi, E)
  - special : <eq> (top-level "lhs - rhs"), <pad>, <eos>
"""
from __future__ import annotations
import re
import sympy as sp

# ---- node-type vocabulary -------------------------------------------------
BINOPS = ["Add", "Mul", "Pow"]
UNOPS  = ["Neg", "sin", "cos", "tan", "exp", "log", "sqrt", "tanh",
          "sinh", "cosh", "Abs", "asin", "acos", "atan",
          "cot", "sec", "csc", "acot", "asinh", "acosh", "atanh", "sign"]
SYMCONST = ["pi", "E"]
SPECIAL  = ["<pad>", "<eos>", "CONST"]
MAX_VARS = 16
VARS = [f"VAR_{i}" for i in range(MAX_VARS)]

NODE_TYPES = SPECIAL + BINOPS + UNOPS + SYMCONST + VARS
NT2ID = {t: i for i, t in enumerate(NODE_TYPES)}
ID2NT = {i: t for t, i in NT2ID.items()}
VOCAB_SIZE = len(NODE_TYPES)

ARITY = {**{op: 2 for op in BINOPS}, **{op: 1 for op in UNOPS}}
# leaves (VAR_i, CONST, pi, E) have arity 0


# ---- sympy func -> node type ----------------------------------------------
_FUNC2NT = {
    sp.sin: "sin", sp.cos: "cos", sp.tan: "tan", sp.exp: "exp",
    sp.log: "log", sp.sqrt: "sqrt", sp.tanh: "tanh", sp.sinh: "sinh",
    sp.cosh: "cosh", sp.Abs: "Abs", sp.asin: "asin", sp.acos: "acos",
    sp.atan: "atan", sp.cot: "cot", sp.sec: "sec", sp.csc: "csc",
    sp.acot: "acot", sp.asinh: "asinh", sp.acosh: "acosh",
    sp.atanh: "atanh", sp.sign: "sign",
}


def _strip_eq(expr_str: str) -> str:
    """'F = m*a' -> 'F - (m*a)' ; 'X - (...) = 0' -> 'X - (...)'."""
    s = expr_str.strip()
    s = re.sub(r"=\s*0\s*$", "", s).strip()
    if s.endswith("="):
        s = s[:-1]
    if "=" in s:
        lhs, _, rhs = s.partition("=")
        return f"({lhs.strip()}) - ({rhs.strip()})"
    return s


_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
# functions/constants that must KEEP their sympy meaning (not become a Symbol)
_RESERVED = {
    "sin", "cos", "tan", "exp", "log", "ln", "sqrt", "tanh", "sinh", "cosh",
    "Abs", "abs", "asin", "acos", "atan", "cot", "sec", "csc", "acot",
    "asinh", "acosh", "atanh", "sign", "pi", "Pow", "Mul", "Add",
}


def _build_locals(text: str) -> dict:
    """Force every identifier to a plain Symbol EXCEPT reserved funcs/pi.
    Fixes I (current), E (energy), N, S, gamma... being read as sympy
    builtins (imaginary unit, Euler's number, etc.)."""
    locs = {}
    for name in set(_IDENT_RE.findall(text)):
        if name in _RESERVED:
            continue
        locs[name] = sp.Symbol(name)
    return locs


def parse_to_sympy(expr_str: str):
    """Return a sympy Expr (residual form, == 0) or None if unparseable."""
    lhs = _strip_eq(expr_str)
    try:
        return sp.sympify(lhs, locals=_build_locals(lhs), evaluate=True)
    except Exception:
        return None


def expr_to_nodes(expr_str: str, var_order=None):
    """
    expr string -> (node_id_seq, const_values, var_names)
    Pre-order flatten. n-ary Add/Mul are binarised left-fold.
    Returns None on failure.
    """
    e = parse_to_sympy(expr_str)
    if e is None:
        return None
    if not hasattr(e, 'free_symbols') or isinstance(e, sp.logic.boolalg.BooleanAtom):
        return None
    free = sorted([str(s) for s in e.free_symbols])
    if var_order is None:
        var_order = free
    if len(var_order) > MAX_VARS:
        return None
    var2slot = {v: i for i, v in enumerate(var_order)}

    seq, consts = [], []

    def emit(nt, const_val=0.0):
        seq.append(NT2ID[nt]); consts.append(const_val)

    def walk(node):
        if node.is_Symbol:
            name = str(node)
            if name not in var2slot:
                raise ValueError(f"unknown var {name}")
            emit(f"VAR_{var2slot[name]}")
        elif node.is_Number:
            if node == sp.pi: emit("pi")
            elif node == sp.E: emit("E")
            else:
                emit("CONST", float(node))
        elif node is sp.pi:
            emit("pi")
        elif node is sp.E:
            emit("E")
        elif node.is_Add or node.is_Mul:
            args = list(node.args)
            nt = "Add" if node.is_Add else "Mul"
            # left-fold binarise: op(op(op(a,b),c),d)
            # emit (n-1) ops then args in order won't preorder cleanly;
            # do recursive binary split
            def fold(lst):
                if len(lst) == 1:
                    walk(lst[0]); return
                emit(nt)
                walk(lst[0])
                fold(lst[1:])
            fold(args)
        elif node.is_Pow:
            emit("Pow"); walk(node.base); walk(node.exp)
        elif node.func in _FUNC2NT:
            emit(_FUNC2NT[node.func]);
            for a in node.args: walk(a)
        elif node.func == sp.Mul:
            pass
        else:
            # unsupported (Derivative, Integral, unknown Function, ...)
            raise ValueError(f"unsupported node {node.func}")

    try:
        walk(e)
    except Exception:
        return None
    seq.append(NT2ID["<eos>"]); consts.append(0.0)
    return seq, consts, var_order


def nodes_to_expr(seq, consts, var_names):
    """Inverse: node-id seq + const values -> sympy expr (best-effort)."""
    pos = 0
    def build():
        nonlocal pos
        if pos >= len(seq): return sp.Integer(0)
        nt = ID2NT[seq[pos]]; cv = consts[pos]; pos += 1
        if nt == "<eos>" or nt == "<pad>":
            return sp.Integer(0)
        if nt.startswith("VAR_"):
            idx = int(nt.split("_")[1])
            name = var_names[idx] if idx < len(var_names) else f"x{idx}"
            return sp.Symbol(name)
        if nt == "CONST": return sp.Float(cv)
        if nt == "pi": return sp.pi
        if nt == "E": return sp.E
        if nt in ARITY:
            ar = ARITY[nt]
            kids = [build() for _ in range(ar)]
            if nt == "Add": return kids[0] + kids[1]
            if nt == "Mul": return kids[0] * kids[1]
            if nt == "Pow": return kids[0] ** kids[1]
            if nt == "Neg": return -kids[0]
            f = {v: k for k, v in _FUNC2NT.items()}.get(nt)
            if f is not None: return f(kids[0])
        return sp.Integer(0)
    try:
        return build()
    except Exception:
        return None


if __name__ == "__main__":
    for s in ["F = m*a", "S - k_B * log(W) = 0", "E - m*c**2 = 0",
              "y - (1/2)*g*t**2 = 0", "p - m_0*v/sqrt(1 - v**2/c**2) = 0"]:
        r = expr_to_nodes(s)
        if r is None:
            print(f"FAIL: {s}"); continue
        seq, consts, vars_ = r
        toks = [ID2NT[i] for i in seq]
        back = nodes_to_expr(seq, consts, vars_)
        print(f"{s}\n  vars={vars_}\n  nodes={toks}\n  back={back}\n")
    print(f"VOCAB_SIZE={VOCAB_SIZE}")
