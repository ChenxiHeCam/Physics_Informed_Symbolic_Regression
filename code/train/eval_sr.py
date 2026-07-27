"""
Corrected SR eval — target-last var convention + constant fitting.

Representation (after the build_cache remap):
  A formula is the IMPLICIT residual R(s_0..s_n)=0 of the full equation.
  VAR slots are target-last:  VAR_0..VAR_{k-1} = the k observed X-columns
  (others, sorted), VAR_k = the target (the data-encoder y-channel). This makes
  the decoder's slots consistent with z_d, which the old sorted(all-symbols)
  convention did not.

Per (X,y) task:
  z_d = data_encoder(X=others, y=target)
  candidates from:
    A) retrieval  : top-R index formulas by cos(z_d, z_f)            (memory)
    B) flow+decode: flow.sample(z_d,K) -> decoder.generate -> AST    (generation)
  each candidate -> skeleton (CONST leaves -> free params) -> fit constants to
  the data by least-squares -> residual-NMSE for ranking.
  Q@k = truth recovered within top-k by numeric equivalence.

--mode train    : sample rows from the pair jsonl (IN-DISTRIBUTION sanity gate —
                  the model has seen these formulas; recovery should be high).
--mode heldout  : real591 (NOT in the 1.9M library — the honest generalization test).
"""
import sys, os, gzip, json, time, argparse, random, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
import sympy as sp
from scipy.optimize import least_squares

from data.ast_grammar import (parse_to_sympy, ID2NT, ARITY, NT2ID, _FUNC2NT,
                              MAX_VARS)
from data.dimensions import class_to_dim, DIM_LEN
from data.dim_analysis import formula_dim
from train.train_manifold import Manifold, PairDataset, load_state_compat

# semantic_class lookup for dimension input (units are known at test time)
_SYM2CLASS = {}
def _load_sym2class():
    if _SYM2CLASS: return
    try:
        for line in open('data/symbol_semantic_tags_20260522.jsonl', encoding='utf-8'):
            if line.strip():
                r = json.loads(line); _SYM2CLASS[r['symbol']] = r.get('semantic_class', 'other')
    except Exception:
        pass

def dims_for_task(others, target, dim_len):
    """(MAX_VARS+1, dim_len) SI dimension vectors: chans 0..k-1 = others, chan MAX_VARS = target."""
    _load_sym2class()
    d = np.zeros((MAX_VARS + 1, dim_len), dtype=np.float32)
    for i, o in enumerate(others):
        d[i] = class_to_dim(_SYM2CLASS.get(o, 'other'))[:dim_len]
    d[MAX_VARS] = class_to_dim(_SYM2CLASS.get(target, 'other'))[:dim_len]
    return d

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# candidate construction from decoded node sequence
# ---------------------------------------------------------------------------

def nodes_to_skeleton(seq, var_names_full):
    """node seq -> (sympy expr with CONST leaves replaced by free params c0.., [params]).
    var_names_full = others + [target]  (target-last convention)."""
    params = []
    pos = 0
    def build():
        nonlocal pos
        if pos >= len(seq):
            return sp.Integer(0)
        nt = ID2NT[seq[pos]]; pos += 1
        if nt in ("<eos>", "<pad>"):
            return sp.Integer(0)
        if nt.startswith("VAR_"):
            idx = int(nt.split("_")[1])
            name = var_names_full[idx] if idx < len(var_names_full) else f"x{idx}"
            return sp.Symbol(name)
        if nt == "CONST":
            c = sp.Symbol(f"c{len(params)}"); params.append(c); return c
        if nt == "pi": return sp.pi
        if nt == "E":  return sp.E
        if nt in ARITY:
            kids = [build() for _ in range(ARITY[nt])]
            if nt == "Add": return kids[0] + kids[1]
            if nt == "Mul": return kids[0] * kids[1]
            if nt == "Pow": return kids[0] ** kids[1]
            if nt == "Neg": return -kids[0]
            f = {v: k for k, v in _FUNC2NT.items()}.get(nt)
            if f is not None: return f(kids[0])
        return sp.Integer(0)
    try:
        e = build()
    except Exception:
        return None, []
    return e, params


def fit_residual(expr, params, symvals, target, c_init=None, n_restarts=4, seed=0):
    """Fit free params so the implicit residual expr(s)=0 holds on the data.
    symvals: dict symbol-name -> np.array of values (others from X, target from y).
    Returns (fitted_expr, residual_nmse). Lower is better. The target symbol must
    appear or the candidate cannot predict it (nmse=1e9)."""
    free = {str(s) for s in expr.free_symbols}
    if target not in free:
        return expr, 1e9
    syms = sorted(free - {str(p) for p in params})
    # all data symbols must be available
    if any(s not in symvals for s in syms):
        return expr, 1e9
    n = len(next(iter(symvals.values())))
    yscale = np.std(symvals[target]) + 1e-9
    try:
        f = sp.lambdify([sp.Symbol(s) for s in syms] + params, expr, 'numpy')
    except Exception:
        return expr, 1e9
    argcols = [symvals[s] for s in syms]
    # per-row scale: on WIDE-dynamic-range data (y spanning many decades) a single
    # global yscale lets the largest-magnitude rows dominate the sum-of-squares, so
    # the fit lands on wrong constants/exponents. Divide each row by its own local
    # scale (|dominant term| ~ |y_row|) so every point contributes comparably.
    yabs = np.abs(np.asarray(symvals[target], float))
    rowscale = np.maximum(yabs, 0.1 * (np.median(yabs[yabs > 0]) if np.any(yabs > 0) else 1.0)) + 1e-9
    def resid(theta):
        try:
            r = np.asarray(f(*argcols, *theta), float)
            r = np.broadcast_to(r, (n,)).astype(float).copy()
            r[~np.isfinite(r)] = 1e3 * yscale
            return r / rowscale
        except Exception:
            return np.full(n, 1e3)
    if not params:
        r = resid([]); return expr, float(np.mean(r**2))
    rng = np.random.default_rng(seed)
    best = None
    inits = []
    if c_init is not None and len(c_init) == len(params):
        inits.append(np.array(c_init, float))
    inits.append(np.ones(len(params)))
    inits.append(np.full(len(params), 2.0))   # exponent-ish prior
    inits.append(np.full(len(params), 3.0))   # cubic / 3-halves regimes
    inits.append(np.full(len(params), 0.5))   # sqrt regimes
    while len(inits) < max(n_restarts, 8):
        inits.append(rng.normal(0, 1.5, len(params)))
    for x0 in inits:
        try:
            sol = least_squares(resid, x0, method='lm', max_nfev=150)
            nm = float(np.mean(resid(sol.x)**2))
            if best is None or nm < best[1]:
                best = (sol.x, nm)
        except Exception:
            continue
    if best is None:
        return expr, 1e9
    # nice-value recovery: physics consts/exponents are usually integers or simple
    # fractions. least_squares (esp. for EXPONENTS, which are non-convex) lands in a
    # wrong local min (a**3 -> a**0.5). EXHAUSTIVE coordinate descent: for each param
    # try EVERY nice value (not just ones near the fit) and keep any strict residual
    # improvement; repeat until stable. This jumps a**0.5 -> a**3 the old ±0.12 snap
    # could never reach. Then a final LM polish refines the non-nice coefficients.
    xb = best[0].copy(); nm = best[1]
    nice = np.array([0, 1, -1, 2, -2, 3, -3, 4, -4, 5, 6, 0.5, -0.5, 1.5, -1.5,
                     2.5, -2.5, 1/3, 2/3, 4/3, -1/3, 0.25, 0.75, 1.25,
                     3/5, 1/5, 2/5, 4/5, 5/2, 3/2], float)
    for _pass in range(3):
        improved = False
        for i in range(len(xb)):
            bv, bnm = xb[i], nm
            for v in nice:
                xt = xb.copy(); xt[i] = v
                nmt = float(np.mean(resid(xt) ** 2))
                if nmt < bnm - 1e-12:
                    bv, bnm = v, nmt
            if bv != xb[i]:
                xb[i] = bv; nm = bnm; improved = True
        if not improved:
            break
    # final LM polish from the snapped point (refines free coefficients without
    # disturbing the recovered integer exponents much); keep only if it improves
    try:
        sol = least_squares(resid, xb, method='lm', max_nfev=120)
        nm2 = float(np.mean(resid(sol.x) ** 2))
        if nm2 < nm - 1e-12:
            xb, nm = sol.x, nm2
    except Exception:
        pass
    fitted = expr.subs({p: float(v) for p, v in zip(params, xb)})
    return fitted, nm


def predict_nmse(expr, symvals, target, nmax=50):
    """Score a candidate by how well it PREDICTS the target from the others on the
    data (solve expr=0 for target). Degenerate fits (residual≡0 via consts->0) do
    not actually involve/constrain the target and get 1e9 — this is what makes the
    ranker robust, unlike scoring the implicit residual directly."""
    t = sp.Symbol(target)
    if not expr.has(t):
        return 1e9
    others = sorted([s for s in symvals if s != target])
    if any(o not in symvals for o in others):
        return 1e9
    y = np.asarray(symvals[target], float)
    nrow = len(y)
    idx = np.arange(nrow) if nrow <= nmax else np.linspace(0, nrow - 1, nmax).astype(int)
    yy = y[idx]
    yvar = np.var(yy) + 1e-12
    # 1) linear-in-target closed form (covers most physics residuals)
    try:
        p = sp.Poly(expr, t)
        if p.degree() == 1:
            a = p.coeff_monomial(t); b = p.coeff_monomial(1)
            fa = sp.lambdify([sp.Symbol(o) for o in others], a, 'numpy')
            fb = sp.lambdify([sp.Symbol(o) for o in others], b, 'numpy')
            cols = [symvals[o][idx] for o in others]
            av = np.broadcast_to(np.asarray(fa(*cols), float), (len(idx),)).astype(float)
            bv = np.broadcast_to(np.asarray(fb(*cols), float), (len(idx),)).astype(float)
            with np.errstate(all='ignore'):
                yhat = -bv / av
            ok = np.isfinite(yhat) & (np.abs(av) > 1e-12)
            if ok.sum() < max(10, 0.3 * len(idx)): return 1e9
            return float(np.mean((yhat[ok] - yy[ok]) ** 2) / yvar)
    except Exception:
        pass
    # 2) nonlinear in target -> brentq per (subsampled) point, bracket around y_j
    try:
        from scipy.optimize import brentq
        f = sp.lambdify([sp.Symbol(o) for o in others] + [t], expr, 'numpy')
        cols = {o: symvals[o] for o in others}
        errs = []
        for j in idx:
            vals = [cols[o][j] for o in others]
            def g(tv):
                try:
                    r = float(f(*vals, tv)); return r if np.isfinite(r) else 1e9
                except Exception:
                    return 1e9
            yj = float(y[j]); root = None
            for span in (0.5, 2.0, 10.0):
                lo, hi = yj - span * abs(yj) - span, yj + span * abs(yj) + span
                try:
                    if g(lo) * g(hi) < 0:
                        root = brentq(g, lo, hi, maxiter=40); break
                except Exception:
                    pass
            errs.append((root - yj) if root is not None else np.nan)
        errs = np.asarray(errs, float); ok = np.isfinite(errs)
        if ok.sum() < max(10, 0.3 * len(idx)): return 1e9
        return float(np.mean(errs[ok] ** 2) / yvar)
    except Exception:
        return 1e9


_TRUTH_SOLVE_CACHE = {}


def _solve_truth_branches(truth, target):
    """Solve the TRUTH for the target symbolically -> lambdified real branches
    y_target(others). Cached per (truth, target). The truth is a clean physics law,
    so sp.solve is reliable (unlike on fitted candidates, which can hang)."""
    key = (str(truth), target)
    if key in _TRUTH_SOLVE_CACHE:
        return _TRUTH_SOLVE_CACHE[key]
    te = parse_to_sympy(truth) if isinstance(truth, str) else truth
    T = sp.Symbol(target)
    others = sorted([str(s) for s in te.free_symbols if str(s) != target]) if te is not None else []
    osym = [sp.Symbol(o) for o in others]
    branches = []
    if te is not None and others:
        try:
            for s in sp.solve(sp.Eq(te, 0), T):
                try:
                    branches.append(sp.lambdify(osym, s, 'numpy'))
                except Exception:
                    pass
        except Exception:
            pass
    out = (others, osym, branches)
    _TRUTH_SOLVE_CACHE[key] = out
    return out


def _rootmatch(cand_expr, target, others, osym, branches, n=40, tol=1e-3, frac=0.8, seed=11):
    """True if the candidate root-matches the truth target over a WIDE off-data
    domain (sign-branch tolerant). Equivalent forms (incl. rearrangements and
    ugly-but-correct fitted exponents) match to ~1e-3 everywhere; data-overfits
    diverge off the fitted points. Validated to beat both num_equiv and a reference check
    on the real591 audit (2026-06-07)."""
    from scipy.optimize import brentq
    if cand_expr is None or not branches:
        return False
    T = sp.Symbol(target)
    if not cand_expr.has(T):
        return False
    if not set(others).issubset({str(s) for s in cand_expr.free_symbols}):
        return False          # candidate must use every variable the truth uses
    try:
        fc = sp.lambdify(osym + [T], cand_expr, 'numpy')
    except Exception:
        return False
    rng = np.random.default_rng(seed)
    diffs = []; matched = 0; total = 0
    for _ in range(n):
        vals = [float(rng.uniform(1.0, 15.0)) for _ in others]
        yt = None
        for fb in branches:
            try:
                yv = complex(fb(*vals))
                if abs(yv.imag) < 1e-6 * (abs(yv.real) + 1):
                    yt = yv.real; break
            except Exception:
                pass
        if yt is None or not np.isfinite(yt):
            continue
        total += 1; a = abs(yt) + 1.0

        def g(x):
            try:
                r = float(fc(*vals, x)); return r if np.isfinite(r) else 1e18
            except Exception:
                return 1e18
        if abs(g(yt + 0.3 * a) - g(yt - 0.3 * a)) < 1e-12:
            continue          # candidate insensitive to target -> degenerate
        root = None
        for span in (0.05, 0.2, 0.8, 3.0):
            lo, hi = yt - span * a, yt + span * a
            try:
                if g(lo) * g(hi) < 0:
                    root = brentq(g, lo, hi, maxiter=60); break
            except Exception:
                pass
        if root is None:
            continue
        rd = min(abs(root - yt), abs(root + yt)) / (abs(yt) + 1e-9)
        diffs.append(rd)
        if rd < tol:
            matched += 1
    if total < 6 or len(diffs) < 5:
        return False
    return bool(np.median(diffs) < tol and matched >= frac * len(diffs))


def is_correct(cand_expr, truth, symvals, target, pn=None):
    """Authoritative SR correctness via off-data root-matching against the truth.
    Solve the truth for the target, then check the candidate root-matches it over a
    wide random domain (sign-branch tolerant). Catches BOTH num_equiv's failure
    modes: overfits that nail the data points (false-pos) and algebraic
    rearrangements / sign branches / ugly fitted exponents (false-neg). Falls back to
    the legacy predict_nmse+num_equiv path only when the truth is not solvable."""
    others, osym, branches = _solve_truth_branches(truth, target)
    if branches:
        return _rootmatch(cand_expr, target, others, osym, branches)
    # fallback: truth unsolvable symbolically -> legacy data-based judge
    if pn is None:
        pn = predict_nmse(cand_expr, symvals, target)
    if pn < 1e-6:
        try:
            te = parse_to_sympy(truth) if isinstance(truth, str) else truth
            tvars = {str(s) for s in te.free_symbols}
            cvars = {str(s) for s in cand_expr.free_symbols}
            if not tvars.issubset(cvars):     # candidate missing a truth variable
                return num_equiv(cand_expr, truth, symvals, target)
        except Exception:
            pass
        return True
    return num_equiv(cand_expr, truth, symvals, target)


def num_equiv(a, b, symvals=None, target=None, n=120, seed=1):
    """Numeric equivalence of two residual expressions, up to a multiplicative
    constant. Variables are sampled FREELY (independent uniform over a couple of
    positive ranges) — NOT from the data domain. This is essential: on the data
    domain both the true residual and any data-fitted candidate are ~0, so they
    would always look 'equivalent' (circular). Off-surface, the true residual is
    nonzero, so only a genuinely equivalent formula matches. Guards reject the
    degenerate all-zero candidate (consts->0)."""
    try:
        ce = parse_to_sympy(a) if isinstance(a, str) else a
        te = parse_to_sympy(b) if isinstance(b, str) else b
        if ce is None or te is None: return False
        allv = sorted(set([str(s) for s in ce.free_symbols] +
                          [str(s) for s in te.free_symbols]))
        if not allv: return False
        rng = np.random.default_rng(seed)
        cf = sp.lambdify([sp.Symbol(v) for v in allv], ce, 'numpy')
        tf = sp.lambdify([sp.Symbol(v) for v in allv], te, 'numpy')
        ycs, yts = [], []
        for lo, hi in [(0.5, 2.5), (0.2, 1.2), (1.0, 4.0)]:
            pts = [rng.uniform(lo, hi, n) for _ in allv]
            ycs.append(np.broadcast_to(np.asarray(cf(*pts), float), (n,)).astype(float))
            yts.append(np.broadcast_to(np.asarray(tf(*pts), float), (n,)).astype(float))
        yc = np.concatenate(ycs); yt = np.concatenate(yts)
        ok = np.isfinite(yc) & np.isfinite(yt) & (np.abs(yt) > 1e-9)
        if ok.sum() < max(30, 0.3 * len(yt)): return False
        yc, yt = yc[ok], yt[ok]
        if np.std(yt) < 1e-9: return False              # truth ~const: ambiguous
        if np.std(yc) < 1e-9: return False              # candidate ~const/degenerate
        if np.std(yc - yt) / (np.std(yt) + 1e-9) < 1e-3:
            return True
        r = yc / yt
        # require a real, non-degenerate proportionality constant
        if abs(np.mean(r)) < 1e-6: return False
        return bool(np.std(r) / abs(np.mean(r)) < 1e-3)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# task loading
# ---------------------------------------------------------------------------

def task_from_pair_row(r):
    """Training jsonl row -> task dict (others sorted, target last)."""
    truth = r['expr']; target = r['target']
    others = list(r['var_names'])          # already sorted-minus-target from gen
    X = np.array(r['X'], float); y = np.array(r['y'], float)
    if X.ndim != 2 or X.shape[1] != len(others): return None
    symvals = {others[i]: X[:, i] for i in range(len(others))}
    symvals[target] = y
    return {'truth': truth, 'target': target, 'others': others,
            'X': X, 'y': y, 'symvals': symvals}


_LEAK_CANON = None
def _is_leaked(truth):
    """True if `truth` is structurally present in the training library."""
    global _LEAK_CANON
    if _LEAK_CANON is None:
        import sympy as sp
        # union of all per-cache leak lists (v6 = built vs cache_v6, 133; v4 = 121).
        # union = conservative: excludes anything leaked into ANY training set, so the
        # held-out set is contamination-free for whichever model we evaluate.
        paths = ['dataset_20260531/real591_leaked_v6.json',
                 'dataset_20260531/real591_leaked_truths.json',
                 'dataset_20260531/v5_physics65_leaked_v6.json']
        _LEAK_CANON = set()
        for p in paths:
            if not os.path.exists(p):
                continue
            for t in json.load(open(p, encoding='utf-8')):
                e = parse_to_sympy(t)
                if e is not None:
                    try: _LEAK_CANON.add(sp.srepr(e))
                    except Exception: pass
    e = parse_to_sympy(truth)
    if e is None: return False
    import sympy as sp
    try: return sp.srepr(e) in _LEAK_CANON
    except Exception: return False


def tasks_from_real591(path, n_max, exclude_leaks=True, well_posed=True, offset=0):
    """offset: skip the first `offset` valid tasks (for a stable dev/test split —
    dev=offset 0, fresh-OOD test=offset 60). well_posed: drop tasks whose truth needs
    a variable with no data (ill-posed missing-variable cases like Friedmann/Lambda)."""
    rows = []; n_leak = 0; n_ill = 0; built = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            tv = r.get('train_values', {}); syms = r.get('symbols', [])
            truth = r.get('eval_truth_surface', '')
            if not tv or not syms or not truth: continue
            te = parse_to_sympy(truth)
            if te is None: continue
            if exclude_leaks and _is_leaked(truth):
                n_leak += 1; continue
            free = [str(s) for s in te.free_symbols]
            usable = [s for s in syms if s in tv and s in free]
            if len(usable) < 2: continue
            if well_posed and (set(free) - set(usable)):   # truth needs a var with no data -> ill-posed
                n_ill += 1; continue
            target = usable[0]
            others = sorted([s for s in usable if s != target])
            if len(others) + 1 > MAX_VARS: continue
            try:
                X = np.array([tv[v] for v in others], float).T
                y = np.array(tv[target], float)
            except Exception:
                continue
            if X.shape[0] < 10: continue
            symvals = {others[i]: X[:, i] for i in range(len(others))}
            symvals[target] = y
            built += 1
            if built <= offset: continue                   # skip first `offset` valid tasks (dev/test split)
            rows.append({'truth': truth, 'target': target, 'others': others,
                         'X': X, 'y': y, 'symvals': symvals, 'law_id': r.get('law_id', '')})
            if len(rows) >= n_max: break
    if exclude_leaks:
        print(f"(excluded {n_leak} leaked, {n_ill} ill-posed missing-var truths)")
    return rows


# ---------------------------------------------------------------------------
# main eval
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_v4_full_flow.pt')
    ap.add_argument('--index', default='sr_model/ckpt/zf_v4_full')
    ap.add_argument('--mode', choices=['train', 'heldout'], default='heldout')
    ap.add_argument('--pairs', default='dataset_20260531/data_pairs_v4_full.jsonl.gz')
    ap.add_argument('--real', default='data/real591.jsonl')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--K', type=int, default=60)        # flow samples
    ap.add_argument('--D', type=int, default=4)         # decodes per flow z_f
    ap.add_argument('--topr', type=int, default=30)     # retrieval depth
    ap.add_argument('--holdout', type=float, default=0.0)  # >0 = fraction of points held out for overfit-robust candidate reranking
    ap.add_argument('--xregion', action='store_true')      # held-out split BY |y| MAGNITUDE (fit on low-|y| region, score on high-|y| region): a wrong formula that fits one regime can't extrapolate to a DIFFERENT regime, but the true law does -> cross-region transfer discriminator
    ap.add_argument('--units', default='')                  # json {law_id: {symbol: [M,L,T,Th,I,N]}} -> activates dimensional-consistency candidate filtering
    ap.add_argument('--max-points', type=int, default=128)  # observation points fed to data encoder (real591 has 160)
    ap.add_argument('--temp', type=float, default=0.9)
    ap.add_argument('--skip', type=int, default=0)      # skip first N rows (train mode)
    ap.add_argument('--dump', default='')                # save per-task top-k candidates (JSON)
    ap.add_argument('--flow2', default='')               # Flow-2 ckpt (retrieval-conditioned)
    ap.add_argument('--lowd-flow', default='')           # low-dim PCA-coord flow ckpt (flow_v6_lowd*.pt)
    ap.add_argument('--no-fine', action='store_true')    # dual model: SKIP fine flow, decode from coarse z_f only (diagnostic)
    args = ap.parse_args()
    UNITS = json.load(open(args.units, encoding='utf-8')) if args.units else {}

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64),
                     n_ftokens=ck.get('n_ftokens', 0), log_feats=ck.get('log_feats', False),
                     class_feats=ck.get('class_feats', False), dim_head=ck.get('dim_head', False),
                     robust_norm=ck.get('robust_norm', False)).to(DEVICE)
    # auto-detect flow n_blocks from ckpt (train_flow.py may have used a different size)
    _fb = max([int(k.split('flow.v.blocks.')[1].split('.')[0])
               for k in ck['model'] if 'flow.v.blocks.' in k], default=-1) + 1
    if _fb > 0 and _fb != 6:
        from models.flow_matching import FlowMatching, FlowMatchingTok
        model.flow = (FlowMatchingTok(d=d, n_blocks=_fb) if model.tok
                      else FlowMatching(d=d, n_blocks=_fb)).to(DEVICE)
        print(f"flow rebuilt with n_blocks={_fb} (from ckpt)")
    load_state_compat(model, ck['model']); model.eval()
    gen_mlen = min(getattr(model, 'dec_max_len', 48), 96)   # gen window (cap for eval speed; covers ~97%)
    zf_index = torch.from_numpy(np.load(args.index + '_index.npy')).to(DEVICE)
    meta = [json.loads(l) for l in open(args.index + '_meta.jsonl', encoding='utf-8')]
    idx_expr = [m['expr'] for m in meta]
    print(f"manifold d={d} | index {len(meta)} formulas | mode={args.mode}")

    flow2 = None; zf_raw = None
    if args.flow2:
        from models.flow_matching import FlowMatching2
        f2ck = torch.load(args.flow2, map_location=DEVICE)
        flow2 = FlowMatching2(d=d).to(DEVICE); flow2.load_state_dict(f2ck['flow2']); flow2.eval()
        zf_raw = torch.from_numpy(np.load(args.index + '_raw.npy')).to(DEVICE)  # raw z_f for hint
        print(f"Flow-2 loaded from {args.flow2} (retrieval-conditioned)")

    # low-dim flow: generates k-dim PCA coords; lift back to z_f for the decoder.
    lowd_flow = None; pca_mu = None; pca_Vk = None
    if args.lowd_flow:
        from models.flow_matching import FlowMatchingTokLowD
        lck = torch.load(args.lowd_flow, map_location=DEVICE)
        dz = lck['d_z']
        lowd_flow = FlowMatchingTokLowD(d=d, d_z=dz, n_blocks=6).to(DEVICE)
        lowd_flow.load_state_dict(lck['flow']); lowd_flow.eval()
        pca = np.load(lck['pca'])
        pca_mu = torch.tensor(pca['mu'], dtype=torch.float32, device=DEVICE)
        pca_Vk = torch.tensor(pca['Vt'][:dz], dtype=torch.float32, device=DEVICE)
        model.flow = lowd_flow      # routed through the tok-mode sampling path below
        print(f"low-dim flow loaded from {args.lowd_flow} (d_z={dz}); lifting coords -> z_f via PCA")

    if args.mode == 'train':
        rows = []
        with gzip.open(args.pairs, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < args.skip: continue
                if not line.strip(): continue
                t = task_from_pair_row(json.loads(line))
                if t: rows.append(t)
                if len(rows) >= args.n: break
    else:
        rows = tasks_from_real591(args.real, args.n)
    print(f"tasks: {len(rows)}\n")

    def zd_of(t):
        others, X, y = t['others'], t['X'], t['y']
        mp = args.max_points
        npt = min(X.shape[0], mp); nv = len(others)
        points = torch.zeros(1, mp, MAX_VARS + 1)
        points[0, :npt, :nv] = torch.from_numpy(X[:npt].astype(np.float32))
        points[0, :npt, MAX_VARS] = torch.from_numpy(y[:npt].astype(np.float32))
        var_mask = torch.zeros(1, MAX_VARS); var_mask[0, :nv] = 1
        point_mask = torch.zeros(1, mp); point_mask[0, :npt] = 1
        dims = None
        if getattr(model.data_enc, 'dim_len', 0):
            dd = dims_for_task(others, t['target'], model.data_enc.dim_len)
            dims = torch.from_numpy(dd).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if model.tok:
                zd, tok = model.data_enc(points.to(DEVICE), var_mask.to(DEVICE),
                                         point_mask.to(DEVICE), dims=dims, return_tokens=True)
                return zd[0], tok[0]            # pooled z_d (retrieval) + token memory
            zd = model.data_enc(points.to(DEVICE), var_mask.to(DEVICE), point_mask.to(DEVICE), dims=dims)
        return zd[0], None   # RAW: decoder/flow trained on un-normalized z_d; normalize only for retrieval cosine

    TOPK = [1, 3, 5, 10, 20]
    hits_retr = {k: 0 for k in TOPK}
    hits_flow = {k: 0 for k in TOPK}
    hits_comb = {k: 0 for k in TOPK}
    retr_ceiling = 0; flow_ceiling = 0; n = 0; t0 = time.time(); dump = []

    for t in rows:
        truth = t['truth']; target = t['target']; others = t['others']
        vnf = others + [target]                      # target-last names
        symvals = t['symvals']
        # held-out split: fit constants on FIT points, rank by predict_nmse on VAL
        # points so candidates that OVERFIT the fit points (low train residual but
        # wrong) are demoted -> the truth (generalizes) rises to top-1. Closes the
        # comb@1 -> flow-ceiling gap (~7-10pp) without needing units.
        if args.holdout > 0 or args.xregion:
            _nv = len(next(iter(symvals.values())))
            _hold = args.holdout if args.holdout > 0 else 0.4
            _cut = max(4, int(_nv * (1 - _hold)))
            if args.xregion:
                # split by |target| magnitude: fit on the LOW-|y| region, validate on
                # the HIGH-|y| region. The true law extrapolates across the regime gap;
                # a candidate that merely overfits the low region diverges on the high.
                _order = np.argsort(np.abs(np.asarray(symvals[target], float)))
                _fi, _vi = _order[:_cut], _order[_cut:]
            else:
                _perm = np.random.default_rng(0).permutation(_nv)
                _fi, _vi = _perm[:_cut], _perm[_cut:]
            symvals_fit = {k: np.asarray(v, float)[_fi] for k, v in symvals.items()}
            symvals_val = {k: np.asarray(v, float)[_vi] for k, v in symvals.items()}
        else:
            symvals_fit = symvals_val = symvals
        zd, ztok = zd_of(t)
        # variable dimensions for this task (for dimensional-consistency pruning).
        # Only trust the check when EVERY variable has a KNOWN dimension (unknown-class
        # vars would be wrongly treated as dimensionless and misfire the penalty).
        _load_sym2class()
        umap = UNITS.get(t.get('law_id', ''), {})
        if umap and all(v in umap for v in vnf):
            # curated SI dims for every var -> activate dimensional-consistency filter
            var_dims = {v: tuple(umap[v][:6]) for v in vnf}
            dims_all_known = True
        else:
            var_dims = {v: tuple(class_to_dim(_SYM2CLASS.get(v, 'other'))[:6]) for v in vnf}
            dims_all_known = all(class_to_dim(_SYM2CLASS.get(v, 'other'))[6] == 0 for v in vnf)

        # A) retrieval candidates (index formulas, as-is) — normalize z_d for cosine.
        # Skip cleanly if the index z_f dim doesn't match the model's z_d (e.g. a
        # 384-dim legacy index vs a 768-dim model): the spaces are incomparable, so
        # retrieval is meaningless here and we report a 0% retrieval ceiling.
        if zd.size(-1) == zf_index.size(-1):
            sims = (torch.nn.functional.normalize(zd, dim=-1) @ zf_index.t())
            top = sims.topk(min(args.topr, zf_index.size(0)))
            retr_exprs = [idx_expr[i] for i in top.indices.cpu().numpy()]
            if any(num_equiv(e, truth, symvals, target) for e in retr_exprs):
                retr_ceiling += 1
        else:
            retr_exprs = []

        # B) flow -> decode candidates
        flow_cands = []
        try:
          with torch.no_grad():
            if model.tok:
                # token architecture: flow + decoder cross-attend to the token memory;
                # decoder ALSO gets pooled z_d (global summary), matching training.
                tok_b = ztok.unsqueeze(0)                                  # (1,T,d)
                zf_flow = model.flow.sample(tok_b, k=args.K, n_steps=20)[0]  # (K,d) or (K,d_z)
                if pca_Vk is not None:                                       # low-dim flow: lift coords -> z_f
                    zf_flow = pca_mu + zf_flow @ pca_Vk
                if torch.isnan(zf_flow).any():
                    raise ValueError('flow.sample produced NaN z_f')
                tok_rep = tok_b.expand(args.K, -1, -1)
                zd_rep = zd.unsqueeze(0).expand(args.K, -1)
                # coarse-to-fine: generate fine z_f_tokens per coarse z_f sample
                zftok_flow = None
                if getattr(model, 'n_ftokens', 0) > 0 and model.flow_fine is not None and not args.no_fine:
                    zftok_flow = model.flow_fine.sample(tok_rep, zf_flow, M=model.n_ftokens,
                                                        k=1, n_steps=20)[:, 0]   # (K,M,d)
                for _ in range(args.D):
                    for (seq, consts) in model.decoder.generate(zd_rep, zf_flow, data_tokens=tok_rep,
                                                                z_f_tokens=zftok_flow,
                                                                max_len=gen_mlen, greedy=False,
                                                                temperature=args.temp):
                        flow_cands.append((seq, consts))
            else:
                zf_flow = model.flow.sample(zd.unsqueeze(0), k=args.K, n_steps=20)[0]  # (K,d)
                zd_rep = zd.unsqueeze(0).expand(args.K, -1)
                for _ in range(args.D):
                    gens = model.decoder.generate(zd_rep, zf_flow, max_len=gen_mlen,
                                                  greedy=False, temperature=args.temp)
                    for (seq, consts) in gens:
                        flow_cands.append((seq, consts))
            # B2) Flow-2: retrieval-conditioned. Hint = RAW z_f of nearest training
            # formula; flow2 refines (z_d, hint) -> z_f. Independent of flow-1.
            if flow2 is not None:
                hint = zf_raw[top.indices[0]].unsqueeze(0)               # (1,d) nearest
                zf2 = flow2.sample(zd.unsqueeze(0), hint, k=args.K, n_steps=20)[0]
                for _ in range(args.D):
                    for (seq, consts) in model.decoder.generate(zd_rep, zf2, max_len=gen_mlen,
                                                                greedy=False, temperature=args.temp):
                        flow_cands.append((seq, consts))
        except (ValueError, RuntimeError) as ex:
            # a numerically degenerate task (e.g. NaN logits from flow/decoder) is
            # skipped as a miss rather than aborting the whole eval.
            print(f'  [skip task: {type(ex).__name__}: {ex}]', flush=True)
            flow_cands = []

        # build + fit each unique flow skeleton
        def eval_cands(cand_list, is_retr):
            scored = []
            seen = set()
            for item in cand_list:
                if is_retr:
                    e = parse_to_sympy(item)
                    if e is None: continue
                    params = []
                    key = sp.srepr(e)
                else:
                    seq, consts = item
                    e, params = nodes_to_skeleton(seq, vnf)
                    if e is None: continue
                    try: key = sp.srepr(e)
                    except Exception: key = str(e)
                if key in seen: continue
                seen.add(key)
                # seed the fit with the decoder's predicted constants (better exponent
                # recovery); retrieval candidates have no params -> None
                ci = None
                if not is_retr:
                    ci = [consts[i] for i in range(len(seq)) if seq[i] == NT2ID['CONST']]
                    if len(ci) != len(params): ci = None
                fitted, nm = fit_residual(e, params, symvals_fit, target, c_init=ci)
                try: cplx = sp.count_ops(fitted)
                except Exception: cplx = 30
                # rank by PREDICTION nmse on HELD-OUT points (overfit-robust): a
                # candidate that nails the fit points but is wrong diverges here.
                pn = predict_nmse(fitted, symvals_val, target)
                score = np.exp(-min(pn, 20)) - 0.01 * np.log1p(cplx)
                # dimensional-consistency prior (only when all var dims are known,
                # else the check is unreliable and could penalize a correct formula)
                if dims_all_known:
                    try:
                        if not formula_dim(fitted, var_dims).consistent:
                            score -= 0.5
                    except Exception:
                        pass
                scored.append((score, fitted, pn))
            scored.sort(key=lambda x: -x[0])
            return scored

        retr_scored = eval_cands(retr_exprs, True)
        flow_scored = eval_cands(flow_cands, False)
        # flow ceiling: any flow candidate CORRECT (predict_nmse-based, catches
        # algebraic rearrangements num_equiv misses)
        if any(is_correct(e, truth, symvals, target, pn) for _, e, pn in flow_scored):
            flow_ceiling += 1
        comb_scored = sorted(retr_scored + flow_scored, key=lambda x: -x[0])

        def tally(scored, store):
            for rk, (_, e, pn) in enumerate(scored, 1):
                if is_correct(e, truth, symvals, target, pn):
                    for k in TOPK:
                        if rk <= k: store[k] += 1
                    return
        tally(retr_scored, hits_retr)
        tally(flow_scored, hits_flow)
        tally(comb_scored, hits_comb)
        if args.dump:
            cds = []
            for rk, (sc, e, pn) in enumerate(flow_scored[:25], 1):
                cds.append({'rank': rk, 'formula': str(e), 'pnmse': float(pn),
                            'correct': bool(is_correct(e, truth, symvals, target, pn))})
            # retrieval neighbours: what formula FAMILY the manifold thinks this data
            # looks like (z_d nearest z_f index entries + cosine sim) — a strong prior
            # for the reference decoder even when the exact formula isn't in the index.
            retr = []
            if zd.size(-1) == zf_index.size(-1):
                for rk2, (ii, sv2) in enumerate(zip(top.indices.cpu().numpy()[:10],
                                                    top.values.cpu().numpy()[:10]), 1):
                    retr.append({'rank': rk2, 'formula': idx_expr[ii], 'cos': float(sv2)})
            dump.append({'i': n, 'truth': truth, 'target': target, 'others': others,
                         'candidates': cds, 'retrieval': retr})
        n += 1
        if n % 25 == 0:
            print(f"  {n}/{len(rows)} | retrC={retr_ceiling} flowC={flow_ceiling} "
                  f"comb@1={hits_comb[1]} @10={hits_comb[10]} ({time.time()-t0:.0f}s)",
                  flush=True)

    print(f"\n=== SR eval ({args.mode}, n={n}, K={args.K} D={args.D} topr={args.topr}) ===")
    print(f"Retrieval ceiling (truth in top-{args.topr}): {retr_ceiling}/{n} = {100*retr_ceiling/n:.1f}%")
    print(f"Flow ceiling (truth among flow cands):       {flow_ceiling}/{n} = {100*flow_ceiling/n:.1f}%")
    print(f"\n{'k':>4} {'retrieval':>12} {'flow+dec':>12} {'combined':>12}")
    for k in TOPK:
        print(f"{k:>4} {100*hits_retr[k]/n:>11.1f}% {100*hits_flow[k]/n:>11.1f}% {100*hits_comb[k]/n:>11.1f}%")
    if args.dump:
        json.dump(dump, open(args.dump, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f"dumped {len(dump)} tasks (top-8 cands each) -> {args.dump}")


if __name__ == '__main__':
    main()
