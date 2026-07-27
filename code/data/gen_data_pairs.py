"""
Generate (X, y) pairs — v4: semantics-aware, valid-domain, precondition-safe.

Improvements over v3:
  - per-target sampling (f=ma -> 3 tasks), 500 points each
  - semantic-class-aware sampling RANGES (velocity/mass/length/... each get a
    physically sensible relative magnitude band; constants like c, hbar, k_B use
    fixed normalized values) so points are physically coherent, not random soup
  - VALID-DOMAIN enforcement: reject points where sqrt(<0), log(<=0), div-by-0;
    verify the brentq root actually zeroes the residual (|r|<1e-4)
  - this implicitly satisfies preconditions like v<c (sqrt(1-v^2/c^2)>=0 forces it)
  - 10-20% of points get 2% relative noise

We do NOT use raw SI magnitudes (c=3e8, hbar=1e-34) — they overflow and make
sampling impossible when constants and variables mix. Instead we use NORMALIZED
natural-unit magnitudes per semantic class, which keeps physical relations and
domain constraints intact while staying numerically stable.
"""
import sys, os, gzip, json, time, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sympy as sp
from scipy.optimize import brentq
from ast_grammar import expr_to_nodes, parse_to_sympy

NODES_DEFAULT = 'dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz'
TAGS = 'data/symbol_semantic_tags_20260522.jsonl'
N_POINTS = 500
NOISE_FRAC = (0.10, 0.20)
NOISE_REL = 0.02

# ---- load semantic tags -------------------------------------------------
_SYM2CLASS = {}
_SYM_UNIT = {}
def load_tags():
    if _SYM2CLASS: return
    try:
        with open(TAGS, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                _SYM2CLASS[r['symbol']] = r.get('semantic_class', 'other')
                _SYM_UNIT[r['symbol']] = r.get('unit')
    except Exception:
        pass

# normalized magnitude bands per semantic class (lo, hi) — natural units
_CLASS_RANGE = {
    'velocity': (0.05, 0.9),         # < 1 (c normalized to 1) -> v<c auto
    'mass': (0.1, 10.0), 'length': (0.1, 10.0), 'time': (0.1, 10.0),
    'energy_generic': (0.1, 10.0), 'energy_kinetic': (0.1, 10.0),
    'energy_potential': (0.1, 10.0), 'energy_thermal': (0.1, 10.0),
    'temperature': (0.5, 5.0), 'force': (0.1, 10.0), 'pressure': (0.1, 10.0),
    'charge': (0.1, 5.0), 'current': (0.1, 5.0), 'frequency': (0.1, 5.0),
    'angular_frequency': (0.1, 5.0), 'angle': (0.05, 3.0),
    'number_density': (0.1, 10.0), 'concentration': (0.1, 5.0),
    'dimensionless_ratio': (0.05, 0.95), 'refractive_index': (1.0, 3.0),
    'wavelength': (0.1, 10.0), 'area': (0.1, 10.0), 'volume': (0.1, 10.0),
    'momentum': (0.1, 10.0), 'power': (0.1, 10.0), 'resistance': (0.1, 10.0),
    'magnetic_field': (0.1, 5.0), 'electric_field': (0.1, 5.0),
    'spin': (0.5, 4.0), 'atomic_number': (1.0, 30.0), 'mass_number': (1.0, 60.0),
    'count_integer': (1.0, 20.0),
}
# fixed normalized values for fundamental constants
_CONST_VAL = {
    'c': 1.0, 'c_EM': 1.0, 'c_light': 1.0,
    'hbar': 1.0, 'h': 1.0, 'h_planck': 1.0,
    'k_B': 1.0, 'k': 1.0, 'k_boltz': 1.0,
    'G': 1.0, 'e': 1.0, 'e_charge': 1.0, 'epsilon_0': 1.0, 'mu_0': 1.0,
    'N_A': 1.0, 'R_gas': 1.0, 'pi': float(np.pi),
}
_DEFAULT_RANGE = (0.3, 3.0)

# curated per-variable-NAME sampling specs (physics-informed, numerically
# safe / normalized). Covers the auto-combo variable names whose semantic class is
# 'other'/unknown, where the class-based ranges fall back to a blind default. Set
# env SR_VAR_RANGES=/path/to/_train_var_ranges.json to activate. Each entry:
#   {"role":"const","value":v} | {"role":"var","lo":..,"hi":..,"scale":"log"|"linear"}
_VAR_OVERRIDE = {}
def _load_var_override():
    global _VAR_OVERRIDE
    import os, json as _json
    p = os.environ.get('SR_VAR_RANGES', '')
    if p and os.path.exists(p):
        try:
            _VAR_OVERRIDE = _json.load(open(p, encoding='utf-8'))
        except Exception:
            _VAR_OVERRIDE = {}
_load_var_override()


def class_range(sym):
    ov = _VAR_OVERRIDE.get(sym)
    if ov is not None and ov.get('role') == 'const':
        return ('const', float(ov.get('value', 1.0)))
    if sym in _CONST_VAL:
        return ('const', _CONST_VAL[sym])
    cls = _SYM2CLASS.get(sym, 'other')
    if _SYM2CLASS.get(sym) == 'fundamental_constant':
        return ('const', 1.0)
    return ('range', _CLASS_RANGE.get(cls, _DEFAULT_RANGE))


from data.dimensions import class_to_dim

# classes that GATE a physical regime via a dimensionless ratio (hν/kT, Θ/T,
# (E-μ)/kT, ...): sample them WIDE in log-space so the ratio sweeps the transition
# (low-T<->high-T, quantum<->classical) instead of sitting in one limit.
_REGIME_LOG = {'temperature', 'frequency', 'angular_frequency', 'energy_generic',
               'energy_kinetic', 'energy_potential', 'energy_thermal', 'gibbs_energy',
               'chemical_potential', 'wavelength', 'wavenumber', 'time', 'length',
               'mass', 'momentum', 'force', 'pressure', 'power', 'number_density',
               'concentration', 'charge', 'current', 'magnetic_field', 'electric_field',
               'frequency', 'decay_rate', 'reaction_rate', 'volume', 'area'}


# Per-call RANGE JITTER: real-world observations span only ~6x (0.8 decade) while
# the default wide sampling spans ~4 decades -> train/test mismatch (model can't
# discriminate exponents/sqrt on narrow windows). With SR_RANGE_JITTER=1, each
# sample_var call draws a RANDOM sub-window (width from ~6x up to the full band)
# at a random center, so the model sees the whole spectrum of observation widths.
_JITTER = os.environ.get('SR_RANGE_JITTER', '') not in ('', '0', 'false')
_MIN_HALF = np.log(5.0) / 2.0          # narrowest window ~5x (covers real rfs ~6x)
_MAX_HALF = np.log(100.0) / 2.0        # widest jitter window ~100x (user spec): wide
                                       # enough to disambiguate, stays physical
                                       # (1000x peaks but pushes vars to extremes)


def _shaped(lo, hi, n, rng):
    """n samples in [lo,hi] with a RANDOMLY chosen distribution shape so the model
    sees normal-core / uniform / long-tail / edge-concentrated data (not always
    uniform) -> robust to any input pattern (user regen spec 2026-06-18)."""
    if not _JITTER or hi <= lo:
        return rng.uniform(lo, hi, n)
    c, w = (lo + hi) / 2.0, (hi - lo)
    u = rng.random()
    if u < 0.45:                                   # normal core (~6x effective: std=w/6)
        x = rng.normal(c, w / 6.0, n)
    elif u < 0.70:                                 # uniform
        x = rng.uniform(lo, hi, n)
    elif u < 0.90:                                 # long-tail (most central, few far)
        x = c + (rng.standard_t(3, n)) * (w / 8.0)
    else:                                          # edge-concentrated (U-shape)
        x = lo + w * rng.beta(0.4, 0.4, n)
    return np.clip(x, lo, hi)


def _logsample(lo_log, hi_log, n, rng):
    """Sample n points over [lo_log, hi_log] (log-space), or (if jitter on) over a
    random sub-window of random width, with a randomly chosen distribution shape."""
    if _JITTER and hi_log - lo_log > 2 * _MIN_HALF:
        hi_half = min((hi_log - lo_log) / 2.0, _MAX_HALF)
        half = rng.uniform(_MIN_HALF, hi_half)
        center = rng.uniform(lo_log + half, hi_log - half)
        lo_log, hi_log = center - half, center + half
    return np.exp(_shaped(lo_log, hi_log, n, rng))


def sample_var(sym, n, rng):
    kind, spec = class_range(sym)
    if kind == 'const':
        return np.full(n, spec)
    # reference per-name override (takes priority over class-based sampling)
    ov = _VAR_OVERRIDE.get(sym)
    if ov is not None and ov.get('role') == 'var':
        lo, hi = float(ov['lo']), float(ov['hi'])
        if ov.get('scale') == 'log' and lo > 0:
            return _logsample(np.log(lo), np.log(hi), n, rng)
        return rng.uniform(lo, hi, n)
    lo, hi = spec
    cls = _SYM2CLASS.get(sym, 'other')
    if cls == 'velocity':
        # v<c (c=1): log-sample toward c so gamma sweeps 1 -> ~20 (relativistic regime)
        return 1.0 - np.exp(rng.uniform(np.log(1e-3), np.log(0.95), n))
    if cls in _REGIME_LOG:
        # wide log span (~4 decades) so dimensionless ratios cross their transition
        return _logsample(np.log(1e-2), np.log(1e2), n, rng)
    # dimensionless (ratio/angle/count) or unknown: natural-range, mixed shape
    return _shaped(lo, hi, n, rng)


_TRIG_FUNCS = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)


def _trig_arg_symbols(e):
    """Symbols appearing INSIDE any trig function. These MUST be sampled over >=1
    period (>=2pi); over a narrow arc (~3 rad, the old 'angle' range) sin is a
    smooth curve indistinguishable from a polynomial -> trig is unidentifiable from
    the data and the model (correctly) outputs a power-law approximation. Verified
    2026-06-18: trig recovery is data-limited (angles spanned only ~2.7 rad), not an
    encoder/decoder fault (oracle 75% vs flow/dec7 29%)."""
    out = set()
    try:
        for node in sp.preorder_traversal(e):
            if isinstance(node, _TRIG_FUNCS):
                for s in node.args[0].free_symbols:
                    out.add(str(s))
    except Exception:
        pass
    return out


def make_samples(expr, rng, n_points=N_POINTS):
    load_tags()
    # POINT-COUNT JITTER: vary #observation-points per formula in [100,300] (user
    # spec) so the encoder sees the spectrum of dataset sizes (real tasks ~160).
    # NB: cache build must use --max-points 300 to actually store up to 300.
    if _JITTER:
        n_points = int(rng.integers(100, 301))
    e = parse_to_sympy(expr)
    if e is None or not hasattr(e, 'free_symbols'):
        return None
    free = sorted([str(s) for s in e.free_symbols])
    if not (2 <= len(free) <= 12):
        return None
    syms = [sp.Symbol(v) for v in free]
    trig_syms = _trig_arg_symbols(e)        # sample these over >=1 period (see helper)
    try:
        fn = sp.lambdify(syms, e, 'numpy')
    except Exception:
        return None

    out = []
    for target in free:
        others = [v for v in free if v != target]
        # constants can't be the target (they're fixed)
        if class_range(target)[0] == 'const':
            continue
        cols = {v: sample_var(v, n_points, rng) for v in others}
        # TRIG ANGLE WIDENING: override trig-argument symbols to span >=1 period so
        # the oscillation is visible (else sin~polynomial over a narrow arc). k
        # jittered 1..2.5 periods; products of two such symbols still resolve at 160 pts.
        for v in others:
            if v in trig_syms:
                k = rng.uniform(1.0, 2.5) if _JITTER else 2.0
                cols[v] = rng.uniform(0.0, 2.0 * np.pi * k, n_points)
        Xo = np.stack([cols[v] for v in others], axis=1)
        ys = np.full(n_points, np.nan)
        # FAST PATH: if e is LINEAR in target, solve algebraically once + vectorized
        # eval (skips 100-300 per-point brentq -> ~100x faster, the gen bottleneck).
        # e linear in t iff de/dt is t-independent; then target = -(e - de/dt*t)/(de/dt).
        fast_ok = False
        tsym = sp.Symbol(target)
        try:
            dedt = sp.diff(e, tsym)
            if dedt != 0 and tsym not in dedt.free_symbols:
                solexpr = sp.expand(e) - dedt * tsym       # the t-free remainder
                solfn = sp.lambdify([sp.Symbol(v) for v in others],
                                    -solexpr / dedt, 'numpy')
                yv = np.asarray(solfn(*[cols[v] for v in others]), float)
                yv = np.broadcast_to(yv, (n_points,)).astype(float).copy()
                if np.isfinite(yv).sum() >= n_points * 0.4:
                    ys = yv; fast_ok = True
        except Exception:
            fast_ok = False
        for j in range(n_points):
            if fast_ok: break                              # linear fast path filled ys
            vals = {v: cols[v][j] for v in others}
            def g(tv):
                arr = [tv if v == target else vals[v] for v in free]
                try:
                    r = fn(*arr)
                    return float(r) if np.isfinite(r) else np.nan
                except Exception:
                    return np.nan
            # target range as bracket guide
            kind, spec = class_range(target)
            blo, bhi = (spec if kind == 'range' else _DEFAULT_RANGE)
            try:
                # search within plausible target band first, then wider
                root = None
                for (a, b) in [(blo*0.1, bhi*5), (-bhi*5, bhi*5)]:
                    ga, gb = g(a), g(b)
                    if np.isfinite(ga) and np.isfinite(gb) and ga*gb < 0:
                        root = brentq(g, a, b, maxiter=40, xtol=1e-7)
                        break
                if root is None:
                    # STRONGER search: sign-scan a log grid (both signs) to bracket a
                    # root the two coarse brackets straddle without a sign change.
                    # recovers good formulas the weak sampler else drops as degenerate.
                    grid = np.concatenate([
                        np.geomspace(max(blo*0.01, 1e-6), bhi*50, 30),
                        -np.geomspace(max(blo*0.01, 1e-6), bhi*50, 18)])
                    gv = [(t, g(t)) for t in grid]
                    gv = [(t, val) for t, val in gv if np.isfinite(val)]
                    for (t1, v1), (t2, v2) in zip(gv, gv[1:]):
                        if v1 * v2 < 0:
                            root = brentq(g, min(t1, t2), max(t1, t2), maxiter=40, xtol=1e-7)
                            break
                if root is not None and abs(g(root)) < 1e-4:  # VERIFY root
                    ys[j] = root
            except Exception:
                pass
        ok = np.isfinite(ys)
        if ok.sum() < n_points * 0.4:
            continue
        Xo, ys = Xo[ok], ys[ok]
        # FINAL validity sweep: residual must truly be ~0 (vectorized)
        try:
            arr = []
            for v in free:
                if v == target:
                    arr.append(ys)
                else:
                    arr.append(Xo[:, others.index(v)])
            resid = np.abs(np.asarray(fn(*arr), dtype=float))
            valid = np.isfinite(resid) & (resid < 1e-4)
            if valid.sum() < len(ys) * 0.6:
                continue
            Xo, ys = Xo[valid], ys[valid]
        except Exception:
            continue
        # GEN-TIME DATA GATE (stronger sampler): reject degenerate (X,y) at the
        # source so target_zero / target_const / var_const never enter the pool.
        # These were ~49% of the old cache and taught the decoder noise mappings.
        if len(ys) < 40:
            continue
        if np.all(ys == 0) or np.std(ys) < 1e-9 * (abs(np.mean(ys)) + 1e-9):
            continue                                    # degenerate target -> drop this target
        if any(np.std(Xo[:, k]) < 1e-9 * (abs(np.mean(Xo[:, k])) + 1e-9)
               for k in range(Xo.shape[1])):
            continue                                    # a constant input column -> drop
        # NOISE (user spec 2026-06-18): 50% of instances CLEAN, 50% get a random
        # relative measurement noise 0-2% on ALL points (real data always has error).
        if len(ys) > 5 and rng.random() < 0.5:
            rel = rng.uniform(0.0, NOISE_REL)        # 0..2% this instance
            ys = ys * (1 + rng.normal(0, rel, len(ys)))
        res = expr_to_nodes(expr)
        if res is None: continue
        seq, consts, var_order = res
        out.append({
            'expr': expr, 'target': target, 'var_names': others,
            'X': np.round(Xo, 6).tolist(), 'y': np.round(ys, 6).tolist(),
            'seq': seq, 'consts': consts, 'formula_vars': var_order,
        })
    return out if out else None


def _worker(t):
    expr, uid, seed = t
    rng = np.random.default_rng(seed)
    try:
        s = make_samples(expr, rng)
    except Exception:
        return None
    if not s: return None
    for p in s: p['uid'] = uid
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', default=NODES_DEFAULT)
    ap.add_argument('--out', default='dataset_20260531/data_pairs_v4.jsonl.gz')
    ap.add_argument('--max-formulas', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--timeout', type=float, default=10.0)
    args = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout

    tasks = []
    with gzip.open(args.nodes, 'rt', encoding='utf-8') as f:
        for line in f:
            if len(tasks) >= args.max_formulas: break
            if not line.strip(): continue
            r = json.loads(line); ex = r.get('expr', '')
            if ex: tasks.append((ex, r.get('id', ''), args.seed + len(tasks)))
    print(f"Loaded {len(tasks)} | {N_POINTS}pts | semantic-aware + valid-domain")

    t0 = time.time(); nok = nsamp = nto = done = 0
    BATCH = 4000
    # Pool created ONCE and reused (was recreated every 200 formulas -> 80-proc
    # respawn dominated -> ~26/s; reuse -> ~hundreds/s). Refresh the pool only every
    # REFRESH formulas to clear any rare hung worker without per-batch respawn cost.
    REFRESH = 80000
    with gzip.open(args.out, 'wt', encoding='utf-8') as w:
        ex = ProcessPoolExecutor(max_workers=args.workers); since_refresh = 0
        try:
            for bs in range(0, len(tasks), BATCH):
                futs = [ex.submit(_worker, t) for t in tasks[bs:bs+BATCH]]
                for fut in futs:
                    try: res = fut.result(timeout=args.timeout)
                    except (FTimeout, Exception): res = None; nto += 1
                    done += 1
                    if res:
                        nok += 1
                        for p in res: w.write(json.dumps(p)+'\n'); nsamp += 1
                print(f"  {done}/{len(tasks)} | ok {nok} | {nsamp} samples | to {nto} | "
                      f"{done/(time.time()-t0):.0f}/s | {time.time()-t0:.0f}s", flush=True)
                since_refresh += BATCH
                if since_refresh >= REFRESH:                 # periodic pool refresh
                    ex.shutdown(wait=False, cancel_futures=True)
                    ex = ProcessPoolExecutor(max_workers=args.workers); since_refresh = 0
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    print(f"\nDone: {done} | {nok} usable | {nsamp} samples ({nsamp/max(nok,1):.1f}/f) | "
          f"{nto} to | {time.time()-t0:.0f}s\nSaved -> {args.out}")


if __name__ == '__main__':
    main()
