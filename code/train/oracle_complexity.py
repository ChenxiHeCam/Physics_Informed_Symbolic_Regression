"""Why does the oracle fail on ~14% even given the TRUE z_f?

Hypothesis: single-vector z_f is a lossy compression -> long/complex formulas exceed
its capacity and decode to a wrong structure. Test: encode each held-out truth ->
z_f -> decode -> structural success; bucket success by formula complexity (token
length, #ops). If failure concentrates in long/complex formulas, z_f CAPACITY is the
cause and z_f-tokenization (a sequence of z_f tokens) is the fix.

CPU (oracle probes must be CPU; GPU contention gives garbage and the GPU is busy).
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch
from data.ast_grammar import parse_to_sympy, expr_to_nodes, MAX_VARS
from train.train_manifold import Manifold, build_ast_edges
from train.eval_sr import _is_leaked, dims_for_task

DEVICE = torch.device('cpu')
CKPT = 'sr_model/ckpt/flow_v6_noaug.pt'   # encoder+decoder identical across v6-fixed variants
REAL = 'data/real591.jsonl'
N = 120


def main():
    ck = torch.load(CKPT, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64)).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    gen_cap = min(getattr(model, 'dec_max_len', 48), 96)
    dec_max = getattr(model, 'dec_max_len', 128)

    def zf_of_seq(seq):
        ee = build_ast_edges(seq)
        ei = torch.tensor(ee).t().contiguous() if ee else torch.zeros(2, 0, dtype=torch.long)
        with torch.no_grad():
            return model.formula_enc(torch.tensor(seq), torch.zeros(len(seq)), ei,
                                     torch.zeros(len(seq), dtype=torch.long), 1)

    rows = []
    with open(REAL, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            tv = r.get('train_values', {}); syms = r.get('symbols', []); truth = r.get('eval_truth_surface', '')
            if not tv or not truth: continue
            te = parse_to_sympy(truth)
            if te is None or _is_leaked(truth): continue
            free = [str(s) for s in te.free_symbols]; usable = [s for s in syms if s in tv and s in free]
            if len(usable) < 2 or not set(free).issubset(set(usable)): continue
            target = usable[0]; others = sorted([s for s in usable if s != target])
            if len(others) + 1 > MAX_VARS: continue
            try:
                X = np.array([tv[v] for v in others], float).T; y = np.array(tv[target], float)
            except Exception:
                continue
            if X.shape[0] < 10: continue
            rows.append({'truth': truth, 'target': target, 'others': others, 'X': X, 'y': y})
            if len(rows) >= N: break
    print(f"held-out tasks: {len(rows)}\n")

    recs = []
    for t in rows:
        vnf = t['others'] + [t['target']]
        res = expr_to_nodes(t['truth'], var_order=vnf)
        if res is None: continue
        seq_true = res[0]; L = len(seq_true)
        nops = sum(1 for tok in seq_true)            # token count proxy
        nvar = len(vnf)
        too_long = L > gen_cap
        zf = zf_of_seq(seq_true)
        X, y = t['X'], t['y']; mp = 128; npt = min(X.shape[0], mp); nv = len(t['others'])
        pts = torch.zeros(1, mp, MAX_VARS + 1)
        pts[0, :npt, :nv] = torch.from_numpy(X[:npt].astype(np.float32))
        pts[0, :npt, MAX_VARS] = torch.from_numpy(y[:npt].astype(np.float32))
        vm = torch.zeros(1, MAX_VARS); vm[0, :nv] = 1; pm = torch.zeros(1, mp); pm[0, :npt] = 1
        dims = None
        if getattr(model.data_enc, 'dim_len', 0):
            dims = torch.from_numpy(dims_for_task(t['others'], t['target'], model.data_enc.dim_len)).unsqueeze(0)
        with torch.no_grad():
            if model.tok:
                zd, tok = model.data_enc(pts, vm, pm, dims=dims, return_tokens=True)
                gen = model.decoder.generate(zd, zf, data_tokens=tok, max_len=gen_cap, greedy=True)
            else:
                zd = model.data_enc(pts, vm, pm, dims=dims)
                gen = model.decoder.generate(zd, zf, max_len=gen_cap, greedy=True)
        seq_o = gen[0][0]
        # generate() stops at tree-completion and omits the trailing EOS that
        # expr_to_nodes appends, so a correct decode has len L-1 and matches the prefix.
        ok = (len(seq_o) >= L - 1) and (seq_o == seq_true[:len(seq_o)])
        recs.append({'L': L, 'nvar': nvar, 'ok': ok, 'too_long': too_long})

    recs = [r for r in recs]
    n = len(recs)
    okrate = np.mean([r['ok'] for r in recs])
    print(f"overall structural oracle: {100*okrate:.1f}%  (n={n})")
    print(f"truths exceeding gen window ({gen_cap} tok): {sum(r['too_long'] for r in recs)} "
          f"({100*np.mean([r['too_long'] for r in recs]):.1f}%) -> auto-fail\n")
    print("success by formula LENGTH (tokens):")
    for lo, hi in [(0, 16), (16, 24), (24, 32), (32, 48), (48, 64), (64, 96), (96, 999)]:
        sub = [r for r in recs if lo <= r['L'] < hi]
        if sub:
            print(f"  len [{lo:3d},{hi:3d}): {100*np.mean([r['ok'] for r in sub]):5.1f}% ok  (n={len(sub)})")
    print("\nsuccess by #variables:")
    for v in range(2, 9):
        sub = [r for r in recs if r['nvar'] == v]
        if sub:
            print(f"  {v} vars: {100*np.mean([r['ok'] for r in sub]):5.1f}% ok  (n={len(sub)})")
    # correlation: failures' mean length vs successes'
    fail_L = [r['L'] for r in recs if not r['ok']]; ok_L = [r['L'] for r in recs if r['ok']]
    print(f"\nmean token-length:  FAIL={np.mean(fail_L):.1f}  OK={np.mean(ok_L):.1f}"
          f"  (if FAIL>>OK -> capacity limit confirmed)")


if __name__ == '__main__':
    main()
