"""
HONEST held-out SR eval on real591 (NOT in our 1.7M training library).

Pipeline per held-out (X,y):
  z_d = data_encoder(X,y)
  candidates from THREE z_f sources, all fed to the conditioning decoder:
    A) retrieval: top-R real-formula z_f by cos(z_d,z_f)  -> hints (not answers)
    B) flow:      flow.sample(z_d, K)                      -> generated z_f
  each z_f -> decoder.generate(z_d, z_f) -> formula string
  also include retrieved formulas AS-IS (pure retrieval baseline)
  rerank all by [sim, fit(exp-NMSE), complexity] ; Q@k by numeric equivalence

Reports separately:
  - retrieval-only Q@k        (does memory help on unseen formulas?)
  - flow+decoder Q@k          (true generation)
  - combined Q@k
"""
import sys, os, json, argparse, random, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
import torch.nn as nn
import sympy as sp

from data.ast_grammar import parse_to_sympy, nodes_to_expr, MAX_VARS
from train.train_manifold import Manifold, PairDataset, build_ast_edges

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOPR = 30


def num_equiv(a, b, n=60, seed=0):
    rng = np.random.default_rng(seed)
    try:
        ce = parse_to_sympy(a) if isinstance(a, str) else a
        te = parse_to_sympy(b) if isinstance(b, str) else b
        if ce is None or te is None: return False
        allv = sorted(set([str(s) for s in ce.free_symbols] + [str(s) for s in te.free_symbols]))
        if not allv: return False
        cf = sp.lambdify([sp.Symbol(v) for v in allv], ce, 'numpy')
        tf = sp.lambdify([sp.Symbol(v) for v in allv], te, 'numpy')
        pts = [rng.uniform(0.5, 2.5, n) for _ in allv]
        yc = np.asarray(cf(*pts), float).flatten(); yt = np.asarray(tf(*pts), float).flatten()
        ok = np.isfinite(yc) & np.isfinite(yt) & (np.abs(yt) > 1e-9)
        if ok.sum() < 20: return False
        if np.std(yc[ok]-yt[ok])/(np.std(yt[ok])+1e-9) < 1e-3: return True
        r = yc[ok]/yt[ok]
        return bool(np.std(r)/(abs(np.mean(r))+1e-9) < 1e-3)
    except Exception:
        return False


def load_real591(path, n_max):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            tv = r.get('train_values', {})
            syms = r.get('symbols', [])
            truth = r.get('eval_truth_surface', '')
            if not tv or not syms or not truth: continue
            # build X,y: target = first symbol that parse_to_sympy of truth has
            te = parse_to_sympy(truth)
            if te is None: continue
            free = [str(s) for s in te.free_symbols]
            usable = [s for s in syms if s in tv and s in free]
            if len(usable) < 2: continue
            target = usable[0]
            others = [s for s in usable if s != target]
            try:
                X = np.array([tv[v] for v in others], dtype=float).T
                y = np.array(tv[target], dtype=float)
            except Exception:
                continue
            if X.shape[0] < 10 or len(others) > MAX_VARS: continue
            rows.append({'expr': truth, 'target': target, 'var_names': others,
                         'X': X.tolist(), 'y': y.tolist(),
                         'seq': [0], 'consts': [0.0]})  # seq unused for eval
            if len(rows) >= n_max: break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_flow.pt')
    ap.add_argument('--index', default='sr_model/ckpt/zf_full')
    ap.add_argument('--real', default='data/real591.jsonl')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--K', type=int, default=50)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d).to(DEVICE); model.load_state_dict(ck['model']); model.eval()
    zf_index = torch.from_numpy(np.load(args.index + '_index.npy')).to(DEVICE)
    meta = [json.loads(l) for l in open(args.index + '_meta.jsonl', encoding='utf-8')]
    idx_expr = [m['expr'] for m in meta]
    idx_meta = meta
    print(f"manifold d={d} | index {len(meta)} formulas")

    rows = load_real591(args.real, args.n)
    print(f"real591 held-out usable: {len(rows)} formulas\n")

    def zd_of(r):
        ds = PairDataset.__new__(PairDataset); ds.rows=[r]; ds.max_points=128; ds.max_seq=48
        b = ds.collate([0])
        b = {k:(v.to(DEVICE) if torch.is_tensor(v) else v) for k,v in b.items()}
        with torch.no_grad():
            zd = model.data_enc(b['points'], b['var_mask'], b['point_mask'])
        return torch.nn.functional.normalize(zd, dim=-1)[0]

    def nmse(expr, X, y, vn):
        try:
            e = parse_to_sympy(expr)
            if e is None: return 1e9
            f = sp.lambdify([sp.Symbol(v) for v in vn], e, 'numpy')
            p = np.asarray(f(*[X[:,i] for i in range(X.shape[1])]), float)
            if p.shape != y.shape or not np.all(np.isfinite(p)): return 1e9
            return float(np.mean((p-y)**2)/(np.var(y)+1e-9))
        except Exception:
            return 1e9

    TOPK=[1,3,5,10,20]
    hits_retr={k:0 for k in TOPK}
    hits_flow={k:0 for k in TOPK}
    hits_comb={k:0 for k in TOPK}
    retr_ceiling=0; n=0; t0=time.time()

    for r in rows:
        X=np.array(r['X'],float); y=np.array(r['y'],float); vn=r['var_names']; truth=r['expr']
        zd = zd_of(r)
        # A) retrieval
        sims = (zd @ zf_index.t())
        top = sims.topk(TOPR)
        retr_exprs = [idx_expr[i] for i in top.indices.cpu().numpy()]
        retr_zf = zf_index[top.indices]                     # (R, d)
        # ceiling: is truth equivalent to ANY retrieved?
        if any(num_equiv(e, truth) for e in retr_exprs): retr_ceiling += 1

        # B) flow candidates -> decode
        with torch.no_grad():
            zf_flow = model.flow.sample(zd.unsqueeze(0), k=args.K, n_steps=20)[0]  # (K,d)
            zd_rep = zd.unsqueeze(0).expand(args.K, -1)
            gens = model.decoder.generate(zd_rep, zf_flow, max_len=48, greedy=False, temperature=1.0)
        flow_exprs = []
        for (seq, consts) in gens:
            e = nodes_to_expr(seq, consts, vn)
            if e is not None: flow_exprs.append(str(e))

        # score+rank helper
        def rank_hits(cands, store):
            scored = []
            for e in cands:
                nm = nmse(e, X, y, vn)
                try: cplx = sp.count_ops(parse_to_sympy(e))
                except Exception: cplx = 30
                scored.append((np.exp(-min(nm,20)) - 0.01*np.log1p(cplx), e))
            scored.sort(key=lambda x:-x[0])
            for rk,(_,e) in enumerate(scored,1):
                if num_equiv(e, truth):
                    for k in TOPK:
                        if rk<=k: store[k]+=1
                    return

        rank_hits(retr_exprs, hits_retr)
        rank_hits(flow_exprs, hits_flow)
        rank_hits(retr_exprs + flow_exprs, hits_comb)
        n += 1
        if n % 25 == 0:
            print(f"  {n}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== HELD-OUT real591 ({n} formulas, K={args.K} flow, top-{TOPR} retr) ===")
    print(f"Retrieval ceiling (truth in top-{TOPR}): {retr_ceiling}/{n} = {100*retr_ceiling/n:.1f}%")
    print(f"\n{'k':>4} {'retrieval':>12} {'flow+dec':>12} {'combined':>12}")
    for k in TOPK:
        print(f"{k:>4} {100*hits_retr[k]/n:>11.1f}% {100*hits_flow[k]/n:>11.1f}% {100*hits_comb[k]/n:>11.1f}%")


if __name__ == '__main__':
    main()
