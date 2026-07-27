"""
Retrieval + range-limited reranker (user's design).

Inference per (X,y):
  1. z_d = data_encoder(X,y)
  2. retrieve top-30 real formulas by cos(z_d, z_f) over the index   (fast, no NMSE)
  3. reranker scores the 30 candidates from manifold-native features
     [sim(z_d,z_f), fit=exp(-NMSE), log(1+complexity), rank_in_retrieval]
     -> reorder
  4. Q@k = truth recovered within top-k (numeric equivalence)

The reranker only ever sees a SMALL, focused candidate set (the 30 retrieved),
so it is easy to train: clear positives/negatives, bounded search space.
"""
import sys, os, gzip, json, time, argparse, random, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
import torch.nn as nn
import sympy as sp

from data.ast_grammar import parse_to_sympy, expr_to_nodes
from train.train_manifold import Manifold, PairDataset, build_ast_edges, MAX_VARS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOPR = 30   # retrieval depth


def numeric_equiv(cand_expr, true_expr, n=60, seed=0):
    rng = np.random.default_rng(seed)
    try:
        ce = parse_to_sympy(cand_expr) if isinstance(cand_expr, str) else cand_expr
        te = parse_to_sympy(true_expr)
        if ce is None or te is None: return False
        allv = sorted(set([str(s) for s in ce.free_symbols] +
                          [str(s) for s in te.free_symbols]))
        if not allv: return False
        cf = sp.lambdify([sp.Symbol(v) for v in allv], ce, 'numpy')
        tf = sp.lambdify([sp.Symbol(v) for v in allv], te, 'numpy')
        pts = [rng.uniform(0.5, 2.5, n) for _ in allv]
        yc = np.asarray(cf(*pts), float).flatten()
        yt = np.asarray(tf(*pts), float).flatten()
        ok = np.isfinite(yc) & np.isfinite(yt) & (np.abs(yt) > 1e-9)
        if ok.sum() < 20: return False
        if np.std(yc[ok]-yt[ok])/(np.std(yt[ok])+1e-9) < 1e-3: return True
        r = yc[ok]/yt[ok]
        if np.std(r)/(abs(np.mean(r))+1e-9) < 1e-3: return True
        return False
    except Exception:
        return False


def nmse_of(expr, X, y, var_names):
    try:
        e = parse_to_sympy(expr)
        if e is None: return 1e9
        f = sp.lambdify([sp.Symbol(v) for v in var_names], e, 'numpy')
        pred = np.asarray(f(*[X[:, i] for i in range(X.shape[1])]), float)
        if pred.shape != y.shape or not np.all(np.isfinite(pred)): return 1e9
        return float(np.mean((pred-y)**2)/(np.var(y)+1e-9))
    except Exception:
        return 1e9


class Reranker(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 32), nn.GELU(),
                                 nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_full.pt')
    ap.add_argument('--index', default='sr_model/ckpt/zf_index')
    ap.add_argument('--pairs', default='dataset_20260531/data_pairs_full.jsonl.gz')
    ap.add_argument('--n-eval', type=int, default=400)
    ap.add_argument('--n-train', type=int, default=2000)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE)
    d = ck['d']
    model = Manifold(d=d).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()

    zf_index = np.load(args.index + '_index.npy')              # (N, d) normalized
    meta = [json.loads(l) for l in open(args.index + '_meta.jsonl', encoding='utf-8')]
    zf_t = torch.from_numpy(zf_index).to(DEVICE)
    index_exprs = [m['expr'] for m in meta]
    print(f"Index: {len(meta)} formulas, d={d}")

    # load eval/train rows
    rows = []
    with gzip.open(args.pairs, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if 2 <= len(r['var_names']) <= 6 and len(r['seq']) <= 40:
                rows.append(r)
            if len(rows) >= args.n_eval + args.n_train: break
    random.shuffle(rows)
    tr, ev = rows[:args.n_train], rows[args.n_train:args.n_train+args.n_eval]
    print(f"train={len(tr)} eval={len(ev)}")

    def encode_zd(rs):
        ds = PairDataset.__new__(PairDataset); ds.rows = rs
        ds.max_points = 128; ds.max_seq = 48
        b = ds.collate(list(range(len(rs))))
        b = {k:(v.to(DEVICE) if torch.is_tensor(v) else v) for k,v in b.items()}
        with torch.no_grad():
            zd = model.data_enc(b['points'], b['var_mask'], b['point_mask'])
            zd = torch.nn.functional.normalize(zd, dim=-1)
        return zd

    def retrieve(zd):
        sims = zd @ zf_t.t()                     # (B, N)
        top = sims.topk(TOPR, dim=1)
        return top.indices.cpu().numpy(), top.values.cpu().numpy()

    def feats_for(idxs, sims, X, y, vn):
        F, cand = [], []
        for rank, (ci, sm) in enumerate(zip(idxs, sims)):
            ex = index_exprs[ci]
            nm = nmse_of(ex, X, y, vn)
            fit = float(np.exp(-min(nm, 20)))
            try: cplx = sp.count_ops(parse_to_sympy(ex))
            except Exception: cplx = 30
            F.append([float(sm), fit, np.log1p(cplx), rank/TOPR])
            cand.append(ex)
        return F, cand

    # ---- train reranker ----
    print("\nTraining range-limited reranker...")
    rr = Reranker().to(DEVICE); opt = torch.optim.Adam(rr.parameters(), 1e-3)
    Xtr, Ytr = [], []
    for bs in range(0, len(tr), 64):
        rs = tr[bs:bs+64]; zd = encode_zd(rs)
        idxs, sims = retrieve(zd)
        for bi, r in enumerate(rs):
            X = np.array(r['X'], float); y = np.array(r['y'], float)
            Fc, cand = feats_for(idxs[bi], sims[bi], X, y, r['var_names'])
            for f, ex in zip(Fc, cand):
                Xtr.append(f); Ytr.append(1.0 if numeric_equiv(ex, r['expr']) else 0.0)
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    Ytr = torch.tensor(Ytr, dtype=torch.float32, device=DEVICE)
    npos = int(Ytr.sum())
    print(f"reranker train: {len(Xtr)} cands, {npos} positive "
          f"({100*npos/len(Xtr):.1f}%) — retrieval recall@{TOPR}={100*npos/len(tr):.0f}%")
    if npos > 0:
        pw = torch.tensor([(len(Ytr)-Ytr.sum())/Ytr.sum().clamp_min(1)], device=DEVICE)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        for _ in range(400):
            opt.zero_grad(); bce(rr(Xtr), Ytr).backward(); opt.step()

    # ---- evaluate ----
    print("\nEvaluating Q@k...")
    TOPK = [1, 3, 5, 10, 20]
    hits = {k:0 for k in TOPK}; retr_hit = 0; n = 0
    for bs in range(0, len(ev), 64):
        rs = ev[bs:bs+64]; zd = encode_zd(rs)
        idxs, sims = retrieve(zd)
        for bi, r in enumerate(rs):
            X = np.array(r['X'], float); y = np.array(r['y'], float)
            Fc, cand = feats_for(idxs[bi], sims[bi], X, y, r['var_names'])
            # is truth anywhere in retrieved 30?
            in_retr = any(numeric_equiv(ex, r['expr']) for ex in cand)
            if in_retr: retr_hit += 1
            with torch.no_grad():
                sc = rr(torch.tensor(Fc, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            order = np.argsort(-sc)
            hit_rank = None
            for rk, oi in enumerate(order, 1):
                if numeric_equiv(cand[oi], r['expr']):
                    hit_rank = rk; break
            if hit_rank:
                for k in TOPK:
                    if hit_rank <= k: hits[k] += 1
            n += 1
        print(f"  {n}/{len(ev)}", flush=True)

    print(f"\n=== Retrieval(top-{TOPR}) + range-limited reranker ===")
    print(f"Evaluated {n} formulas")
    print(f"Retrieval recall@{TOPR}: {retr_hit}/{n} = {100*retr_hit/n:.1f}%  (ceiling)")
    for k in TOPK:
        print(f"  Q@{k}: {hits[k]}/{n} = {100*hits[k]/n:.1f}%")


if __name__ == '__main__':
    main()
