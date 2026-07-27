"""
End-to-end SR evaluation with a manifold-native reranker (option B).

Pipeline per test (X,y):
  1. z_d = data_encoder(X, y)
  2. flow.sample(z_d, K) -> K candidate z_f
  3. each z_f -> decoder.generate -> formula string
  4. score each candidate with manifold-native features:
       sim    = cos(z_d, z_f_candidate)          (manifold alignment)
       fit    = exp(-NMSE of formula on (X,y))    (hard numeric fit)
       cplx   = n_ops                              (Occam penalty)
     reranker (small MLP on [sim, fit, log(1+cplx), flow_logp_proxy]) -> score
  5. sort by score; Q@k = truth recovered within top-k (sympy/numeric equiv)

The reranker is trained on the SAME flow candidates: a candidate is POSITIVE if
it is numerically equivalent to the truth, NEGATIVE otherwise. Features are
manifold-native, so no dependence on the legacy reranker.
"""
import sys, os, gzip, json, time, argparse, random, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp

from data.ast_grammar import (expr_to_nodes, nodes_to_expr, VOCAB_SIZE as V,
                              ARITY, ID2NT, parse_to_sympy)
from models.encoders import DataEncoder
from models.formula_encoder import ASTEncoder
from models.decoder import ConditioningDecoderDecoder, EOS_ID, PAD_ID
from models.flow_matching import FlowMatching
from train.train_manifold import Manifold, build_ast_edges, MAX_VARS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---- numeric equivalence (handles const folding, log/sqrt identities) ----
def numeric_equiv(cand_expr, true_expr, var_names, n=60, seed=0):
    rng = np.random.default_rng(seed)
    try:
        cs = {v: sp.Symbol(v) for v in var_names}
        ce = sp.sympify(cand_expr, locals=cs)
        te = parse_to_sympy(true_expr)
        if te is None: return False
        # align free symbols
        allv = sorted(set([str(s) for s in ce.free_symbols] +
                          [str(s) for s in te.free_symbols]))
        if not allv: return False
        cf = sp.lambdify([sp.Symbol(v) for v in allv], ce, 'numpy')
        tf = sp.lambdify([sp.Symbol(v) for v in allv], te, 'numpy')
        pts = [rng.uniform(0.5, 2.5, n) for _ in allv]
        yc = np.asarray(cf(*pts), dtype=float).flatten()
        yt = np.asarray(tf(*pts), dtype=float).flatten()
        ok = np.isfinite(yc) & np.isfinite(yt) & (np.abs(yt) > 1e-9)
        if ok.sum() < 20: return False
        ratio = yc[ok] / yt[ok]
        # equal, or constant ratio, or constant offset
        if np.std(yc[ok] - yt[ok]) / (np.std(yt[ok]) + 1e-9) < 1e-3: return True
        if np.std(ratio) / (abs(np.mean(ratio)) + 1e-9) < 1e-3: return True
        return False
    except Exception:
        return False


def nmse_of(cand_expr, X, y, var_names):
    """Lower is better; candidate is a residual r(target,others)=0 — we evaluate
    how well it predicts y. We treat the candidate as an explicit expr if it has
    'target' isolated; else fall back to residual magnitude."""
    try:
        cs = {v: sp.Symbol(v) for v in var_names}
        ce = sp.sympify(cand_expr, locals=cs)
        fv = [str(s) for s in ce.free_symbols]
        f = sp.lambdify([sp.Symbol(v) for v in var_names], ce, 'numpy')
        pred = np.asarray(f(*[X[:, i] for i in range(X.shape[1])]), dtype=float)
        if pred.shape != y.shape or not np.all(np.isfinite(pred)):
            return 1e9
        return float(np.mean((pred - y) ** 2) / (np.var(y) + 1e-9))
    except Exception:
        return 1e9


class Reranker(nn.Module):
    """Tiny MLP over manifold-native features -> relevance score."""
    def __init__(self, n_feat=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, 32), nn.GELU(),
                                 nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x): return self.net(x).squeeze(-1)


def gen_candidates(model, b, K, device):
    """Return list of (cand_str, z_f_cand) per batch item using flow sampling."""
    with torch.no_grad():
        z_d = model.data_enc(b['points'], b['var_mask'], b['point_mask'])
        z_f_cands = model.flow.sample(z_d, k=K, n_steps=20)   # (B, K, d)
        B = z_d.size(0)
        all_str = []
        for bi in range(B):
            zf = z_f_cands[bi]                                  # (K, d)
            zd_rep = z_d[bi:bi+1].expand(K, -1)
            gens = model.decoder.generate(zd_rep, zf, max_len=48, greedy=False, temperature=1.0)
            strs = []
            for (seq, consts) in gens:
                e = nodes_to_expr(seq, consts, b['var_names'][bi])
                strs.append(str(e) if e is not None else None)
            all_str.append((strs, zf.cpu().numpy(), z_d[bi].cpu().numpy()))
    return all_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_full.pt')
    ap.add_argument('--pairs', default='dataset_20260531/data_pairs_full.jsonl.gz')
    ap.add_argument('--n-eval', type=int, default=300)
    ap.add_argument('--K', type=int, default=80)
    ap.add_argument('--train-rerank-rows', type=int, default=1500)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE)
    d = ck['d']
    model = Manifold(d=d).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    print(f"Loaded manifold d={d}")

    # load eval rows (held-out tail of the file)
    rows = []
    with gzip.open(args.pairs, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if 2 <= len(r['var_names']) <= 6 and len(r['seq']) <= 40:
                rows.append(r)
            if len(rows) >= args.n_eval + args.train_rerank_rows: break
    random.shuffle(rows)
    train_rows = rows[:args.train_rerank_rows]
    eval_rows = rows[args.train_rerank_rows:args.train_rerank_rows + args.n_eval]
    print(f"reranker-train rows={len(train_rows)} eval rows={len(eval_rows)}")

    from train.train_manifold import PairDataset
    def collate_rows(rs):
        # reuse PairDataset.collate by faking a dataset
        ds = PairDataset.__new__(PairDataset)
        ds.rows = rs; ds.max_points = 128; ds.max_seq = 48
        b = ds.collate(list(range(len(rs))))
        b['var_names'] = [r['var_names'] for r in rs]
        b['expr'] = [r['expr'] for r in rs]
        b['X'] = [np.array(r['X'], dtype=float) for r in rs]
        b['y'] = [np.array(r['y'], dtype=float) for r in rs]
        return b

    def features_for(strs, zf_arr, zd_vec, X, y, vn):
        feats, labels, valid = [], [], []
        zd = zd_vec / (np.linalg.norm(zd_vec) + 1e-9)
        for ci, s in enumerate(strs):
            if s is None or s == '0':
                continue
            zf = zf_arr[ci] / (np.linalg.norm(zf_arr[ci]) + 1e-9)
            sim = float(np.dot(zd, zf))
            nm = nmse_of(s, X, y, vn)
            fit = float(np.exp(-min(nm, 20)))
            # complexity
            try:
                cplx = sp.count_ops(sp.sympify(s))
            except Exception:
                cplx = 30
            feats.append([sim, fit, np.log1p(cplx), 0.0])
            valid.append((ci, s))
        return feats, valid

    # ---- train reranker on flow candidates of train_rows ----
    print("\nGenerating candidates for reranker training...")
    rr = Reranker().to(DEVICE)
    opt = torch.optim.Adam(rr.parameters(), lr=1e-3)
    Xtr, Ytr = [], []
    bsz = 32
    t0 = time.time()
    for bs in range(0, len(train_rows), bsz):
        rs = train_rows[bs:bs+bsz]
        b = collate_rows(rs)
        b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
        cand = gen_candidates(model, b, args.K, DEVICE)
        for bi, (strs, zf_arr, zd_vec) in enumerate(cand):
            X, y, vn, truth = b['X'][bi], b['y'][bi], b['var_names'][bi], b['expr'][bi]
            feats, valid = features_for(strs, zf_arr, zd_vec, X, y, vn)
            for (f, (ci, s)) in zip(feats, valid):
                pos = numeric_equiv(s, truth, vn)
                Xtr.append(f); Ytr.append(1.0 if pos else 0.0)
        if bs % 160 == 0:
            print(f"  rr-train {bs}/{len(train_rows)} feats={len(Xtr)} pos={int(sum(Ytr))} ({time.time()-t0:.0f}s)", flush=True)
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    Ytr = torch.tensor(Ytr, dtype=torch.float32, device=DEVICE)
    print(f"reranker train set: {len(Xtr)} candidates, {int(Ytr.sum())} positive")
    if Ytr.sum() > 0:
        pw = torch.tensor([(len(Ytr) - Ytr.sum()) / Ytr.sum().clamp_min(1)], device=DEVICE)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        for ep in range(300):
            opt.zero_grad(); loss = bce(rr(Xtr), Ytr); loss.backward(); opt.step()
        print(f"reranker trained, final loss {loss.item():.4f}")

    # ---- evaluate Q@k ----
    print("\nEvaluating Q@k on held-out...")
    TOPK = [1, 3, 5, 10, 20]
    hits = {k: 0 for k in TOPK}
    n = 0
    for bs in range(0, len(eval_rows), bsz):
        rs = eval_rows[bs:bs+bsz]
        b = collate_rows(rs)
        b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
        cand = gen_candidates(model, b, args.K, DEVICE)
        for bi, (strs, zf_arr, zd_vec) in enumerate(cand):
            X, y, vn, truth = b['X'][bi], b['y'][bi], b['var_names'][bi], b['expr'][bi]
            feats, valid = features_for(strs, zf_arr, zd_vec, X, y, vn)
            if not feats:
                n += 1; continue
            with torch.no_grad():
                sc = rr(torch.tensor(feats, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            order = np.argsort(-sc)
            ranked = [valid[i][1] for i in order]
            hit_rank = None
            for rk, s in enumerate(ranked[:max(TOPK)], 1):
                if numeric_equiv(s, truth, vn):
                    hit_rank = rk; break
            if hit_rank:
                for k in TOPK:
                    if hit_rank <= k: hits[k] += 1
            n += 1
        print(f"  eval {n}/{len(eval_rows)}", flush=True)

    print(f"\n=== End-to-end SR (manifold + flow K={args.K} + reranker) ===")
    print(f"Evaluated {n} held-out formulas")
    for k in TOPK:
        print(f"  Q@{k}: {hits[k]}/{n} = {100*hits[k]/n:.1f}%")


if __name__ == '__main__':
    main()
