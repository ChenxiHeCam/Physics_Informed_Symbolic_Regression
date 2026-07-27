"""Evaluate the masked-diffusion decoder via MaskGIT iterative generation, same
fit/judge as the other pipelines. For each task: build conditioning (frozen
full512 z_d + flow + feature tokens), then for several candidate lengths start
all-[MASK] and iteratively reveal highest-confidence tokens. Fit + judge.

Usage: python eval_mask_diff.py --diff mask_diff.pt --base full512.pt --real <set> --n 60
"""
import sys, os, json, time, argparse, warnings, math
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import sympy as sp

from data.ast_grammar import MAX_VARS, VOCAB_SIZE, NT2ID
from models.decoder import PAD_ID, EOS_ID, ARITY_BY_ID
from train.train_manifold import Manifold, load_state_compat
from train.train_mask_diffusion import MaskDiffusion, FeatTok, MASK_ID
from train.eval_sr import (tasks_from_real591, nodes_to_skeleton, fit_residual,
                           predict_nmse, is_correct)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def valid_prefix(seq):
    """Trim a token sequence to the first complete AST (preorder, arity-closed)."""
    open_slots = 1; out = []
    for tk in seq:
        if tk in (PAD_ID, EOS_ID, MASK_ID): break
        out.append(tk); open_slots += int(ARITY_BY_ID[tk]) - 1
        if open_slots <= 0: break
    return out if open_slots <= 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diff', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--real', required=True)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--lengths', default='5,7,9,13,17,21')
    ap.add_argument('--rounds', type=int, default=8)      # MaskGIT reveal steps
    ap.add_argument('--samples', type=int, default=6)     # stochastic samples per length
    ap.add_argument('--max-points', type=int, default=128)
    ap.add_argument('--holdout', type=float, default=0.3)
    args = ap.parse_args()

    dk = torch.load(args.diff, map_location=DEVICE)
    bck = torch.load(args.base, map_location=DEVICE); d = bck['d']
    base = Manifold(d=d, dim_len=bck.get('dim_len', 0), tok=bck.get('tok', False),
                    n_tokens=bck.get('n_tokens', 16), n_gin=bck.get('n_gin', 4),
                    dec_layers=bck.get('dec_layers', 6), dec_max_len=bck.get('dec_max_len', 64),
                    n_ftokens=bck.get('n_ftokens', 0), log_feats=bck.get('log_feats', False),
                    class_feats=bck.get('class_feats', False), dim_head=bck.get('dim_head', False),
                    robust_norm=bck.get('robust_norm', False)).to(DEVICE)
    load_state_compat(base, bck['model']); base.eval()
    model = MaskDiffusion(d=d, n_layers=dk['n_layers'], max_len=48, k_flow=dk['k_flow']).to(DEVICE)
    model.load_state_dict(dk['model']); model.eval()
    ftok = FeatTok(d).to(DEVICE); ftok.load_state_dict(dk['ftok']); ftok.eval()
    lengths = [int(x) for x in args.lengths.split(',')]

    rows = tasks_from_real591(args.real, args.n)
    hits = {1: 0, 5: 0, 10: 0}; ceiling = 0; n = 0; t0 = time.time()
    for t in rows:
        target, others, truth = t['target'], t['others'], t['truth']
        sv = t['symvals']; nv = len(sv[target])
        perm = np.random.default_rng(0).permutation(nv); cut = max(4, int(nv * 0.7))
        sv_fit = {k: np.asarray(v, float)[perm[:cut]] for k, v in sv.items()}
        sv_val = {k: np.asarray(v, float)[perm[cut:]] for k, v in sv.items()}
        mp = min(args.max_points, nv)
        pts = np.zeros((1, mp, MAX_VARS + 1), np.float32)
        for k, s in enumerate(others): pts[0, :, k] = np.asarray(sv[s], np.float32)[:mp]
        pts[0, :, MAX_VARS] = np.asarray(sv[target], np.float32)[:mp]
        vm = np.zeros((1, MAX_VARS), np.float32); vm[0, :len(others)] = 1
        pm = np.ones((1, mp), np.float32)
        P, V, M = (torch.from_numpy(x).to(DEVICE) for x in (pts, vm, pm))
        with torch.no_grad():
            zd = base.data_enc(P, V, M)
            zf = base.flow.sample(zd, k=dk['k_flow'], n_steps=20)
            feat = ftok(P, V, M)
            cands_tok = []
            for L in lengths:
                mem = model.cond_mem(zd, zf, feat).expand(args.samples, -1, -1)
                seq = torch.full((args.samples, L), MASK_ID, dtype=torch.long, device=DEVICE)
                for r in range(args.rounds):
                    logits = model(seq, mem)                       # (S,L,V)
                    probs = logits.softmax(-1)
                    conf, pred = probs.max(-1)
                    ismask = seq == MASK_ID
                    # reveal a growing fraction of the most-confident masked tokens
                    keep_frac = (r + 1) / args.rounds
                    for b in range(args.samples):
                        mpos = torch.where(ismask[b])[0]
                        if len(mpos) == 0: continue
                        k = max(1, int(math.ceil(len(mpos) * (keep_frac))))
                        order = conf[b, mpos].argsort(descending=True)[:k]
                        sel = mpos[order]
                        seq[b, sel] = pred[b, sel]
                    if not (seq == MASK_ID).any(): break
                # fill any leftover masks greedily
                if (seq == MASK_ID).any():
                    logits = model(seq, mem); pred = logits.argmax(-1)
                    seq = torch.where(seq == MASK_ID, pred, seq)
                for b in range(args.samples):
                    vp = valid_prefix(seq[b].tolist())
                    if vp: cands_tok.append(vp)
        vnf = others + [target]; scored = []; seen = set()
        for vp in cands_tok:
            e, params = nodes_to_skeleton(vp, vnf)
            if e is None: continue
            try: key = sp.srepr(e)
            except Exception: key = str(e)
            if key in seen: continue
            seen.add(key)
            fitted, _ = fit_residual(e, params, sv_fit, target)
            pn = predict_nmse(fitted, sv_val, target)
            try: cplx = sp.count_ops(fitted)
            except Exception: cplx = 30
            scored.append((np.exp(-min(pn, 20)) - 0.01 * np.log1p(cplx), fitted, pn))
        scored.sort(key=lambda x: -x[0])
        if any(is_correct(e, truth, sv, target, pn) for _, e, pn in scored): ceiling += 1
        for rk, (_, e, pn) in enumerate(scored, 1):
            if is_correct(e, truth, sv, target, pn):
                for k in hits:
                    if rk <= k: hits[k] += 1
                break
        n += 1
        if n % 20 == 0:
            print(f'  {n}/{len(rows)} | ceil={ceiling} @1={hits[1]} ({time.time()-t0:.0f}s)', flush=True)
    print(f'\n=== mask-diffusion eval n={n} ===')
    print(f'candidate ceiling: {ceiling}/{n} = {100*ceiling/n:.1f}%')
    for k in sorted(hits): print(f'  @{k}: {100*hits[k]/n:.1f}%')


if __name__ == '__main__':
    main()
