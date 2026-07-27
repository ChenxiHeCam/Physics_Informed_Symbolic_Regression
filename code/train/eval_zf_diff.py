"""Evaluate the latent z_f DIFFUSION generator head-to-head vs the rectified
flow: same frozen full512 decoder, same fit/judge machinery. Reports the
candidate ceiling + Q@k from diffusion samples (optionally unioned with flow).

Usage:
  python eval_zf_diff.py --diff sr_model/ckpt/zf_diff.pt --index sr_model/ckpt/zf_full512 \
      --real <set.jsonl> --n 90 --K 60 --D 4 [--with-flow]
"""
import sys, os, json, time, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import sympy as sp

from data.ast_grammar import MAX_VARS, NT2ID
from train.train_manifold import Manifold, load_state_compat
from train.train_zf_diffusion import ZfDiffusion
from train.eval_sr import (tasks_from_real591, nodes_to_skeleton, fit_residual,
                           predict_nmse, is_correct)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diff', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--real', required=True)
    ap.add_argument('--n', type=int, default=90)
    ap.add_argument('--K', type=int, default=60)
    ap.add_argument('--D', type=int, default=4)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--ddim-steps', type=int, default=50)
    ap.add_argument('--max-points', type=int, default=128)
    ap.add_argument('--holdout', type=float, default=0.3)
    ap.add_argument('--with-flow', action='store_true')   # union diffusion+flow cands
    ap.add_argument('--pure', action='store_true')        # pure LDM: condition only on z_d
    ap.add_argument('--dump', default='')                 # per-task solved dump
    args = ap.parse_args()

    dk = torch.load(args.diff, map_location=DEVICE)
    bck = torch.load(dk['base_ckpt'] if os.path.exists(dk['base_ckpt'])
                     else 'sr_model/ckpt/manifold_full512.pt', map_location=DEVICE)
    d = bck['d']
    base = Manifold(d=d, dim_len=bck.get('dim_len', 0), tok=bck.get('tok', False),
                    n_tokens=bck.get('n_tokens', 16), n_gin=bck.get('n_gin', 4),
                    dec_layers=bck.get('dec_layers', 6), dec_max_len=bck.get('dec_max_len', 64),
                    n_ftokens=bck.get('n_ftokens', 0), log_feats=bck.get('log_feats', False),
                    class_feats=bck.get('class_feats', False), dim_head=bck.get('dim_head', False),
                    robust_norm=bck.get('robust_norm', False)).to(DEVICE)
    load_state_compat(base, bck['model']); base.eval()

    diff = ZfDiffusion(d=d, n_blocks=dk['n_blocks']).to(DEVICE)
    diff.load_state_dict(dk['model']); diff.eval()
    k_flow, k_nbr = dk.get('k_flow', 6), dk.get('k_nbr', 8)

    zf_index = torch.from_numpy(np.load(args.index + '_index.npy')).to(DEVICE)
    zf_raw = torch.from_numpy(np.load(args.index + '_raw.npy')).to(DEVICE)

    rows = tasks_from_real591(args.real, args.n)
    hits = {1: 0, 3: 0, 5: 0, 10: 0}; ceiling = 0; n = 0; t0 = time.time(); solved_dump = []
    for t in rows:
        target, others, truth = t['target'], t['others'], t['truth']
        sv = t['symvals']
        nv = len(sv[target])
        perm = np.random.default_rng(0).permutation(nv)
        cut = max(4, int(nv * (1 - args.holdout)))
        sv_fit = {k: np.asarray(v, float)[perm[:cut]] for k, v in sv.items()}
        sv_val = {k: np.asarray(v, float)[perm[cut:]] for k, v in sv.items()}
        mp = min(args.max_points, nv)
        pts = np.zeros((1, mp, MAX_VARS + 1), np.float32)
        for k, s in enumerate(others):
            pts[0, :, k] = np.asarray(sv[s], np.float32)[:mp]
        pts[0, :, MAX_VARS] = np.asarray(sv[target], np.float32)[:mp]
        vm = np.zeros((1, MAX_VARS), np.float32); vm[0, :len(others)] = 1
        pm = np.ones((1, mp), np.float32)
        P, V, M = (torch.from_numpy(x).to(DEVICE) for x in (pts, vm, pm))
        with torch.no_grad():
            zd = base.data_enc(P, V, M)
            zf_flow = None if args.pure else base.flow.sample(zd, k=k_flow, n_steps=20)
            zf_nbr = None
            if not args.pure:
                zn = torch.nn.functional.normalize(zd, dim=-1)
                ids = (zn @ zf_index.t()).topk(k_nbr, dim=-1).indices
                zf_nbr = zf_raw[ids[0]].unsqueeze(0)
            zf_gen = diff.sample(zd, zf_flow, zf_nbr, k=args.K,
                                 n_steps=args.ddim_steps)[0]          # (K,d)
            if args.with_flow:
                zf_more = base.flow.sample(zd, k=args.K, n_steps=20)[0]
                zf_gen = torch.cat([zf_gen, zf_more], 0)
            zd_rep = zd.expand(zf_gen.size(0), -1)
            cands = []
            for _ in range(args.D):
                for (seq, consts) in base.decoder.generate(zd_rep, zf_gen,
                                                           max_len=60, greedy=False,
                                                           temperature=args.temp,
                                                           n_vars=len(others) + 1):
                    cands.append((seq, consts))
        vnf = others + [target]
        scored = []; seen = set()
        for (seq, consts) in cands:
            e, params = nodes_to_skeleton(seq, vnf)
            if e is None: continue
            try: key = sp.srepr(e)
            except Exception: key = str(e)
            if key in seen: continue
            seen.add(key)
            ci = [consts[i] for i in range(len(seq)) if seq[i] == NT2ID['CONST']]
            if len(ci) != len(params): ci = None
            fitted, _ = fit_residual(e, params, sv_fit, target, c_init=ci)
            pn = predict_nmse(fitted, sv_val, target)
            try: cplx = sp.count_ops(fitted)
            except Exception: cplx = 30
            scored.append((np.exp(-min(pn, 20)) - 0.01 * np.log1p(cplx), fitted, pn))
        scored.sort(key=lambda x: -x[0])
        _sv = any(is_correct(e, truth, sv, target, pn) for _, e, pn in scored)
        solved_dump.append({'i': n, 'truth': truth, 'solved': bool(_sv),
                            'top': str(scored[0][1]) if scored else None})
        if _sv:
            ceiling += 1
        for rk, (_, e, pn) in enumerate(scored, 1):
            if is_correct(e, truth, sv, target, pn):
                for k in hits:
                    if rk <= k: hits[k] += 1
                break
        n += 1
        if n % 25 == 0:
            print(f'  {n}/{len(rows)} | ceil={ceiling} @1={hits[1]} ({time.time()-t0:.0f}s)',
                  flush=True)
    tag = 'diff+flow' if args.with_flow else 'diff'
    print(f'\n=== zf {tag} eval n={n} (K={args.K} D={args.D}) ===')
    print(f'candidate ceiling: {ceiling}/{n} = {100*ceiling/n:.1f}%')
    for k in sorted(hits):
        print(f'  @{k}: {100*hits[k]/n:.1f}%')
    if args.dump:
        import json as _json
        _json.dump(solved_dump, open(args.dump, 'w', encoding='utf-8'))
        print('solved-dump ->', args.dump)


if __name__ == '__main__':
    main()
