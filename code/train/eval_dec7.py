"""Evaluate decoder v7 (captioning decoder) on an eval set with the SAME
fit/ranking/judge machinery as eval_sr (apples-to-apples vs the neural baseline).

Candidates per task: dec7.generate with conditioning = data tokens + feature
tokens + K flow samples (frozen flow) + top-k retrieval neighbour z_f, sampled
D times at temperature t.

Usage:
  python eval_dec7.py --dec sr_model/ckpt/dec7.pt --index sr_model/ckpt/zf_full512 \
      --real <set.jsonl> --n 90 --K 32 --D 4
"""
import sys, os, json, time, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import sympy as sp

from data.ast_grammar import MAX_VARS, VOCAB_SIZE
from models.decoder import ConditioningDecoderDecoder
from train.train_manifold import Manifold, load_state_compat
from train.train_dec7 import FeatTokenizer
from train.eval_sr import (tasks_from_real591, nodes_to_skeleton, fit_residual,
                           predict_nmse, is_correct)
from data.ast_grammar import NT2ID

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dec', required=True)
    ap.add_argument('--index', required=True)            # prefix
    ap.add_argument('--real', required=True)
    ap.add_argument('--n', type=int, default=90)
    ap.add_argument('--K', type=int, default=32)         # generated candidates
    ap.add_argument('--D', type=int, default=2)          # sampling rounds
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--max-points', type=int, default=128)
    ap.add_argument('--holdout', type=float, default=0.3)
    ap.add_argument('--dump', default='')
    ap.add_argument('--pure', action='store_true')       # pure decoder: data tokens only
    args = ap.parse_args()
    dump = []

    dk = torch.load(args.dec, map_location=DEVICE)
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
    base.data_enc.pma_tok.load_state_dict(dk['pma_tok'])
    base.data_enc.out_tok.load_state_dict(dk['out_tok'])

    POINTER = dk.get('pointer', False)
    dec = ConditioningDecoderDecoder(vocab_size=VOCAB_SIZE, d=d,
                                     cond_layers=dk['cond_layers'],
                                     dec_layers=dk['dec_layers'], max_len=dk['dec']['decoder.pos_emb.weight'].shape[0],
                                     moe=dk.get('moe', False),
                                     n_experts=dk.get('n_experts', 8),
                                     pointer=POINTER).to(DEVICE)
    dec.load_state_dict(dk['dec']); dec.eval()
    ftok = FeatTokenizer(d, dim_len=max(bck.get('dim_len', 7), 1)).to(DEVICE)
    ftok.load_state_dict(dk['ftok']); ftok.eval()

    zf_index = torch.from_numpy(np.load(args.index + '_index.npy')).to(DEVICE)
    zf_raw = torch.from_numpy(np.load(args.index + '_raw.npy')).to(DEVICE)
    k_flow, k_nbr = dk.get('k_flow', 6), dk.get('k_nbr', 8)
    Ln = dk.get('nbr_toklen', 40)
    idx2seq = None
    if POINTER:
        # tokenize the index formulas once -> (M, Ln) for copy lookup
        from data.ast_grammar import expr_to_nodes
        from models.decoder import PAD_ID
        metas = [json.loads(l) for l in open(args.index + '_meta.jsonl', encoding='utf-8')]
        idx2seq = np.full((len(metas), Ln), PAD_ID, np.int64)
        for i, m in enumerate(metas):
            r = expr_to_nodes(m.get('formula', m.get('expr', '')))
            if r is not None:
                s = r[0][:Ln]; idx2seq[i, :len(s)] = s
        print(f'pointer: tokenized {len(metas)} index formulas', flush=True)

    rows = tasks_from_real591(args.real, args.n)
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    ceiling = 0; n = 0; t0 = time.time()
    for t in rows:
        target, others, truth = t['target'], t['others'], t['truth']
        sv = t['symvals']
        nv = len(sv[target])
        if args.holdout > 0:
            perm = np.random.default_rng(0).permutation(nv)
            cut = max(4, int(nv * (1 - args.holdout)))
            fi, vi = perm[:cut], perm[cut:]
            sv_fit = {k: np.asarray(v, float)[fi] for k, v in sv.items()}
            sv_val = {k: np.asarray(v, float)[vi] for k, v in sv.items()}
        else:
            sv_fit = sv_val = sv
        mp = min(args.max_points, nv)
        pts = np.zeros((1, mp, MAX_VARS + 1), np.float32)
        for k, s in enumerate(others):
            pts[0, :, k] = np.asarray(sv[s], np.float32)[:mp]
        pts[0, :, MAX_VARS] = np.asarray(sv[target], np.float32)[:mp]
        vm = np.zeros((1, MAX_VARS), np.float32); vm[0, :len(others)] = 1
        pm = np.ones((1, mp), np.float32)
        P = torch.from_numpy(pts).to(DEVICE); V = torch.from_numpy(vm).to(DEVICE)
        M = torch.from_numpy(pm).to(DEVICE)
        with torch.no_grad():
            zd = base.data_enc(P, V, M)
            zf_flow = None if args.pure else base.flow.sample(zd, k=k_flow, n_steps=20)
            _, dtok = base.data_enc(P, V, M, return_tokens=True)
            feat = ftok(P, V, M, None)
            data_tokens = torch.cat([dtok, feat], 1)
            z_extra = None
            if not args.pure:
                zn = torch.nn.functional.normalize(zd, dim=-1)
                ids = (zn @ zf_index.t()).topk(k_nbr, dim=-1).indices       # (1,K)
                z_extra = zf_raw[ids[0]].unsqueeze(0)
            nbr_ids_t = nbr_pad_t = None
            if POINTER and not args.pure and idx2seq is not None:
                ns = idx2seq[ids[0].cpu().numpy()].reshape(1, -1)       # (1, K*Ln)
                from models.decoder import PAD_ID as _PAD
                nbr_ids_t = torch.from_numpy(ns).to(DEVICE)
                nbr_pad_t = torch.from_numpy(ns != _PAD).to(DEVICE)
            # generate: replicate conditioning K times for diverse sampling
            R = args.K
            gens = []
            for _ in range(args.D):
                out = dec.generate(None, None,
                                   z_extra=z_extra.expand(R, -1, -1) if z_extra is not None else None,
                                   data_tokens=data_tokens.expand(R, -1, -1),
                                   z_f_tokens=zf_flow.expand(R, -1, -1) if zf_flow is not None else None,
                                   nbr_ids=nbr_ids_t.expand(R, -1) if nbr_ids_t is not None else None,
                                   nbr_pad=nbr_pad_t.expand(R, -1) if nbr_pad_t is not None else None,
                                   n_vars=len(others) + 1,   # VAR_0..nv-1 + target slot
                                   max_len=60, greedy=False, temperature=args.temp)
                gens.extend(out)
        vnf = others + [target]
        scored = []; seen = set()
        for (seq, consts) in gens:
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
            score = np.exp(-min(pn, 20)) - 0.01 * np.log1p(cplx)
            scored.append((score, fitted, pn))
        scored.sort(key=lambda x: -x[0])
        if any(is_correct(e, truth, sv, target, pn) for _, e, pn in scored):
            ceiling += 1
        for rk, (_, e, pn) in enumerate(scored, 1):
            if is_correct(e, truth, sv, target, pn):
                for k in hits:
                    if rk <= k: hits[k] += 1
                break
        if args.dump:
            solved = any(is_correct(e, truth, sv, target, pn) for _, e, pn in scored[:10])
            dump.append({'i': n, 'truth': truth, 'target': target, 'others': others,
                         'solved': bool(solved),
                         'cands': [{'formula': str(e), 'pnmse': float(pn), 'score': float(s)}
                                   for s, e, pn in scored[:8]]})
        n += 1
        if n % 25 == 0:
            print(f'  {n}/{len(rows)} | ceil={ceiling} @1={hits[1]} @10={hits[10]} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    print(f'\n=== dec7 eval n={n} (K={args.K} D={args.D} temp={args.temp}) ===')
    print(f'candidate ceiling: {ceiling}/{n} = {100*ceiling/n:.1f}%')
    for k in sorted(hits):
        print(f'  @{k}: {100*hits[k]/n:.1f}%')
    if args.dump:
        json.dump(dump, open(args.dump, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'dumped {len(dump)} tasks -> {args.dump}')


if __name__ == '__main__':
    main()
