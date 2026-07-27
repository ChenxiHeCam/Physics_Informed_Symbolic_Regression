"""Union + overlap analysis of the three generators on one cleaned eval set.

For each task, generate candidates from flow, LDM (zf-diffusion), and dec7;
fit+judge each pool separately AND the union pool with the same machinery.
Records which pipelines solved each task -> overlap matrix + union ceiling.

Usage:
  python eval_union.py --dec sr_model/ckpt/dec7_v2b.pt --diff sr_model/ckpt/zf_diff.pt \
      --index sr_model/ckpt/zf_full512 --real <cleaned.jsonl> --n 400 \
      --Kf 60 --Kd 64 --Kl 60 --D 4 --out _union_<set>.json
"""
import sys, os, json, time, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import sympy as sp

from data.ast_grammar import MAX_VARS, VOCAB_SIZE, NT2ID
from models.decoder import ConditioningDecoderDecoder
from train.train_manifold import Manifold, load_state_compat
from train.train_dec7 import FeatTokenizer
from train.train_zf_diffusion import ZfDiffusion
from train.eval_sr import (tasks_from_real591, nodes_to_skeleton, fit_residual,
                           predict_nmse, is_correct)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def score_pool(cands, vnf, target, sv_fit, sv_val):
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
    return scored


def solved(scored, truth, sv, target, topk=10):
    for rk, (_, e, pn) in enumerate(scored, 1):
        if is_correct(e, truth, sv, target, pn):
            return True, rk
    return False, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dec', required=True)
    ap.add_argument('--diff', default='')
    ap.add_argument('--base', default='')   # override base_ckpt stored in --dec (use robust-decoder+good-flow ckpt)
    ap.add_argument('--index', required=True)
    ap.add_argument('--real', required=True)
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--Kf', type=int, default=60)
    ap.add_argument('--Kd', type=int, default=64)
    ap.add_argument('--Kl', type=int, default=60)
    ap.add_argument('--D', type=int, default=4)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--max-points', type=int, default=128)
    ap.add_argument('--holdout', type=float, default=0.3)
    ap.add_argument('--pure', action='store_true')   # pure v9: dec7/LDM condition only on data/z_d
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    dk = torch.load(args.dec, map_location=DEVICE)
    _bp = args.base or dk.get('base_ckpt', 'sr_model/ckpt/manifold_full512.pt')
    if not os.path.exists(_bp): _bp = 'sr_model/ckpt/manifold_full512.pt'
    bck = torch.load(_bp, map_location=DEVICE)
    print(f'base latent: {_bp}', flush=True)
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

    dec = ConditioningDecoderDecoder(vocab_size=VOCAB_SIZE, d=d, cond_layers=dk['cond_layers'],
                                     dec_layers=dk['dec_layers'], max_len=64).to(DEVICE)
    dec.load_state_dict(dk['dec']); dec.eval()
    ftok = FeatTokenizer(d, dim_len=max(bck.get('dim_len', 7), 1)).to(DEVICE)
    ftok.load_state_dict(dk['ftok']); ftok.eval()

    diff = None
    if args.diff:
        df = torch.load(args.diff, map_location=DEVICE)
        diff = ZfDiffusion(d=d, n_blocks=df['n_blocks']).to(DEVICE)
        diff.load_state_dict(df['model']); diff.eval()

    zf_index = torch.from_numpy(np.load(args.index + '_index.npy')).to(DEVICE)
    zf_raw = torch.from_numpy(np.load(args.index + '_raw.npy')).to(DEVICE)
    k_flow, k_nbr = dk.get('k_flow', 6), dk.get('k_nbr', 8)

    rows = tasks_from_real591(args.real, args.n)
    keys = ['flow', 'ldm', 'dec7', 'union']
    solv = {k: 0 for k in keys}
    overlap = {}              # frozenset of solvers -> count
    none_solved = []
    dump_tasks = []
    n = 0; t0 = time.time()
    for t in rows:
        target, others, truth = t['target'], t['others'], t['truth']
        sv = t['symvals']; nv = len(sv[target])
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
        vnf = others + [target]
        with torch.no_grad():
            zd, dtok = base.data_enc(P, V, M, return_tokens=True)
            zf_flow = None if args.pure else base.flow.sample(dtok, k=k_flow, n_steps=20)
            feat = ftok(P, V, M, None)
            data_tokens = torch.cat([dtok, feat], 1)
            zf_nbr = None
            if not args.pure:
                zn = torch.nn.functional.normalize(zd, dim=-1)
                ids = (zn @ zf_index.t()).topk(k_nbr, dim=-1).indices
                zf_nbr = zf_raw[ids[0]].unsqueeze(0)
            # --- flow candidates ---
            fcand = []
            zf_f = base.flow.sample(dtok, k=args.Kf, n_steps=20)[0]
            zd_rep = zd.expand(zf_f.size(0), -1)
            dtok_rep = dtok.expand(zf_f.size(0), -1, -1)   # role-0 memory the robust decoder needs
            for _ in range(args.D):
                fcand += base.decoder.generate(zd_rep, zf_f, data_tokens=dtok_rep, max_len=60, greedy=False,
                                               temperature=args.temp, n_vars=len(others) + 1)
            # --- LDM candidates (optional; skipped when --diff not given) ---
            lcand = []
            if diff is not None:
                zf_l = diff.sample(zd, zf_flow, zf_nbr, k=args.Kl, n_steps=50)[0]   # pure -> None,None
                zd_rep2 = zd.expand(zf_l.size(0), -1)
                for _ in range(args.D):
                    lcand += base.decoder.generate(zd_rep2, zf_l, max_len=60, greedy=False,
                                                   temperature=args.temp, n_vars=len(others) + 1)
            # --- dec7 candidates ---
            dcand = []
            for _ in range(args.D):
                dcand += dec.generate(None, None,
                                      z_extra=None if zf_nbr is None else zf_nbr.expand(args.Kd, -1, -1),
                                      data_tokens=data_tokens.expand(args.Kd, -1, -1),
                                      z_f_tokens=None if zf_flow is None else zf_flow.expand(args.Kd, -1, -1),
                                      n_vars=len(others) + 1, max_len=60, greedy=False,
                                      temperature=args.temp)
        scored_f = score_pool(fcand, vnf, target, sv_fit, sv_val)
        scored_l = score_pool(lcand, vnf, target, sv_fit, sv_val)
        scored_d = score_pool(dcand, vnf, target, sv_fit, sv_val)
        sf = solved(scored_f, truth, sv, target)[0]
        sl = solved(scored_l, truth, sv, target)[0]
        sd = solved(scored_d, truth, sv, target)[0]
        su = solved(score_pool(fcand + lcand + dcand, vnf, target, sv_fit, sv_val),
                    truth, sv, target)[0]
        solv['flow'] += sf; solv['ldm'] += sl; solv['dec7'] += sd; solv['union'] += su
        who = frozenset([p for p, s in [('flow', sf), ('ldm', sl), ('dec7', sd)] if s])
        overlap[who] = overlap.get(who, 0) + 1
        if not who:
            none_solved.append({'truth': truth, 'target': target})
        # per-task dump of each model's top-2 generations (for manual review)
        def _top2(sc): return [f'{str(e)}' for _, e, pn in sc[:2]]
        dump_tasks.append({'i': n, 'truth': truth, 'target': target, 'others': others,
                           'flow': {'solved_num': bool(sf), 'gens': _top2(scored_f)},
                           'ldm': {'solved_num': bool(sl), 'gens': _top2(scored_l)},
                           'dec7': {'solved_num': bool(sd), 'gens': _top2(scored_d)}})
        n += 1
        if n % 25 == 0:
            print(f'  {n}/{len(rows)} | F={solv["flow"]} L={solv["ldm"]} D={solv["dec7"]} '
                  f'U={solv["union"]} ({time.time()-t0:.0f}s)', flush=True)

    print(f'\n=== union/overlap n={n} ({os.path.basename(args.real)}) ===')
    for k in keys:
        print(f'  {k:6s} ceiling: {solv[k]}/{n} = {100*solv[k]/n:.1f}%')
    print('  --- overlap (who solves) ---')
    for who, c in sorted(overlap.items(), key=lambda x: -x[1]):
        label = '+'.join(sorted(who)) if who else 'NONE'
        print(f'    {label:20s} {c}')
    if args.out:
        json.dump({'set': args.real, 'n': n, 'solv': solv,
                   'overlap': {'+'.join(sorted(w)) or 'NONE': c for w, c in overlap.items()},
                   'none_solved': none_solved, 'tasks': dump_tasks},
                  open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('saved ->', args.out)


if __name__ == '__main__':
    main()
