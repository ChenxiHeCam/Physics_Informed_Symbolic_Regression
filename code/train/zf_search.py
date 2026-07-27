"""Test-time z_f search: break the flow's coverage ceiling.

The flow only samples its learned (in-distribution) region -> ~40% ceiling. But the
oracle proves that if we HAD the right z_f (even OOD), the decoder recovers the law
(86% structural). So instead of trusting the flow to predict z_f, we SEARCH z_f
space: seed with the flow's K samples, then hill-climb (perturb z_f -> decode -> score
by data-fit predict_nmse -> keep improvements). Decode is discrete/non-differentiable,
so this is black-box search. Reaching a correct z_f the raw K samples missed = a
ceiling lift.

Reports raw-flow ceiling vs search-augmented ceiling on held-out tasks.
"""
import sys, os, argparse, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch
from train.train_manifold import Manifold
from train.eval_sr import (tasks_from_real591, nodes_to_skeleton, fit_residual,
                           predict_nmse, is_correct, dims_for_task)
from data.ast_grammar import MAX_VARS
from models.decoder import CONST_ID

DEVICE = torch.device('cpu')   # prototype on CPU to not contend with GPU training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/flow_v6_noaug.pt')
    ap.add_argument('--real', default='data/real591.jsonl')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--K', type=int, default=16)        # flow seeds
    ap.add_argument('--M', type=int, default=4)         # survivors per round
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--perturb', type=int, default=8)   # children per survivor
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64)).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    mlen = min(getattr(model, 'dec_max_len', 48), 96)
    rows = tasks_from_real591(args.real, args.n)
    print(f"tasks {len(rows)}  K={args.K} rounds={args.rounds} perturb={args.perturb}\n")

    def encode(t):
        others, X, y = t['others'], t['X'], t['y']
        mp = 128; npt = min(X.shape[0], mp); nv = len(others)
        pts = torch.zeros(1, mp, MAX_VARS + 1)
        pts[0, :npt, :nv] = torch.from_numpy(X[:npt].astype(np.float32))
        pts[0, :npt, MAX_VARS] = torch.from_numpy(y[:npt].astype(np.float32))
        vm = torch.zeros(1, MAX_VARS); vm[0, :nv] = 1
        pm = torch.zeros(1, mp); pm[0, :npt] = 1
        dims = None
        if getattr(model.data_enc, 'dim_len', 0):
            dims = torch.from_numpy(dims_for_task(others, t['target'], model.data_enc.dim_len)).unsqueeze(0)
        with torch.no_grad():
            zd, tok = model.data_enc(pts, vm, pm, dims=dims, return_tokens=True)
        return zd[0], tok[0]

    def decode_score(zf_mat, zd, tok, t):
        """zf_mat (M,d) -> list of (pnmse, is_corr, expr). Greedy decode each."""
        M = zf_mat.size(0)
        vnf = t['others'] + [t['target']]; sv = t['symvals']
        zd_rep = zd.unsqueeze(0).expand(M, -1)
        tok_rep = tok.unsqueeze(0).expand(M, -1, -1)
        out = []
        with torch.no_grad():
            gens = model.decoder.generate(zd_rep, zf_mat, data_tokens=tok_rep, max_len=mlen, greedy=True)
        for (seq, consts) in gens:
            e, params = nodes_to_skeleton(seq, vnf)
            if e is None:
                out.append((1e9, False, None)); continue
            c_init = [consts[i] for i in range(len(seq)) if seq[i] == CONST_ID]
            fit, _ = fit_residual(e, params, sv, t['target'],
                                  c_init=(c_init if len(c_init) == len(params) else None))
            pn = predict_nmse(fit, sv, t['target'])
            out.append((pn, is_correct(fit, t['truth'], sv, t['target'], pn), fit))
        return out

    raw_hit = 0; rawbig_hit = 0; search_hit = 0; gained = []
    for ti, t in enumerate(rows):
        t0 = time.time()
        zd, tok = encode(t)
        budget = args.K + args.rounds * args.M * args.perturb     # total decodes search spends
        with torch.no_grad():
            seeds = model.flow.sample(tok.unsqueeze(0), k=args.K, n_steps=20)[0]   # (K,d)
            big = model.flow.sample(tok.unsqueeze(0), k=budget, n_steps=20)[0]     # budget-matched raw control
        sc = decode_score(seeds, zd, tok, t)
        raw_correct = any(c for _, c, _ in sc)
        # budget-matched raw-flow control: equal #decodes, pure sampling (no search)
        rawbig_correct = any(c for _, c, _ in decode_score(big, zd, tok, t))
        # per-dim search scale from the seed spread
        sigma0 = seeds.std(0, keepdim=True).clamp_min(1e-3)
        # survivors = top-M seeds by pnmse
        order = sorted(range(len(sc)), key=lambda i: sc[i][0])[:args.M]
        surv = seeds[order].clone()
        best_pn = min(s[0] for s in sc)
        found = raw_correct
        for r in range(args.rounds):
            sigma = sigma0 * (0.6 ** r)               # anneal
            kids = []
            for m in range(surv.size(0)):
                kids.append(surv[m].unsqueeze(0).expand(args.perturb, -1)
                            + sigma * torch.randn(args.perturb, surv.size(1)))
            kids = torch.cat(kids, 0)
            ksc = decode_score(kids, zd, tok, t)
            if any(c for _, c, _ in ksc):
                found = True
            # next survivors = best-M among current survivors' kids by pnmse
            korder = sorted(range(len(ksc)), key=lambda i: ksc[i][0])[:args.M]
            if ksc[korder[0]][0] < best_pn:
                best_pn = ksc[korder[0]][0]
                surv = kids[korder].clone()
        raw_hit += raw_correct; rawbig_hit += rawbig_correct; search_hit += found
        if found and not rawbig_correct:
            gained.append(t['truth'][:60])
        print(f"  task {ti:2d} raw={'HIT' if raw_correct else '   '} "
              f"rawBIG={'HIT' if rawbig_correct else '   '} "
              f"search={'HIT' if found else '   '} best_pn={best_pn:.1e} ({time.time()-t0:.0f}s)", flush=True)
    n = len(rows)
    print(f"\n=== z_f search on {n} held-out (budget={args.K + args.rounds*args.M*args.perturb} decodes) ===")
    print(f"raw-flow ceiling (K={args.K}):       {raw_hit}/{n} = {100*raw_hit/n:.1f}%")
    print(f"raw-flow ceiling (budget-matched):  {rawbig_hit}/{n} = {100*rawbig_hit/n:.1f}%")
    print(f"SEARCH-augmented ceiling:           {search_hit}/{n} = {100*search_hit/n:.1f}%")
    print(f"gained by SEARCH over budget-raw ({len(gained)}): {gained}")


if __name__ == '__main__':
    main()
