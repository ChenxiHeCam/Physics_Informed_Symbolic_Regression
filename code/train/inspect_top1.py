"""
Eyeball held-out: for each task print TRUTH vs the TOP-1 ranked candidate (what
Q@1 would report), its data-fit, and whether num_equiv accepts it. Lets us see
if top-1 is actually right-but-rejected (false neg) or genuinely wrong.
"""
import sys, os, json, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch, sympy as sp
from data.ast_grammar import MAX_VARS
from train.train_manifold import Manifold
from train.eval_sr import (nodes_to_skeleton, fit_residual, num_equiv, predict_nmse,
                           tasks_from_real591, dims_for_task)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_v4_units.pt')
    ap.add_argument('--real', default='data/real591.jsonl')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--K', type=int, default=80)
    ap.add_argument('--D', type=int, default=4)
    ap.add_argument('--dump', default='')   # JSON out: per-task truth + top1 + pnmse
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0)).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    rows = tasks_from_real591(args.real, args.n)
    print(f"tasks: {len(rows)}\n")

    nhit1 = nceil = nfalseneg = 0; dump = []
    for ti, t in enumerate(rows):
        truth = t['truth']; target = t['target']; others = t['others']
        vnf = others + [target]; sv = t['symvals']; X, y = t['X'], t['y']
        mp = 128; npt = min(X.shape[0], mp); nv = len(others)
        pts = torch.zeros(1, mp, MAX_VARS + 1)
        pts[0, :npt, :nv] = torch.from_numpy(X[:npt].astype(np.float32))
        pts[0, :npt, MAX_VARS] = torch.from_numpy(y[:npt].astype(np.float32))
        vm = torch.zeros(1, MAX_VARS); vm[0, :nv] = 1
        pm = torch.zeros(1, mp); pm[0, :npt] = 1
        dims = None
        if getattr(model.data_enc, 'dim_len', 0):
            dims = torch.from_numpy(dims_for_task(others, target, model.data_enc.dim_len)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            zd = model.data_enc(pts.to(DEVICE), vm.to(DEVICE), pm.to(DEVICE), dims=dims)
            zf = model.flow.sample(zd, k=args.K, n_steps=20)[0]
            zdr = zd.expand(args.K, -1)
            cands = []
            for _ in range(args.D):
                for (seq, cs) in model.decoder.generate(zdr, zf, max_len=48, greedy=False, temperature=0.9):
                    cands.append(seq)
        # stage 1: cheap residual fit for ALL unique candidates
        prelim, seen = [], set()
        for seq in cands:
            e, params = nodes_to_skeleton(seq, vnf)
            if e is None: continue
            try: key = sp.srepr(e)
            except Exception: key = str(e)
            if key in seen: continue
            seen.add(key)
            fitted, rn = fit_residual(e, params, sv, target)
            prelim.append((rn, fitted))
        # stage 2: expensive predict_nmse + num_equiv only on the top-18 by residual
        prelim.sort(key=lambda x: x[0])
        scored, anyeq = [], False
        for rn, fitted in prelim[:30]:
            pn = predict_nmse(fitted, sv, target)
            eq = num_equiv(fitted, truth, sv, target)
            if eq: anyeq = True
            scored.append((pn, fitted, eq))
        scored.sort(key=lambda x: x[0])
        if anyeq: nceil += 1
        if scored:
            pn1, top1, eq1 = scored[0]
            if eq1: nhit1 += 1
            # false-neg flag: top1 fits data well but judged not-equiv
            fn = (pn1 < 1e-2 and not eq1)
            if fn: nfalseneg += 1
            tag = 'HIT@1' if eq1 else ('GOODFIT!' if fn else 'miss ')
            print(f"[{tag}] {truth[:58]}")
            print(f"        top1: {str(top1)[:66]}  pnmse={pn1:.1e} eq={eq1}")
            dump.append({'i': ti, 'truth': truth, 'target': target, 'others': others,
                         'top1': str(top1), 'pnmse': float(pn1), 'num_equiv': bool(eq1)})
        else:
            print(f"[nocand] {truth[:58]}")
            dump.append({'i': ti, 'truth': truth, 'target': target, 'others': others,
                         'top1': None, 'pnmse': 1e9, 'num_equiv': False})
    if args.dump:
        json.dump(dump, open(args.dump, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f"dumped {len(dump)} -> {args.dump}")
    print(f"\nQ@1={nhit1}/{len(rows)}={100*nhit1/len(rows):.1f}%  ceiling={nceil}/{len(rows)}="
          f"{100*nceil/len(rows):.1f}%  top1-goodfit-but-rejected(suspect false-neg)={nfalseneg}")


if __name__ == '__main__':
    main()
