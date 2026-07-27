"""
Diagnostic: isolate where in-distribution SR recovery fails.
For a few TRAINING rows (model has seen them), print:
  - cos(z_d, z_f_true)                     : data<->formula alignment for THIS row
  - full-index retrieval rank of an equivalent formula
  - ORACLE decode: decoder.generate(z_d, z_f_true) -> tests decoder given perfect z_f
  - FLOW decode  : decoder.generate(z_d, flow.sample(z_d)) -> tests flow+decoder
  - cos(sampled z_f, z_f_true)             : how close flow gets to the true latent
This tells us: retrieval vs decoder vs flow is the bottleneck, and whether
num_equiv is a false-negative source.
"""
import sys, os, gzip, json, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
import sympy as sp

from data.ast_grammar import parse_to_sympy, ID2NT, MAX_VARS
from train.train_manifold import Manifold, build_ast_edges
from train.eval_sr import (nodes_to_skeleton, fit_residual, num_equiv,
                           task_from_pair_row)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_v4_full.pt')
    ap.add_argument('--index', default='sr_model/ckpt/zf_v4_full')
    ap.add_argument('--pairs', default='dataset_20260531/data_pairs_v4_full.jsonl.gz')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--skip', type=int, default=800000)
    ap.add_argument('--K', type=int, default=40)
    ap.add_argument('--max-nv', type=int, default=99)
    ap.add_argument('--max-slen', type=int, default=999)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d).to(DEVICE); model.load_state_dict(ck['model']); model.eval()
    zf_index = torch.from_numpy(np.load(args.index + '_index.npy')).to(DEVICE)
    idx_expr = [json.loads(l)['expr'] for l in open(args.index + '_meta.jsonl', encoding='utf-8')]
    print(f"d={d} index={len(idx_expr)}\n")

    rows = []
    with gzip.open(args.pairs, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < args.skip: continue
            if not line.strip(): continue
            r = json.loads(line)
            if len(r['var_names']) > args.max_nv: continue
            if len(r['seq']) > args.max_slen: continue
            t = task_from_pair_row(r)
            if t: t['seq'] = r['seq']; rows.append(t)
            if len(rows) >= args.n: break

    def collate_one(t):
        others, X, y = t['others'], t['X'], t['y']
        mp = 128; npt = min(X.shape[0], mp); nv = len(others)
        points = torch.zeros(1, mp, MAX_VARS + 1)
        points[0, :npt, :nv] = torch.from_numpy(X[:npt].astype(np.float32))
        points[0, :npt, MAX_VARS] = torch.from_numpy(y[:npt].astype(np.float32))
        vm = torch.zeros(1, MAX_VARS); vm[0, :nv] = 1
        pm = torch.zeros(1, mp); pm[0, :npt] = 1
        return points.to(DEVICE), vm.to(DEVICE), pm.to(DEVICE)

    def zf_of_seq(seq):
        enc_types, enc_const, enc_batch, enc_edges = [], [], [], []
        for tk in seq: enc_types.append(tk); enc_const.append(0.0); enc_batch.append(0)
        for (a, b) in build_ast_edges(seq): enc_edges.append((a, b))
        ei = (torch.tensor(enc_edges).t().contiguous() if enc_edges else torch.zeros(2,0,dtype=torch.long))
        with torch.no_grad():
            z = model.formula_enc(torch.tensor(enc_types, device=DEVICE),
                                  torch.tensor(enc_const, device=DEVICE),
                                  ei.to(DEVICE), torch.tensor(enc_batch, device=DEVICE), 1)
        return torch.nn.functional.normalize(z, dim=-1)[0]

    for ti, t in enumerate(rows):
        truth = t['truth']; target = t['target']; others = t['others']
        vnf = others + [target]; symvals = t['symvals']
        pts, vm, pm = collate_one(t)
        with torch.no_grad():
            zd = torch.nn.functional.normalize(model.data_enc(pts, vm, pm), dim=-1)[0]
        zf_true = zf_of_seq(t['seq'])
        cos_dt = float(zd @ zf_true)

        print(f"=== [{ti}] truth: {truth}")
        print(f"    target={target} others={others} | cos(z_d,z_f_true)={cos_dt:.3f}")

        # retrieval
        sims = (zd @ zf_index.t())
        top = sims.topk(30)
        retr = [idx_expr[i] for i in top.indices.cpu().numpy()]
        eq_ranks = [r+1 for r, e in enumerate(retr) if num_equiv(e, truth, symvals, target)]
        print(f"    retrieval top-3: {[e[:45] for e in retr[:3]]}")
        print(f"    retrieval: equiv-to-truth at ranks {eq_ranks[:5]} (top sim={top.values[0]:.3f})")

        # ORACLE decode from true z_f
        with torch.no_grad():
            gen = model.decoder.generate(zd.unsqueeze(0), zf_true.unsqueeze(0),
                                         max_len=48, greedy=True)
        seq_o, _ = gen[0]
        e_o, params_o = nodes_to_skeleton(seq_o, vnf)
        fit_o, nm_o = fit_residual(e_o, params_o, symvals, target)
        ok_o = num_equiv(fit_o, truth, symvals, target)
        print(f"    ORACLE decode (greedy, true z_f): {str(fit_o)[:60]} | nmse={nm_o:.2e} equiv={ok_o}")

        # FLOW decode
        with torch.no_grad():
            zf_flow = model.flow.sample(zd.unsqueeze(0), k=args.K, n_steps=20)[0]
            cos_ft = float(torch.nn.functional.cosine_similarity(zf_flow, zf_true.unsqueeze(0), dim=-1).max())
            zd_rep = zd.unsqueeze(0).expand(args.K, -1)
            gens = model.decoder.generate(zd_rep, zf_flow, max_len=48, greedy=False, temperature=0.9)
        best = None; any_eq = False
        for (seq, consts) in gens:
            e, params = nodes_to_skeleton(seq, vnf)
            if e is None: continue
            fit, nm = fit_residual(e, params, symvals, target)
            if num_equiv(fit, truth, symvals, target): any_eq = True
            if best is None or nm < best[1]: best = (fit, nm)
        print(f"    FLOW max cos(z_f_samp,z_f_true)={cos_ft:.3f} | best flow cand: "
              f"{str(best[0])[:55] if best else None} nmse={best[1]:.2e} | any_equiv={any_eq}")
        print()


if __name__ == '__main__':
    main()
