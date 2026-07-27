"""Clean oracle probe for the dual-z_f (n_ftokens) model, using the validated
tasks_from_real591 loader (oracle_check's own train_values loader mis-conditions the
multi-source v8 decoder). Encode truth -> z_f + z_f_tokens, encode data -> tokens,
decode -> structural + root-match equiv oracle."""
import sys, os, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch
from data.ast_grammar import expr_to_nodes, MAX_VARS
from train.train_manifold import Manifold, build_ast_edges, load_state_compat
from train.eval_sr import (tasks_from_real591, nodes_to_skeleton, fit_residual,
                           is_correct, dims_for_task)
from models.decoder import CONST_ID

DEVICE = torch.device('cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--real', default='data/real591.jsonl')
    ap.add_argument('--n', type=int, default=80)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    nft = ck.get('n_ftokens', 0)
    m = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False), n_tokens=ck.get('n_tokens', 16),
                 n_gin=ck.get('n_gin', 4), dec_layers=ck.get('dec_layers', 6),
                 dec_max_len=ck.get('dec_max_len', 64), n_ftokens=nft,
                 log_feats=ck.get('log_feats', False)).to(DEVICE)
    load_state_compat(m, ck['model']); m.eval()
    mlen = min(getattr(m, 'dec_max_len', 96), 96)
    rows = tasks_from_real591(args.real, args.n)
    print(f"v8 oracle: ckpt step {ck.get('step','?')} n_ftokens={nft} tasks={len(rows)}")

    struct = 0; equiv = 0; n = 0
    for t in rows:
        vnf = t['others'] + [t['target']]; sv = t['symvals']
        res = expr_to_nodes(t['truth'], var_order=vnf)
        if res is None: continue
        seq_true = res[0]; L = len(seq_true)
        ee = build_ast_edges(seq_true); ei = torch.tensor(ee).t().contiguous() if ee else torch.zeros(2, 0, dtype=torch.long)
        with torch.no_grad():
            if nft > 0:
                zf, zftok = m.formula_enc(torch.tensor(seq_true), torch.zeros(L), ei, torch.zeros(L, dtype=torch.long), 1, return_tokens=True)
            else:
                zf = m.formula_enc(torch.tensor(seq_true), torch.zeros(L), ei, torch.zeros(L, dtype=torch.long), 1); zftok = None
            X, y = t['X'], t['y']; mp = 128; npt = min(X.shape[0], mp); nv = len(t['others'])
            pts = torch.zeros(1, mp, MAX_VARS + 1)
            pts[0, :npt, :nv] = torch.from_numpy(X[:npt].astype(np.float32)); pts[0, :npt, MAX_VARS] = torch.from_numpy(y[:npt].astype(np.float32))
            vm = torch.zeros(1, MAX_VARS); vm[0, :nv] = 1; pm = torch.zeros(1, mp); pm[0, :npt] = 1
            dims = None
            if getattr(m.data_enc, 'dim_len', 0):
                dims = torch.from_numpy(dims_for_task(t['others'], t['target'], m.data_enc.dim_len)).unsqueeze(0)
            zd, tok = m.data_enc(pts, vm, pm, dims=dims, return_tokens=True)
            gen = m.decoder.generate(zd, zf, data_tokens=tok, z_f_tokens=zftok, max_len=mlen, greedy=True)
        seq_o = gen[0][0]; n += 1
        if len(seq_o) >= L - 1 and seq_o == seq_true[:len(seq_o)]:
            struct += 1
        e_o, params_o = nodes_to_skeleton(seq_o, vnf)
        if e_o is not None:
            cons = gen[0][1]; ci = [cons[i] for i in range(len(seq_o)) if seq_o[i] == CONST_ID]
            fit, _ = fit_residual(e_o, params_o, sv, t['target'], c_init=(ci if len(ci) == len(params_o) else None))
            if is_correct(fit, t['truth'], sv, t['target']):
                equiv += 1
    print(f"structural oracle: {struct}/{n} = {100*struct/max(n,1):.1f}%")
    print(f"root-match equiv:  {equiv}/{n} = {100*equiv/max(n,1):.1f}%")


if __name__ == '__main__':
    main()
