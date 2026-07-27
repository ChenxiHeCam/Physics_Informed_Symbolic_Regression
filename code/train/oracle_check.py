"""
Oracle ceiling + false-negative audit on held-out real591.

For each usable held-out task:
  z_f_true = formula_encoder(expr_to_nodes(truth, var_order=others+[target]))   [RAW]
  z_d      = data_encoder(X,y)                                                  [RAW]
  decode greedily from (z_d, z_f_true) -> skeleton -> fit consts -> fitted
  check num_equiv(fitted, truth)

If the model reconstructs the truth from the TRUE z_f but num_equiv says False,
that is an eval false-negative (printed for manual inspection). The oracle hit
rate is the upper bound the flow can chase (decoder is near-perfect given z_f).
"""
import sys, os, json, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch, sympy as sp
from data.ast_grammar import parse_to_sympy, expr_to_nodes, ID2NT, MAX_VARS
from train.train_manifold import Manifold, build_ast_edges, load_state_compat
from train.eval_sr import nodes_to_skeleton, fit_residual, num_equiv, _is_leaked, dims_for_task
from models.decoder import EOS_ID, CONST_ID

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_v4_full.pt')
    ap.add_argument('--real', default='data/real591.jsonl')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--show', type=int, default=20)
    ap.add_argument('--clean', action='store_true')   # exclude leaked truths
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64),
                     n_ftokens=ck.get('n_ftokens', 0), log_feats=ck.get('log_feats', False)).to(DEVICE)
    load_state_compat(model, ck['model']); model.eval()

    def zf_of_seq(seq):
        et, ec, eb, ee = [], [], [], []
        for tk in seq: et.append(tk); ec.append(0.0); eb.append(0)
        for (a, b) in build_ast_edges(seq): ee.append((a, b))
        ei = torch.tensor(ee).t().contiguous() if ee else torch.zeros(2,0,dtype=torch.long)
        with torch.no_grad():
            args5 = (torch.tensor(et,device=DEVICE), torch.tensor(ec,device=DEVICE),
                     ei.to(DEVICE), torch.tensor(eb,device=DEVICE), 1)
            if getattr(model, 'n_ftokens', 0) > 0:
                return model.formula_enc(*args5, return_tokens=True)   # (z_f, z_f_tokens) RAW
            return model.formula_enc(*args5), None   # RAW

    rows = []
    with open(args.real, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            tv = r.get('train_values', {}); syms = r.get('symbols', [])
            truth = r.get('eval_truth_surface', '')
            if not tv or not truth: continue
            te = parse_to_sympy(truth)
            if te is None: continue
            if args.clean and _is_leaked(truth): continue          # exclude leaked
            free = [str(s) for s in te.free_symbols]
            usable = [s for s in syms if s in tv and s in free]
            if len(usable) < 2: continue
            if not set(free).issubset(set(usable)): continue      # fully observed only
            target = usable[0]
            others = sorted([s for s in usable if s != target])
            if len(others) + 1 > MAX_VARS: continue
            try:
                X = np.array([tv[v] for v in others], float).T
                y = np.array(tv[target], float)
            except Exception:
                continue
            if X.shape[0] < 10: continue
            sv = {others[i]: X[:, i] for i in range(len(others))}; sv[target] = y
            rows.append({'truth': truth, 'target': target, 'others': others,
                         'X': X, 'y': y, 'symvals': sv})
            if len(rows) >= args.n: break
    print(f"usable held-out tasks: {len(rows)}\n")

    oracle_hit = 0; recon_struct = 0; recon_tf = 0; shown = 0; n = 0
    for t in rows:
        truth = t['truth']; target = t['target']; others = t['others']
        vnf = others + [target]; sv = t['symvals']; X, y = t['X'], t['y']
        res = expr_to_nodes(truth, var_order=vnf)
        if res is None: continue
        seq_true = res[0]
        zf, zftok = zf_of_seq(seq_true)
        mp = 128; npt = min(X.shape[0], mp); nv = len(others)
        pts = torch.zeros(1, mp, MAX_VARS+1)
        pts[0,:npt,:nv] = torch.from_numpy(X[:npt].astype(np.float32))
        pts[0,:npt,MAX_VARS] = torch.from_numpy(y[:npt].astype(np.float32))
        vm = torch.zeros(1,MAX_VARS); vm[0,:nv]=1
        pm = torch.zeros(1,mp); pm[0,:npt]=1
        dims = None
        if getattr(model.data_enc, 'dim_len', 0):
            dd = dims_for_task(others, target, model.data_enc.dim_len)
            dims = torch.from_numpy(dd).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mlen = getattr(model, 'dec_max_len', 48)
            if model.tok:
                zd, tok = model.data_enc(pts.to(DEVICE), vm.to(DEVICE), pm.to(DEVICE),
                                         dims=dims, return_tokens=True)
                gen = model.decoder.generate(zd, zf, data_tokens=tok, z_f_tokens=zftok,
                                             max_len=mlen, greedy=True)
            else:
                zd = model.data_enc(pts.to(DEVICE), vm.to(DEVICE), pm.to(DEVICE), dims=dims)  # RAW
                gen = model.decoder.generate(zd, zf, max_len=mlen, greedy=True)
        seq_o = gen[0][0]
        struct_ok = (seq_o == seq_true[:len(seq_o)])
        if struct_ok: recon_struct += 1
        # TEACHER-FORCED exact: feed true prefix, argmax-predict each next node.
        # TF>>free => exposure bias (decoding problem); TF~=free => z_f capacity limit.
        # Only on truths representable in the 48-token window (longer ones can't be
        # produced by free-greedy either, so they'd fail both equally).
        L = len(seq_true)
        tf_ok = False
        if L <= getattr(model, 'dec_max_len', 48):
            with torch.no_grad():
                bos = torch.full((1, 1), EOS_ID, dtype=torch.long, device=DEVICE)
                inp_t = torch.cat([bos, torch.tensor(seq_true[:-1], device=DEVICE).unsqueeze(0)], dim=1)
                inp_c = torch.zeros(1, L, device=DEVICE)
                if model.tok:
                    logits_tf, _ = model.decoder(None, zf, inp_t, inp_c, data_tokens=tok, z_f_tokens=zftok)
                else:
                    logits_tf, _ = model.decoder(zd, zf, inp_t, inp_c)
                pred_tf = logits_tf[0].argmax(-1).tolist()
            tf_ok = (pred_tf == seq_true)
        if tf_ok: recon_tf += 1
        e_o, params_o = nodes_to_skeleton(seq_o, vnf)
        # use the decoder's PREDICTED constant values (const_head) as the fit init —
        # the model already estimates exponents/coeffs; seeding the optimizer there
        # lands the right local min more often (esp. exponents).
        cons_o = gen[0][1]
        c_init = [cons_o[i] for i in range(len(seq_o)) if seq_o[i] == CONST_ID]
        fit_o, nm_o = fit_residual(e_o, params_o, sv, target,
                                   c_init=(c_init if len(c_init) == len(params_o) else None))
        ok = num_equiv(fit_o, truth, sv, target)
        if ok: oracle_hit += 1
        n += 1
        # show cases, especially candidate structurally-right but num_equiv False
        if shown < args.show and (struct_ok or not ok):
            flag = "FALSE-NEG?" if (struct_ok and not ok) else ("HIT" if ok else "miss")
            print(f"[{flag}] truth: {truth[:70]}")
            print(f"   oracle decode: {str(fit_o)[:70]} | nmse={nm_o:.2e} struct_ok={struct_ok} equiv={ok}")
            shown += 1
    print(f"\n=== oracle (true z_f) on {n} held-out ===")
    print(f"teacher-forced exact:      {recon_tf}/{n} = {100*recon_tf/max(n,1):.1f}%   (decoder capacity given true prefix)")
    print(f"free-greedy structural:    {recon_struct}/{n} = {100*recon_struct/max(n,1):.1f}%   (oracle ceiling)")
    print(f"num_equiv oracle hit:      {oracle_hit}/{n} = {100*oracle_hit/max(n,1):.1f}%")
    gap = 100*(recon_tf - recon_struct)/max(n,1)
    print(f"-> TF-vs-free gap = {gap:.1f}pp  ({'EXPOSURE BIAS (decoding lever)' if gap > 10 else 'z_f CAPACITY (encoder/latent lever)'})")


if __name__ == '__main__':
    main()
