"""Oracle-via-PCA: how much oracle do we lose if z_f is truncated to k PCA dims?

Fit PCA on the z_f cloud, then for each held-out truth decode from the PCA-k
RECONSTRUCTED true z_f (mu + project-to-k + lift-back). If oracle holds at k, the
tail dims beyond k carry no decode-relevant info -> a k-dim flow target loses
nothing. Where oracle starts dropping = the floor on how far we can reduce.

Uses the validated root-match is_correct for the equivalence metric.
"""
import sys, os, json, random, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch
from data.ast_grammar import parse_to_sympy, expr_to_nodes, MAX_VARS
from train.train_manifold import Manifold, build_ast_edges, MemmapPairDataset
from train.eval_sr import nodes_to_skeleton, fit_residual, is_correct, _is_leaked, dims_for_task
from models.decoder import CONST_ID

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'sr_model/ckpt/manifold_v6_fixed.pt'
CACHE = 'dataset_20260531/cache_v6'
REAL = 'data/real591.jsonl'
KS = [32, 64, 96, 128, 192, 256, 384, 768]
N_FIT = 16000
N_EVAL = 80


def main():
    ck = torch.load(CKPT, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64)).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    mlen = getattr(model, 'dec_max_len', 48)

    # 1) fit PCA on a z_f sample
    ds = MemmapPairDataset(CACHE, 0, max_points=128, max_seq=mlen)
    idxs = random.sample(range(len(ds)), min(N_FIT, len(ds)))
    zs = []
    with torch.no_grad():
        for s in range(0, len(idxs), 256):
            b = ds.collate(idxs[s:s+256]); b = {k:(v.to(DEVICE) if torch.is_tensor(v) else v) for k,v in b.items()}
            zs.append(model.formula_enc(b['enc_types'], b['enc_const'], b['enc_edges'],
                                        b['enc_batch'], b['n_graphs']).float().cpu().numpy())
    Z = np.concatenate(zs, 0)
    mu = Z.mean(0)
    _, _, Vt = np.linalg.svd(Z - mu, full_matrices=False)   # Vt: (d,d) components
    muT = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    VtT = torch.tensor(Vt, dtype=torch.float32, device=DEVICE)
    np.savez('dataset_20260531/zf_pca_v6.npz', mu=mu, Vt=Vt)
    print(f"PCA fit on {Z.shape}; saved basis -> dataset_20260531/zf_pca_v6.npz\n")

    def reconstruct_k(zf, k):
        if k >= d: return zf
        c = (zf - muT) @ VtT[:k].t()      # (1,k) coords
        return muT + c @ VtT[:k]          # (1,d) lifted back

    def zf_of_seq(seq):
        et=[tk for tk in seq]; ec=[0.0]*len(seq); eb=[0]*len(seq)
        ee=build_ast_edges(seq)
        ei=torch.tensor(ee).t().contiguous() if ee else torch.zeros(2,0,dtype=torch.long)
        with torch.no_grad():
            return model.formula_enc(torch.tensor(et,device=DEVICE), torch.tensor(ec,device=DEVICE),
                                     ei.to(DEVICE), torch.tensor(eb,device=DEVICE), 1)

    # 2) load held-out tasks (clean)
    rows = []
    with open(REAL, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            tv = r.get('train_values', {}); syms = r.get('symbols', []); truth = r.get('eval_truth_surface', '')
            if not tv or not truth: continue
            te = parse_to_sympy(truth)
            if te is None or _is_leaked(truth): continue
            free=[str(s) for s in te.free_symbols]; usable=[s for s in syms if s in tv and s in free]
            if len(usable) < 2 or not set(free).issubset(set(usable)): continue
            target=usable[0]; others=sorted([s for s in usable if s!=target])
            if len(others)+1 > MAX_VARS: continue
            try:
                X=np.array([tv[v] for v in others],float).T; y=np.array(tv[target],float)
            except Exception: continue
            if X.shape[0] < 10: continue
            sv={others[i]:X[:,i] for i in range(len(others))}; sv[target]=y
            rows.append({'truth':truth,'target':target,'others':others,'X':X,'y':y,'symvals':sv})
            if len(rows) >= N_EVAL: break
    print(f"held-out tasks: {len(rows)}\n")

    struct = {k: 0 for k in KS}; equiv = {k: 0 for k in KS}; n = 0
    for t in rows:
        truth=t['truth']; target=t['target']; others=t['others']; vnf=others+[target]; sv=t['symvals']
        X,y=t['X'],t['y']
        res=expr_to_nodes(truth, var_order=vnf)
        if res is None: continue
        seq_true=res[0]; zf_full=zf_of_seq(seq_true)
        mp=128; npt=min(X.shape[0],mp); nv=len(others)
        pts=torch.zeros(1,mp,MAX_VARS+1); pts[0,:npt,:nv]=torch.from_numpy(X[:npt].astype(np.float32))
        pts[0,:npt,MAX_VARS]=torch.from_numpy(y[:npt].astype(np.float32))
        vm=torch.zeros(1,MAX_VARS); vm[0,:nv]=1; pm=torch.zeros(1,mp); pm[0,:npt]=1
        dims=None
        if getattr(model.data_enc,'dim_len',0):
            dims=torch.from_numpy(dims_for_task(others,target,model.data_enc.dim_len)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if model.tok:
                zd,tok=model.data_enc(pts.to(DEVICE),vm.to(DEVICE),pm.to(DEVICE),dims=dims,return_tokens=True)
            else:
                zd=model.data_enc(pts.to(DEVICE),vm.to(DEVICE),pm.to(DEVICE),dims=dims); tok=None
        n+=1
        for k in KS:
            zf=reconstruct_k(zf_full,k)
            with torch.no_grad():
                if model.tok:
                    gen=model.decoder.generate(zd,zf,data_tokens=tok,max_len=mlen,greedy=True)
                else:
                    gen=model.decoder.generate(zd,zf,max_len=mlen,greedy=True)
            seq_o=gen[0][0]
            if seq_o==seq_true[:len(seq_o)]: struct[k]+=1
            e_o,params_o=nodes_to_skeleton(seq_o,vnf)
            cons_o=gen[0][1]; c_init=[cons_o[i] for i in range(len(seq_o)) if seq_o[i]==CONST_ID]
            fit_o,_=fit_residual(e_o,params_o,sv,target,c_init=(c_init if len(c_init)==len(params_o) else None))
            if is_correct(fit_o,truth,sv,target): equiv[k]+=1
    print(f"=== oracle-via-PCA on {n} held-out (structural / root-match equiv) ===")
    print(f"{'k':>5} {'structural':>12} {'equiv':>12}")
    for k in KS:
        print(f"{k:>5} {100*struct[k]/max(n,1):>11.1f}% {100*equiv[k]/max(n,1):>11.1f}%")


if __name__ == '__main__':
    main()
