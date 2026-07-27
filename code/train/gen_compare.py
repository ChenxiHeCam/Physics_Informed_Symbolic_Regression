"""
Pinpoint the free-generation failure: for simple training rows, compare
  TRUE seq   vs   teacher-forced argmax (given true prefix)   vs   generate().
If TF-argmax == TRUE but generate diverges -> exposure bias / unconstrained
decoding (the grammar mask permits out-of-range VAR slots). Fixable.
"""
import sys, os, gzip, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch
from train.train_manifold import Manifold, build_ast_edges
from train.eval_sr import task_from_pair_row
from models.decoder import EOS_ID
from data.ast_grammar import ID2NT, MAX_VARS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_v4_full.pt')
    ap.add_argument('--pairs', default='dataset_20260531/data_pairs_v4_full.jsonl.gz')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--max-nv', type=int, default=3)
    ap.add_argument('--max-slen', type=int, default=14)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d).to(DEVICE); model.load_state_dict(ck['model']); model.eval()

    rows = []
    with gzip.open(args.pairs, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if len(r['var_names']) > args.max_nv or len(r['seq']) > args.max_slen: continue
            t = task_from_pair_row(r)
            if t: t['seq'] = r['seq']; rows.append(t)
            if len(rows) >= args.n: break

    def zf_of_seq(seq):
        et, ec, eb, ee = [], [], [], []
        for tk in seq: et.append(tk); ec.append(0.0); eb.append(0)
        for (a, b) in build_ast_edges(seq): ee.append((a, b))
        ei = torch.tensor(ee).t().contiguous() if ee else torch.zeros(2,0,dtype=torch.long)
        with torch.no_grad():
            z = model.formula_enc(torch.tensor(et,device=DEVICE), torch.tensor(ec,device=DEVICE),
                                  ei.to(DEVICE), torch.tensor(eb,device=DEVICE), 1)
        return z   # RAW (decoder was trained on un-normalized z_f)

    for t in rows:
        seq = t['seq']; others = t['others']; nv = len(others)
        X, y = t['X'], t['y']
        mp = 128; npt = min(X.shape[0], mp)
        pts = torch.zeros(1, mp, MAX_VARS+1)
        pts[0,:npt,:nv] = torch.from_numpy(X[:npt].astype(np.float32))
        pts[0,:npt,MAX_VARS] = torch.from_numpy(y[:npt].astype(np.float32))
        vm = torch.zeros(1,MAX_VARS); vm[0,:nv]=1
        pm = torch.zeros(1,mp); pm[0,:npt]=1
        with torch.no_grad():
            zd = model.data_enc(pts.to(DEVICE), vm.to(DEVICE), pm.to(DEVICE))  # RAW
        zf = zf_of_seq(seq)
        # teacher-forced argmax
        tgt = torch.tensor(seq, device=DEVICE).unsqueeze(0)
        bos = torch.full((1,1), EOS_ID, device=DEVICE)
        inp = torch.cat([bos, tgt[:,:-1]], dim=1)
        inpc = torch.zeros_like(inp, dtype=torch.float)
        with torch.no_grad():
            logits, _ = model.decoder(zd, zf, inp, inpc)
            tf = logits.argmax(-1)[0].cpu().tolist()
        # generate (greedy)
        with torch.no_grad():
            g = model.decoder.generate(zd, zf, max_len=len(seq)+4, greedy=True)
        gen = g[0][0]
        print(f"truth: {t['truth']}  (nv={nv}, valid VAR_0..VAR_{nv})")
        print(f"  TRUE seq : {[ID2NT[i] for i in seq]}")
        print(f"  TF argmax: {[ID2NT[i] for i in tf]}")
        print(f"  GENERATE : {[ID2NT[i] for i in gen]}")
        print()


if __name__ == '__main__':
    main()
