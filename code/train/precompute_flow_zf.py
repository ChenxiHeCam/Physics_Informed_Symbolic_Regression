"""Precompute the flow's z_f sample for each training row (one-time, batched).

The decoder-robustness fine-tune needs, per formula, the IMPRECISE z_f the flow
actually produces from that formula's data — but calling flow.sample() every training
step is what made the earlier attempt ~25h. Here we batch it once: data -> tokens ->
flow.sample(k=1) -> flow_zf[row], saved to a memmap the fine-tune reads instantly.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch
from train.train_manifold import Manifold, MemmapPairDataset, load_state_compat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/flow_v6_noaug.pt')
    ap.add_argument('--cache', default='dataset_20260531/cache_v6')
    ap.add_argument('--max-rows', type=int, default=300000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--out', default='dataset_20260531/flow_zf_v6noaug.npy')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64),
                     n_ftokens=ck.get('n_ftokens', 0), log_feats=ck.get('log_feats', False)).to(DEVICE)
    load_state_compat(model, ck['model']); model.eval()
    ds = MemmapPairDataset(args.cache, args.max_rows, max_points=128, max_seq=ck.get('dec_max_len', 128))
    N = len(ds)
    out = np.lib.format.open_memmap(args.out, mode='w+', dtype=np.float32, shape=(N, d))
    print(f"precomputing flow z_f for {N} rows -> {args.out} (d={d})")

    import time; t0 = time.time()
    for s in range(0, N, args.batch):
        idxs = list(range(s, min(s + args.batch, N)))
        b = ds.collate(idxs); b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            if model.tok:
                _, toks = model.data_enc(b['points'], b['var_mask'], b['point_mask'],
                                         dims=b.get('dims'), return_tokens=True)
                zf = model.flow.sample(toks, k=1, n_steps=20)[:, 0]      # (B,d)
            else:
                zd = model.data_enc(b['points'], b['var_mask'], b['point_mask'], dims=b.get('dims'))
                zf = model.flow.sample(zd, k=1, n_steps=20)[:, 0]
        out[s:s + len(idxs)] = zf.float().cpu().numpy()
        if s % (args.batch * 200) == 0:
            print(f"  {s}/{N} ({time.time()-t0:.0f}s)", flush=True)
    out.flush()
    print(f"done -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
