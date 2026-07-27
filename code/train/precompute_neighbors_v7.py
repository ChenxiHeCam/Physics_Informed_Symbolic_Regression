"""Precompute per-cache-row retrieval neighbours for decoder v7 conditioning:
frozen full512 data encoder -> z_d -> top-K cosine against the zf index.
Saves (N, K) int32 ids into the index/raw matrices.

Usage:
  python precompute_neighbors_v7.py --cache <dir> --base <full512.pt> \
      --index sr_model/ckpt/zf_full512_index.npy --out nbr_v7.npy --max-rows 2000000
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from train.train_manifold import Manifold, MemmapPairDataset, load_state_compat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True)
ap.add_argument('--base', required=True)
ap.add_argument('--index', required=True)        # normalized index .npy
ap.add_argument('--out', required=True)
ap.add_argument('--max-rows', type=int, default=2000000)
ap.add_argument('--max-points', type=int, default=128)
ap.add_argument('--batch', type=int, default=512)
ap.add_argument('--k', type=int, default=8)
args = ap.parse_args()

ck = torch.load(args.base, map_location=DEVICE)
d = ck['d']
base = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64),
                n_ftokens=ck.get('n_ftokens', 0), log_feats=ck.get('log_feats', False),
                class_feats=ck.get('class_feats', False), dim_head=ck.get('dim_head', False),
                robust_norm=ck.get('robust_norm', False)).to(DEVICE)
load_state_compat(base, ck['model'])
base.eval()

zf_index = torch.from_numpy(np.load(args.index)).to(DEVICE)        # (M,d) normalized
ds = MemmapPairDataset(args.cache, args.max_rows, max_points=args.max_points)
N = len(ds)
out = np.zeros((N, args.k), np.int32)
t0 = time.time()
with torch.no_grad():
    for s in range(0, N, args.batch):
        idxs = list(range(s, min(s + args.batch, N)))
        b = ds.collate(idxs)
        dims = b.get('dims')
        zd = base.data_enc(b['points'].to(DEVICE), b['var_mask'].to(DEVICE),
                           b['point_mask'].to(DEVICE),
                           dims=dims.to(DEVICE) if dims is not None else None)
        zn = torch.nn.functional.normalize(zd, dim=-1)
        top = (zn @ zf_index.t()).topk(args.k, dim=-1).indices
        out[s:s + len(idxs)] = top.cpu().numpy().astype(np.int32)
        if (s // args.batch) % 100 == 0:
            print(f'  {s}/{N} ({time.time()-t0:.0f}s)', flush=True)
np.save(args.out, out)
print(f'saved {out.shape} -> {args.out}')
