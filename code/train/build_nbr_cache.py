"""Build retrieval-neighbour ids for every cache row: encode each row's data z_d
with the manifold, knn into the zf index (normalized formula latents) -> (N,K) int32
ids into zf_raw. Used by train_dec7 / train_zf_diffusion (--neighbors) for the
retrieval-conditioning channel.

Usage:
  python build_nbr_cache.py --ckpt sr_model/ckpt/manifold_v8.pt \
      --cache dataset_20260531/cache_union --index sr_model/ckpt/zf_v8 \
      --k 8 --out sr_model/ckpt/nbr_v8.npy
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from train.train_manifold import Manifold, MemmapPairDataset, load_state_compat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--index', required=True)          # prefix -> <prefix>_index.npy (normalized)
    ap.add_argument('--max-rows', type=int, default=10_000_000)
    ap.add_argument('--max-points', type=int, default=128)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64),
                     n_ftokens=ck.get('n_ftokens', 0), log_feats=ck.get('log_feats', False),
                     class_feats=ck.get('class_feats', False),
                     robust_norm=ck.get('robust_norm', False)).to(DEVICE)
    load_state_compat(model, ck['model']); model.eval()
    print(f"loaded manifold d={d}", flush=True)

    zf_index = torch.from_numpy(np.load(args.index + '_index.npy')).to(DEVICE)  # (M,d) normalized
    M = zf_index.size(0)
    print(f"index: {M} formulas", flush=True)

    ds = MemmapPairDataset(args.cache, args.max_rows, max_points=args.max_points)
    N = len(ds)
    out = np.zeros((N, args.k), dtype=np.int32)
    t0 = time.time()
    with torch.no_grad():
        for bs in range(0, N, args.batch):
            idxs = list(range(bs, min(bs + args.batch, N)))
            b = ds.collate(idxs)
            b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
            if model.tok:
                z_d, _ = model.data_enc(b['points'], b['var_mask'], b['point_mask'],
                                        dims=b.get('dims'), return_tokens=True)
            else:
                z_d = model.data_enc(b['points'], b['var_mask'], b['point_mask'],
                                     dims=b.get('dims'))
            zn = torch.nn.functional.normalize(z_d, dim=-1)
            ids = (zn @ zf_index.t()).topk(args.k, dim=-1).indices       # (B,k)
            out[bs:bs + len(idxs)] = ids.cpu().numpy().astype(np.int32)
            if bs % (args.batch * 50) == 0:
                el = time.time() - t0
                print(f"  {bs}/{N} ({el:.0f}s, {bs/max(el,1):.0f}/s)", flush=True)
    np.save(args.out, out)
    print(f"saved {out.shape} -> {args.out} ({out.nbytes/1e6:.0f}MB) time={time.time()-t0:.0f}s",
          flush=True)


if __name__ == '__main__':
    main()
