"""z_f intrinsic-dimension probe.

Encode a sample of training formulas through the frozen v6 formula encoder, then
PCA the z_f cloud. If 768-dim z_f lives in a ~k-dim subspace (k<<768), that is the
mechanism behind the flow's poor samp_cos: the flow generates in 768-D but must hit
a low-D manifold, wasting most directions. Reports cumulative explained variance
and the participation ratio (effective dim).
"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
from train.train_manifold import Manifold, MemmapPairDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'sr_model/ckpt/manifold_v6_fixed.pt'
CACHE = 'dataset_20260531/cache_v6'
N = 16000
BATCH = 256


def main():
    ck = torch.load(CKPT, map_location=DEVICE)
    d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6),
                     dec_max_len=ck.get('dec_max_len', 64)).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    ds = MemmapPairDataset(CACHE, 0, max_points=128, max_seq=ck.get('dec_max_len', 128))
    idxs = random.sample(range(len(ds)), min(N, len(ds)))
    zs = []
    with torch.no_grad():
        for s in range(0, len(idxs), BATCH):
            b = ds.collate(idxs[s:s + BATCH])
            b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
            zf = model.formula_enc(b['enc_types'], b['enc_const'], b['enc_edges'],
                                   b['enc_batch'], b['n_graphs'])
            zs.append(zf.float().cpu().numpy())
    Z = np.concatenate(zs, 0)
    print(f"z_f sample: {Z.shape}  (dim={d})")
    # center, SVD
    Zc = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    var = s ** 2
    cum = np.cumsum(var) / var.sum()
    pr = (var.sum() ** 2) / (var ** 2).sum()       # participation ratio = effective dim
    print(f"participation ratio (effective dim): {pr:.1f} / {d}")
    print(f"raw z_f norm: mean={np.linalg.norm(Z,axis=1).mean():.3f} std={np.linalg.norm(Z,axis=1).std():.3f}")
    for k in [16, 32, 64, 96, 128, 192, 256, 384, 512]:
        if k <= len(cum):
            print(f"  top-{k:4d} dims explain {100*cum[k-1]:.1f}% variance")
    # dims to reach thresholds
    for thr in (0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, thr) + 1)
        print(f"  {int(thr*100)}% variance needs {k} dims")


if __name__ == '__main__':
    main()
