"""
Train Flow 2 = retrieval-conditioned flow p(z_f | z_d, z_hint).

Frozen manifold encoders. For each batch we form a HINT for every sample as its
in-batch nearest neighbour by cos(z_d_i, z_f_j) (j != i) — i.e. "the most similar
OTHER formula's z_f". The flow learns to REFINE that hint into the true z_f given
the data z_d. At inference the hint comes from the real z_f index (nearest
training formula to the held-out z_d), so the flow starts near a known region and
only has to move to the (possibly novel) target — breaking the from-scratch flow's
train-distribution ceiling.
"""
import sys, os, time, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.train_manifold import Manifold, MemmapPairDataset
from models.flow_matching import FlowMatching2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_v4_units.pt')
    ap.add_argument('--cache', default='dataset_20260531/cache_v4_dims')
    ap.add_argument('--max-rows', type=int, default=2000000)
    ap.add_argument('--batch', type=int, default=384)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--noise', type=float, default=0.03)
    ap.add_argument('--balanced', action='store_true')
    ap.add_argument('--save', default='sr_model/ckpt/flow2_v4_units.pt')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0)).to(DEVICE)
    model.load_state_dict(ck['model'])
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()

    flow2 = FlowMatching2(d=d, n_blocks=6).to(DEVICE); flow2.train()
    ds = MemmapPairDataset(args.cache, args.max_rows)
    opt = torch.optim.AdamW(flow2.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    print(f"flow2 params {sum(p.numel() for p in flow2.parameters()):,} | d={d} dim_len={ck.get('dim_len',0)}")

    sample_w = None
    if args.balanced:
        nv = np.asarray(ds.nv[:len(ds)], np.int64); sl = np.asarray(ds.slen[:len(ds)], np.int64)//8
        key = nv*100+sl; u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        w = 1.0/cnt[inv]; sample_w = w/w.sum(); print(f"balanced over {len(u)} bins")

    t0 = time.time()
    for step in range(1, args.steps+1):
        if sample_w is not None:
            idxs = np.random.choice(len(ds), min(args.batch, len(ds)), replace=True, p=sample_w).tolist()
        else:
            idxs = random.sample(range(len(ds)), min(args.batch, len(ds)))
        b = ds.collate(idxs)
        b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            z_d = model.data_enc(b['points'], b['var_mask'], b['point_mask'], dims=b.get('dims'))
            z_f = model.formula_enc(b['enc_types'], b['enc_const'], b['enc_edges'],
                                    b['enc_batch'], b['n_graphs'])
            # in-batch nearest neighbour hint (exclude self)
            zd_n = F.normalize(z_d, dim=-1); zf_n = F.normalize(z_f, dim=-1)
            sim = zd_n @ zf_n.t()
            sim.fill_diagonal_(-1e4)
            j = sim.argmax(1)
            z_hint = z_f[j]                         # raw z_f of nearest neighbour
            if args.noise > 0:
                z_d = z_d + args.noise * torch.randn_like(z_d)
        loss = flow2.loss(z_f, z_d, z_hint)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(flow2.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                samp = flow2.sample(z_d[:64], z_hint[:64], k=1, n_steps=20).squeeze(1)
                cos = F.cosine_similarity(samp, z_f[:64], dim=-1).mean()
                cos_hint = F.cosine_similarity(z_hint[:64], z_f[:64], dim=-1).mean()
            print(f"step {step:5d} loss={loss.item():.4f} samp_cos={cos.item():.3f} "
                  f"(hint_cos={cos_hint.item():.3f}) ({time.time()-t0:.0f}s)", flush=True)

    torch.save({'flow2': flow2.state_dict(), 'd': d, 'dim_len': ck.get('dim_len', 0),
                'base_ckpt': args.ckpt}, args.save)
    print(f"Saved -> {args.save}")


if __name__ == '__main__':
    main()
