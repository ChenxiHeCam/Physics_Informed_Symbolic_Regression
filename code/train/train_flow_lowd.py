"""Phase B-dim: train a LOW-DIM flow on PCA coords of z_f.

Frozen manifold => z_f fixed. Project z_f to k PCA dims (basis from zf_pca_v6.npz),
train a FlowMatchingTokLowD to generate those k coords conditioned on data tokens.
The k-dim target is ~3x smaller than 768 -> easier to hit precisely. Eval lifts the
sampled coords back to z_f (mu + coords @ Vt[:k]) for the frozen decoder.

Reports samp_cos in BOTH coord space and lifted-z_f space (the latter is what the
decoder sees and what the old 768-d flow scored ~0.63 on).
"""
import sys, os, time, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch, torch.nn as nn
from train.train_manifold import Manifold, MemmapPairDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_v6_fixed.pt')
    ap.add_argument('--cache', default='dataset_20260531/cache_v6')
    ap.add_argument('--pca', default='dataset_20260531/zf_pca_v6.npz')
    ap.add_argument('--k', type=int, default=256)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--noise', type=float, default=0.03)
    ap.add_argument('--save', default='sr_model/ckpt/flow_v6_lowd256.pt')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64)).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    pca = np.load(args.pca)
    mu = torch.tensor(pca['mu'], dtype=torch.float32, device=DEVICE)             # (d,)
    Vk = torch.tensor(pca['Vt'][:args.k], dtype=torch.float32, device=DEVICE)    # (k,d)
    print(f"PCA basis: mu{tuple(mu.shape)} Vk{tuple(Vk.shape)} k={args.k}")

    from models.flow_matching import FlowMatchingTokLowD
    flow = FlowMatchingTokLowD(d=d, d_z=args.k, n_blocks=6).to(DEVICE)
    flow.train()
    print(f"low-d flow params {sum(p.numel() for p in flow.parameters()):,}  device={DEVICE}")

    ds = MemmapPairDataset(args.cache, 0, max_points=128, max_seq=ck.get('dec_max_len', 128))
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def to_code(z_f):
        return (z_f - mu) @ Vk.t()        # (B,k)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        idxs = random.sample(range(len(ds)), min(args.batch, len(ds)))
        b = ds.collate(idxs); b = {k:(v.to(DEVICE) if torch.is_tensor(v) else v) for k,v in b.items()}
        with torch.no_grad():
            z_d, cond = model.data_enc(b['points'], b['var_mask'], b['point_mask'],
                                       dims=b.get('dims'), return_tokens=True)   # cond=tokens
            z_f = model.formula_enc(b['enc_types'], b['enc_const'], b['enc_edges'],
                                    b['enc_batch'], b['n_graphs'])
            code = to_code(z_f)
            if args.noise > 0:
                cond = cond + args.noise * torch.randn_like(cond)
        loss = flow.loss(code, cond)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 1.0); opt.step(); sched.step()
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                samp = flow.sample(cond[:64], k=1, n_steps=20).squeeze(1)        # (64,k)
                cos_code = torch.nn.functional.cosine_similarity(samp, code[:64], dim=-1).mean()
                # lift back to z_f and measure cosine in the space the decoder sees
                zf_samp = mu + samp @ Vk; zf_true = mu + code[:64] @ Vk
                cos_zf = torch.nn.functional.cosine_similarity(zf_samp, zf_true, dim=-1).mean()
            print(f"step {step:5d} loss={loss.item():.4f} samp_cos_code={cos_code.item():.3f} "
                  f"samp_cos_zf={cos_zf.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({'flow': flow.state_dict(), 'd': d, 'd_z': args.k, 'pca': args.pca,
                'base_ckpt': args.ckpt}, args.save)
    print(f"Saved -> {args.save}")


if __name__ == '__main__':
    main()
