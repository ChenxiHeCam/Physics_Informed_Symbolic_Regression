"""Latent diffusion on the manifold: denoise z_f conditioned on
[z_d, K frozen-flow z_f samples, top-k retrieval neighbour z_f].

Purpose: head-to-head vs rectified flow — can a stochastic denoiser with RICHER
conditioning (flow's own samples + neighbours) reach z_f regions the flow
misses? Decoding uses the FROZEN full512 decoder, so the comparison is pure
generator-vs-generator under the same judge.

Training: v-prediction DDPM (cosine schedule) on z_f_true = frozen formula_enc.
Sampling: DDIM.

Usage:
  python train_zf_diffusion.py --cache <dir> --base <full512.pt> \
      --neighbors nbr.npy --zf-raw zf_raw.npy --steps 40000 --save zf_diff.pt
"""
import sys, os, math, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.train_manifold import Manifold, MemmapPairDataset, load_state_compat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def timestep_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class DiTBlock(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.mod = nn.Linear(d, 9 * d)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)

    def forward(self, x, cond_tokens, temb):
        m = self.mod(temb).unsqueeze(1).chunk(9, -1)
        h = self.n1(x) * (1 + m[0]) + m[1]
        x = x + m[2] * self.attn(h, h, h)[0]
        h = self.n2(x) * (1 + m[3]) + m[4]
        x = x + m[5] * self.cross(h, cond_tokens, cond_tokens)[0]
        h = self.n3(x) * (1 + m[6]) + m[7]
        return x + m[8] * self.ff(h)


class ZfDiffusion(nn.Module):
    """Single-token DiT over z_f with cross-attn conditioning."""
    def __init__(self, d=512, n_blocks=8, heads=8, T=1000):
        super().__init__()
        self.d, self.T = d, T
        self.in_proj = nn.Linear(d, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.role = nn.Embedding(3, d)   # 0=z_d 1=flow-sample 2=neighbour
        self.blocks = nn.ModuleList([DiTBlock(d, heads) for _ in range(n_blocks)])
        self.out = nn.Linear(d, d)
        # cosine schedule
        s = 0.008
        ts = torch.linspace(0, T, T + 1)
        f = torch.cos((ts / T + s) / (1 + s) * math.pi / 2) ** 2
        ab = (f / f[0]).clamp(1e-5, 1.0)
        self.register_buffer('abar', ab)

    def cond_tokens(self, zd, zf_flow, zf_nbr):
        toks = [zd.unsqueeze(1) + self.role.weight[0]]
        if zf_flow is not None:
            toks.append(zf_flow + self.role.weight[1])
        if zf_nbr is not None:
            toks.append(zf_nbr + self.role.weight[2])
        return torch.cat(toks, 1)

    def forward(self, z_t, t, cond):
        temb = self.t_mlp(timestep_emb(t, self.d))
        x = self.in_proj(z_t).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cond, temb)
        return self.out(x.squeeze(1))    # v-prediction

    def loss(self, z0, cond):
        B = z0.size(0)
        t = torch.randint(1, self.T, (B,), device=z0.device)
        ab = self.abar[t].unsqueeze(-1)
        eps = torch.randn_like(z0)
        z_t = ab.sqrt() * z0 + (1 - ab).sqrt() * eps
        v_target = ab.sqrt() * eps - (1 - ab).sqrt() * z0
        v = self(z_t, t, cond)
        return F.mse_loss(v, v_target)

    @torch.no_grad()
    def sample(self, zd, zf_flow=None, zf_nbr=None, k=16, n_steps=50):
        B = zd.size(0)
        cond = self.cond_tokens(zd, zf_flow, zf_nbr)
        cond = cond.repeat_interleave(k, 0)
        z = torch.randn(B * k, self.d, device=zd.device)
        ts = torch.linspace(self.T - 1, 1, n_steps, device=zd.device).long()
        for i, t in enumerate(ts):
            tt = torch.full((B * k,), int(t), device=zd.device, dtype=torch.long)
            ab = self.abar[tt].unsqueeze(-1)
            v = self(z, tt, cond)
            z0 = ab.sqrt() * z - (1 - ab).sqrt() * v
            eps = (1 - ab).sqrt() * z + ab.sqrt() * v
            t_next = ts[i + 1] if i + 1 < len(ts) else torch.tensor(0, device=zd.device)
            abn = self.abar[t_next].unsqueeze(-1) if t_next > 0 else torch.ones_like(ab)
            z = abn.sqrt() * z0 + (1 - abn).sqrt() * eps
        return z.view(B, k, self.d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--neighbors', default='')
    ap.add_argument('--zf-raw', default='')
    ap.add_argument('--max-rows', type=int, default=8200000)
    ap.add_argument('--max-points', type=int, default=128)
    ap.add_argument('--batch', type=int, default=96)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--n-blocks', type=int, default=8)
    ap.add_argument('--k-flow', type=int, default=6)
    ap.add_argument('--k-nbr', type=int, default=8)
    ap.add_argument('--ckpt-every', type=int, default=10000)
    ap.add_argument('--clean-index', default='')        # npz clean mask -> train ONLY on clean rows
    ap.add_argument('--pure', action='store_true')      # PURE pretrain: condition ONLY on z_d
                                                        # (no flow z_f / no nbr); arch keeps the slots
    ap.add_argument('--boost-file', default='')         # npz {boost} mask -> SPECIALIZE: oversample
    ap.add_argument('--boost-frac', type=float, default=0.0)  # fraction of each batch from boosted class
    ap.add_argument('--save', required=True)
    args = ap.parse_args()

    ck = torch.load(args.base, map_location=DEVICE)
    d = ck['d']
    base = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                    n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                    dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64),
                    n_ftokens=ck.get('n_ftokens', 0), log_feats=ck.get('log_feats', False),
                    class_feats=ck.get('class_feats', False), dim_head=ck.get('dim_head', False),
                    robust_norm=ck.get('robust_norm', False)).to(DEVICE)
    load_state_compat(base, ck['model']); base.eval()
    for p in base.parameters():
        p.requires_grad = False

    model = ZfDiffusion(d=d, n_blocks=args.n_blocks).to(DEVICE)
    print(f'diffusion params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    nbr_ids = np.load(args.neighbors) if args.neighbors and os.path.exists(args.neighbors) else None
    zf_raw = (torch.from_numpy(np.load(args.zf_raw)).to(DEVICE)
              if args.zf_raw and os.path.exists(args.zf_raw) else None)

    ds = MemmapPairDataset(args.cache, args.max_rows, max_points=args.max_points)
    n_train = len(ds)
    if nbr_ids is not None:
        n_train = min(n_train, len(nbr_ids))
    clean_pool = None
    if args.clean_index and os.path.exists(args.clean_index):
        cmask = np.load(args.clean_index)['clean'][:n_train]
        clean_pool = np.nonzero(cmask)[0]
        print(f"clean-index: LDM trains on {len(clean_pool)}/{n_train} clean rows "
              f"({100*len(clean_pool)/n_train:.1f}%)", flush=True)
    boost_pool = None
    if args.boost_file and os.path.exists(args.boost_file) and args.boost_frac > 0:
        bmask = np.load(args.boost_file)['boost'][:n_train]
        boost_pool = np.nonzero(bmask)[0]
        print(f"SPECIALIZE: boost-frac {args.boost_frac} of each batch from "
              f"{len(boost_pool)} boosted-class rows", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler('cuda')
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if clean_pool is not None:
            idxs = clean_pool[rng.integers(0, len(clean_pool), args.batch)]
        else:
            idxs = rng.integers(0, n_train, args.batch)
        if boost_pool is not None:                       # SPECIALIZE: swap in boosted-class rows
            nb = int(args.batch * args.boost_frac)
            idxs[:nb] = boost_pool[rng.integers(0, len(boost_pool), nb)]
        b = ds.collate(list(idxs))
        for k in b:
            if torch.is_tensor(b[k]):
                b[k] = b[k].to(DEVICE)
        with torch.no_grad():
            zd = base.data_enc(b['points'], b['var_mask'], b['point_mask'], dims=b.get('dims'))
            out = base.formula_enc(b['enc_types'], b['enc_const'], b['enc_edges'],
                                   b['enc_batch'], b['n_graphs'])
            zf_true = out[0] if isinstance(out, tuple) else out
            zf_flow = None if args.pure else base.flow.sample(zd, k=args.k_flow, n_steps=8)
            zf_nbr = None
            if not args.pure and nbr_ids is not None and zf_raw is not None:
                ids = torch.from_numpy(nbr_ids[idxs][:, :args.k_nbr].astype(np.int64)).to(DEVICE)
                zf_nbr = zf_raw[ids]
        cond = model.cond_tokens(zd, zf_flow, zf_nbr)
        with torch.amp.autocast('cuda'):
            loss = model.loss(zf_true.float(), cond)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 200 == 0:
            print(f'step {step:6d} loss={loss.item():.4f} ({time.time()-t0:.0f}s)', flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0:
            torch.save({'model': model.state_dict(), 'd': d, 'n_blocks': args.n_blocks,
                        'k_flow': args.k_flow, 'k_nbr': args.k_nbr, 'base_ckpt': args.base,
                        'step': step}, args.save)
    torch.save({'model': model.state_dict(), 'd': d, 'n_blocks': args.n_blocks,
                'k_flow': args.k_flow, 'k_nbr': args.k_nbr, 'base_ckpt': args.base,
                'step': args.steps}, args.save)
    print(f'Saved -> {args.save}')


if __name__ == '__main__':
    main()
