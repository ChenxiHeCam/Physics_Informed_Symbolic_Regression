"""Discrete (masked-token) diffusion decoder for formula generation — a MaskGIT-
style alternative to the autoregressive decoder and the z_f flow/LDM.

Why: LDM failed because it denoises the CONTINUOUS z_f (the bottleneck modality).
This denoises the DISCRETE formula token sequence directly, conditioned on the
data — sidestepping z_f. Bidirectional (non-causal) so it can compose/edit whole
subtrees at once (good for the nested-rational / sqrt-over-sum structures the AR
decoder misses).

Conditioning (frozen full512 + cheap features, data-first like dec7):
  pooled z_d, K frozen-flow z_f samples, per-channel feature tokens.

Train: mask a random fraction (cosine schedule) of formula tokens with [MASK];
predict the originals at masked positions (cross-entropy).
Infer (separate): start all-[MASK], iteratively predict + reveal high-confidence.

Usage:
  python train_mask_diffusion.py --cache <dir> --base <full512.pt> \
      --steps 40000 --save mask_diff.pt
"""
import sys, os, math, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.ast_grammar import MAX_VARS, VOCAB_SIZE
from models.decoder import PAD_ID
from train.train_manifold import Manifold, MemmapPairDataset, load_state_compat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MASK_ID = VOCAB_SIZE          # extra id for [MASK]


class FeatTok(nn.Module):
    """Per-channel data features -> tokens (loglog slope/corr, magnitude stats)."""
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, points, var_mask, point_mask):
        B, N, C = points.shape
        eps = 1e-9
        pm = point_mask.unsqueeze(-1)
        y = points[..., MAX_VARS:MAX_VARS + 1]
        ly = torch.log(y.abs() + eps); lx = torch.log(points.abs() + eps)
        cnt = pm.sum(1).clamp_min(2.0)
        mx = (lx * pm).sum(1) / cnt; my = (ly * pm).sum(1) / cnt
        vx = (((lx - mx.unsqueeze(1)) * pm) ** 2).sum(1) / cnt
        cov = (((lx - mx.unsqueeze(1)) * (ly - my.unsqueeze(1))) * pm).sum(1) / cnt
        slope = (cov / vx.clamp_min(1e-6)).clamp(-8, 8)
        corr = (cov / (vx.clamp_min(1e-9)).sqrt() / (((ly - my.unsqueeze(1)) ** 2 * pm).sum(1) / cnt).clamp_min(1e-9).sqrt()).clamp(-1, 1)
        amax = torch.log10((points.abs() * pm + (1 - pm)).amax(1).clamp_min(1e-12))
        amean = torch.log10((points.abs() * pm).sum(1) / cnt + eps)
        fneg = ((points < 0).float() * pm).sum(1) / cnt
        feats = torch.stack([slope, corr, amean, amax, fneg, cnt.expand(-1, C) * 0], -1).nan_to_num(0.0)
        chan = torch.cat([var_mask, var_mask.new_ones(B, 1)], 1)
        return self.proj(feats) * chan.unsqueeze(-1)


class MaskDiffusion(nn.Module):
    def __init__(self, d=512, n_layers=12, heads=8, max_len=48, k_flow=6):
        super().__init__()
        self.d = d; self.max_len = max_len; self.k_flow = k_flow
        self.tok = nn.Embedding(VOCAB_SIZE + 1, d)        # +1 = [MASK]
        self.pos = nn.Embedding(max_len, d)
        self.role = nn.Embedding(3, d)                    # 0 z_d 1 flow 2 feat
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True, activation='gelu')
        self.blocks = nn.TransformerEncoder(enc, n_layers)
        self.cross = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True)
                                    for _ in range(n_layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, VOCAB_SIZE)
        self.layers = n_layers

    def cond_mem(self, zd, zf_flow, feat):
        toks = [zd.unsqueeze(1) + self.role.weight[0],
                zf_flow + self.role.weight[1],
                feat + self.role.weight[2]]
        return torch.cat(toks, 1)

    def forward(self, seq_in, mem):
        # interleave self-attn (TransformerEncoder) with cross-attn to condition
        L = seq_in.size(1)
        x = self.tok(seq_in) + self.pos(torch.arange(L, device=seq_in.device))[None]
        for i in range(self.layers):
            x = self.blocks.layers[i](x)
            a, _ = self.cross[i](self.ln[i](x), mem, mem)
            x = x + a
        return self.head(x)


def cosine_mask_frac(rng_u):
    # fraction masked ~ cos schedule: more often heavily masked
    return float(np.cos(rng_u * math.pi / 2))   # u~U(0,1) -> frac in (0,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--max-rows', type=int, default=970000)
    ap.add_argument('--max-points', type=int, default=128)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--n-layers', type=int, default=12)
    ap.add_argument('--k-flow', type=int, default=6)
    ap.add_argument('--ckpt-every', type=int, default=10000)
    ap.add_argument('--save', required=True)
    args = ap.parse_args()

    ck = torch.load(args.base, map_location=DEVICE); d = ck['d']
    base = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                    n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                    dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64),
                    n_ftokens=ck.get('n_ftokens', 0), log_feats=ck.get('log_feats', False),
                    class_feats=ck.get('class_feats', False), dim_head=ck.get('dim_head', False),
                    robust_norm=ck.get('robust_norm', False)).to(DEVICE)
    load_state_compat(base, ck['model']); base.eval()
    for p in base.parameters(): p.requires_grad = False

    model = MaskDiffusion(d=d, n_layers=args.n_layers, max_len=48, k_flow=args.k_flow).to(DEVICE)
    ftok = FeatTok(d).to(DEVICE)
    params = list(model.parameters()) + list(ftok.parameters())
    print(f'mask-diffusion params: {sum(p.numel() for p in params)/1e6:.1f}M')

    ds = MemmapPairDataset(args.cache, args.max_rows, max_points=args.max_points)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler('cuda')
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idxs = rng.integers(0, len(ds), args.batch)
        b = ds.collate(list(idxs))
        pts = b['points'].to(DEVICE); vm = b['var_mask'].to(DEVICE); pm = b['point_mask'].to(DEVICE)
        dims = b.get('dims'); dims = dims.to(DEVICE) if dims is not None else None
        tgt = b['tgt_types'].to(DEVICE); tm = b['tgt_mask'].to(DEVICE)
        with torch.no_grad():
            zd = base.data_enc(pts, vm, pm, dims=dims)
            zf = base.flow.sample(zd, k=args.k_flow, n_steps=8)
        feat = ftok(pts, vm, pm)
        mem = model.cond_mem(zd, zf, feat)
        # build masked input: mask a cosine-scheduled fraction of REAL tokens with
        # [MASK] (vectorized Bernoulli per sample — no per-sample GPU sync loop).
        u = rng.uniform(0, 1, args.batch).astype(np.float32)
        fr = np.clip(np.cos(u * math.pi / 2), 0.15, 1.0)
        frt = torch.from_numpy(fr).to(DEVICE).unsqueeze(1)        # (B,1)
        real = tm.bool()
        rscore = torch.rand_like(tgt, dtype=torch.float)
        mask = (rscore < frt) & real
        # guarantee >=1 masked real token per row
        empty = (mask & real).sum(1) == 0
        if empty.any():
            first_real = real.float().argmax(1)
            mask[empty, first_real[empty]] = True
        seq_in = tgt.clone()
        seq_in[mask] = MASK_ID
        seq_in[~real] = PAD_ID
        with torch.amp.autocast('cuda'):
            logits = model(seq_in, mem)
            # loss only on masked (real) positions
            lt = logits.reshape(-1, VOCAB_SIZE)[mask.reshape(-1)]
            yt = tgt.reshape(-1)[mask.reshape(-1)]
            loss = F.cross_entropy(lt, yt) if yt.numel() > 0 else logits.sum() * 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 200 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1).reshape(-1)[mask.reshape(-1)] == yt).float().mean() if yt.numel() else torch.tensor(0.)
            print(f'step {step:6d} loss={loss.item():.3f} mask_acc={acc.item():.3f} ({time.time()-t0:.0f}s)', flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0:
            torch.save({'model': model.state_dict(), 'ftok': ftok.state_dict(), 'd': d,
                        'n_layers': args.n_layers, 'k_flow': args.k_flow,
                        'base_ckpt': args.base, 'step': step}, args.save)
    torch.save({'model': model.state_dict(), 'ftok': ftok.state_dict(), 'd': d,
                'n_layers': args.n_layers, 'k_flow': args.k_flow,
                'base_ckpt': args.base, 'step': args.steps}, args.save)
    print(f'Saved -> {args.save}')


if __name__ == '__main__':
    main()
