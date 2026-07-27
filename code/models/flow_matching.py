"""
Flow matching in z_f space, conditioned on z_d (DiT-style velocity net).

Why flow matching (not GFlowNet / not plain diffusion):
  - we generate in a CONTINUOUS latent (z_f), GFlowNet is for discrete trees
  - flow matching trains stably & samples in FEW ODE steps (we want many z_f
    candidates fast)
  - DiT backbone = AdaLN-Zero conditioning (same mechanism as the decoder),
    forces z_d to actually steer the velocity field.

Training (conditional flow matching, rectified-flow / linear path):
  z0 ~ N(0, I)
  z1 = target z_f
  z_t = (1-t) z0 + t z1
  target velocity u = z1 - z0
  loss = || v_theta(z_t, t, z_d) - u ||^2

Sampling (K candidates):
  z ~ N(0,I)  (K different seeds)
  integrate dz/dt = v_theta(z, t, z_d) from t=0 to 1  (Euler, n_steps)
  -> K candidate z_f's, each decoded to a formula.

z_d is encoded from observations with SLIGHTLY larger noise than the encoder
training (user spec) so the flow learns to map dirty data to the formula
distribution.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLNZeroBlock(nn.Module):
    """DiT block: AdaLN-Zero modulated self-MLP, conditioned on (t, z_d)."""
    def __init__(self, d, hidden_mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * hidden_mult), nn.GELU(),
            nn.Linear(d * hidden_mult, d),
        )
        self.to_mod = nn.Linear(d, 3 * d)
        nn.init.zeros_(self.to_mod.weight); nn.init.zeros_(self.to_mod.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.to_mod(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.mlp(h)


class TimeEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.d = d

    def forward(self, t):
        # sinusoidal time embedding
        half = self.d // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) *
                          (torch.log(torch.tensor(10000.0)) / (half - 1)))
        ang = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.size(-1) < self.d:
            emb = F.pad(emb, (0, self.d - emb.size(-1)))
        return self.lin(emb)


class VelocityNet(nn.Module):
    """v_theta(z_t, t, z_d) -> velocity in R^d, DiT-style."""
    def __init__(self, d=512, n_blocks=6):
        super().__init__()
        self.in_proj = nn.Linear(d, d)
        self.t_embed = TimeEmbed(d)
        self.cond_proj = nn.Linear(d, d)      # from z_d
        self.blocks = nn.ModuleList([AdaLNZeroBlock(d) for _ in range(n_blocks)])
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, z_t, t, z_d):
        cond = self.t_embed(t) + self.cond_proj(z_d)
        x = self.in_proj(z_t)
        for blk in self.blocks:
            x = blk(x, cond)
        return self.out(x)


class FlowMatching(nn.Module):
    def __init__(self, d=512, n_blocks=6):
        super().__init__()
        self.v = VelocityNet(d, n_blocks)
        self.d = d

    def loss(self, z_f, z_d):
        """Conditional flow-matching (linear/rectified path)."""
        B = z_f.size(0)
        z0 = torch.randn_like(z_f)
        t = torch.rand(B, device=z_f.device)
        z_t = (1 - t)[:, None] * z0 + t[:, None] * z_f
        u = z_f - z0                              # target velocity
        v = self.v(z_t, t, z_d)
        return F.mse_loss(v, u)

    @torch.no_grad()
    def sample(self, z_d, k=16, n_steps=20):
        """
        Generate k candidate z_f per data item.
        z_d: (B, d). Returns (B, k, d).
        """
        B, d = z_d.shape
        device = z_d.device
        # expand for k seeds
        zd_rep = z_d.unsqueeze(1).expand(B, k, d).reshape(B * k, d)
        z = torch.randn(B * k, d, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B * k,), i * dt, device=device)
            z = z + dt * self.v(z, t, zd_rep)
        return z.view(B, k, d)


class VelocityNet2(nn.Module):
    """v(z_t, t, z_d, z_hint): conditions on data z_d AND a retrieved hint z_f
    (nearest neighbour by similarity). The hint anchors generation near a known
    formula region so the flow can REFINE toward novel held-out z_f instead of
    sampling the whole (train-bounded) distribution from scratch."""
    def __init__(self, d=512, n_blocks=6):
        super().__init__()
        self.in_proj = nn.Linear(d, d)
        self.t_embed = TimeEmbed(d)
        self.cond_proj = nn.Linear(d, d)      # from z_d
        self.hint_proj = nn.Linear(d, d)      # from retrieved z_f hint
        self.blocks = nn.ModuleList([AdaLNZeroBlock(d) for _ in range(n_blocks)])
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, z_t, t, z_d, z_hint):
        cond = self.t_embed(t) + self.cond_proj(z_d) + self.hint_proj(z_hint)
        x = self.in_proj(z_t)
        for blk in self.blocks:
            x = blk(x, cond)
        return self.out(x)


class FlowMatching2(nn.Module):
    """Retrieval-conditioned flow: p(z_f | z_d, z_hint)."""
    def __init__(self, d=512, n_blocks=6):
        super().__init__()
        self.v = VelocityNet2(d, n_blocks)
        self.d = d

    def loss(self, z_f, z_d, z_hint):
        B = z_f.size(0)
        z0 = torch.randn_like(z_f)
        t = torch.rand(B, device=z_f.device)
        z_t = (1 - t)[:, None] * z0 + t[:, None] * z_f
        u = z_f - z0
        return F.mse_loss(self.v(z_t, t, z_d, z_hint), u)

    @torch.no_grad()
    def sample(self, z_d, z_hint, k=16, n_steps=20):
        """z_d,(z_hint): (B,d). Returns (B,k,d)."""
        B, d = z_d.shape
        device = z_d.device
        zd_rep = z_d.unsqueeze(1).expand(B, k, d).reshape(B * k, d)
        zh_rep = z_hint.unsqueeze(1).expand(B, k, d).reshape(B * k, d)
        z = torch.randn(B * k, d, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B * k,), i * dt, device=device)
            z = z + dt * self.v(z, t, zd_rep, zh_rep)
        return z.view(B, k, d)


class VelocityNetTok(nn.Module):
    """v(z_t, t, data_tokens): the flow conditions on the RICH data-token memory
    (cross-attention) instead of a single pooled z_d, so it can generate z_f the
    pooled-z_d flow can't reach (pushes the flow ceiling toward the oracle)."""
    def __init__(self, d=512, n_blocks=6, heads=8):
        super().__init__()
        self.in_proj = nn.Linear(d, d)
        self.t_embed = TimeEmbed(d)
        self.blocks = nn.ModuleList([AdaLNZeroBlock(d) for _ in range(n_blocks)])
        self.cross = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True)
                                    for _ in range(n_blocks)])
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, z_t, t, data_tokens):
        cond = self.t_embed(t)
        x = self.in_proj(z_t).unsqueeze(1)              # (B,1,d) query
        for blk, cr in zip(self.blocks, self.cross):
            a, _ = cr(x, data_tokens, data_tokens)      # attend to data memory
            x = x + a
            x = blk(x.squeeze(1), cond).unsqueeze(1)
        return self.out(x.squeeze(1))


class FlowMatchingTok(nn.Module):
    """Flow matching conditioned on data tokens: p(z_f | data_tokens)."""
    def __init__(self, d=512, n_blocks=6):
        super().__init__()
        self.v = VelocityNetTok(d, n_blocks); self.d = d

    def loss(self, z_f, data_tokens):
        B = z_f.size(0)
        z0 = torch.randn_like(z_f); t = torch.rand(B, device=z_f.device)
        z_t = (1 - t)[:, None] * z0 + t[:, None] * z_f
        return F.mse_loss(self.v(z_t, t, data_tokens), z_f - z0)

    @torch.no_grad()
    def sample(self, data_tokens, k=16, n_steps=20):
        B, T, d = data_tokens.shape
        dt_rep = data_tokens.unsqueeze(1).expand(B, k, T, d).reshape(B * k, T, d)
        z = torch.randn(B * k, d, device=data_tokens.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B * k,), i * dt, device=data_tokens.device)
            z = z + dt * self.v(z, t, dt_rep)
        return z.view(B, k, d)


class SeqFlowBlock(nn.Module):
    """DiT-style block over a SEQUENCE of latent tokens: self-attn (over the M noisy
    z_f tokens) + cross-attn (to the data-token memory) + MLP, each AdaLN-Zero
    modulated by a global condition (coarse z_f + time)."""
    def __init__(self, d, heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.to_mod = nn.Linear(d, 9 * d)
        nn.init.zeros_(self.to_mod.weight); nn.init.zeros_(self.to_mod.bias)

    def forward(self, x, cond, data_tokens):
        m = self.to_mod(cond).unsqueeze(1).chunk(9, dim=-1)   # each (B,1,d) -> broadcast over M
        s1, sh1, g1, s2, sh2, g2, s3, sh3, g3 = m
        h = self.n1(x) * (1 + s1) + sh1
        x = x + g1 * self.self_attn(h, h, h)[0]
        h = self.n2(x) * (1 + s2) + sh2
        x = x + g2 * self.cross_attn(h, data_tokens, data_tokens)[0]
        h = self.n3(x) * (1 + s3) + sh3
        x = x + g3 * self.mlp(h)
        return x


class VelocityNetTokSeq(nn.Module):
    """v(z_t, t, data_tokens, z_coarse) for a SEQUENCE target z_f_tokens (B,M,d)."""
    def __init__(self, d=512, n_blocks=6, heads=8):
        super().__init__()
        self.in_proj = nn.Linear(d, d)
        self.t_embed = TimeEmbed(d)
        self.coarse_proj = nn.Linear(d, d)
        self.pos = nn.Parameter(torch.randn(1, 64, d) * 0.02)   # up to 64 z_f tokens
        self.blocks = nn.ModuleList([SeqFlowBlock(d, heads) for _ in range(n_blocks)])
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, z_t, t, data_tokens, z_coarse):
        B, M, d = z_t.shape
        cond = self.t_embed(t) + self.coarse_proj(z_coarse)     # (B,d)
        x = self.in_proj(z_t) + self.pos[:, :M]
        for blk in self.blocks:
            x = blk(x, cond, data_tokens)
        return self.out(x)


class FlowMatchingTokSeq(nn.Module):
    """Fine flow: p(z_f_tokens | data_tokens, z_f_coarse). Coarse-to-fine — the coarse
    pooled z_f (easy, low-dim, from FlowMatchingTok) anchors the harder sequence
    generation, which carries the capacity that lifts the oracle ceiling."""
    def __init__(self, d=512, n_blocks=6):
        super().__init__()
        self.v = VelocityNetTokSeq(d, n_blocks); self.d = d

    def loss(self, z_tokens, data_tokens, z_coarse):
        B = z_tokens.size(0)
        z0 = torch.randn_like(z_tokens); t = torch.rand(B, device=z_tokens.device)
        z_t = (1 - t)[:, None, None] * z0 + t[:, None, None] * z_tokens
        return F.mse_loss(self.v(z_t, t, data_tokens, z_coarse), z_tokens - z0)

    @torch.no_grad()
    def sample(self, data_tokens, z_coarse, M, k=1, n_steps=20):
        """Returns (B, k, M, d). z_coarse (B,d), data_tokens (B,T,d)."""
        B, T, d = data_tokens.shape
        dt_rep = data_tokens.unsqueeze(1).expand(B, k, T, d).reshape(B * k, T, d)
        zc_rep = z_coarse.unsqueeze(1).expand(B, k, d).reshape(B * k, d)
        z = torch.randn(B * k, M, d, device=data_tokens.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B * k,), i * dt, device=data_tokens.device)
            z = z + dt * self.v(z, t, dt_rep, zc_rep)
        return z.view(B, k, M, d)


class VelocityNetTokLowD(nn.Module):
    """Velocity net whose TARGET is a low-dim z_f code (d_z << d) but which still
    cross-attends to the full d-dim data-token memory. Internal width = d (so the
    cross-attention matches the token dim); only the input/output projections are
    d_z. Generating a smaller target is far easier to hit precisely -> higher
    samp_cos -> the flow cashes in more of the oracle ceiling."""
    def __init__(self, d=768, d_z=256, n_blocks=6, heads=8):
        super().__init__()
        self.in_proj = nn.Linear(d_z, d)
        self.t_embed = TimeEmbed(d)
        self.blocks = nn.ModuleList([AdaLNZeroBlock(d) for _ in range(n_blocks)])
        self.cross = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True)
                                    for _ in range(n_blocks)])
        self.out = nn.Linear(d, d_z)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, z_t, t, data_tokens):
        cond = self.t_embed(t)
        x = self.in_proj(z_t).unsqueeze(1)              # (B,1,d) query
        for blk, cr in zip(self.blocks, self.cross):
            a, _ = cr(x, data_tokens, data_tokens)      # attend to data memory
            x = x + a
            x = blk(x.squeeze(1), cond).unsqueeze(1)
        return self.out(x.squeeze(1))


class FlowMatchingTokLowD(nn.Module):
    """Flow matching on a low-dim z_f code (PCA coords), conditioned on data tokens."""
    def __init__(self, d=768, d_z=256, n_blocks=6):
        super().__init__()
        self.v = VelocityNetTokLowD(d, d_z, n_blocks); self.d = d; self.d_z = d_z

    def loss(self, z_code, data_tokens):
        B = z_code.size(0)
        z0 = torch.randn_like(z_code); t = torch.rand(B, device=z_code.device)
        z_t = (1 - t)[:, None] * z0 + t[:, None] * z_code
        return F.mse_loss(self.v(z_t, t, data_tokens), z_code - z0)

    @torch.no_grad()
    def sample(self, data_tokens, k=16, n_steps=20):
        B, T, d = data_tokens.shape
        dt_rep = data_tokens.unsqueeze(1).expand(B, k, T, d).reshape(B * k, T, d)
        z = torch.randn(B * k, self.d_z, device=data_tokens.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B * k,), i * dt, device=data_tokens.device)
            z = z + dt * self.v(z, t, dt_rep)
        return z.view(B, k, self.d_z)


if __name__ == "__main__":
    d = 256
    fm = FlowMatching(d=d, n_blocks=4)
    B = 8
    z_f = torch.randn(B, d)
    z_d = torch.randn(B, d)
    l = fm.loss(z_f, z_d)
    print("flow loss", l.item())
    samp = fm.sample(z_d, k=16, n_steps=10)
    print("sampled z_f candidates", samp.shape)   # (B, 16, d)
    print("params", sum(p.numel() for p in fm.parameters()))
