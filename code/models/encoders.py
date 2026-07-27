"""
Two encoders feeding a shared R^d manifold.

data_encoder  : variable-size observation set (X, y) -> z_d in R^d
                Set-Transformer (permutation-invariant, handles any #points/#vars)
formula_encoder: formula AST graph -> z_f in R^d
                GIN message passing + Transformer over node embeddings

Design choices (per SR_PLAN + emergence-first decision):
  - both project to the SAME R^d (d=512)
  - matching head is a plain dot product (linear bottleneck) to force the
    structure to emerge into the embedding geometry (word2vec-style), not
    into a fancy head.
  - z_f is high-dimensional to leave capacity for later finetuning on more data.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Data encoder : Set-Transformer
# =========================================================================

class MAB(nn.Module):
    """Multihead Attention Block (Set-Transformer, Lee et al. 2019)."""
    def __init__(self, dim, heads=8, ln=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ln1 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if ln else nn.Identity()

    def forward(self, q, kv, key_padding_mask=None):
        a, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        h = self.ln1(q + a)
        h = self.ln2(h + self.ff(h))
        return h


class ISAB(nn.Module):
    """Induced Set Attention Block — O(n) attention via m inducing points."""
    def __init__(self, dim, heads=8, m=16):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, dim) * 0.02)
        self.mab1 = MAB(dim, heads)
        self.mab2 = MAB(dim, heads)

    def forward(self, x, key_padding_mask=None):
        B = x.size(0)
        I = self.I.expand(B, -1, -1)
        h = self.mab1(I, x, key_padding_mask=key_padding_mask)  # inducing attends to set
        return self.mab2(x, h)                                  # set attends back


class AdapterBlock(nn.Module):
    """Pre-LN zero-gate residual block: x + gate*MLP(LN(x)). gate zero-init -> the
    block is IDENTITY at init, so appending adapters GROWS the encoder
    function-preserving (unlike post-LN ISAB which can't be zero-gated cleanly).
    Continue-training then lets the new capacity learn."""
    def __init__(self, d, mult=4):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Linear(d * mult, d))
        self.gate = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return x + self.gate * self.mlp(self.ln(x))


class PMA(nn.Module):
    """Pooling by Multihead Attention -> k seed vectors (k=1 -> single z_d)."""
    def __init__(self, dim, heads=8, k=1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, dim) * 0.02)
        self.mab = MAB(dim, heads)

    def forward(self, x, key_padding_mask=None):
        B = x.size(0)
        S = self.S.expand(B, -1, -1)
        return self.mab(S, x, key_padding_mask=key_padding_mask)  # (B, k, dim)


class DataEncoder(nn.Module):
    """
    Observation set -> z_d.
    Input: per-sample feature = concat([x_1..x_V_padded, y]) plus a var-mask.
    We embed each (point) token, run ISAB layers, pool with PMA.
    Handles variable #points (set dim) and variable #vars (feature dim, padded).
    """
    def __init__(self, max_vars=16, d=512, heads=8, n_isab=8, m_induce=16, dim_len=0,
                 n_tokens=16, nondim=False, log_feats=False, class_feats=False,
                 robust_norm=False, n_adapt=0):
        super().__init__()
        self.max_vars = max_vars
        self.dim_len = dim_len   # per-channel SI dimension vector length (0 = off)
        self.n_tokens = n_tokens # data-token memory size for flow/decoder cross-attn
        self.nondim = nondim     # non-dimensionalize input (z-score each column) -> no value-scale leak
        self.robust_norm = robust_norm  # ALWAYS-ON robust per-column standardization that survives WIDE
                                        # dynamic range (many decades). The old path fed RAW values (up to
                                        # ~1e4) straight into in_proj -> activations saturate, z_d collapses
                                        # on wide/multi-region data (manifold_300 retr ceiling -> 0). Here
                                        # every value channel is signed-log-compressed then median/MAD
                                        # standardized -> O(1) regardless of regime; units carried by `dims`.
        self.class_feats = class_feats  # inject explicit R^2 goodness-of-fit for {poly,exp,log,power}
                                        # into pooled z_d -> z_d DISCRIMINATES function class -> flow can
                                        # target exp/log z_f regions (fixes the flow degeneracy on transcendentals)
        self.log_feats = log_feats  # add log|x|,log|y| channels so the set-transformer can see
                                    # FUNCTION CLASS: exp -> log|y| linear in x; power -> linear in log|x|;
                                    # trig -> neither (the encoder was BLIND to this: cos(exp,poly) z_d = 0.995)
        # each observation point: [x_0..x_{max_vars-1}, y]  (padded x with mask)
        in_dim = max_vars + 1
        # + var-presence mask, + (optional) per-channel dimension block over the
        # max_vars+1 channels (units are always known at test time => safe input)
        extra = (max_vars + 1) * dim_len
        self.in_feat = in_dim + max_vars + extra + ((max_vars + 1) if log_feats else 0)
        self.in_proj = nn.Sequential(
            nn.Linear(self.in_feat, d),
            nn.GELU(), nn.Linear(d, d),
        )
        self.isabs = nn.ModuleList([ISAB(d, heads, m_induce) for _ in range(n_isab)])
        self.adapt = nn.ModuleList([AdapterBlock(d) for _ in range(n_adapt)])  # f-preserving growth
        self.pma = PMA(d, heads, k=1)              # -> pooled z_d (alignment/flow/retrieval)
        self.pma_tok = PMA(d, heads, k=n_tokens)   # -> token memory (flow + decoder cross-attn)
        self.out = nn.Linear(d, d)
        self.out_tok = nn.Linear(d, d)
        if class_feats:
            self.cf_proj = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))

    @staticmethod
    def _class_feats(points, var_mask, point_mask, max_vars):
        """(B,5) R^2 goodness-of-fit for [poly: y~X, exp: log|y|~X, log: y~log|X|,
        power: log|y|~log|X|, TRIG: y~sin(X),cos(X)]. Scale-invariant function-class
        signature; the right channel fires ~1 for each class. The trig channel needs
        angle vars sampled over >=1 period to fire (see gen_data_pairs trig widening,
        2026-06-18) — over a narrow arc sin~poly and the poly channel fires instead."""
        B, N, _ = points.shape
        dev = points.device
        y = points[..., max_vars]
        X = points[..., :max_vars]
        pm = point_mask.float() if point_mask is not None else points.new_ones(B, N)
        vm = var_mask.float()
        eps = 1e-5

        def r2(target, design):
            A = torch.cat([points.new_ones(B, N, 1), design], -1)
            Aw = A * pm.unsqueeze(-1)
            AtA = torch.einsum('bnk,bnj->bkj', Aw, A)
            Att = torch.einsum('bnk,bn->bk', Aw, target * pm)
            AtA = AtA + eps * torch.eye(AtA.size(-1), device=dev).unsqueeze(0)
            beta = torch.linalg.solve(AtA, Att.unsqueeze(-1)).squeeze(-1)
            pred = torch.einsum('bnk,bk->bn', A, beta)
            ss_res = ((target - pred) ** 2 * pm).sum(1)
            cnt = pm.sum(1).clamp_min(1)
            ybar = (target * pm).sum(1) / cnt
            ss_tot = ((target - ybar.unsqueeze(1)) ** 2 * pm).sum(1)
            return (1 - ss_res / ss_tot.clamp_min(eps)).clamp(-1, 1)

        Xm = X * vm.unsqueeze(1)
        sy = torch.sign(y) * torch.log1p(y.abs())
        sX = torch.sign(Xm) * torch.log1p(Xm.abs())
        # TRIG channel: y ~ sum_i a_i*sin(x_i) + b_i*cos(x_i). Fires when y is
        # oscillatory in some input (only detectable if that input spans >=1 period).
        trig_design = torch.cat([torch.sin(Xm), torch.cos(Xm)], dim=-1) * vm.repeat(1, 2).unsqueeze(1)
        return torch.stack([r2(y, Xm), r2(sy, Xm), r2(y, sX), r2(sy, sX),
                            r2(y, trig_design)], -1)

    def _robust_std(self, vals, point_mask):
        """Per-column median/MAD standardization over the REAL points. Robust to the
        heavy tails of wide log-distributed data (a few huge rows don't crush the rest,
        unlike RMS-divide). Output is O(1), zero-centred, ~unit spread per column."""
        B, N, C = vals.shape
        m = (point_mask.unsqueeze(-1) if point_mask is not None else vals.new_ones(B, N, 1))
        cnt = m.sum(1).clamp_min(1.0)                        # (B,1)
        mean = (vals * m).sum(1) / cnt                       # (B,C)  center
        centered = (vals - mean.unsqueeze(1)) * m
        mad = (centered.abs().sum(1) / cnt).clamp_min(1e-6)  # (B,C)  robust scale
        return (vals - mean.unsqueeze(1)) / mad.unsqueeze(1)

    def _nondim(self, points, var_mask, point_mask):
        """Per-column non-dimensionalization: divide each variable/y column by its
        RMS over the real points so absolute scale (unit-dependent) carries no info;
        only the relationships remain. Structure preserved; const-fitting absorbs scale."""
        B, N, C = points.shape
        m = (point_mask.unsqueeze(-1) if point_mask is not None else points.new_ones(B, N, 1))
        cnt = m.sum(1).clamp_min(1.0)
        rms = ((points * m) ** 2).sum(1) / cnt          # (B, C)
        rms = rms.sqrt().clamp_min(1e-6).unsqueeze(1)    # (B,1,C)
        return points / rms

    def forward(self, points, var_mask, point_mask=None, dims=None, return_tokens=False):
        """
        points    : (B, N, max_vars+1)  padded observation points (last col = y)
        dims      : (B, max_vars+1, dim_len)  per-channel SI dimension vectors
        returns z_d (B,d); if return_tokens also tokens (B, n_tokens, d).
        """
        B, N, _ = points.shape
        cf = None
        if self.class_feats:
            cf = self._class_feats(points, var_mask, point_mask, self.max_vars)  # (B,4) pre-nondim (scale-free)
        slog_raw = torch.sign(points) * torch.log1p(points.abs())  # decade-compressed view (pre-norm)
        if self.robust_norm:
            # linear channel: median/MAD-standardized (informative on narrow data);
            # slog channel: decade-compressed THEN standardized (the view that survives
            # WIDE range without collapsing). Encoder always has >=1 O(1) informative view.
            points = self._robust_std(points, point_mask)
        elif self.nondim:
            points = self._nondim(points, var_mask, point_mask)
        vm = var_mask.unsqueeze(1).expand(B, N, self.max_vars)
        feats = [points, vm]
        if self.dim_len:
            if dims is None:
                dims = points.new_zeros(B, self.max_vars + 1, self.dim_len)
            db = dims.reshape(B, 1, -1).expand(B, N, -1)   # constant across points
            feats.append(db)
        if self.log_feats:
            # signed-log of every channel: exp grows linearly here, power-laws don't ->
            # gives the set-transformer the view it needs to tell function classes apart.
            slog = self._robust_std(slog_raw, point_mask) if self.robust_norm else slog_raw
            feats.append(slog)
        h = self.in_proj(torch.cat(feats, dim=-1))
        kpm = (point_mask == 0) if point_mask is not None else None
        for isab in self.isabs:
            h = isab(h, key_padding_mask=kpm)
        for ad in self.adapt:                                       # f-preserving capacity (gate=0 at init)
            h = ad(h)
        z = self.out(self.pma(h, key_padding_mask=kpm).squeeze(1))   # (B, d) pooled z_d
        if cf is not None:
            z = z + self.cf_proj(cf)                                 # inject function-class signal
        if return_tokens:
            tok = self.out_tok(self.pma_tok(h, key_padding_mask=kpm))  # (B, n_tokens, d)
            return z, tok
        return z


# =========================================================================
# Formula encoder : GIN + Transformer over AST graph
# =========================================================================

try:
    from torch_geometric.nn import GINConv, global_mean_pool
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False


class FormulaEncoder(nn.Module):
    """
    AST graph -> z_f.
    Nodes carry: node-type embedding + (for CONST) a scalar value feature.
    GIN layers do structural message passing; a Transformer over node embeddings
    captures long-range; pooled to z_f.
    """
    def __init__(self, vocab_size, d=512, n_gin=4, n_tf=6, heads=8):
        super().__init__()
        self.type_emb = nn.Embedding(vocab_size, d)
        self.const_proj = nn.Linear(1, d)
        if _HAS_PYG:
            self.gins = nn.ModuleList([
                GINConv(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)))
                for _ in range(n_gin)
            ])
        else:
            self.gins = None
        enc_layer = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(enc_layer, n_tf)
        self.pool_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.out = nn.Linear(d, d)

    def encode_nodes(self, node_types, const_vals):
        h = self.type_emb(node_types)
        # add const value feature where node_type == CONST handled by caller mask
        h = h + self.const_proj(const_vals.unsqueeze(-1))
        return h

    def forward_graph(self, x, edge_index, batch):
        """PyG-style: x=(total_nodes, d), edge_index, batch index."""
        h = x
        if self.gins is not None:
            for gin in self.gins:
                h = h + F.gelu(gin(h, edge_index))
        return h

    def forward_seq(self, node_types, const_vals, pad_mask=None):
        """
        Sequence form (no explicit edges) — Transformer only.
        node_types: (B, L) ; const_vals: (B, L) ; pad_mask: (B, L) 1=real
        returns z_f: (B, d)
        """
        h = self.encode_nodes(node_types, const_vals)
        B = h.size(0)
        cls = self.pool_tok.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        if pad_mask is not None:
            full_mask = torch.cat([torch.ones(B, 1, device=h.device), pad_mask], dim=1)
            kpm = (full_mask == 0)
        else:
            kpm = None
        h = self.tf(h, src_key_padding_mask=kpm)
        return self.out(h[:, 0])


# =========================================================================
# Linear-bottleneck matching head (dot product) — forces emergence
# =========================================================================

def align_logits(z_d, z_f, temp=0.07):
    """CLIP-style similarity matrix; plain dot product after L2 norm."""
    zd = F.normalize(z_d, dim=-1)
    zf = F.normalize(z_f, dim=-1)
    return (zd @ zf.t()) / temp


if __name__ == "__main__":
    import torch
    d = 256
    de = DataEncoder(max_vars=16, d=d, n_isab=4)
    fe = FormulaEncoder(vocab_size=38, d=d, n_gin=2, n_tf=2)
    B, N = 4, 100
    pts = torch.randn(B, N, 17)
    vm = torch.zeros(B, 16); vm[:, :3] = 1
    z_d = de(pts, vm)
    nt = torch.randint(0, 38, (B, 12)); cv = torch.randn(B, 12)
    z_f = fe.forward_seq(nt, cv)
    print("z_d", z_d.shape, "z_f", z_f.shape)
    print("align", align_logits(z_d, z_f).shape)
    print("params: data_enc", sum(p.numel() for p in de.parameters()),
          "formula_enc", sum(p.numel() for p in fe.parameters()))
