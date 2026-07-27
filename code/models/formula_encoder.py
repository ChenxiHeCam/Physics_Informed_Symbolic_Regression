"""
Formula encoder — three levels.

Level 1  (intra-formula): GIN over the AST graph  -> z_f_local
          encodes the formula's own syntactic structure (a+b)*c != a+(b*c).

Level 2  (inter-formula): Graph-Transformer over a sampled DAG neighborhood
          - full attention discovers IMPLICIT relations (edges the DAG missed)
          - per-relation edge-bias injects EXPLICIT DAG edges
            (derivation / co_occurrence / limiting_case / ...) so the model
            can use abstract relations like "these two are often used together".
          -> z_f  (structure + relational context)

Emergence-first: we never hand-assign dimensions to relations. The GNN/edge-bias
only give the network the ABILITY to see structure; how it encodes that into the
R^d geometry stays emergent.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GINConv
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False


# =========================================================================
# Level 1 : GIN over AST
# =========================================================================

class ASTEncoder(nn.Module):
    """AST graph -> per-formula embedding. Dual output (opt-in n_ftokens>0):
      z_f        : (n_graphs, d)        pooled global gist (alignment / coarse flow)
      z_f_tokens : (n_graphs, M, d)     PMA over the AST nodes -> fine detail memory
                   (the decoder cross-attends to these; gives capacity + disambiguates
                    variable-position/structure that the single pooled vector loses)."""
    def __init__(self, vocab_size, d=512, n_gin=4, n_ftokens=0):
        super().__init__()
        self.type_emb = nn.Embedding(vocab_size, d)
        self.const_proj = nn.Linear(1, d)
        if _HAS_PYG:
            self.gins = nn.ModuleList([
                GINConv(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)),
                        train_eps=True)
                for _ in range(n_gin)
            ])
        else:
            self.gins = None
        self.norm = nn.LayerNorm(d)
        self.n_ftokens = n_ftokens
        if n_ftokens > 0:
            from models.encoders import PMA
            self.pma_tok = PMA(d, heads=8, k=n_ftokens)
            self.out_tok = nn.Linear(d, d)

    def forward(self, node_types, const_vals, edge_index, batch, n_graphs, return_tokens=False):
        """returns z_f (n_graphs,d); if return_tokens also z_f_tokens (n_graphs,M,d)."""
        h = self.type_emb(node_types) + self.const_proj(const_vals.unsqueeze(-1))
        if self.gins is not None and edge_index.numel() > 0:
            for gin in self.gins:
                h = h + F.gelu(gin(h, edge_index))
        h = self.norm(h)
        d = h.size(-1)
        out = torch.zeros(n_graphs, d, device=h.device, dtype=h.dtype)
        cnt = torch.zeros(n_graphs, 1, device=h.device, dtype=h.dtype)
        out.index_add_(0, batch, h)
        cnt.index_add_(0, batch, torch.ones_like(h[:, :1]))
        z_f = out / cnt.clamp_min(1.0)
        if not return_tokens or self.n_ftokens == 0:
            return z_f
        # densify flattened nodes -> (n_graphs, max_nodes, d) + mask, then PMA -> M tokens
        from torch_geometric.utils import to_dense_batch
        padded, mask = to_dense_batch(h, batch, batch_size=n_graphs)   # (G, maxN, d), (G, maxN)
        z_f_tokens = self.out_tok(self.pma_tok(padded, key_padding_mask=(~mask)))
        return z_f, z_f_tokens


# =========================================================================
# Level 2 : Graph-Transformer over DAG neighborhood
# =========================================================================

class GraphTransformerLayer(nn.Module):
    """
    One layer of attention with per-relation edge bias.
      attn(i,j) = (q_i . k_j)/sqrt(dh) + edge_bias[rel(i,j)]
    edge_bias is a learned scalar per (relation_type, head).
    Nodes with no DAG edge still attend freely (bias=0) -> implicit discovery.
    """
    def __init__(self, d, heads, n_relations):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        # +1 relation slot for "no edge" (index 0), learned bias per head
        self.edge_bias = nn.Parameter(torch.zeros(n_relations + 1, heads))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

    def forward(self, x, rel_mat, pad_mask=None):
        """
        x       : (B, N, d)    node embeddings (B sampled neighborhoods)
        rel_mat : (B, N, N)    relation-type id per pair (0 = no edge)
        pad_mask: (B, N)       1=real node, 0=pad
        """
        B, N, d = x.shape
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)  # (B,h,N,dh)
        k = self.k(x).view(B, N, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / (self.dh ** 0.5)         # (B,h,N,N)
        # edge bias: gather (B,N,N,h) -> (B,h,N,N)
        bias = self.edge_bias[rel_mat]                             # (B,N,N,h)
        att = att + bias.permute(0, 3, 1, 2)
        if pad_mask is not None:
            m = (pad_mask == 0).view(B, 1, 1, N)
            att = att.masked_fill(m, float("-inf"))
        att = att.softmax(dim=-1)
        h = (att @ v).transpose(1, 2).contiguous().view(B, N, d)
        x = self.ln1(x + self.o(h))
        x = self.ln2(x + self.ff(x))
        return x


class FormulaEncoder(nn.Module):
    """
    Full formula encoder. Two call modes:
      encode_ast(...)        -> z_f_local (level 1 only; cheap, for retrieval index)
      forward(...)           -> z_f       (level 1 + level 2 with DAG neighborhood)
    """
    def __init__(self, vocab_size, d=512, n_gin=4, n_gt=4, heads=8, n_relations=8):
        super().__init__()
        self.ast = ASTEncoder(vocab_size, d, n_gin)
        self.gt_layers = nn.ModuleList([
            GraphTransformerLayer(d, heads, n_relations) for _ in range(n_gt)
        ])
        self.out = nn.Linear(d, d)
        self.d = d

    def encode_ast(self, node_types, const_vals, edge_index, batch, n_graphs):
        return self.ast(node_types, const_vals, edge_index, batch, n_graphs)

    def forward(self, neigh_local, rel_mat, pad_mask=None, center_idx=0):
        """
        neigh_local : (B, N, d)  level-1 embeddings of center + its DAG neighbors
        rel_mat     : (B, N, N)  per-pair relation id (0=no edge)
        pad_mask    : (B, N)
        center_idx  : which node in the neighborhood is the target formula
        returns z_f : (B, d)  context-aware embedding of the center formula
        """
        x = neigh_local
        for layer in self.gt_layers:
            x = layer(x, rel_mat, pad_mask)
        return self.out(x[:, center_idx])


if __name__ == "__main__":
    V, d = 38, 256
    fe = FormulaEncoder(V, d=d, n_gin=2, n_gt=2, heads=8, n_relations=6)

    # --- level 1 smoke: 3 ASTs batched PyG-style ---
    nt = torch.randint(0, V, (20,))
    cv = torch.randn(20)
    ei = torch.randint(0, 20, (2, 30))
    batch = torch.cat([torch.zeros(7), torch.ones(7), torch.full((6,), 2)]).long()
    zl = fe.encode_ast(nt, cv, ei, batch, 3)
    print("z_f_local", zl.shape)

    # --- level 2 smoke: B neighborhoods of N formulas ---
    B, N = 4, 32
    neigh = torch.randn(B, N, d)
    rel = torch.randint(0, 7, (B, N, N))   # 0=no edge, 1..6 relation types
    pad = torch.ones(B, N); pad[:, 28:] = 0
    zf = fe(neigh, rel, pad, center_idx=0)
    print("z_f", zf.shape)
    print("params", sum(p.numel() for p in fe.parameters()))
