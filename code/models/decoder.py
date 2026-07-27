"""
Conditioning decoder-decoder: (z_d, z_f) -> formula AST.

Two stages (T5/BART-style, but tree-structured output):
  Stage 1 (condition encoder): pack [z_d, z_f] (+ optional extra z_f's) into a
           short memory sequence via a small transformer encoder.
  Stage 2 (grammar-constrained AST decoder): autoregressively emit AST nodes in
           pre-order. Each step:
             - transformer decoder block with self-attn + CROSS-ATTN to memory
             - AdaLN-Zero modulation from a pooled condition vector
               (forces conditioning to engage; zero-init => starts as identity,
                cannot be silently bypassed — fixes the v6 "gate->0.006" failure)
             - grammar mask: only node-types valid at the current open slot get
               probability mass => output is ALWAYS a syntactically valid AST,
               yet free to compose formulas never seen in training.

Generates novel formulas: vocabulary is fixed (all operators present) but
composition is unbounded — like an LM's finite tokens, infinite sentences,
minus the invalid-syntax outputs.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ast_grammar import (
    NODE_TYPES, NT2ID, ID2NT, VOCAB_SIZE, ARITY, BINOPS, UNOPS,
    SYMCONST, VARS, MAX_VARS,
)

# precompute arity per node-id; leaves = 0
ARITY_BY_ID = torch.zeros(VOCAB_SIZE, dtype=torch.long)
for nt, ar in ARITY.items():
    ARITY_BY_ID[NT2ID[nt]] = ar
EOS_ID = NT2ID["<eos>"]
PAD_ID = NT2ID["<pad>"]
CONST_ID = NT2ID["CONST"]


# =========================================================================
# AdaLN-Zero modulation
# =========================================================================

class AdaLNZero(nn.Module):
    """Per-layer scale/shift/gate from condition; zero-init so layer starts as
    identity and conditioning is FORCED to be learned (can't be ignored)."""
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.to_mod = nn.Linear(d, 3 * d)
        nn.init.zeros_(self.to_mod.weight)
        nn.init.zeros_(self.to_mod.bias)

    def forward(self, x, cond):
        # cond: (B, d) -> (B,1,d)
        scale, shift, gate = self.to_mod(cond).unsqueeze(1).chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


# =========================================================================
# Condition encoder (stage 1)
# =========================================================================

class ConditionEncoder(nn.Module):
    """Pack conditioning into memory (B, M, d) + pooled cond (B, d).
    Sources (role-embedded): DATA TOKENS (rich, queryable observation memory),
    z_f primary (flow), z_f extra (retrieval), and optionally pooled z_d. The
    data-token memory replaces the single pooled z_d so the decoder can attend to
    fine-grained data per generation step (breaks the single-z_d bottleneck)."""
    def __init__(self, d, n_layers=2, heads=8):
        super().__init__()
        # 0=data-token 1=z_f-primary(pooled) 2=z_f-extra(retrieval) 3=z_d 4=z_f-token(fine)
        self.role_emb = nn.Embedding(5, d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, n_layers)
        self.pool = nn.Linear(d, d)

    def forward(self, z_d, z_f, z_extra=None, data_tokens=None, z_f_tokens=None):
        toks = []
        if data_tokens is not None:           # (B, T, d) rich data memory
            toks.append(data_tokens + self.role_emb.weight[0])
        if z_f_tokens is not None:            # (B, M, d) fine formula memory (capacity + disambiguation)
            toks.append(z_f_tokens + self.role_emb.weight[4])
        if z_d is not None:                   # (B, d) pooled (optional / back-compat)
            toks.append(z_d.unsqueeze(1) + self.role_emb.weight[3])
        if z_f is not None:                   # (B, d) pooled z_f gist
            toks.append(z_f.unsqueeze(1) + self.role_emb.weight[1])
        if z_extra is not None:               # (B, K, d) retrieval z_f hints
            toks.append(z_extra + self.role_emb.weight[2])
        mem = torch.cat(toks, dim=1)          # (B, M, d)
        mem = self.tf(mem)
        cond = self.pool(mem.mean(dim=1))     # (B, d)
        return mem, cond


# =========================================================================
# AST decoder block (stage 2)
# =========================================================================

class MoEFFN(nn.Module):
    """Top-2 mixture-of-experts FFN. Failures cluster by STRUCTURE FAMILY
    (rational / transcendental / polynomial) so experts can specialize; capacity
    grows without per-token compute. Returns (out, aux_loss) where aux_loss is the
    load-balance term (Switch-style) to prevent router collapse."""
    def __init__(self, d, n_experts=8, k=2, mult=2):
        super().__init__()
        self.n_experts, self.k = n_experts, k
        self.gate = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Linear(d * mult, d))
            for _ in range(n_experts)])

    def forward(self, x):
        B, L, d = x.shape
        xf = x.reshape(-1, d)                                  # (T, d)
        logits = self.gate(xf)                                 # (T, E)
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.k, dim=-1)                # (T, k)
        topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(xf)
        for slot in range(self.k):
            idx = topi[:, slot]                                # (T,)
            w = topv[:, slot:slot + 1]
            for e in range(self.n_experts):
                m = idx == e
                if m.any():
                    out[m] += w[m] * self.experts[e](xf[m])
        # load-balance aux: mean prob per expert * fraction routed to it
        with torch.no_grad():
            one = torch.zeros_like(probs).scatter_(1, topi, 1.0)
            frac = one.mean(0)                                 # fraction tokens per expert
        aux = (frac * probs.mean(0)).sum() * self.n_experts
        return out.reshape(B, L, d), aux


class DecoderBlock(nn.Module):
    def __init__(self, d, heads, moe=False, n_experts=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.adaln1 = AdaLNZero(d)
        self.adaln2 = AdaLNZero(d)
        self.adaln3 = AdaLNZero(d)
        self.moe = moe
        if moe:
            self.ff = MoEFFN(d, n_experts=n_experts, k=2, mult=2)
        else:
            self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

    def forward(self, x, mem, cond, causal_mask):
        h, g1 = self.adaln1(x, cond)
        a, _ = self.self_attn(h, h, h, attn_mask=causal_mask)
        x = x + g1 * a
        h, g2 = self.adaln2(x, cond)
        c, _ = self.cross_attn(h, mem, mem)
        x = x + g2 * c
        h, g3 = self.adaln3(x, cond)
        if self.moe:
            ffo, aux = self.ff(h)
            x = x + g3 * ffo
            self._aux = aux
        else:
            x = x + g3 * self.ff(h)
            self._aux = None
        return x


class ASTDecoder(nn.Module):
    """Grammar-constrained pre-order AST decoder with conditioning."""
    def __init__(self, vocab_size=VOCAB_SIZE, d=512, n_layers=6, heads=8, max_len=128,
                 moe=False, n_experts=8, pointer=False):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([DecoderBlock(d, heads, moe=moe, n_experts=n_experts)
                                     for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab_size)
        self.const_head = nn.Linear(d, 1)        # regress CONST value
        self.max_len = max_len
        self.pointer = pointer
        if pointer:
            # copy mechanism: attend over RETRIEVED-NEIGHBOUR formula tokens and
            # copy whole subtrees the model can't compose from scratch. The gate
            # blends generate-from-vocab with copy-from-neighbour per step.
            self.nbr_tok_emb = nn.Embedding(vocab_size, d)
            self.copy_attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.copy_gate = nn.Linear(d, 1)
        self.register_buffer("arity", ARITY_BY_ID, persistent=False)

    def _pointer_mix(self, x, gen_logits, nbr_emb, nbr_ids, nbr_pad):
        """Blend vocab generation with copy-from-neighbour. Returns log-probs (B,L,V)."""
        gen_p = gen_logits.softmax(-1)                                   # (B,L,V)
        kpm = (nbr_pad == 0) if nbr_pad is not None else None
        _, attw = self.copy_attn(x, nbr_emb, nbr_emb, key_padding_mask=kpm,
                                 need_weights=True, average_attn_weights=True)  # (B,L,Tn)
        gate = torch.sigmoid(self.copy_gate(x))                          # (B,L,1)
        copy_p = x.new_zeros(gen_p.shape)
        ids = nbr_ids.unsqueeze(1).expand(-1, x.size(1), -1)             # (B,L,Tn)
        copy_p.scatter_add_(2, ids, attw)                               # tokens -> vocab
        mixed = (1 - gate) * gen_p + gate * copy_p
        return (mixed + 1e-9).log()

    def _causal(self, L, device):
        return torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

    def forward(self, mem, cond, node_types, const_vals, nbr_ids=None, nbr_pad=None):
        """
        Teacher-forced training.
        node_types: (B, L) target node ids ; const_vals: (B, L)
        nbr_ids: (B, Tn) retrieved-neighbour token ids (pointer mode).
        returns (type_logits OR logprobs) (B,L,V), const_pred (B,L). When pointer,
        the first return is LOG-PROBS (use NLL); else raw logits (use CE).
        """
        B, L = node_types.shape
        pos = torch.arange(L, device=node_types.device)
        x = self.tok_emb(node_types) + self.pos_emb(pos).unsqueeze(0)
        cmask = self._causal(L, node_types.device)
        aux = 0.0
        for blk in self.blocks:
            x = blk(x, mem, cond, cmask)
            if getattr(blk, '_aux', None) is not None:
                aux = aux + blk._aux
        self.aux_loss = aux
        gen_logits = self.head(x)
        if self.pointer and nbr_ids is not None:
            nbr_emb = self.nbr_tok_emb(nbr_ids)
            out = self._pointer_mix(x, gen_logits, nbr_emb, nbr_ids, nbr_pad)  # log-probs
        else:
            out = gen_logits
        return out, self.const_head(x).squeeze(-1)

    @torch.no_grad()
    def generate(self, mem, cond, max_len=None, temperature=1.0, greedy=False,
                 n_vars=None, nbr_ids=None, nbr_pad=None):
        """
        Grammar-constrained generation. Tracks #open child slots; stops when the
        tree is complete. Returns list of (node_ids, const_vals) per batch item.
        n_vars: optional int or (B,) — only VAR_0..VAR_{n-1} are allowed, masking
        out variables that don't exist in the task (kills variable hallucination).
        """
        device = mem.device
        B = mem.size(0)
        max_len = max_len or self.max_len
        var_ban = None
        if n_vars is not None:
            from data.ast_grammar import VARS as _VARS, NT2ID as _N
            var_ids = torch.tensor([_N[v] for v in _VARS], device=device)
            nv = (torch.full((B,), int(n_vars), device=device)
                  if isinstance(n_vars, int) else torch.as_tensor(n_vars, device=device))
            # ban VAR_k for k >= n_vars[b]
            ks = torch.arange(len(var_ids), device=device)
            var_ban = (ks.unsqueeze(0) >= nv.unsqueeze(1))   # (B, n_var_tokens)
        seqs = [[] for _ in range(B)]
        consts = [[] for _ in range(B)]
        open_slots = [1] * B          # start: need 1 expression (the root)
        done = [False] * B
        # Input begins with BOS (= EOS_ID, matching training shift-right).
        # Each step we feed [BOS, gen_1, ..., gen_{t-1}] and predict gen_t.
        inp = torch.full((B, 1), EOS_ID, dtype=torch.long, device=device)
        for step in range(max_len):
            if all(done): break
            L = inp.size(1)
            pos = torch.arange(L, device=device)
            x = self.tok_emb(inp) + self.pos_emb(pos).unsqueeze(0)
            cmask = self._causal(L, device)
            for blk in self.blocks:
                x = blk(x, mem, cond, cmask)
            gen_logits = self.head(x[:, -1])           # (B, V)
            cpred = self.const_head(x[:, -1]).squeeze(-1)
            if self.pointer and nbr_ids is not None:
                # pointer: convert last-step hidden to mixed log-probs, treat as logits
                lp = self._pointer_mix(x[:, -1:], gen_logits.unsqueeze(1),
                                       self.nbr_tok_emb(nbr_ids), nbr_ids, nbr_pad)
                logits = lp[:, -1]
            else:
                logits = gen_logits
            # grammar mask
            mask = self._grammar_mask(open_slots, done, device)
            if var_ban is not None:
                mask[:, var_ids] = mask[:, var_ids] & ~var_ban
            logits = logits.masked_fill(~mask, float("-inf"))
            if greedy:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.distributions.Categorical(logits=(logits / temperature)).sample()
            inp = torch.cat([inp, nxt.unsqueeze(1)], dim=1)
            for b in range(B):
                if done[b]:
                    continue
                nid = int(nxt[b])
                seqs[b].append(nid); consts[b].append(float(cpred[b]))
                # update open slots: consumed 1 slot, opened ARITY[nid]
                open_slots[b] += int(self.arity[nid]) - 1
                if open_slots[b] <= 0:
                    done[b] = True
        return [(seqs[b], consts[b]) for b in range(B)]

    def _grammar_mask(self, open_slots, done, device):
        """Allow only node-types that keep the tree valid. Here every non-special
        node fills exactly one open slot, so any operator/leaf is allowed while a
        slot is open; <eos>/<pad> are disallowed mid-tree (tree closes implicitly
        when open_slots hits 0)."""
        B = len(open_slots)
        mask = torch.ones(B, VOCAB_SIZE, dtype=torch.bool, device=device)
        # forbid <pad> and <eos> as emitted nodes (tree completion is structural)
        mask[:, PAD_ID] = False
        mask[:, EOS_ID] = False
        for b in range(B):
            if done[b]:
                mask[b, :] = False
                mask[b, PAD_ID] = True
        return mask


# =========================================================================
# Full decoder-decoder
# =========================================================================

class ConditioningDecoderDecoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d=512, cond_layers=2,
                 dec_layers=6, heads=8, max_len=128, moe=False, n_experts=8,
                 pointer=False):
        super().__init__()
        self.cond_enc = ConditionEncoder(d, cond_layers, heads)
        self.decoder = ASTDecoder(vocab_size, d, dec_layers, heads, max_len,
                                  moe=moe, n_experts=n_experts, pointer=pointer)
        self.moe = moe
        self.pointer = pointer

    @property
    def aux_loss(self):
        return getattr(self.decoder, 'aux_loss', 0.0)

    def forward(self, z_d, z_f, node_types, const_vals, z_extra=None, data_tokens=None,
                z_f_tokens=None, nbr_ids=None, nbr_pad=None):
        mem, cond = self.cond_enc(z_d, z_f, z_extra, data_tokens=data_tokens, z_f_tokens=z_f_tokens)
        return self.decoder(mem, cond, node_types, const_vals, nbr_ids=nbr_ids, nbr_pad=nbr_pad)

    @torch.no_grad()
    def generate(self, z_d, z_f, z_extra=None, data_tokens=None, z_f_tokens=None,
                 n_vars=None, nbr_ids=None, nbr_pad=None, **kw):
        mem, cond = self.cond_enc(z_d, z_f, z_extra, data_tokens=data_tokens, z_f_tokens=z_f_tokens)
        return self.decoder.generate(mem, cond, n_vars=n_vars, nbr_ids=nbr_ids,
                                     nbr_pad=nbr_pad, **kw)


if __name__ == "__main__":
    d = 256
    model = ConditioningDecoderDecoder(d=d, dec_layers=4)
    B, L = 4, 16
    z_d = torch.randn(B, d); z_f = torch.randn(B, d)
    nt = torch.randint(2, VOCAB_SIZE, (B, L)); cv = torch.randn(B, L)
    logits, cpred = model(z_d, z_f, nt, cv)
    print("train logits", logits.shape, "const", cpred.shape)
    # generation
    gen = model.generate(z_d, z_f, max_len=20, greedy=True)
    print("gen example node count:", [len(g[0]) for g in gen])
    toks = [ID2NT[i] for i in gen[0][0]]
    print("gen[0] nodes:", toks)
    print("params", sum(p.numel() for p in model.parameters()))
