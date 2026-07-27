"""
Path-B overfit test: formula_encoder -> z_f -> decoder -> original formula.

Goal: prove the architecture CAN reconstruct. Take a small batch of real
formulas, train encoder+decoder end-to-end to overfit. If teacher-forced
token accuracy and free-generation exact-match climb toward 1.0, the
decoder-decoder design is sound. If it can't even overfit, the design is broken.

This isolates the DECODER path (no data encoder, no flow matching).
z_f comes from the formula's own AST via the formula encoder.
"""
import sys, os, gzip, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.ast_grammar import (expr_to_nodes, nodes_to_expr, ID2NT, NT2ID,
                              VOCAB_SIZE as V)
from models.formula_encoder import ASTEncoder
from models.decoder import ConditioningDecoderDecoder, CONST_ID, PAD_ID, EOS_ID

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = 256
N_FORMULAS = 64
STEPS = 800
LR = 3e-4

NODES = 'dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz'
if not os.path.exists(NODES):
    NODES = 'dataset_20260531/unified_nodes.jsonl.gz'

# ---- load a small batch of parseable real formulas ----
formulas = []
with gzip.open(NODES, 'rt', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        expr = r.get('expr', '')
        res = expr_to_nodes(expr)
        if res is None: continue
        seq, consts, vars_ = res
        if 3 <= len(seq) <= 24:   # keep short for overfit
            formulas.append((expr, seq, consts, vars_))
        if len(formulas) >= N_FORMULAS: break
print(f"Loaded {len(formulas)} short parseable formulas")

# ---- build AST graph batch for the formula encoder (level-1 only) ----
def build_ast_graph(seq):
    """Pre-order seq -> node list + parent->child edges (undirected)."""
    from data.ast_grammar import ARITY, ID2NT
    types = list(seq)
    edges = []
    # reconstruct tree edges from pre-order + arity
    pos = [0]
    def walk(parent):
        i = pos[0]; pos[0] += 1
        nt = ID2NT[types[i]]
        ar = ARITY.get(nt, 0)
        for _ in range(ar):
            child = pos[0]
            edges.append((i, child)); edges.append((child, i))
            walk(i)
        return i
    try:
        walk(-1)
    except Exception:
        pass
    return types, edges

# pad target sequences
maxL = max(len(s) for _, s, _, _ in formulas)
def pad(seq, consts):
    L = len(seq)
    s = seq + [PAD_ID] * (maxL - L)
    c = consts + [0.0] * (maxL - L)
    m = [1] * L + [0] * (maxL - L)
    return s, c, m

# precompute encoder graph batch + decoder targets
enc_types, enc_batch, enc_edges = [], [], []
tgt_types, tgt_consts, tgt_mask = [], [], []
node_off = 0
for gi, (expr, seq, consts, vars_) in enumerate(formulas):
    types, edges = build_ast_graph(seq)
    enc_types.extend(types)
    enc_batch.extend([gi] * len(types))
    for (a, b) in edges:
        enc_edges.append((a + node_off, b + node_off))
    node_off += len(types)
    s, c, m = pad(seq, consts)
    tgt_types.append(s); tgt_consts.append(c); tgt_mask.append(m)

enc_types = torch.tensor(enc_types, device=DEVICE)
enc_const = torch.zeros(len(enc_types), device=DEVICE)  # const feature for encoder nodes (approx 0)
enc_batch = torch.tensor(enc_batch, device=DEVICE)
enc_ei = torch.tensor(enc_edges, device=DEVICE).t().contiguous() if enc_edges else torch.zeros(2,0,dtype=torch.long,device=DEVICE)
tgt_types = torch.tensor(tgt_types, device=DEVICE)
tgt_consts = torch.tensor(tgt_consts, device=DEVICE)
tgt_mask = torch.tensor(tgt_mask, device=DEVICE).float()
B = len(formulas)

# ---- models ----
encoder = ASTEncoder(V, d=D, n_gin=3).to(DEVICE)
decoder = ConditioningDecoderDecoder(vocab_size=V, d=D, dec_layers=4, max_len=maxL+2).to(DEVICE)
opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

print(f"encoder {sum(p.numel() for p in encoder.parameters()):,}  "
      f"decoder {sum(p.numel() for p in decoder.parameters()):,}  device={DEVICE}")

# teacher-forcing: input = shifted-right targets
def shift(types, consts):
    bos = torch.full((B, 1), EOS_ID, device=DEVICE)  # BOS proxy
    inp_t = torch.cat([bos, types[:, :-1]], dim=1)
    bos_c = torch.zeros(B, 1, device=DEVICE)
    inp_c = torch.cat([bos_c, consts[:, :-1]], dim=1)
    return inp_t, inp_c

t0 = time.time()
for step in range(1, STEPS + 1):
    encoder.train(); decoder.train()
    z_f = encoder(enc_types, enc_const, enc_ei, enc_batch, B)   # (B, D)
    # path B: condition on z_f only (z_d = zeros placeholder for this isolation test)
    z_d = torch.zeros_like(z_f)
    inp_t, inp_c = shift(tgt_types, tgt_consts)
    logits, cpred = decoder(z_d, z_f, inp_t, inp_c)
    # type loss (masked CE) + const loss (masked MSE on CONST positions)
    type_loss = F.cross_entropy(logits.reshape(-1, V), tgt_types.reshape(-1),
                                reduction='none').reshape(B, -1)
    type_loss = (type_loss * tgt_mask).sum() / tgt_mask.sum()
    # signed-log compress huge consts (some are 3e21) to keep MSE finite
    def slog(x): return torch.sign(x) * torch.log1p(torch.abs(x))
    const_pos = (tgt_types == CONST_ID).float()
    cl = ((slog(cpred) - slog(tgt_consts)) ** 2) * const_pos
    const_loss = cl.sum() / const_pos.sum().clamp_min(1)
    loss = type_loss + 0.05 * const_loss
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(list(encoder.parameters())+list(decoder.parameters()), 1.0)
    opt.step()

    if step % 100 == 0 or step == 1:
        with torch.no_grad():
            pred = logits.argmax(-1)
            tok_acc = ((pred == tgt_types).float() * tgt_mask).sum() / tgt_mask.sum()
        print(f"step {step:4d}  loss={loss.item():.4f}  type_tok_acc={tok_acc.item():.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

# ---- free generation exact-match ----
encoder.eval(); decoder.eval()
with torch.no_grad():
    z_f = encoder(enc_types, enc_const, enc_ei, enc_batch, B)
    z_d = torch.zeros_like(z_f)
    gens = decoder.generate(z_d, z_f, max_len=maxL+2, greedy=True)

exact = 0
shown = 0
for i, (expr, seq, consts, vars_) in enumerate(formulas):
    gen_seq, gen_consts = gens[i]
    # compare node-type sequence (ignore trailing)
    match = gen_seq[:len(seq)] == seq[:len(gen_seq)] and len(gen_seq) >= len(seq) - 1
    if match: exact += 1
    if shown < 6:
        back = nodes_to_expr(gen_seq, gen_consts, vars_)
        print(f"\n  TRUTH: {expr[:55]}")
        print(f"  GEN  : {back}")
        shown += 1

print(f"\n=== Path-B overfit result ===")
print(f"Free-generation structural exact: {exact}/{B} ({100*exact/B:.0f}%)")
print(f"(if this is high, decoder-decoder CAN reconstruct from z_f)")
