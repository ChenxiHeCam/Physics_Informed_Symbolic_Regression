"""Equivalence-augmented flow training.

Frozen v6 manifold. Train a fresh flow, but with prob p_aug replace each formula's
TARGET z_f with the z_f of a random algebraically-equivalent form (multiply-through
rearrangement). The data condition is UNCHANGED (same law, same observations), so the
flow learns: given this data, produce ANY of the law's equivalent z_f's. This turns
each law's single point-target into a spread cloud (cos~0.66 apart) -> the flow hits
one more often -> higher ceiling. The frozen decoder decodes whichever form lands;
the root-match judge accepts any equivalent form.

Run twice for a clean A/B: --p-aug 0 (control) vs --p-aug 0.5 (treatment).
"""
import sys, os, time, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch, torch.nn as nn
from train.train_manifold import Manifold, MemmapPairDataset, build_ast_edges
from train.equiv_augment import gen_equiv_forms
from data.ast_grammar import parse_to_sympy, expr_to_nodes

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_graph(seqs):
    """List of token-seqs -> formula-encoder graph batch tensors."""
    enc_types, enc_const, enc_batch, enc_edges = [], [], [], []
    off = 0
    for bi, seq in enumerate(seqs):
        for t in seq:
            enc_types.append(int(t)); enc_const.append(0.0); enc_batch.append(bi)
        for (a, b) in build_ast_edges(seq):
            enc_edges.append((a + off, b + off))
        off += len(seq)
    ei = (torch.tensor(enc_edges).t().contiguous() if enc_edges
          else torch.zeros(2, 0, dtype=torch.long))
    return (torch.tensor(enc_types, device=DEVICE), torch.tensor(enc_const, device=DEVICE),
            ei.to(DEVICE), torch.tensor(enc_batch, device=DEVICE), len(seqs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_v6_fixed.pt')
    ap.add_argument('--cache', default='dataset_20260531/cache_v6')
    ap.add_argument('--max-rows', type=int, default=300000)
    ap.add_argument('--p-aug', type=float, default=0.5)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--steps', type=int, default=25000)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--noise', type=float, default=0.03)
    ap.add_argument('--save', default='sr_model/ckpt/flow_v6_aug.pt')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64)).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    from models.flow_matching import FlowMatchingTok
    flow = FlowMatchingTok(d=d, n_blocks=6).to(DEVICE); flow.train()
    print(f"fresh flow params {sum(p.numel() for p in flow.parameters()):,}  p_aug={args.p_aug}")

    ds = MemmapPairDataset(args.cache, args.max_rows, max_points=128, max_seq=ck.get('dec_max_len', 128))
    N = len(ds); mlen = ck.get('dec_max_len', 128)
    print(f"dataset rows: {N}")
    # row formula strings (only the rows we train on)
    exprs = []
    with open(os.path.join(args.cache, 'exprs.txt'), encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= N: break
            exprs.append(line.strip())
    aug_cache = {}                                   # row -> list[seq] (lazy)

    def aug_seqs(idx):
        if idx in aug_cache:
            return aug_cache[idx]
        out = []
        try:
            te = parse_to_sympy(exprs[idx])
            if te is not None and len(te.free_symbols) >= 2:
                vn = sorted([str(s) for s in te.free_symbols])
                for fs in gen_equiv_forms(exprs[idx], vn, max_len=mlen, max_forms=2):
                    r = expr_to_nodes(fs, var_order=vn)
                    if r is not None and len(r[0]) <= mlen:
                        out.append(r[0])
        except Exception:
            pass
        aug_cache[idx] = out
        return out

    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idxs = random.sample(range(N), min(args.batch, N))
        b = ds.collate(idxs); b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
        # choose target seqs: original, or (prob p_aug) a random equivalent form
        chosen = []
        for bi, idx in enumerate(idxs):
            picked = None
            if args.p_aug > 0 and random.random() < args.p_aug:
                a = aug_seqs(idx)
                if a:
                    picked = random.choice(a)
            if picked is None:
                L = int(ds.slen[idx]); picked = [int(x) for x in ds.seq[idx, :L]]
            chosen.append(picked)
        with torch.no_grad():
            z_d, cond = model.data_enc(b['points'], b['var_mask'], b['point_mask'],
                                       dims=b.get('dims'), return_tokens=True)
            et, ec, ee, eb, ng = build_graph(chosen)
            z_f = model.formula_enc(et, ec, ee, eb, ng)
            if args.noise > 0:
                cond = cond + args.noise * torch.randn_like(cond)
        loss = flow.loss(z_f, cond)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 1.0); opt.step(); sched.step()
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                samp = flow.sample(cond[:64], k=1, n_steps=20).squeeze(1)
                cos = torch.nn.functional.cosine_similarity(samp, z_f[:64], dim=-1).mean()
            print(f"step {step:5d} loss={loss.item():.4f} samp_cos={cos.item():.3f} "
                  f"augcached={len(aug_cache)} ({time.time()-t0:.0f}s)", flush=True)

    model.flow = flow                              # fold trained flow into the manifold
    torch.save({'model': model.state_dict(), 'd': d, 'dim_len': ck.get('dim_len', 0),
                'tok': ck.get('tok', False), 'n_tokens': ck.get('n_tokens', 16),
                'n_gin': ck.get('n_gin', 4), 'dec_layers': ck.get('dec_layers', 6),
                'dec_max_len': ck.get('dec_max_len', 64)}, args.save)
    print(f"Saved -> {args.save}")


if __name__ == '__main__':
    main()
