"""
Fix the v6 data-encoder collapse, then retrain the flow — the real blocker.

v6's z_f space + decoder are good (oracle 84.7%), but the DATA encoder collapsed:
align (z_d<->z_f InfoNCE) froze at ln(batch) because batch=64 gave too few fresh
negatives for the hard "infer formula from numbers" contrastive (v5 had batch 256
and it worked; a stale memory bank did NOT help). So the flow can't map data->z_f
(end-to-end 7% vs v5's 37%) even though the oracle is great.

Fix: FREEZE formula_enc + decoder (keep the good z_f geometry as a fixed target),
retrain ONLY data_enc + flow. That small footprint (no decoder, formula_enc no_grad)
fits a BIG batch -> many fresh in-batch negatives -> align finally learns -> tokens
become formula-discriminative -> flow can map data->z_f.

Losses: align(z_d, z_f_frozen) [big batch] + flow.loss(z_f_frozen, tokens).
"""
import sys, os, time, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
import torch.nn as nn

from train.train_manifold import Manifold, MemmapPairDataset, info_nce, MAX_VARS
from models.flow_matching import FlowMatchingTok
from models.encoders import DataEncoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_v6_big.pt')
    ap.add_argument('--cache', default='dataset_20260531/cache_v6')
    ap.add_argument('--max-rows', type=int, default=9000000)
    ap.add_argument('--batch', type=int, default=320)        # BIG batch = many fresh negatives
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--w-align', type=float, default=1.0)
    ap.add_argument('--w-flow', type=float, default=1.0)
    ap.add_argument('--noise', type=float, default=0.03)
    ap.add_argument('--balanced', action='store_true')
    ap.add_argument('--ckpt-every', type=int, default=0)
    ap.add_argument('--save', default='sr_model/ckpt/manifold_v6_fixed.pt')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6),
                     dec_max_len=ck.get('dec_max_len', 64)).to(DEVICE)
    model.load_state_dict(ck['model'])
    # v6's data encoder COLLAPSED to a constant output (batch-std=0) — a dead degenerate
    # state align can't escape. Re-initialize it FRESH so it learns data->z_f from scratch
    # against the (good, frozen) z_f target. Same for the flow.
    model.data_enc = DataEncoder(max_vars=MAX_VARS, d=d, n_isab=6,
                                 dim_len=ck.get('dim_len', 0),
                                 n_tokens=ck.get('n_tokens', 16),
                                 nondim=ck.get('tok', False)).to(DEVICE)
    model.flow = FlowMatchingTok(d=d, n_blocks=6).to(DEVICE)
    print("data encoder + flow RE-INITIALIZED (v6's data_enc was collapsed to constant)")
    # freeze formula_enc + decoder (fixed z_f target); train data_enc + flow
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.data_enc.parameters(): p.requires_grad_(True)
    for p in model.flow.parameters(): p.requires_grad_(True)
    model.temp.requires_grad_(True)
    train_params = ([model.temp] + list(model.data_enc.parameters())
                    + list(model.flow.parameters()))
    model.formula_enc.eval(); model.decoder.eval()
    model.data_enc.train(); model.flow.train()

    ds = MemmapPairDataset(args.cache, args.max_rows)
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    print(f"trainable params {sum(p.numel() for p in train_params):,}  batch={args.batch}  device={DEVICE}")

    sample_w = None
    if args.balanced:
        nvb = np.asarray(ds.nv[:len(ds)], dtype=np.int64)
        slb = np.asarray(ds.slen[:len(ds)], dtype=np.int64) // 8
        key = nvb * 100 + slb
        uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv].astype(np.float64); sample_w = w / w.sum()
        print(f"balanced over {len(uniq)} bins")

    def _save(path, step):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model': model.state_dict(), 'd': d, 'dim_len': ck.get('dim_len', 0),
                    'tok': ck.get('tok', False), 'n_tokens': ck.get('n_tokens', 16),
                    'n_gin': ck.get('n_gin', 4), 'dec_layers': ck.get('dec_layers', 6),
                    'dec_max_len': ck.get('dec_max_len', 64), 'step': step}, path)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        if sample_w is not None:
            idxs = np.random.choice(len(ds), size=min(args.batch, len(ds)), replace=True, p=sample_w).tolist()
        else:
            idxs = random.sample(range(len(ds)), min(args.batch, len(ds)))
        b = ds.collate(idxs)
        b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
        # data encoder WITH grad; formula encoder frozen (no grad) -> fixed z_f target
        z_d, toks = model.data_enc(b['points'], b['var_mask'], b['point_mask'],
                                   dims=b.get('dims'), return_tokens=True)
        with torch.no_grad():
            z_f = model.formula_enc(b['enc_types'], b['enc_const'],
                                    b['enc_edges'], b['enc_batch'], b['n_graphs'])
        l_align = info_nce(z_d, z_f, model.temp)
        cond = toks + args.noise * torch.randn_like(toks) if args.noise > 0 else toks
        l_flow = model.flow.loss(z_f, cond)                  # tokens have grad -> shaped to be flow-predictive
        loss = args.w_align * l_align + args.w_flow * l_flow
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                zd = torch.nn.functional.normalize(z_d, dim=-1)
                zf = torch.nn.functional.normalize(z_f, dim=-1)
                sim = zd @ zf.t()
                r1 = (sim.argmax(1) == torch.arange(sim.size(0), device=DEVICE)).float().mean()
                samp = model.flow.sample(toks[:64], k=1, n_steps=20).squeeze(1)
                scos = torch.nn.functional.cosine_similarity(samp, z_f[:64], dim=-1).mean()
            print(f"step {step:5d} align={l_align.item():.3f} flow={l_flow.item():.4f} | "
                  f"retr@1={r1.item():.3f} samp_cos={scos.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0:
            _save(args.save.replace('.pt', f'_step{step}.pt'), step); print(f"  [ckpt] step{step}", flush=True)
    _save(args.save, args.steps)
    print(f"Saved -> {args.save}")


if __name__ == '__main__':
    main()
