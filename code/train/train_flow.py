"""
Phase 2: dedicated flow-matching training.

Load the pretrained manifold (frozen encoders), train ONLY the flow velocity
field to model p(z_f | z_d) well, on the full pair set. This is what makes
flow-sampled z_f decode into correct formulas.

Slightly larger noise on z_d input (user spec) so flow learns to map dirty
data to the formula distribution.
"""
import sys, os, gzip, json, time, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
import torch.nn as nn

from train.train_manifold import Manifold, PairDataset, MemmapPairDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_full.pt')
    ap.add_argument('--pairs', default='dataset_20260531/data_pairs_full.jsonl.gz')
    ap.add_argument('--cache', default='')  # memmap cache dir (full-data, memory-safe)
    ap.add_argument('--max-rows', type=int, default=600000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--noise', type=float, default=0.03)   # extra conditioning noise
    ap.add_argument('--balanced', action='store_true')     # uniform across complexity
    ap.add_argument('--reset-flow', action='store_true')   # re-init flow (warmup flow chased a moving target -> start clean on frozen z_f)
    ap.add_argument('--clean-index', default='')           # npz clean mask -> train flow ONLY on clean rows
    ap.add_argument('--boost-file', default='')            # npz {boost} -> SPECIALIZE: oversample class
    ap.add_argument('--boost-frac', type=float, default=0.0)
    ap.add_argument('--save', default='sr_model/ckpt/manifold_flow.pt')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE)
    d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6),
                     dec_max_len=ck.get('dec_max_len', 64),
                     n_ftokens=ck.get('n_ftokens', 0), log_feats=ck.get('log_feats', False)).to(DEVICE)
    from train.train_manifold import load_state_compat
    load_state_compat(model, ck['model'])
    if args.reset_flow:
        # frozen manifold => z_f is now FIXED; re-init the flow and train it from
        # scratch on that fixed target (the joint-warmup flow drifted and is useless).
        from models.flow_matching import FlowMatchingTok, FlowMatching
        model.flow = (FlowMatchingTok(d=d, n_blocks=10) if model.tok
                      else FlowMatching(d=d, n_blocks=10)).to(DEVICE)
        print("flow re-initialized (training fresh on frozen z_f)")
    # freeze everything except flow
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.flow.parameters(): p.requires_grad_(True)
    model.eval(); model.flow.train()

    ds = (MemmapPairDataset(args.cache, args.max_rows, max_points=400) if args.cache
          else PairDataset(args.pairs, args.max_rows))
    clean_pool = None
    if args.clean_index and os.path.exists(args.clean_index):
        cmask = np.load(args.clean_index)['clean'][:len(ds)]
        clean_pool = np.nonzero(cmask)[0]
        print(f"clean-index: flow trains on {len(clean_pool)}/{len(ds)} clean rows "
              f"({100*len(clean_pool)/len(ds):.1f}%)")
    boost_pool = None
    if args.boost_file and os.path.exists(args.boost_file) and args.boost_frac > 0:
        bmask = np.load(args.boost_file)['boost'][:len(ds)]
        boost_pool = np.nonzero(bmask)[0]
        print(f"SPECIALIZE: boost-frac {args.boost_frac} from {len(boost_pool)} boosted rows")
    opt = torch.optim.AdamW(model.flow.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    print(f"flow params {sum(p.numel() for p in model.flow.parameters()):,}  device={DEVICE}")

    # complexity-balanced sampling: each step samples uniformly across complexity
    # bins (by #vars x seq-length bucket) so rare complex formulas are not drowned
    # out by abundant simple ones — flow learns z_f across the whole spectrum.
    sample_w = None
    if args.balanced and args.cache:
        nv = np.asarray(ds.nv[:len(ds)], dtype=np.int64)
        slb = (np.asarray(ds.slen[:len(ds)], dtype=np.int64) // 8)  # seq-length buckets
        key = nv * 100 + slb
        uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv].astype(np.float64)      # inverse bin frequency
        sample_w = w / w.sum()
        print(f"balanced sampling over {len(uniq)} complexity bins")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        if sample_w is not None:
            idxs = np.random.choice(len(ds), size=min(args.batch, len(ds)),
                                    replace=True, p=sample_w).tolist()
        elif clean_pool is not None:
            idxs = np.random.choice(clean_pool, size=min(args.batch, len(clean_pool)),
                                    replace=False).tolist()
        else:
            idxs = random.sample(range(len(ds)), min(args.batch, len(ds)))
        if boost_pool is not None:                       # SPECIALIZE: swap in boosted-class rows
            nb = int(args.batch * args.boost_frac)
            idxs[:nb] = np.random.choice(boost_pool, size=nb, replace=False).tolist()
        b = ds.collate(idxs)
        b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            if model.tok:
                z_d, cond = model.data_enc(b['points'], b['var_mask'], b['point_mask'],
                                           dims=b.get('dims'), return_tokens=True)  # cond = tokens
            else:
                cond = model.data_enc(b['points'], b['var_mask'], b['point_mask'], dims=b.get('dims'))
            z_f = model.formula_enc(b['enc_types'], b['enc_const'],
                                    b['enc_edges'], b['enc_batch'], b['n_graphs'])
            if args.noise > 0:
                cond = cond + args.noise * torch.randn_like(cond)   # dirtier conditioning
        loss = model.flow.loss(z_f, cond)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.flow.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == 1:
            # sample-reconstruction proxy: how close is a sampled z_f to true z_f?
            with torch.no_grad():
                samp = model.flow.sample(cond[:64], k=1, n_steps=20).squeeze(1)
                cos = torch.nn.functional.cosine_similarity(
                    samp, z_f[:64], dim=-1).mean()
            print(f"step {step:5d} flow_loss={loss.item():.4f} samp_cos={cos.item():.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({'model': model.state_dict(), 'd': d, 'dim_len': ck.get('dim_len', 0),
                'tok': ck.get('tok', False), 'n_tokens': ck.get('n_tokens', 16),
                'n_gin': ck.get('n_gin', 4), 'dec_layers': ck.get('dec_layers', 6),
                'dec_max_len': ck.get('dec_max_len', 64),
                'n_ftokens': ck.get('n_ftokens', 0),
                'log_feats': ck.get('log_feats', False)}, args.save)
    print(f"Saved -> {args.save}")


if __name__ == '__main__':
    main()
