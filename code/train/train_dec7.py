"""decoder v7: reference-style autoregressive captioning decoder, data-first with
flow/retrieval as auxiliary conditioning. Representation (data_enc / formula_enc /
flow) is FROZEN from the proven full512 checkpoint; only the new BIG decoder,
the feature tokenizer and the data-token heads train.

Conditioning memory (all via the existing ConditionEncoder roles):
  role0 data   : encoder token memory (pma_tok/out_tok, unfrozen — they were
                 untrained in no-tok ckpts) + FEATURE tokens (loglog slope/corr,
                 magnitude stats, SI dims — the features that proved decisive in
                 the reference diagnostic, now fed to our OWN model)
  role2 z_extra: top-K retrieval neighbour z_f (precomputed ids -> raw z_f)
  role3 z_d    : pooled data vector (frozen path)
  role4 z_f_tok: K flow samples from the FROZEN flow (decoder learns when to
                 trust them — they are near-truth on memorized tasks, noise on
                 novel compositions; see flow-proximity diagnostic 2026-06-13)

Usage:
  python train_dec7.py --cache <dir> --base sr_model/ckpt/manifold_full512.pt \
      --neighbors <ids.npy> --zf-raw sr_model/ckpt/zf_full512_raw.npy \
      --steps 60000 --batch 48 --save sr_model/ckpt/dec7.pt
"""
import sys, os, json, math, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.ast_grammar import MAX_VARS, VOCAB_SIZE
from models.decoder import ConditioningDecoderDecoder, PAD_ID, EOS_ID, CONST_ID
from train.train_manifold import Manifold, MemmapPairDataset, load_state_compat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatTokenizer(nn.Module):
    """Per-variable explicit features -> conditioning tokens.
    For each channel c (vars + target): [loglog slope vs target, loglog corr,
    log10|min|, log10|med|, log10|max|, frac_negative, dims(7)] -> Linear -> d.
    Computed on GPU from the raw padded points (cheap closed-form regressions)."""
    def __init__(self, d, dim_len=7):
        super().__init__()
        self.dim_len = dim_len
        self.proj = nn.Sequential(nn.Linear(6 + dim_len, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, points, var_mask, point_mask, dims):
        B, N, C = points.shape
        eps = 1e-9
        pm = point_mask.unsqueeze(-1)                          # (B,N,1)
        y = points[..., MAX_VARS:MAX_VARS + 1]                 # (B,N,1)
        ly = torch.log(y.abs() + eps)
        lx = torch.log(points.abs() + eps)                     # (B,N,C)
        cnt = pm.sum(1).clamp_min(2.0)                         # (B,1)
        mx = (lx * pm).sum(1) / cnt                            # (B,C)
        my = (ly * pm).sum(1) / cnt                            # (B,1)
        vx = (((lx - mx.unsqueeze(1)) * pm) ** 2).sum(1) / cnt
        vy = (((ly - my.unsqueeze(1)) * pm) ** 2).sum(1) / cnt
        cov = (((lx - mx.unsqueeze(1)) * (ly - my.unsqueeze(1))) * pm).sum(1) / cnt
        slope = (cov / vx.clamp_min(1e-6)).clamp(-8, 8)        # (B,C)
        corr = (cov / (vx.clamp_min(1e-9) * vy.clamp_min(1e-9)).sqrt()).clamp(-1, 1)
        big = points.masked_fill(pm == 0, float('nan'))
        amin = torch.log10(big.abs().nan_to_num(nan=1.0).clamp_min(1e-12).amin(1))
        amed = torch.log10((points.abs() * pm).sum(1) / cnt + eps)    # mean|.| proxy for median
        amax = torch.log10(big.abs().nan_to_num(nan=1.0).amax(1).clamp_min(1e-12))
        fneg = ((points < 0).float() * pm).sum(1) / cnt
        if dims is None:
            dims = points.new_zeros(B, C, self.dim_len)
        feats = torch.stack([slope, corr, amin, amed, amax, fneg], -1)  # (B,C,6)
        feats = torch.cat([feats.nan_to_num(0.0), dims[..., :self.dim_len]], -1)
        tok = self.proj(feats)                                  # (B,C,d)
        chan_mask = torch.cat([var_mask, var_mask.new_ones(B, 1)], 1)  # target always on
        return tok * chan_mask.unsqueeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--base', required=True)               # frozen full512 ckpt
    ap.add_argument('--neighbors', default='')             # (N,K) int32 ids into zf_raw
    ap.add_argument('--zf-raw', default='')                # raw z_f matrix for neighbour ids
    ap.add_argument('--max-rows', type=int, default=2000000)
    ap.add_argument('--max-points', type=int, default=128)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--dec-layers', type=int, default=16)
    ap.add_argument('--cond-layers', type=int, default=4)
    ap.add_argument('--k-flow', type=int, default=6)
    ap.add_argument('--k-nbr', type=int, default=8)
    ap.add_argument('--nbr-dropout', type=float, default=0.5)   # per-sample blank the WHOLE retrieval block -> decoder can't rely on the (leaky: train formulas ARE in the index) neighbor z_f
    ap.add_argument('--flow-steps', type=int, default=8)
    ap.add_argument('--w-const', type=float, default=0.3)
    ap.add_argument('--ckpt-every', type=int, default=10000)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--trans-frac', type=float, default=0.0)   # fraction of each batch forced
                                                               # from TRANSCENDENTAL formulas
                                                               # (exp/log/trig in target seq) —
                                                               # the v1 autopsy's dominant failure
    ap.add_argument('--resume', default='')     # resume dec/ftok/token-head states
    ap.add_argument('--clean-index', default='') # npz with clean mask + struct tags ->
                                                 # train ONLY on clean rows (cache_v6 is
                                                 # ~49% pathological: target_zero/var_const)
    ap.add_argument('--struct-frac', type=float, default=0.0)  # fraction of batch from
                                                 # structure-rich clean rows (deep rational /
                                                 # sqrt / div) — the v3 missing-structure recipe
    ap.add_argument('--moe', action='store_true')             # mixture-of-experts FFN decoder
    ap.add_argument('--n-experts', type=int, default=8)
    ap.add_argument('--w-aux', type=float, default=0.01)      # MoE load-balance weight
    ap.add_argument('--pointer', action='store_true')         # copy-from-neighbour pointer-generator
    ap.add_argument('--pure', action='store_true')            # PURE pretrain: condition ONLY on data
                                                              # tokens (no flow z_f / no nbr); arch keeps slots
    ap.add_argument('--nbr-toklen', type=int, default=40)     # max tokens per neighbour formula
    ap.add_argument('--save', required=True)
    args = ap.parse_args()

    # ---- frozen base (representation) ----
    ck = torch.load(args.base, map_location=DEVICE)
    d = ck['d']
    base = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                    n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                    dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64),
                    n_ftokens=ck.get('n_ftokens', 0), log_feats=ck.get('log_feats', False),
                    class_feats=ck.get('class_feats', False), dim_head=ck.get('dim_head', False),
                    robust_norm=ck.get('robust_norm', False)).to(DEVICE)
    load_state_compat(base, ck['model'])
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    # unfreeze ONLY the token heads of the data encoder (untrained in no-tok ckpt)
    for m in (base.data_enc.pma_tok, base.data_enc.out_tok):
        for p in m.parameters():
            p.requires_grad = True

    # ---- new big decoder + feature tokenizer ----
    dec = ConditioningDecoderDecoder(vocab_size=VOCAB_SIZE, d=d,
                                     cond_layers=args.cond_layers,
                                     dec_layers=args.dec_layers,
                                     max_len=80, moe=args.moe,
                                     n_experts=args.n_experts,
                                     pointer=args.pointer).to(DEVICE)
    ftok = FeatTokenizer(d, dim_len=max(ck.get('dim_len', 7), 1)).to(DEVICE)
    if args.resume and os.path.exists(args.resume):
        rk = torch.load(args.resume, map_location=DEVICE)
        # strict=False so a MoE decoder can WARM-START from a non-MoE ckpt: shared
        # attn/emb/heads load, the 8 expert FFNs init fresh (or vice-versa).
        miss = dec.load_state_dict(rk['dec'], strict=False)
        ftok.load_state_dict(rk['ftok'])
        if args.moe:
            print(f'resume(strict=False): {len(miss.missing_keys)} new keys '
                  f'(MoE experts init fresh)')
        base.data_enc.pma_tok.load_state_dict(rk['pma_tok'])
        base.data_enc.out_tok.load_state_dict(rk['out_tok'])
        print(f'resumed from {args.resume} (step {rk.get("step")})')

    # ---- neighbour assets ----
    nbr_ids = np.load(args.neighbors) if args.neighbors and os.path.exists(args.neighbors) else None
    zf_raw = None
    if args.zf_raw and os.path.exists(args.zf_raw):
        zf_raw = torch.from_numpy(np.load(args.zf_raw)).to(DEVICE)   # (M,d) raw

    ds = MemmapPairDataset(args.cache, args.max_rows, max_points=args.max_points)
    n_train = len(ds)
    if nbr_ids is not None:
        n_train = min(n_train, len(nbr_ids))
        print(f'neighbors cover {len(nbr_ids)} rows -> training on {n_train}')

    params = [p for p in dec.parameters()] + [p for p in ftok.parameters()] + \
             [p for m in (base.data_enc.pma_tok, base.data_enc.out_tok) for p in m.parameters()]
    n_trainable = sum(p.numel() for p in params)
    print(f'trainable params: {n_trainable/1e6:.1f}M (decoder '
          f'{sum(p.numel() for p in dec.parameters())/1e6:.1f}M)')
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler('cuda')

    # clean-row restriction (+ structure pools) — v3
    clean_pool = None; struct_pool = None
    if args.clean_index and os.path.exists(args.clean_index):
        z = np.load(args.clean_index)
        clean = z['clean'][:n_train]
        clean_pool = np.where(clean)[0]
        if 'tag_struct' in z:                      # precise Pow(Add) missing-structure tag
            rich = z['tag_struct'][:n_train] & clean
        else:
            rich = (z['tag_deep'] | z['tag_sqrt'] | z['tag_div'])[:n_train] & clean
        struct_pool = np.where(rich)[0]
        print(f'clean rows: {len(clean_pool)}/{n_train} ({100*len(clean_pool)/n_train:.1f}%); '
              f'structure-rich clean: {len(struct_pool)} -> {args.struct_frac:.0%}/batch')

    # transcendental row subset for oversampling
    trans_idx = None
    if args.trans_frac > 0:
        from data.ast_grammar import NT2ID as _N
        tids = [_N[t] for t in ('exp', 'log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh')
                if t in _N]
        m = np.isin(ds.seq[:n_train], tids).any(axis=1)
        if clean_pool is not None:
            cm = np.zeros(n_train, bool); cm[clean_pool] = True; m = m & cm
        trans_idx = np.where(m)[0]
        print(f'transcendental rows: {len(trans_idx)}/{n_train} '
              f'({100*len(trans_idx)/n_train:.1f}%) -> {args.trans_frac:.0%} of each batch')

    rng = np.random.default_rng(0)
    t0 = time.time()
    def base_sample(m):
        # draw m indices from the clean pool if given, else uniform
        if clean_pool is not None:
            return rng.choice(clean_pool, m)
        return rng.integers(0, n_train, m)

    for step in range(1, args.steps + 1):
        parts = []
        rem = args.batch
        if struct_pool is not None and len(struct_pool) and args.struct_frac > 0:
            ks = int(args.batch * args.struct_frac)
            parts.append(rng.choice(struct_pool, ks)); rem -= ks
        if trans_idx is not None and len(trans_idx) > 0 and args.trans_frac > 0:
            kt = int(args.batch * args.trans_frac)
            parts.append(rng.choice(trans_idx, kt)); rem -= kt
        parts.append(base_sample(max(rem, 0)))
        idxs = np.concatenate(parts)
        if os.environ.get('DEC7_IDXLOG'):
            with open(os.environ['DEC7_IDXLOG'], 'w') as _f:
                _f.write(f'step {step} idxs ' + ' '.join(str(int(x)) for x in idxs) + '\n')
                _f.flush()
        b = ds.collate(list(idxs))
        points = b['points'].to(DEVICE); vm = b['var_mask'].to(DEVICE)
        pm = b['point_mask'].to(DEVICE)
        dims = b.get('dims'); dims = dims.to(DEVICE) if dims is not None else None
        tt = b['tgt_types'].to(DEVICE); tc = b['tgt_consts'].to(DEVICE)
        tm = b['tgt_mask'].to(DEVICE)

        with torch.no_grad():
            zf_flow = None
            if not args.pure:
                zd = base.data_enc(points, vm, pm, dims=dims)          # frozen pooled
                zf_flow = base.flow.sample(zd, k=args.k_flow, n_steps=args.flow_steps)  # (B,K,d)
        # data tokens: trainable heads over frozen trunk -> rerun WITH grad on heads
        _, dtok = base.data_enc(points, vm, pm, dims=dims, return_tokens=True)
        feat = ftok(points, vm, pm, dims)                              # (B,C,d)
        data_tokens = torch.cat([dtok, feat], 1)
        z_extra = None
        if not args.pure and nbr_ids is not None and zf_raw is not None:
            ids = torch.from_numpy(nbr_ids[idxs][:, :args.k_nbr].astype(np.int64)).to(DEVICE)
            z_extra = zf_raw[ids]                                      # (B,K,d)
            if args.nbr_dropout > 0:                                  # retrieval dropout (user): train formulas ARE in the latent index -> self-retrieval leaks the answer; blank the whole block per-sample so decoder learns retrieval is an optional hint, not a crutch
                keep = (torch.rand(z_extra.size(0), 1, 1, device=DEVICE) >= args.nbr_dropout).float()
                z_extra = z_extra * keep

        # pointer: gather the neighbour formulas' TOKEN sequences (index pos == cache
        # row, since the index was built sequentially) to copy subtrees from.
        nbr_ids_t = nbr_pad_t = None
        if args.pointer and not args.pure and nbr_ids is not None:
            nb = nbr_ids[idxs][:, :args.k_nbr]                          # (B,K) cache rows
            Ln = args.nbr_toklen
            seqs = ds.seq[nb.reshape(-1)][:, :Ln]                       # (B*K, Ln) int16, pad=-1
            seqs = seqs.reshape(len(idxs), -1)                         # (B, K*Ln)
            pad = (seqs >= 0)
            seqs = np.where(pad, seqs, PAD_ID).astype(np.int64)
            nbr_ids_t = torch.from_numpy(seqs).to(DEVICE)
            nbr_pad_t = torch.from_numpy(pad).to(DEVICE)

        # teacher forcing: input = [BOS, t_0..t_{L-2}], predict t_0..t_{L-1}
        inp = torch.cat([torch.full_like(tt[:, :1], EOS_ID), tt[:, :-1]], 1)
        with torch.amp.autocast('cuda'):
            out, cpred = dec(None, None, inp, tc,
                             z_extra=z_extra, data_tokens=data_tokens,
                             z_f_tokens=zf_flow, nbr_ids=nbr_ids_t, nbr_pad=nbr_pad_t)
            if args.pointer:
                ce = F.nll_loss(out.reshape(-1, VOCAB_SIZE), tt.reshape(-1),
                                ignore_index=PAD_ID)     # out is log-probs
            else:
                ce = F.cross_entropy(out.reshape(-1, VOCAB_SIZE), tt.reshape(-1),
                                     ignore_index=PAD_ID)
            cmask = tm * (tt == CONST_ID).float()
            # signed-log space (mirrors train_manifold): raw consts reach 1e21+
            slog = lambda x: torch.sign(x) * torch.log1p(torch.abs(x))
            closs = ((slog(cpred) - slog(tc)) ** 2 * cmask).sum() / cmask.sum().clamp_min(1.0)
            loss = ce + args.w_const * closs
            if args.moe:
                loss = loss + args.w_aux * dec.aux_loss
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            print(f'step {step}: non-finite loss, batch skipped', flush=True)
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True)   # NaN/inf grads: drop step, keep weights
            scaler.update()
            continue
        scaler.step(opt); scaler.update(); sched.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                pred = out.argmax(-1)
                acc = ((pred == tt).float() * tm).sum() / tm.sum()
            print(f'step {step:6d} loss={loss.item():.3f} ce={ce.item():.3f} '
                  f'const={closs.item():.3f} tok_acc={acc.item():.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0:
            _save(args, dec, ftok, base, ck, step)
    _save(args, dec, ftok, base, ck, args.steps)
    print(f'Saved -> {args.save}')


def _save(args, dec, ftok, base, ck, step):
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    torch.save({'dec': dec.state_dict(), 'ftok': ftok.state_dict(),
                'pma_tok': base.data_enc.pma_tok.state_dict(),
                'out_tok': base.data_enc.out_tok.state_dict(),
                'base_ckpt': args.base, 'd': ck['d'],
                'dec_layers': args.dec_layers, 'cond_layers': args.cond_layers,
                'k_flow': args.k_flow, 'k_nbr': args.k_nbr,
                'moe': args.moe, 'n_experts': args.n_experts,
                'pointer': args.pointer, 'nbr_toklen': args.nbr_toklen,
                'step': step}, args.save)


if __name__ == '__main__':
    main()
