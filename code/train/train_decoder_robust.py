"""Decoder-robustness fine-tune (fast version, precomputed flow z_f).

The end-to-end gap (oracle ~75% vs flow-fed ~40%) is because the decoder was trained
teacher-forced on the TRUE z_f, so it DEMANDS a precise z_f the flow can't deliver
(samp_cos ~0.63). Here we freeze the encoders+flow and fine-tune ONLY the decoder on
a mix of the FLOW's actual (imprecise) z_f [precomputed per row] and the true z_f, so
the decoder learns to recover the true formula even from an imprecise z_f by leaning
on the data-token memory. Directly attacks the "decoder demands precise z_f" wall on
the single-vector model where the flow CAN generate (no token-sequence to produce).

Run on single-vector flow_v6_noaug (NOT the dual model, whose fine flow is broken).
"""
import sys, os, time, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from train.train_manifold import Manifold, MemmapPairDataset, load_state_compat
from data.ast_grammar import VOCAB_SIZE as V
from models.decoder import CONST_ID, EOS_ID

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def slog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/flow_v6_noaug.pt')
    ap.add_argument('--cache', default='dataset_20260531/cache_v6')
    ap.add_argument('--flow-zf', default='dataset_20260531/flow_zf_v6noaug.npy')
    ap.add_argument('--max-rows', type=int, default=300000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--p-flow', type=float, default=0.7)      # fraction decoded from flow z_f
    ap.add_argument('--p-drop', type=float, default=0.0)      # fraction with z_f ZEROED -> decode from data only (data-primary; coverage not bounded by flow z_f)
    ap.add_argument('--noise', type=float, default=0.1)       # jitter on precomputed flow z_f (sample diversity)
    ap.add_argument('--save', default='sr_model/ckpt/manifold_v6_robustdec.pt')
    ap.add_argument('--ckpt-every', type=int, default=1000)
    ap.add_argument('--neighbors', default='')                # (N,K) int ids into zf_raw -> feed retrieval z_extra (role 2) to the flow decoder. empty=OFF (keeps flow dec clean; retrieval otherwise lives only in dec7)
    ap.add_argument('--zf-raw', default='')                   # (M,d) z_f bank for retrieval
    ap.add_argument('--k-nbr', type=int, default=4)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64),
                     n_ftokens=ck.get('n_ftokens', 0), log_feats=ck.get('log_feats', False)).to(DEVICE)
    load_state_compat(model, ck['model']); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.decoder.parameters(): p.requires_grad_(True)
    model.decoder.train()
    print(f"decoder params (trainable) {sum(p.numel() for p in model.decoder.parameters()):,}  device={DEVICE}")

    ds = MemmapPairDataset(args.cache, args.max_rows, max_points=128, max_seq=ck.get('dec_max_len', 128))
    flow_zf = np.load(args.flow_zf, mmap_mode='r')               # (N,d) precomputed flow samples
    nbr_ids = np.load(args.neighbors) if args.neighbors and os.path.exists(args.neighbors) else None
    zf_raw = (torch.from_numpy(np.load(args.zf_raw)).to(DEVICE)
              if nbr_ids is not None and args.zf_raw and os.path.exists(args.zf_raw) else None)
    if nbr_ids is None or zf_raw is None:
        nbr_ids = zf_raw = None                                   # retrieval fully OFF unless both present
    N = min(len(ds), flow_zf.shape[0], len(nbr_ids) if nbr_ids is not None else len(ds))
    if nbr_ids is not None:
        print(f"retrieval ON (flow decoder): neighbors {nbr_ids.shape}, zf_raw {tuple(zf_raw.shape)}, k={args.k_nbr}")
    # std of the flow z_f cloud, for noise scaling
    zf_std = float(np.std(flow_zf[:20000]))
    print(f"rows {N} | flow_zf {flow_zf.shape} std~{zf_std:.3f}")

    opt = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idxs = random.sample(range(N), min(args.batch, N))
        b = ds.collate(idxs); b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
        B = b['tgt_types'].size(0)
        with torch.no_grad():
            z_d, toks = model.data_enc(b['points'], b['var_mask'], b['point_mask'],
                                       dims=b.get('dims'), return_tokens=True)
            z_f_true = model.formula_enc(b['enc_types'], b['enc_const'], b['enc_edges'],
                                         b['enc_batch'], b['n_graphs'])
            z_f_flow = torch.from_numpy(np.asarray(flow_zf[idxs])).to(DEVICE)        # (B,d) precomputed
            if args.noise > 0:
                z_f_flow = z_f_flow + args.noise * zf_std * torch.randn_like(z_f_flow)
            use_flow = (torch.rand(B, device=DEVICE) < args.p_flow).float().unsqueeze(1)
            z_f_used = use_flow * z_f_flow + (1 - use_flow) * z_f_true
            # data-primary dropout: zero z_f -> decoder MUST generate from data tokens
            if args.p_drop > 0:
                drop = (torch.rand(B, device=DEVICE) < args.p_drop).float().unsqueeze(1)
                z_f_used = (1 - drop) * z_f_used
        bos = torch.full((B, 1), EOS_ID, device=DEVICE)
        inp_t = torch.cat([bos, b['tgt_types'][:, :-1]], dim=1)
        inp_c = torch.cat([torch.zeros(B, 1, device=DEVICE), b['tgt_consts'][:, :-1]], dim=1)
        z_extra = None
        if nbr_ids is not None:                                   # retrieved-neighbour z_f as role-2 memory
            nb = torch.from_numpy(nbr_ids[idxs][:, :args.k_nbr].astype(np.int64)).to(DEVICE)
            z_extra = zf_raw[nb]                                  # (B, k_nbr, d)
        logits, cpred = model.decoder(z_d, z_f_used, inp_t, inp_c, data_tokens=toks, z_extra=z_extra)
        ce = F.cross_entropy(logits.reshape(-1, V), b['tgt_types'].reshape(-1), reduction='none')
        ce = (ce.reshape(B, -1) * b['tgt_mask']).sum() / b['tgt_mask'].sum()
        cpos = (b['tgt_types'] == CONST_ID).float()
        closs = (((slog(cpred) - slog(b['tgt_consts'])) ** 2) * cpos).sum() / cpos.sum().clamp_min(1)
        loss = ce + 0.05 * closs
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0); opt.step(); sched.step()
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                pred = logits.argmax(-1)
                m = use_flow.squeeze(1).bool()
                acc_all = (((pred == b['tgt_types']).float() * b['tgt_mask']).sum() / b['tgt_mask'].sum())
                if m.any():
                    fm = b['tgt_mask'][m]
                    acc_flow = (((pred[m] == b['tgt_types'][m]).float() * fm).sum() / fm.sum().clamp_min(1))
                else:
                    acc_flow = torch.tensor(0.0)
            print(f"step {step:5d} loss={loss.item():.3f} ce={ce.item():.3f} tok_acc={acc_all.item():.3f} "
                  f"tok_acc_flowzf={acc_flow.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0:
            out = {k: ck.get(k) for k in ('d', 'dim_len', 'tok', 'n_tokens', 'n_gin', 'dec_layers', 'dec_max_len', 'n_ftokens', 'log_feats')}
            out['model'] = model.state_dict()
            torch.save(out, args.save)
            print(f"  [ckpt @ {step} -> {args.save}]", flush=True)

    out = {k: ck.get(k) for k in ('d', 'dim_len', 'tok', 'n_tokens', 'n_gin', 'dec_layers', 'dec_max_len', 'n_ftokens', 'log_feats')}
    out['model'] = model.state_dict()
    torch.save(out, args.save)
    print(f"Saved -> {args.save}")


if __name__ == '__main__':
    main()
