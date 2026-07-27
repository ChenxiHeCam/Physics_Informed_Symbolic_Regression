"""
Build the retrieval z_f index from the CACHE (target-last seqs), so the index
latents live in the same VAR-slot convention the manifold was trained on and
align with z_d. Dedups by expr string (keeps first target variant per formula).

Output:
  <prefix>_index.npy   (N, d) float32   normalized formula latents
  <prefix>_meta.jsonl  N lines          {expr}
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch

from train.train_manifold import Manifold, build_ast_edges, MAX_VARS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_v4_full.pt')
    ap.add_argument('--cache', default='dataset_20260531/cache_v4_full')
    ap.add_argument('--max', type=int, default=400000)   # max unique formulas
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--out-prefix', default='sr_model/ckpt/zf_v4_full')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False),
                     n_tokens=ck.get('n_tokens', 16), n_gin=ck.get('n_gin', 4),
                     dec_layers=ck.get('dec_layers', 6), dec_max_len=ck.get('dec_max_len', 64),
                     n_ftokens=ck.get('n_ftokens', 0), log_feats=ck.get('log_feats', False),
                     class_feats=ck.get('class_feats', False),
                     robust_norm=ck.get('robust_norm', False)).to(DEVICE)
    try:
        model.load_state_dict(ck['model'])
    except Exception:
        from train.train_manifold import load_state_compat
        load_state_compat(model, ck['model'])
    model.eval()
    fe = model.formula_enc
    print(f"Loaded manifold d={d}")

    seq_all = np.load(os.path.join(args.cache, 'seq.npy'))
    slen = np.load(os.path.join(args.cache, 'slen.npy'))
    exprs = open(os.path.join(args.cache, 'exprs.txt'), encoding='utf-8').read().splitlines()

    # dedup by expr, keep first occurrence
    items, seen = [], set()
    for i in range(len(slen)):
        ex = exprs[i] if i < len(exprs) else ''
        if not ex or ex in seen: continue
        seen.add(ex)
        L = int(slen[i])
        items.append((ex, [int(x) for x in seq_all[i, :L]]))
        if len(items) >= args.max: break
    print(f"Encoding {len(items)} unique formulas...")

    all_z = np.zeros((len(items), d), dtype=np.float32)
    all_raw = np.zeros((len(items), d), dtype=np.float32)   # un-normalized, for Flow-2 hint
    t0 = time.time()
    with torch.no_grad():
        for bs in range(0, len(items), args.batch):
            batch = items[bs:bs+args.batch]
            enc_types, enc_const, enc_batch, enc_edges = [], [], [], []
            node_off = 0
            for bi, (_, seq) in enumerate(batch):
                for t in seq:
                    enc_types.append(t); enc_const.append(0.0); enc_batch.append(bi)
                for (a, b) in build_ast_edges(seq):
                    enc_edges.append((a+node_off, b+node_off))
                node_off += len(seq)
            ei = (torch.tensor(enc_edges).t().contiguous() if enc_edges
                  else torch.zeros(2, 0, dtype=torch.long))
            z = fe(torch.tensor(enc_types, device=DEVICE),
                   torch.tensor(enc_const, device=DEVICE),
                   ei.to(DEVICE),
                   torch.tensor(enc_batch, device=DEVICE),
                   len(batch))
            all_raw[bs:bs+len(batch)] = z.cpu().numpy()
            z = torch.nn.functional.normalize(z, dim=-1)
            all_z[bs:bs+len(batch)] = z.cpu().numpy()
            if bs % (args.batch*40) == 0:
                print(f"  {bs}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    np.save(args.out_prefix + '_index.npy', all_z)
    np.save(args.out_prefix + '_raw.npy', all_raw)
    with open(args.out_prefix + '_meta.jsonl', 'w', encoding='utf-8') as w:
        for ex, _ in items:
            w.write(json.dumps({'expr': ex}) + '\n')
    print(f"Saved {len(items)} z_f -> {args.out_prefix}_index.npy + _raw.npy ({all_z.nbytes/1e6:.0f}MB)")


if __name__ == '__main__':
    main()
