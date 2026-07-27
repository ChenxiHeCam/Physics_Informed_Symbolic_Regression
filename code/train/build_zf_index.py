"""
Build the retrieval z_f index: encode real formulas -> z_f, store for nearest
-neighbor lookup. This gives "retrieval z_f": given a data z_d, find the most
similar real-formula z_f's from the library.

Output:
  zf_index.npy      (N, d) float32   formula latents
  zf_meta.jsonl     N lines          {uid, expr, seq, consts, var_names}
"""
import sys, os, gzip, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch

from data.ast_grammar import expr_to_nodes, ARITY, ID2NT, VOCAB_SIZE as V
from train.train_manifold import Manifold, build_ast_edges, MAX_VARS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_full.pt')
    ap.add_argument('--nodes', default='dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz')
    ap.add_argument('--max', type=int, default=400000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--out-prefix', default='sr_model/ckpt/zf')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE)
    d = ck['d']
    model = Manifold(d=d).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    fe = model.formula_enc
    print(f"Loaded manifold d={d}")

    # collect parseable formulas
    items = []
    with gzip.open(args.nodes, 'rt', encoding='utf-8') as f:
        for line in f:
            if len(items) >= args.max: break
            if not line.strip(): continue
            r = json.loads(line)
            expr = r.get('expr', '')
            res = expr_to_nodes(expr)
            if res is None: continue
            seq, consts, var_order = res
            if len(seq) > 48: continue
            items.append({'uid': r.get('id', ''), 'expr': expr,
                          'seq': seq, 'consts': consts, 'var_names': var_order})
    print(f"Encoding {len(items)} formulas...")

    all_z = np.zeros((len(items), d), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for bs in range(0, len(items), args.batch):
            batch = items[bs:bs+args.batch]
            enc_types, enc_const, enc_batch, enc_edges = [], [], [], []
            node_off = 0
            for bi, it in enumerate(batch):
                seq = it['seq']
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
            z = torch.nn.functional.normalize(z, dim=-1)
            all_z[bs:bs+len(batch)] = z.cpu().numpy()
            if bs % (args.batch*20) == 0:
                print(f"  {bs}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    np.save(args.out_prefix + '_index.npy', all_z)
    with open(args.out_prefix + '_meta.jsonl', 'w', encoding='utf-8') as w:
        for it in items:
            w.write(json.dumps(it) + '\n')
    print(f"Saved {len(items)} z_f -> {args.out_prefix}_index.npy ({all_z.nbytes/1e6:.0f}MB)")
    print(f"      meta -> {args.out_prefix}_meta.jsonl  time={time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
