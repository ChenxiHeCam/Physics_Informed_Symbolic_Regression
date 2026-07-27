"""
Decisive test: does the decoder actually USE the (z_d, z_f) conditioning, or is
it a glorified unconditional AST language model?

Teacher-forced token accuracy with:
  (a) correct (z_d, z_f)
  (b) shuffled z_f  (formula latent from a DIFFERENT example)
  (c) zeroed  z_f
  (d) shuffled z_d
If (b)/(c)/(d) accuracy ~= (a), the decoder ignores conditioning -> generation
can never be correct. Also reports the mean |gate| of the decoder AdaLN-Zero
modulations (near 0 => conditioning bypassed).
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch, torch.nn.functional as F
from train.train_manifold import Manifold, MemmapPairDataset
from models.decoder import EOS_ID, CONST_ID, PAD_ID
from data.ast_grammar import VOCAB_SIZE as V

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tok_acc(model, b, zd, zf):
    bos = torch.full((zf.size(0), 1), EOS_ID, device=DEVICE)
    inp_t = torch.cat([bos, b['tgt_types'][:, :-1]], dim=1)
    inp_c = torch.cat([torch.zeros(zf.size(0), 1, device=DEVICE), b['tgt_consts'][:, :-1]], dim=1)
    with torch.no_grad():
        logits, _ = model.decoder(zd, zf, inp_t, inp_c)
        acc = ((logits.argmax(-1) == b['tgt_types']).float() * b['tgt_mask']).sum() / b['tgt_mask'].sum()
    return float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_model/ckpt/manifold_v4_full.pt')
    ap.add_argument('--cache', default='dataset_20260531/cache_v4_full')
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    model = Manifold(d=d).to(DEVICE); model.load_state_dict(ck['model']); model.eval()
    ds = MemmapPairDataset(args.cache, 2000000)
    import random
    idxs = random.sample(range(len(ds)), args.batch)
    b = ds.collate(idxs)
    b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
    with torch.no_grad():
        zd, zf = model.encode(b)
    roll = torch.roll(torch.arange(zf.size(0)), 1)
    print(f"ckpt={args.ckpt} d={d} batch={args.batch}\n")
    print(f"(a) correct (z_d, z_f)      tok_acc = {tok_acc(model, b, zd, zf):.4f}")
    print(f"(b) SHUFFLED z_f            tok_acc = {tok_acc(model, b, zd, zf[roll]):.4f}")
    print(f"(c) ZEROED   z_f            tok_acc = {tok_acc(model, b, zd, torch.zeros_like(zf)):.4f}")
    print(f"(d) SHUFFLED z_d            tok_acc = {tok_acc(model, b, zd[roll], zf):.4f}")
    print(f"(e) BOTH shuffled           tok_acc = {tok_acc(model, b, zd[roll], zf[roll]):.4f}")

    # inspect AdaLN-Zero gate magnitudes in the decoder blocks
    bos = torch.full((zf.size(0), 1), EOS_ID, device=DEVICE)
    inp_t = torch.cat([bos, b['tgt_types'][:, :-1]], dim=1)
    mem, cond = model.decoder.cond_enc(zd, zf)
    gates = []
    for blk in model.decoder.decoder.blocks:
        for adaln in [blk.adaln1, blk.adaln2, blk.adaln3]:
            _, _, g = adaln.to_mod(cond).chunk(3, dim=-1)
            gates.append(float(g.abs().mean()))
    print(f"\nAdaLN-Zero mean|gate| per sublayer: "
          f"{['%.3f'%x for x in gates]}")
    print(f"overall mean|gate| = {np.mean(gates):.4f}  (near 0 => conditioning bypassed)")


if __name__ == '__main__':
    main()
