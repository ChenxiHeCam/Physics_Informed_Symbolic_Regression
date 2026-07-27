"""Probe: does the data encoder distinguish FUNCTION CLASS (exp/trig/log/power)?

Encodes synthetic (x,y) sets for several pure function classes and reports the
between-class z_d cosine. The baseline encoder was BLIND: cos(exp, poly)=0.995.
If --log-feats training works, cos(exp, poly) should drop well below that.
Run on any manifold ckpt: python encoder_class_probe.py --ckpt <path>
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch
from train.train_manifold import Manifold, load_state_compat
from data.ast_grammar import MAX_VARS

DEVICE = torch.device('cpu')
FUNCS = {
    'poly2': lambda x, r: x**2 * r.uniform(0.5, 2),
    'poly3': lambda x, r: x**3 * r.uniform(0.5, 2),
    'exp':   lambda x, r: np.exp(x * r.uniform(0.8, 1.5)),
    'trig':  lambda x, r: np.sin(3 * x * r.uniform(0.8, 1.2)),
    'log':   lambda x, r: np.log(x * r.uniform(1, 3) + 1),
    'sqrt':  lambda x, r: np.sqrt(x * r.uniform(0.5, 2)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--lo', type=float, default=0.5)
    ap.add_argument('--hi', type=float, default=2.5)
    ap.add_argument('--n-sets', type=int, default=8)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=DEVICE); d = ck['d']
    m = Manifold(d=d, dim_len=ck.get('dim_len', 0), tok=ck.get('tok', False), n_tokens=ck.get('n_tokens', 16),
                 n_gin=ck.get('n_gin', 4), dec_layers=ck.get('dec_layers', 6),
                 dec_max_len=ck.get('dec_max_len', 64), n_ftokens=ck.get('n_ftokens', 0),
                 log_feats=ck.get('log_feats', False)).to(DEVICE)
    load_state_compat(m, ck['model']); m.eval()
    print(f"ckpt step {ck.get('step','?')} | log_feats={ck.get('log_feats', False)} | range [{args.lo},{args.hi}]")

    def enc(f):
        rng = np.random.default_rng(0); zs = []
        for s in range(args.n_sets):
            x = rng.uniform(args.lo, args.hi, 120); y = f(x, rng)
            pts = torch.zeros(1, 128, MAX_VARS + 1); npt = 120
            pts[0, :npt, 0] = torch.from_numpy(x.astype('float32')); pts[0, :npt, MAX_VARS] = torch.from_numpy(y.astype('float32'))
            vm = torch.zeros(1, MAX_VARS); vm[0, 0] = 1; pm = torch.zeros(1, 128); pm[0, :npt] = 1
            with torch.no_grad():
                out = m.data_enc(pts, vm, pm, dims=None, return_tokens=True) if m.tok else (m.data_enc(pts, vm, pm, dims=None), None)
            zs.append(out[0][0].numpy())
        return np.array(zs)

    Z = {k: enc(f) for k, f in FUNCS.items()}
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print("between-class z_d cosine (LOWER = encoder separates classes better):")
    keys = list(FUNCS)
    for i, a in enumerate(keys):
        row = '  '.join(f'{cos(Z[a].mean(0), Z[b].mean(0)):+.2f}' for b in keys)
        print(f"  {a:6s} {row}")
    # headline numbers
    print(f"\n  cos(exp, poly2) = {cos(Z['exp'].mean(0), Z['poly2'].mean(0)):.3f}  (baseline was 0.995)")
    print(f"  cos(exp, trig)  = {cos(Z['exp'].mean(0), Z['trig'].mean(0)):.3f}")
    print(f"  cos(trig, poly2)= {cos(Z['trig'].mean(0), Z['poly2'].mean(0)):.3f}")


if __name__ == '__main__':
    main()
