"""
Phase 1: pretrain the shared R^d manifold.

Ties together:
  data_encoder    (X,y) -> z_d
  formula_encoder AST   -> z_f
  decoder         (z_d,z_f) -> reconstruct formula   (path B)
  [flow matching trained in a later phase / jointly optional]

Loss (v1 core; DAG signals added once basic alignment emerges):
  L = w_align * InfoNCE(z_d, z_f)          # CLIP-style, dot-product bottleneck
    + w_recon * CE(decoder(z_d,z_f), AST)  # path-B reconstruction
    + w_flow  * flow_matching(z_f | z_d)    # learn p(z_f|z_d) for sampling

Emergence-first: matching head is a plain normalized dot product. No explicit
axis assignment. We probe emergence after training.
"""
import sys, os, gzip, json, time, argparse, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.ast_grammar import (expr_to_nodes, VOCAB_SIZE as V, ARITY, ID2NT, NT2ID)
from models.encoders import DataEncoder
from models.formula_encoder import ASTEncoder
from models.decoder import ConditioningDecoderDecoder, CONST_ID, PAD_ID, EOS_ID
from models.flow_matching import FlowMatching, FlowMatchingTok, FlowMatchingTokSeq

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_VARS = 16


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_ast_edges(seq):
    """pre-order node seq -> undirected parent-child edge list."""
    types = list(seq)
    edges = []
    pos = [0]
    def walk():
        i = pos[0]; pos[0] += 1
        nt = ID2NT[types[i]]; ar = ARITY.get(nt, 0)
        for _ in range(ar):
            c = pos[0]
            edges.append((i, c)); edges.append((c, i))
            walk()
    try:
        walk()
    except Exception:
        pass
    return edges


class PairDataset:
    """Streams (X,y) + AST pairs from the generated jsonl.gz into memory (subset)."""
    def __init__(self, path, max_rows, max_points=128, max_seq=48):
        self.rows = []
        self.max_points = max_points
        self.max_seq = max_seq
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if len(r['seq']) > max_seq: continue
                if len(r['var_names']) > MAX_VARS: continue
                self.rows.append(r)
                if len(self.rows) >= max_rows: break
        print(f"Loaded {len(self.rows)} training pairs")

    def __len__(self): return len(self.rows)

    def collate(self, idxs):
        B = len(idxs)
        mp, ms = self.max_points, self.max_seq
        # data side
        points = torch.zeros(B, mp, MAX_VARS + 1)
        var_mask = torch.zeros(B, MAX_VARS)
        point_mask = torch.zeros(B, mp)
        # formula AST (level-1 graph batch)
        enc_types, enc_const, enc_batch, enc_edges = [], [], [], []
        node_off = 0
        # decoder targets
        tgt_types = torch.full((B, ms), PAD_ID, dtype=torch.long)
        tgt_consts = torch.zeros(B, ms)
        tgt_mask = torch.zeros(B, ms)

        for bi, idx in enumerate(idxs):
            r = self.rows[idx]
            X = np.array(r['X'], dtype=np.float32)
            y = np.array(r['y'], dtype=np.float32)
            nv = X.shape[1]; npt = min(X.shape[0], mp)
            points[bi, :npt, :nv] = torch.from_numpy(X[:npt])
            points[bi, :npt, MAX_VARS] = torch.from_numpy(y[:npt])
            var_mask[bi, :nv] = 1
            point_mask[bi, :npt] = 1
            seq, consts = r['seq'], r['consts']
            edges = build_ast_edges(seq)
            for t in seq:
                enc_types.append(t); enc_const.append(0.0); enc_batch.append(bi)
            for (a, b) in edges:
                enc_edges.append((a + node_off, b + node_off))
            node_off += len(seq)
            L = min(len(seq), ms)
            tgt_types[bi, :L] = torch.tensor(seq[:L])
            for k in range(L):
                cv = consts[k]
                # clamp huge consts (some are 3e21) to avoid float overflow
                if not np.isfinite(cv): cv = 0.0
                tgt_consts[bi, k] = float(np.clip(cv, -1e6, 1e6))
            tgt_mask[bi, :L] = 1

        ei = (torch.tensor(enc_edges).t().contiguous()
              if enc_edges else torch.zeros(2, 0, dtype=torch.long))
        return {
            'points': points, 'var_mask': var_mask, 'point_mask': point_mask,
            'enc_types': torch.tensor(enc_types), 'enc_const': torch.tensor(enc_const),
            'enc_batch': torch.tensor(enc_batch), 'enc_edges': ei, 'n_graphs': B,
            'tgt_types': tgt_types, 'tgt_consts': tgt_consts, 'tgt_mask': tgt_mask,
        }


class MemmapPairDataset:
    """Memory-safe full-data loader backed by a build_cache.py cache directory.

    Same collate() output as PairDataset, but rows live on disk in a memmapped
    float32 file; only the sampled batch ever enters RAM. This is what makes
    full 2.68M-row training fit in memory.
    """
    def __init__(self, cache_dir, max_rows=0, max_points=128, max_seq=48):
        self.max_points = max_points
        self.max_seq = max_seq
        self.dir = cache_dir
        self.ptr = np.load(os.path.join(cache_dir, 'ptr.npy'))
        self.npts = np.load(os.path.join(cache_dir, 'npts.npy'))
        self.nv = np.load(os.path.join(cache_dir, 'nv.npy'))
        self.seq = np.load(os.path.join(cache_dir, 'seq.npy'))      # (N, MS) int16, pad=-1
        self.slen = np.load(os.path.join(cache_dir, 'slen.npy'))
        self.max_seq = int(self.seq.shape[1])   # use cache's ACTUAL seq width (auto-adapts to ms=80 rebuild; never truncate long formulas)
        self.consts = np.load(os.path.join(cache_dir, 'consts.npy'))
        dp = os.path.join(cache_dir, 'dims.npy')
        self.dims = np.load(dp) if os.path.exists(dp) else None   # (N, MAX_VARS+1, dim_len)
        self.flat = np.memmap(os.path.join(cache_dir, 'flat.f32'),
                              dtype=np.float32, mode='r')
        self.N = len(self.npts)
        if max_rows and max_rows < self.N:
            self.N = max_rows
        print(f"Memmap cache {cache_dir}: {self.N} rows "
              f"(flat {self.flat.shape[0]*4/1e9:.2f}GB on disk)")

    def __len__(self): return self.N

    def collate(self, idxs):
        B = len(idxs)
        mp, ms = self.max_points, self.max_seq
        points = torch.zeros(B, mp, MAX_VARS + 1)
        var_mask = torch.zeros(B, MAX_VARS)
        point_mask = torch.zeros(B, mp)
        enc_types, enc_const, enc_batch, enc_edges = [], [], [], []
        node_off = 0
        tgt_types = torch.full((B, ms), PAD_ID, dtype=torch.long)
        tgt_consts = torch.zeros(B, ms)
        tgt_mask = torch.zeros(B, ms)
        dims = (torch.zeros(B, MAX_VARS + 1, self.dims.shape[-1])
                if self.dims is not None else None)

        for bi, idx in enumerate(idxs):
            if dims is not None:
                dims[bi] = torch.from_numpy(self.dims[idx].astype(np.float32))
            nv = int(self.nv[idx]); npt = int(self.npts[idx])
            a, b = int(self.ptr[idx]), int(self.ptr[idx + 1])
            mat = np.asarray(self.flat[a:b]).reshape(npt, nv + 1)
            # RANGE JITTER (per-epoch value-range variation): sometimes keep only points
            # inside a RANDOM percentile window on a RANDOM input axis, so the effective
            # value-range (not just count/which-points/noise) differs every epoch. Falls
            # back to full if the window leaves too few points.
            if nv >= 1 and npt > 300 and np.random.random() < 0.5:
                k = int(np.random.randint(0, nv)); xk = mat[:, k]
                lo, hiw = np.percentile(xk, [np.random.uniform(0, 40),
                                             np.random.uniform(60, 100)])
                keep = (xk >= lo) & (xk <= hiw)
                if keep.sum() >= 150:
                    mat = mat[keep]; npt = len(mat)
            # PER-EPOCH AUGMENTATION (user spec): stored pool ~700 pts; each epoch draw a
            # RANDOM count in [150,400] and a RANDOM subset (different distribution every
            # epoch), + re-rolled 0-2% measurement noise on y (50% of instances). This is
            # the multi-epoch diversity the big point pool was generated for.
            hi = min(mp, npt)
            n_use = min(hi, int(np.random.randint(150, 401))) if npt > 150 else min(hi, npt)
            sel = (np.random.choice(npt, n_use, replace=False) if n_use < npt
                   else np.arange(npt))
            sub = mat[sel].astype(np.float32)
            # some cached y overflowed float32 -> inf (build_cache 'overflow in cast'); the
            # data-encoder's signed-log(inf)=inf -> NaN blows up training. Sanitize here.
            np.nan_to_num(sub, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
            np.clip(sub, -1e6, 1e6, out=sub)
            if n_use > 5 and np.random.random() < 0.5:
                sub[:, nv] = sub[:, nv] * (1 + np.random.normal(0, np.random.uniform(0, 0.02), n_use))
            points[bi, :n_use, :nv] = torch.from_numpy(sub[:, :nv].copy())
            points[bi, :n_use, MAX_VARS] = torch.from_numpy(sub[:, nv].copy())
            var_mask[bi, :nv] = 1
            point_mask[bi, :n_use] = 1
            L = int(self.slen[idx])
            seq = [int(x) for x in self.seq[idx, :L]]
            edges = build_ast_edges(seq)
            for t in seq:
                enc_types.append(t); enc_const.append(0.0); enc_batch.append(bi)
            for (ea, eb) in edges:
                enc_edges.append((ea + node_off, eb + node_off))
            node_off += len(seq)
            Lt = min(L, ms)
            tgt_types[bi, :Lt] = torch.from_numpy(self.seq[idx, :Lt].astype(np.int64))
            tgt_consts[bi, :Lt] = torch.from_numpy(self.consts[idx, :Lt].copy())
            tgt_mask[bi, :Lt] = 1

        ei = (torch.tensor(enc_edges).t().contiguous()
              if enc_edges else torch.zeros(2, 0, dtype=torch.long))
        out = {
            'points': points, 'var_mask': var_mask, 'point_mask': point_mask,
            'enc_types': torch.tensor(enc_types), 'enc_const': torch.tensor(enc_const),
            'enc_batch': torch.tensor(enc_batch), 'enc_edges': ei, 'n_graphs': B,
            'tgt_types': tgt_types, 'tgt_consts': tgt_consts, 'tgt_mask': tgt_mask,
        }
        if dims is not None:
            out['dims'] = dims
        return out


class FormulaPoolDataset:
    """Formula-only (NO data): seq/consts for ALL parseable formulas, including
    the ~hundreds-of-k that have no (X,y) (PDEs, long, unrecoverable). Drives a
    formula-autoencoding aux (z_f -> decode -> formula) so z_f covers the FULL
    pool, not just formulas with data -> directly lifts the oracle ceiling."""
    def __init__(self, cache_dir, max_rows=0):
        self.seq = np.load(os.path.join(cache_dir, 'seq.npy'))
        self.consts = np.load(os.path.join(cache_dir, 'consts.npy'))
        self.slen = np.load(os.path.join(cache_dir, 'slen.npy'))
        self.ms = self.seq.shape[1]
        self.N = len(self.slen)
        if max_rows and max_rows < self.N:
            self.N = max_rows
        print(f"Formula pool {cache_dir}: {self.N} formulas (seq width {self.ms})")

    def __len__(self): return self.N

    def collate(self, idxs):
        B, ms = len(idxs), self.ms
        enc_types, enc_const, enc_batch, enc_edges = [], [], [], []
        node_off = 0
        tgt_types = torch.full((B, ms), PAD_ID, dtype=torch.long)
        tgt_consts = torch.zeros(B, ms); tgt_mask = torch.zeros(B, ms)
        for bi, idx in enumerate(idxs):
            L = int(self.slen[idx])
            seq = [int(x) for x in self.seq[idx, :L]]
            for t in seq:
                enc_types.append(t); enc_const.append(0.0); enc_batch.append(bi)
            for (ea, eb) in build_ast_edges(seq):
                enc_edges.append((ea + node_off, eb + node_off))
            node_off += len(seq)
            Lt = min(L, ms)
            tgt_types[bi, :Lt] = torch.from_numpy(self.seq[idx, :Lt].astype(np.int64))
            tgt_consts[bi, :Lt] = torch.from_numpy(self.consts[idx, :Lt].copy())
            tgt_mask[bi, :Lt] = 1
        ei = (torch.tensor(enc_edges).t().contiguous()
              if enc_edges else torch.zeros(2, 0, dtype=torch.long))
        return {'enc_types': torch.tensor(enc_types), 'enc_const': torch.tensor(enc_const),
                'enc_batch': torch.tensor(enc_batch), 'enc_edges': ei, 'n_graphs': B,
                'tgt_types': tgt_types, 'tgt_consts': tgt_consts, 'tgt_mask': tgt_mask}


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

class Manifold(nn.Module):
    """tok=False: legacy (pooled z_d -> flow + decoder).
    tok=True : data-token architecture — encoder emits a token memory; flow and
               decoder CROSS-ATTEND to it (z_d only for InfoNCE alignment/retrieval,
               NOT fed to the decoder). nondim input so value-scale leaks no type."""
    def __init__(self, d=512, dim_len=0, tok=False, n_tokens=16,
                 n_gin=4, dec_layers=6, dec_max_len=64, n_isab=6, n_ftokens=0, log_feats=False,
                 class_feats=False, dim_head=False, robust_norm=False, flow_layers=6,
                 n_adapt=0):
        super().__init__()
        self.tok = tok
        self.n_ftokens = n_ftokens        # >0 = dual z_f (pooled + token sequence)
        self.dec_max_len = dec_max_len
        self.data_enc = DataEncoder(max_vars=MAX_VARS, d=d, n_isab=n_isab, dim_len=dim_len,
                                    n_tokens=n_tokens, nondim=tok, log_feats=log_feats,
                                    class_feats=class_feats, robust_norm=robust_norm,
                                    n_adapt=n_adapt)
        self.formula_enc = ASTEncoder(V, d=d, n_gin=n_gin, n_ftokens=n_ftokens)
        self.decoder = ConditioningDecoderDecoder(vocab_size=V, d=d, dec_layers=dec_layers,
                                                  max_len=dec_max_len)
        self.flow = FlowMatchingTok(d=d, n_blocks=flow_layers) if tok else FlowMatching(d=d, n_blocks=flow_layers)
        # fine flow: p(z_f_tokens | data_tokens, coarse z_f) — coarse-to-fine
        self.flow_fine = FlowMatchingTokSeq(d=d, n_blocks=6) if n_ftokens > 0 else None
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.dim_head_on = dim_head
        if dim_head:   # z_f -> result SI dimension (6 base exponents) — makes z_f dimension-aware
            self.dim_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 6))

    def encode(self, b):
        toks = None
        if self.tok:
            z_d, toks = self.data_enc(b['points'], b['var_mask'], b['point_mask'],
                                      dims=b.get('dims'), return_tokens=True)
        else:
            z_d = self.data_enc(b['points'], b['var_mask'], b['point_mask'], dims=b.get('dims'))
        if self.n_ftokens > 0:
            z_f, z_f_tokens = self.formula_enc(b['enc_types'], b['enc_const'], b['enc_edges'],
                                               b['enc_batch'], b['n_graphs'], return_tokens=True)
            return z_d, z_f, toks, z_f_tokens
        z_f = self.formula_enc(b['enc_types'], b['enc_const'],
                               b['enc_edges'], b['enc_batch'], b['n_graphs'])
        return z_d, z_f, toks


def load_state_compat(model, sd):
    """Load a possibly-older state_dict, padding GROWN params (e.g. decoder role_emb
    4->5 after adding the z_f-token role) by copying the overlap and leaving new
    rows/cols at init. Lets pre-dual ckpts load into the dual-capable model."""
    msd = model.state_dict()
    for k, v in list(sd.items()):
        if k in msd and tuple(msd[k].shape) != tuple(v.shape):
            nw = msd[k].clone()
            sl = tuple(slice(0, min(a, b)) for a, b in zip(nw.shape, v.shape))
            nw[sl] = v[sl]
            sd[k] = nw
    return model.load_state_dict(sd, strict=False)


def info_nce(z_d, z_f, temp):
    zd = F.normalize(z_d, dim=-1); zf = F.normalize(z_f, dim=-1)
    logits = (zd @ zf.t()) / temp.clamp(0.01, 0.5)
    labels = torch.arange(z_d.size(0), device=z_d.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def hypersphere_uniformity(z, t=2.0):
    """Wang & Isola uniformity: spread z EVENLY on the unit hypersphere so the z_f latent
    has NO holes/clusters -> the (rectified) flow can sample anywhere and land on a
    decodable z_f. This is the generative-prior the AE lacked (LDM lesson), sphere-native
    so it coexists with the L2-normalized z_f. Opposes alignment/DAG (which pull related
    formulas together); a small weight keeps clusters navigable while filling the gaps."""
    z = F.normalize(z, dim=-1)
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().clamp_min(1e-12).log()


def info_nce_bank(z_d, z_f, temp, bank):
    """MoCo-style InfoNCE: z_d must match its z_f against the in-batch z_f PLUS a
    queue of past z_f (extra negatives). At d=768/batch=64 the 63 in-batch negatives
    were too few for the data-encoder to learn a discriminative z_d (align froze at
    ln(64)); the bank gives thousands of negatives without extra GPU/batch.
    bank: (M,d) normalized past z_f (may be empty). Returns (loss, new_zf_detached)."""
    t = temp.clamp(0.01, 0.5)
    zd = F.normalize(z_d, dim=-1); zf = F.normalize(z_f, dim=-1)
    labels = torch.arange(z_d.size(0), device=z_d.device)
    cands = zf if bank is None or bank.numel() == 0 else torch.cat([zf, bank], 0)
    logits = (zd @ cands.t()) / t                    # (B, B+M): z_d -> z_f (+bank negs)
    loss = F.cross_entropy(logits, labels)
    loss = loss + F.cross_entropy((zf @ zd.t()) / t, labels)  # symmetric in-batch z_f->z_d
    return 0.5 * loss, zf.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='dataset_20260531/data_pairs_smoke.jsonl.gz')
    ap.add_argument('--cache', default='')  # memmap cache dir (full-data, memory-safe)
    ap.add_argument('--d', type=int, default=384)
    ap.add_argument('--max-rows', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--w-align', type=float, default=1.0)
    ap.add_argument('--w-recon', type=float, default=1.0)
    ap.add_argument('--w-flow', type=float, default=0.5)
    ap.add_argument('--resume', default='')                 # continue from a ckpt
    ap.add_argument('--clean-index', default='')            # npz clean mask -> train ONLY on
                                                            # clean rows (drops target_zero etc.)
    ap.add_argument('--freeze-enc', action='store_true')    # freeze encoders, train ONLY flow+decoder
                                                            # (specialist: own decoder co-trains with flow)
    ap.add_argument('--boost-file', default='')             # npz {boost} -> oversample focus class
    ap.add_argument('--boost-frac', type=float, default=0.0)
    ap.add_argument('--balanced', action='store_true')      # uniform across complexity
    ap.add_argument('--type-balanced', action='store_true') # upsample rare formula types
    ap.add_argument('--dim-len', type=int, default=0)       # SI dimension input width (7=on)
    ap.add_argument('--tok', action='store_true')           # data-token architecture (flow+decoder cross-attend)
    ap.add_argument('--n-tokens', type=int, default=16)     # token-memory size
    ap.add_argument('--dag', action='store_true')           # DAG-neighbor z_f contrastive aux (needs neighbors.npy)
    ap.add_argument('--w-dag', type=float, default=0.3)     # weight of the DAG aux loss
    ap.add_argument('--n-gin', type=int, default=4)         # formula-encoder GIN depth (8 = deeper z_f)
    ap.add_argument('--dec-layers', type=int, default=6)    # decoder depth
    ap.add_argument('--flow-layers', type=int, default=6)   # flow velocity-net depth (grown models: 16)
    ap.add_argument('--dec-max-len', type=int, default=80)  # decoder window (>=MAX_SEQ; 80 to fit long formulas)
    ap.add_argument('--fpool', default='')                  # formula-only AE: jsonl of {seq,consts} for ALL formulas (incl data-less)
    ap.add_argument('--w-fae', type=float, default=0.5)     # weight of formula-only autoencoding loss
    ap.add_argument('--ckpt-every', type=int, default=0)    # save intermediate ckpt every N steps (0=only final)
    ap.add_argument('--align-bank', type=int, default=0)    # MoCo memory-bank size for align InfoNCE (0=off; fixes z_d collapse at small batch)
    ap.add_argument('--equiv', action='store_true')         # equivalence-InfoNCE: pull a formula's algebraic rearrangements' z_f together (fix structural-hash encoder)
    ap.add_argument('--w-equiv', type=float, default=0.3)   # weight of the equivalence aux loss
    ap.add_argument('--class-feats', action='store_true')   # inject R^2 {poly,exp,log,power} goodness-of-fit into z_d (function-class discriminative -> flow targets exp/log regions)
    ap.add_argument('--max-points', type=int, default=128)  # observation points per task (300 = more data constraint)
    ap.add_argument('--dim-head', action='store_true')      # z_f -> SI dimension supervision head (dimension-aware z_f)
    ap.add_argument('--w-dim', type=float, default=0.3)     # weight of the dimension-supervision loss
    ap.add_argument('--n-ftokens', type=int, default=0)     # >0 = dual z_f (pooled gist + token sequence) + coarse-to-fine flow + multi-source decoder
    ap.add_argument('--log-feats', action='store_true')     # add log|x|,log|y| channels to data encoder (see function CLASS: exp/trig/log/power; fixes cos(exp,poly)=0.995 blindness)
    ap.add_argument('--robust-norm', action='store_true')   # always-on median/MAD per-column input standardization (linear + decade-compressed slog), survives WIDE dynamic range without z_d collapse
    ap.add_argument('--w-var', type=float, default=0.0)     # VICReg variance reg on z_d/z_f (prevents the recurring z_d collapse; try 1.0)
    ap.add_argument('--w-unif', type=float, default=0.0)    # hypersphere-uniformity reg on z_f: fills latent holes -> flow-samplable (LDM-style generative prior). try 0.1-0.5
    ap.add_argument('--transe', action='store_true')        # TransE relational loss (replaces InfoNCE-DAG): z_a + R[rel] ~= z_b on a separate proj head (related stay NEAR but navigable-separated, not collapsed)
    ap.add_argument('--w-transe', type=float, default=0.3)
    ap.add_argument('--transe-dim', type=int, default=128)  # relational projection dim
    ap.add_argument('--save', default='sr_model/ckpt/manifold.pt')
    args = ap.parse_args()

    ds = (MemmapPairDataset(args.cache, args.max_rows, max_points=args.max_points) if args.cache
          else PairDataset(args.pairs, args.max_rows, max_points=args.max_points))
    dag_nb = None
    if args.dag:
        nbp = os.path.join(args.cache, 'neighbors.npy')
        dag_nb = np.load(nbp)[:len(ds)]
        print(f"DAG aux ON: {nbp} | {(dag_nb >= 0).mean()*100:.1f}% rows have a neighbor")
    fpool = None
    if args.fpool:
        fpool = FormulaPoolDataset(args.fpool)
        print(f"formula-only AE ON: {len(fpool)} formulas (full pool incl data-less)")
    # equivalence aux: SimCLR-style, augmentation = algebraic rearrangement. Generated
    # on the fly (multiply-through, no sp.solve) and memoized per row.
    equiv_exprs = None; equiv_cache = {}
    if args.equiv:
        from train.equiv_augment import gen_equiv_forms
        from data.ast_grammar import parse_to_sympy as _p2s
        equiv_exprs = open(os.path.join(args.cache, 'exprs.txt'), encoding='utf-8').read().splitlines()[:len(ds)]
        print(f"equivalence aux ON: w_equiv={args.w_equiv} (pull rearrangement z_f together)")

        def _equiv_seq(idx, max_len):
            if idx in equiv_cache:
                forms = equiv_cache[idx]
            else:
                forms = []
                try:
                    te = _p2s(equiv_exprs[idx])
                    if te is not None and len(te.free_symbols) >= 2:
                        vn = sorted([str(s) for s in te.free_symbols])
                        for fstr in gen_equiv_forms(equiv_exprs[idx], vn, max_len=max_len, max_forms=3):
                            r = expr_to_nodes(fstr, var_order=vn)
                            if r is not None and len(r[0]) <= max_len:
                                forms.append([int(x) for x in r[0]])
                except Exception:
                    pass
                equiv_cache[idx] = forms
            return random.choice(forms) if forms else None
    model = Manifold(d=args.d, dim_len=args.dim_len, tok=args.tok, n_tokens=args.n_tokens,
                     n_gin=args.n_gin, dec_layers=args.dec_layers,
                     dec_max_len=args.dec_max_len, n_ftokens=args.n_ftokens,
                     log_feats=args.log_feats, class_feats=args.class_feats,
                     dim_head=args.dim_head, robust_norm=args.robust_norm,
                     flow_layers=args.flow_layers).to(DEVICE)
    # TransE relational head (training-only: relations get baked into z_f geometry,
    # so inference needs neither R nor the projection — pure z_f from the AST encoder).
    transe_proj = None; rel_emb = None; tr_nbr = tr_rel = tr_dir = None
    if args.transe:
        tz = np.load(os.path.join(args.cache, 'neighbors_typed.npz'), allow_pickle=True)
        tr_nbr = tz['nbr'][:len(ds)]; tr_rel = tz['rel'][:len(ds)]; tr_dir = tz['dir'][:len(ds)]
        n_rel = len(tz['relations'])
        transe_proj = nn.Linear(args.d, args.transe_dim).to(DEVICE)
        rel_emb = nn.Parameter(torch.randn(n_rel, args.transe_dim, device=DEVICE) * 0.02)
        print(f"TransE relational head ON: {n_rel} relations, proj dim {args.transe_dim}, "
              f"{(tr_rel >= 0).mean()*100:.1f}% rows have a typed edge")
    if args.resume:
        ck = torch.load(args.resume, map_location=DEVICE)
        sd = ck['model']
        # weight surgery: resuming a no-dims ckpt into a dims model — the in_proj
        # first layer has more input cols now; copy the overlap, zero-fill the new
        # dimension columns so the model starts IDENTICAL and learns units from 0.
        new_sd = model.state_dict()
        k = 'data_enc.in_proj.0.weight'
        if k in sd and sd[k].shape != new_sd[k].shape:
            old = sd[k]; w = new_sd[k].clone(); w.zero_(); w[:, :old.shape[1]] = old
            sd[k] = w
            print(f"in_proj surgery: {tuple(old.shape)} -> {tuple(w.shape)} (new cols zero-init)")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"resumed from {args.resume} (missing={len(missing)} unexpected={len(unexpected)})")
    if args.freeze_enc:                                  # SPECIALIST: freeze encoders,
        for p in model.data_enc.parameters(): p.requires_grad_(False)
        for p in model.formula_enc.parameters(): p.requires_grad_(False)
        opt_params = list(model.flow.parameters()) + list(model.decoder.parameters())
        if model.flow_fine is not None:
            opt_params += list(model.flow_fine.parameters())
        print(f"freeze-enc: training ONLY flow+decoder ({sum(p.numel() for p in opt_params):,} params)")
    else:
        opt_params = list(model.parameters())
    if args.transe:
        opt_params += list(transe_proj.parameters()) + [rel_emb]
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    print(f"params {sum(p.numel() for p in model.parameters()):,}  device={DEVICE}")

    # MoCo memory bank of past (normalized) z_f as extra align negatives
    align_bank = None; bank_ptr = 0; bank_n = 0
    if args.align_bank > 0:
        align_bank = torch.zeros(args.align_bank, args.d, device=DEVICE)
        print(f"align memory-bank ON: {args.align_bank} negatives")

    sample_w = None
    if (args.balanced or args.type_balanced) and args.cache:
        nvb = np.asarray(ds.nv[:len(ds)], dtype=np.int64)
        slb = np.asarray(ds.slen[:len(ds)], dtype=np.int64) // 8
        key = nvb * 100 + slb
        uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv].astype(np.float64)
        msg = f"balanced over {len(uniq)} complexity bins"
        if args.type_balanced:
            # upsample rare-but-representable hard types (damped osc exp*trig,
            # sum-of-exponentials, abs) that the model fails on for lack of data
            exprs = open(os.path.join(args.cache, 'exprs.txt'), encoding='utf-8').read().splitlines()
            N = len(ds); boost = np.ones(N, dtype=np.float64)
            for i in range(N):
                e = exprs[i] if i < len(exprs) else ''
                ne = e.count('exp'); has_trig = ('sin' in e or 'cos' in e or 'tan' in e)
                b = 1.0
                if ne >= 1 and has_trig: b *= 8.0      # damped oscillation
                if ne >= 2: b *= 4.0                   # sum of exponentials
                if 'Abs' in e or 'abs' in e: b *= 4.0  # absolute value
                if ('log' in e) and has_trig: b *= 3.0
                boost[i] = b
            w = w * boost
            msg += f" + type-boost (mean {boost.mean():.2f}, max {boost.max():.0f})"
        sample_w = w / w.sum()
        print(msg)

    # clean-row restriction: zero sampling weight on pathological rows (or build a
    # clean pool when no weighting). cache_v6 is ~24% target_zero garbage.
    clean_pool = None
    if args.clean_index and os.path.exists(args.clean_index):
        cmask = np.load(args.clean_index)['clean'][:len(ds)]
        if sample_w is not None:
            sample_w = sample_w * cmask
            sample_w = sample_w / sample_w.sum()
        else:
            clean_pool = np.where(cmask)[0]
        print(f"clean-index: {int(cmask.sum())}/{len(ds)} clean rows "
              f"({100*cmask.sum()/len(ds):.1f}%)")
    boost_pool = None
    if args.boost_file and os.path.exists(args.boost_file) and args.boost_frac > 0:
        boost_pool = np.where(np.load(args.boost_file)['boost'][:len(ds)])[0]
        print(f"SPECIALIZE: boost-frac {args.boost_frac} from {len(boost_pool)} boosted rows")

    def slog(x): return torch.sign(x) * torch.log1p(torch.abs(x))

    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        if sample_w is not None:
            idxs = np.random.choice(len(ds), size=min(args.batch, len(ds)),
                                    replace=True, p=sample_w).tolist()
        elif clean_pool is not None:
            idxs = np.random.choice(clean_pool, size=min(args.batch, len(clean_pool))).tolist()
        else:
            idxs = random.sample(range(len(ds)), min(args.batch, len(ds)))
        if boost_pool is not None:                       # SPECIALIZE: oversample boosted class
            nb = int(args.batch * args.boost_frac)
            idxs[:nb] = np.random.choice(boost_pool, size=nb).tolist()
        b = ds.collate(idxs)
        b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}

        enc = model.encode(b)
        if model.n_ftokens > 0:
            z_d, z_f, toks, z_f_tokens = enc
        else:
            z_d, z_f, toks = enc; z_f_tokens = None

        # anti-collapse floor: only penalize z_d dims whose batch-std drops BELOW 0.3
        # (i.e. actually collapsing). Threshold 1.0 over-constrained and killed align;
        # 0.3 lets align organize z_d freely at normal std but blocks the std->0 collapse.
        l_var = F.relu(0.3 - z_d.std(0)).mean()

        # align (pooled z_d <-> z_f always — z_d is the retrieval/alignment handle)
        if align_bank is not None:
            bank_view = align_bank[:bank_n] if bank_n > 0 else None
            l_align, zf_new = info_nce_bank(z_d, z_f, model.temp, bank_view)
            # enqueue this batch's z_f into the ring buffer
            bs = zf_new.size(0)
            if bs >= align_bank.size(0):
                align_bank.copy_(zf_new[-align_bank.size(0):]); bank_ptr = 0; bank_n = align_bank.size(0)
            else:
                end = bank_ptr + bs
                if end <= align_bank.size(0):
                    align_bank[bank_ptr:end] = zf_new
                else:
                    first = align_bank.size(0) - bank_ptr
                    align_bank[bank_ptr:] = zf_new[:first]; align_bank[:bs - first] = zf_new[first:]
                bank_ptr = end % align_bank.size(0)
                bank_n = min(bank_n + bs, align_bank.size(0))
        else:
            l_align = info_nce(z_d, z_f, model.temp)
        # recon (path B): teacher-forced. tok-mode feeds DATA TOKENS (no pooled z_d
        # to the decoder — the token memory subsumes it); legacy feeds pooled z_d.
        bos = torch.full((z_f.size(0), 1), EOS_ID, device=DEVICE)
        inp_t = torch.cat([bos, b['tgt_types'][:, :-1]], dim=1)
        inp_c = torch.cat([torch.zeros(z_f.size(0), 1, device=DEVICE), b['tgt_consts'][:, :-1]], dim=1)
        if model.tok:
            # feed BOTH pooled z_d (clean global summary) AND token memory (local) +
            # z_f. z_d also gets recon gradient here -> becomes discriminative ->
            # un-sticks the InfoNCE alignment (else z_d only sees the frozen align loss).
            logits, cpred = model.decoder(z_d, z_f, inp_t, inp_c, data_tokens=toks,
                                          z_f_tokens=z_f_tokens)
        else:
            logits, cpred = model.decoder(z_d, z_f, inp_t, inp_c)
        ce = F.cross_entropy(logits.reshape(-1, V), b['tgt_types'].reshape(-1), reduction='none')
        ce = (ce.reshape(z_f.size(0), -1) * b['tgt_mask']).sum() / b['tgt_mask'].sum()
        cpos = (b['tgt_types'] == CONST_ID).float()
        closs = (((slog(cpred) - slog(b['tgt_consts'])) ** 2) * cpos).sum() / cpos.sum().clamp_min(1)
        l_recon = ce + 0.05 * closs
        # flow matching on z_f: tok-mode conditions on the token memory, else pooled z_d
        if model.tok:
            l_flow = model.flow.loss(z_f.detach(), toks.detach())
        else:
            l_flow = model.flow.loss(z_f.detach(), z_d.detach())
        # fine flow (coarse-to-fine): p(z_f_tokens | data_tokens, coarse z_f)
        l_flow_fine = torch.zeros((), device=DEVICE)
        if model.n_ftokens > 0:
            l_flow_fine = model.flow_fine.loss(z_f_tokens.detach(), toks.detach(), z_f.detach())

        # DAG aux: pull z_f of DAG-related formulas together (InfoNCE, neighbor=positive).
        # Encodes physical kinship (derivation/special_case/...) into z_f geometry so
        # flow/retrieval can hop between related forms. Formula-side only (no data encode).
        l_dag = torch.zeros((), device=DEVICE)
        if dag_nb is not None:
            nb = dag_nb[idxs]
            vmask = nb >= 0
            if vmask.sum() >= 4:
                vi = np.where(vmask)[0]
                bn = ds.collate(nb[vmask].tolist())
                bn = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in bn.items()}
                zf_nb = model.formula_enc(bn['enc_types'], bn['enc_const'],
                                          bn['enc_edges'], bn['enc_batch'], bn['n_graphs'])
                zf_anchor = z_f[torch.from_numpy(vi).to(DEVICE)]
                l_dag = info_nce(zf_anchor, zf_nb, model.temp)

        # equivalence aux (SimCLR, augmentation = algebraic rearrangement): pull each
        # formula's z_f toward a random equivalent-form z_f. Fixes the "structural hash"
        # encoder so the SAME law's rewritings collapse to one point (and DIFFERENT laws
        # of the same shape can separate), giving the flow a single clean target.
        l_equiv = torch.zeros((), device=DEVICE)
        if args.equiv:
            eseqs = []; eanchor = []
            for j, idx in enumerate(idxs):
                s = _equiv_seq(idx, args.dec_max_len)
                if s is not None:
                    eseqs.append(s); eanchor.append(j)
            if len(eseqs) >= 4:
                et, ec, eb, ee = [], [], [], []; off = 0
                for bi, seq in enumerate(eseqs):
                    for t in seq:
                        et.append(t); ec.append(0.0); eb.append(bi)
                    for (a, b2) in build_ast_edges(seq):
                        ee.append((a + off, b2 + off))
                    off += len(seq)
                ei = torch.tensor(ee).t().contiguous() if ee else torch.zeros(2, 0, dtype=torch.long)
                zf_eq = model.formula_enc(torch.tensor(et, device=DEVICE), torch.tensor(ec, device=DEVICE),
                                          ei.to(DEVICE), torch.tensor(eb, device=DEVICE), len(eseqs))
                zf_anc = z_f[torch.tensor(eanchor, device=DEVICE)]
                l_equiv = info_nce(zf_anc, zf_eq, model.temp)

        # TransE relational aux: z_a + dir*R[rel] ~= z_b on a separate proj head.
        # Related formulas stay NEAR but separated by a learned per-relation offset
        # (navigable: walk the DAG in latent), NOT collapsed onto each other.
        l_transe = torch.zeros((), device=DEVICE)
        if args.transe:
            ia = np.asarray(idxs)
            rr = tr_rel[ia]; nn_ = tr_nbr[ia]; dd = tr_dir[ia]
            vmask = (rr >= 0) & (nn_ >= 0)
            if vmask.sum() >= 4:
                vi = np.where(vmask)[0]
                bn = ds.collate(nn_[vmask].tolist())
                bn = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in bn.items()}
                if model.n_ftokens > 0:
                    z_nbr, _ = model.formula_enc(bn['enc_types'], bn['enc_const'], bn['enc_edges'],
                                                 bn['enc_batch'], bn['n_graphs'], return_tokens=True)
                else:
                    z_nbr = model.formula_enc(bn['enc_types'], bn['enc_const'],
                                              bn['enc_edges'], bn['enc_batch'], bn['n_graphs'])
                za = F.normalize(transe_proj(z_f[torch.from_numpy(vi).to(DEVICE)]), dim=-1)
                zb = F.normalize(transe_proj(z_nbr), dim=-1)
                R = rel_emb[torch.from_numpy(rr[vmask].astype(np.int64)).to(DEVICE)]
                sgn = torch.from_numpy(dd[vmask].astype(np.float32)).to(DEVICE).unsqueeze(1)
                pred = za + sgn * R
                pos_d = ((pred - zb) ** 2).sum(-1)
                # negative = corrupted tail (shuffle within batch). Margin loss prevents
                # the trivial collapse (proj->const) that a pure positive MSE allows.
                perm = torch.randperm(zb.size(0), device=DEVICE)
                neg_d = ((pred - zb[perm]) ** 2).sum(-1)
                l_transe = F.relu(1.0 + pos_d - neg_d).mean()

        # formula-only autoencoding: z_f -> decode (NO data tokens) -> reconstruct AST.
        # Trains the formula encoder+decoder on the FULL pool (incl data-less formulas)
        # so z_f generalizes to far more formula structures -> raises the oracle ceiling.
        l_fae = torch.zeros((), device=DEVICE)
        if fpool is not None:
            fidx = random.sample(range(len(fpool)), min(args.batch, len(fpool)))
            fb = fpool.collate(fidx)
            fb = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in fb.items()}
            zf_p_tok = None
            if model.n_ftokens > 0:
                zf_p, zf_p_tok = model.formula_enc(fb['enc_types'], fb['enc_const'], fb['enc_edges'],
                                                   fb['enc_batch'], fb['n_graphs'], return_tokens=True)
            else:
                zf_p = model.formula_enc(fb['enc_types'], fb['enc_const'],
                                         fb['enc_edges'], fb['enc_batch'], fb['n_graphs'])
            B2 = zf_p.size(0)
            bos2 = torch.full((B2, 1), EOS_ID, device=DEVICE)
            inp_t2 = torch.cat([bos2, fb['tgt_types'][:, :-1]], dim=1)
            inp_c2 = torch.cat([torch.zeros(B2, 1, device=DEVICE), fb['tgt_consts'][:, :-1]], dim=1)
            logits_p, cpred_p = model.decoder(None, zf_p, inp_t2, inp_c2, data_tokens=None,
                                              z_f_tokens=zf_p_tok)
            ce_p = F.cross_entropy(logits_p.reshape(-1, V), fb['tgt_types'].reshape(-1), reduction='none')
            l_fae = (ce_p.reshape(B2, -1) * fb['tgt_mask']).sum() / fb['tgt_mask'].sum()

        # dimension supervision: z_f -> result SI dim, supervised by cache dims[target]
        # (chan MAX_VARS), masked to vars with a KNOWN dim (flag==0). Makes z_f
        # dimension-aware -> flow can target dimensionally-correct z_f regions.
        l_dim = torch.zeros((), device=DEVICE)
        if model.dim_head_on and b.get('dims') is not None:
            dtgt = b['dims'][:, MAX_VARS, :6].float()                 # (B,6) target SI exponents
            known = (b['dims'][:, MAX_VARS, 6] == 0).float().unsqueeze(1)  # known-dim mask
            if known.sum() > 0:
                pred = model.dim_head(z_f)
                l_dim = ((pred - dtgt) ** 2 * known).sum() / known.sum().clamp_min(1) / 6.0
        l_unif = hypersphere_uniformity(z_f) if args.w_unif > 0 else torch.zeros((), device=DEVICE)
        loss = (args.w_var * l_var
                + args.w_align * l_align + args.w_recon * l_recon + args.w_flow * l_flow
                + (args.w_dag * l_dag if dag_nb is not None else 0.0)
                + (args.w_fae * l_fae if fpool is not None else 0.0)
                + (args.w_equiv * l_equiv if args.equiv else 0.0)
                + (args.w_transe * l_transe if args.transe else 0.0)
                + (args.w_dim * l_dim if model.dim_head_on else 0.0)
                + (args.w_unif * l_unif if args.w_unif > 0 else 0.0)   # z_f hypersphere smoothing
                + (args.w_flow * l_flow_fine if model.n_ftokens > 0 else 0.0))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                zd = F.normalize(z_d, dim=-1); zf = F.normalize(z_f, dim=-1)
                sim = (zd @ zf.t())
                r1 = (sim.argmax(1) == torch.arange(sim.size(0), device=DEVICE)).float().mean()
                tok = ((logits.argmax(-1) == b['tgt_types']).float() * b['tgt_mask']).sum() / b['tgt_mask'].sum()
            dag_s = f" dag={l_dag.item():.3f}" if dag_nb is not None else ""
            fae_s = f" fae={l_fae.item():.3f}" if fpool is not None else ""
            eq_s = f" equiv={l_equiv.item():.3f}" if args.equiv else ""
            tr_s = f" transe={l_transe.item():.3f}" if args.transe else ""
            ff_s = f" flowfine={l_flow_fine.item():.3f}{tr_s}" if model.n_ftokens > 0 else tr_s
            ff_s += f" var={l_var.item():.3f}" if args.w_var > 0 else ""
            ff_s += f" unif={l_unif.item():.3f}" if args.w_unif > 0 else ""
            print(f"step {step:4d} loss={loss.item():.3f} align={l_align.item():.3f} "
                  f"recon={l_recon.item():.3f} flow={l_flow.item():.3f}{ff_s}{dag_s}{fae_s}{eq_s} | "
                  f"retr@1={r1.item():.3f} tok_acc={tok.item():.3f} ({time.time()-t0:.0f}s)",
                  flush=True)

        def _save(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({'model': model.state_dict(), 'd': args.d, 'dim_len': args.dim_len,
                        'tok': args.tok, 'n_tokens': args.n_tokens, 'n_gin': args.n_gin,
                        'dec_layers': args.dec_layers, 'dec_max_len': args.dec_max_len,
                        'n_ftokens': args.n_ftokens, 'log_feats': args.log_feats,
                        'class_feats': args.class_feats, 'dim_head': args.dim_head,
                        'robust_norm': args.robust_norm, 'flow_layers': args.flow_layers,
                        'step': step}, path)

        # periodic checkpoint: probe oracle on these mid-run, don't lose progress
        if args.ckpt_every and step % args.ckpt_every == 0:
            cp = args.save.replace('.pt', f'_step{step}.pt')
            _save(cp); print(f"  [ckpt] -> {cp}", flush=True)

    _save(args.save)
    print(f"Saved -> {args.save}")


if __name__ == '__main__':
    main()
