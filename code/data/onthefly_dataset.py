"""On-the-fly dataset for the A100 rebuild: reads the pre-SOLVED records
({expr, target, vars, y_expr, units, ranges}) and generates (X,y) FRESH each access
by sampling inputs (full regen method: jitter window / mixed distribution / trig
widening / 50-50 noise) and evaluating the explicit y_expr (cheap, no root-finding).
This is what makes per-epoch resampling fast on the A100's ~10 CPU cores: the
expensive solve was done once (locally); here we only sample + eval.

Produces rows in the same {X, y, seq, consts, var_names, dims} shape PairDataset
expects, and reuses its collate via subclassing.
"""
import os, gzip, json, glob, random
import numpy as np
import sympy as sp

# reuse the regen-method sampling + tokenization from the offline pipeline
from data.gen_data_pairs import (sample_var, _shaped, _trig_arg_symbols, _JITTER,
                                 parse_to_sympy, NOISE_REL)
from data.ast_grammar import expr_to_nodes
from train.train_manifold import build_ast_edges
from models.decoder import PAD_ID
from data.dim_infer import infer_dims
from data.dimensions import class_to_dim, DIM_LEN
try:
    from data.gen_data_pairs import _SYM2CLASS
except Exception:
    _SYM2CLASS = {}

MAX_VARS = 16


class SolvedOnTheFlyDataset:
    def __init__(self, solved_glob, max_rows=0, max_points=300, max_seq=64, seed=0):
        self.max_points = max_points
        self.max_seq = max_seq
        self.recs = []           # parsed, pre-tokenized records
        files = sorted(glob.glob(solved_glob))
        rng = np.random.default_rng(seed)
        for fp in files:
            op = gzip.open(fp, 'rt', encoding='utf-8') if fp.endswith('.gz') else open(fp, encoding='utf-8')
            with op as f:
                for line in f:
                    if not line.strip(): continue
                    try: r = json.loads(line)
                    except Exception: continue
                    rec = self._prep(r)
                    if rec is not None: self.recs.append(rec)
                    if max_rows and len(self.recs) >= max_rows: break
            if max_rows and len(self.recs) >= max_rows: break
        print(f"SolvedOnTheFly: {len(self.recs)} usable records from {len(files)} shards", flush=True)

    def _prep(self, r):
        """One-time per record: tokenize expr (the label) + lambdify y_expr (the data
        generator) + precompute dims. Returns None if any step fails."""
        try:
            others = r['vars']
            if not (1 <= len(others) <= MAX_VARS): return None
            yexpr = sp.sympify(r['y_expr'])
            yfn = sp.lambdify([sp.Symbol(v) for v in others], yexpr, 'numpy')
            res = expr_to_nodes(r['expr'])
            if res is None: return None
            seq, consts, var_order = res
            if len(seq) > self.max_seq: return None
            # trig-arg symbols (widen to >=1 period); and target sympy
            e = parse_to_sympy(r['expr'])
            trig = _trig_arg_symbols(e) if e is not None else set()
            # dims: SI dimension vectors per channel (others..., target) via inference
            dims = self._dims(r, others, var_order)
            return {'others': others, 'yfn': yfn, 'seq': seq, 'consts': consts,
                    'var_order': var_order, 'trig': trig, 'dims': dims,
                    'rng_overrides': r.get('ranges', {})}
        except Exception:
            return None

    def _dims(self, r, others, var_order):
        """(MAX_VARS+1, DIM_LEN) SI dims. Anchor on tag-known vars + infer; channels
        0..nv-1 are `others` in var_order, last channel is the target."""
        drow = np.zeros((MAX_VARS + 1, DIM_LEN), dtype=np.float32)
        target = r['target']
        allv = others + [target]
        try:
            anch = {v: class_to_dim(_SYM2CLASS[v])[:6] for v in allv
                    if v in _SYM2CLASS and class_to_dim(_SYM2CLASS[v])[6] == 0}
            inferred = infer_dims(r['expr'], anch) if anch else {}
        except Exception:
            inferred = {}
        for ci, v in enumerate(others):
            d = inferred.get(v) if inferred.get(v) is not None else class_to_dim(_SYM2CLASS.get(v, 'other'))
            drow[ci] = np.asarray(d, dtype=np.float32)[:DIM_LEN]
        dt = inferred.get(target) if inferred.get(target) is not None else class_to_dim(_SYM2CLASS.get(target, 'other'))
        drow[MAX_VARS] = np.asarray(dt, dtype=np.float32)[:DIM_LEN]
        return drow

    def __len__(self): return len(self.recs)

    def make_row(self, idx, rng=None):
        """FRESH (X,y) for record idx: sample inputs (regen method) + eval y_expr +
        50/50 noise. Different every call -> per-epoch diversity."""
        rng = rng or np.random.default_rng()
        rec = self.recs[idx]
        others = rec['others']
        n = int(rng.integers(100, self.max_points + 1)) if _JITTER else 160
        cols = {v: sample_var(v, n, rng) for v in others}
        for v in others:                                   # trig widening (>=1 period)
            if v in rec['trig']:
                k = rng.uniform(1.0, 2.5) if _JITTER else 2.0
                cols[v] = rng.uniform(0.0, 2.0 * np.pi * k, n)
        Xo = np.stack([cols[v] for v in others], axis=1).astype(np.float32)
        try:
            y = np.asarray(rec['yfn'](*[cols[v] for v in others]), float)
            y = np.broadcast_to(y, (n,)).astype(np.float32).copy()
        except Exception:
            return None
        m = np.isfinite(y) & np.all(np.isfinite(Xo), axis=1)
        if m.sum() < 30: return None
        Xo, y = Xo[m], y[m]
        if np.std(y) < 1e-9 * (abs(np.mean(y)) + 1e-9): return None   # degenerate
        if rng.random() < 0.5:                             # 50% noisy (0-2% rel)
            y = y * (1 + rng.normal(0, rng.uniform(0, NOISE_REL), len(y))).astype(np.float32)
        return {'X': Xo, 'y': y, 'seq': rec['seq'], 'consts': rec['consts'],
                'var_names': others, 'dims': rec['dims']}

    def collate(self, idxs):
        """Build a training batch (same format as MemmapPairDataset.collate, incl
        dims) by sampling FRESH data per index. None rows (degenerate sample) are
        replaced by a random valid one so the batch stays full."""
        import torch
        B = len(idxs); mp, ms = self.max_points, self.max_seq
        rng = np.random.default_rng()
        rows = []
        for idx in idxs:
            r = self.make_row(idx, rng)
            tries = 0
            while r is None and tries < 8:
                r = self.make_row(int(rng.integers(len(self.recs))), rng); tries += 1
            if r is not None: rows.append(r)
        B = len(rows)
        points = torch.zeros(B, mp, MAX_VARS + 1)
        var_mask = torch.zeros(B, MAX_VARS); point_mask = torch.zeros(B, mp)
        dims = torch.zeros(B, MAX_VARS + 1, rows[0]['dims'].shape[-1])
        enc_types, enc_const, enc_batch, enc_edges = [], [], [], []; node_off = 0
        tgt_types = torch.full((B, ms), PAD_ID, dtype=torch.long)
        tgt_consts = torch.zeros(B, ms); tgt_mask = torch.zeros(B, ms)
        for bi, r in enumerate(rows):
            X = r['X']; y = r['y']; nv = X.shape[1]; npt = min(X.shape[0], mp)
            points[bi, :npt, :nv] = torch.from_numpy(X[:npt])
            points[bi, :npt, MAX_VARS] = torch.from_numpy(y[:npt])
            var_mask[bi, :nv] = 1; point_mask[bi, :npt] = 1
            dims[bi] = torch.from_numpy(r['dims'])
            seq, consts = r['seq'], r['consts']
            for t in seq: enc_types.append(t); enc_const.append(0.0); enc_batch.append(bi)
            for (a, b) in build_ast_edges(seq): enc_edges.append((a + node_off, b + node_off))
            node_off += len(seq)
            L = min(len(seq), ms)
            tgt_types[bi, :L] = torch.tensor(seq[:L])
            for k in range(L):
                cv = consts[k]
                tgt_consts[bi, k] = 0.0 if not np.isfinite(cv) else float(np.clip(cv, -1e6, 1e6))
            tgt_mask[bi, :L] = 1
        ei = (torch.tensor(enc_edges).t().contiguous() if enc_edges else torch.zeros(2, 0, dtype=torch.long))
        return {'points': points, 'var_mask': var_mask, 'point_mask': point_mask,
                'enc_types': torch.tensor(enc_types), 'enc_const': torch.tensor(enc_const),
                'enc_batch': torch.tensor(enc_batch), 'enc_edges': ei, 'n_graphs': B,
                'tgt_types': tgt_types, 'tgt_consts': tgt_consts, 'tgt_mask': tgt_mask, 'dims': dims}
