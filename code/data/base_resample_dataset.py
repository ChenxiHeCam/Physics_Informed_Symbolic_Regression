"""Training dataset that RE-SAMPLES from the dense 700-point base cache.

The expensive work (solve target -> y_expr, eval 700 wide points) was done ONCE
offline (_build_base700.py -> flat.f16 + meta.jsonl). Here we only:
  - slice an instance's 700 (X,y) points from the bfloat16 memmap (microsecond),
  - RE-SAMPLE a per-epoch-different view: random count 100-300, optional sub-window
    (narrow range -> the validated wide-train/narrow-test jitter), 50% noise,
  - tokenize the expr label (seq/consts) ONCE per unique formula (cached),
  - emit a batch in the exact PairDataset format (+ SI dims) the model consumes.

No sympy, no eval, no root-finding at train time -> the A100's ~10 CPU cores keep up.
"""
import os, json, glob, hashlib
import numpy as np
import ml_dtypes
BF16 = ml_dtypes.bfloat16

from data.ast_grammar import expr_to_nodes
from train.train_manifold import build_ast_edges
from models.decoder import PAD_ID
from data.dim_infer import infer_dims
from data.dimensions import class_to_dim, DIM_LEN
try:
    from data.gen_data_pairs import NOISE_REL
except Exception:
    NOISE_REL = 0.02
try:
    from data.gen_data_pairs import _SYM2CLASS
except Exception:
    _SYM2CLASS = {}

MAX_VARS = 16


class BaseResampleDataset:
    def __init__(self, base_dir, max_rows=0, max_points=300, min_points=100,
                 max_seq=64, jitter=True):
        self.max_points = max_points
        self.min_points = min_points
        self.max_seq = max_seq
        self.jitter = jitter
        flat_path = os.path.join(base_dir, 'flat.f16')
        meta_path = os.path.join(base_dir, 'meta.jsonl')
        self.flat = np.memmap(flat_path, dtype=BF16, mode='r')   # 1-D bf16 element stream
        self.recs = []           # usable instances (label tokenized lazily/cached)
        self._seq_cache = {}     # expr -> (seq, consts) tokenization (one-time per formula)
        self._dim_cache = {}     # (expr,target) -> dims row
        with open(meta_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try: m = json.loads(line)
                except Exception: continue
                nv = m['nv']
                if not (1 <= nv <= MAX_VARS): continue
                # validate the label tokenizes + fits; skip junk once up-front
                seq = self._seq_for(m['expr'])
                if seq is None or len(seq[0]) > max_seq: continue
                self.recs.append(m)
                if max_rows and len(self.recs) >= max_rows: break
        print(f"BaseResample: {len(self.recs)} usable instances from {base_dir}", flush=True)

    def _seq_for(self, expr):
        if expr in self._seq_cache: return self._seq_cache[expr]
        try:
            res = expr_to_nodes(expr)
            out = (res[0], res[1]) if res is not None else None
        except Exception:
            out = None
        self._seq_cache[expr] = out
        return out

    def _dims_for(self, m):
        """(MAX_VARS+1, DIM_LEN) SI dims; channels 0..nv-1 = vars (file order), last =
        target. Anchored on tag-known vars + inferred from the expr. Cached."""
        key = (m['expr'], m['target'])
        if key in self._dim_cache: return self._dim_cache[key]
        others = m['vars']; target = m['target']
        drow = np.zeros((MAX_VARS + 1, DIM_LEN), dtype=np.float32)
        try:
            anch = {v: class_to_dim(_SYM2CLASS[v])[:6] for v in others + [target]
                    if v in _SYM2CLASS and class_to_dim(_SYM2CLASS[v])[6] == 0}
            inferred = infer_dims(m['expr'], anch) if anch else {}
        except Exception:
            inferred = {}
        for ci, v in enumerate(others):
            d = inferred.get(v) if inferred.get(v) is not None else class_to_dim(_SYM2CLASS.get(v, 'other'))
            drow[ci] = np.asarray(d, dtype=np.float32)[:DIM_LEN]
        dt = inferred.get(target) if inferred.get(target) is not None else class_to_dim(_SYM2CLASS.get(target, 'other'))
        drow[MAX_VARS] = np.asarray(dt, dtype=np.float32)[:DIM_LEN]
        self._dim_cache[key] = drow
        return drow

    def __len__(self): return len(self.recs)

    def _slice(self, m):
        """Read instance m's (npt, nv+1) base matrix from the bf16 memmap -> float32."""
        npt, nv = m['npt'], m['nv']
        w = nv + 1
        a = np.asarray(self.flat[m['ptr']: m['ptr'] + npt * w], dtype=np.float32)
        return a.reshape(npt, w)

    def make_row(self, idx, rng):
        """A per-epoch-different VIEW of instance idx's 700 base points."""
        m = self.recs[idx]
        mat = self._slice(m)                       # (npt, nv+1) float32
        npt, nv = mat.shape[0], m['nv']
        X = mat[:, :nv]; y = mat[:, nv]
        keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[keep], y[keep]
        if len(y) < self.min_points: return None
        # optional SUB-WINDOW (narrow range jitter: train wide -> test narrow, +8.3pp
        # validated). Narrow along the single highest-variance input axis so enough
        # points survive in any dimension count.
        if self.jitter and nv >= 1 and rng.random() < 0.5 and len(y) > self.min_points * 2:
            ax = int(np.argmax(X.std(0) + 1e-9))
            lo, hi = X[:, ax].min(), X[:, ax].max()
            span = hi - lo
            if span > 0:
                frac = rng.uniform(0.30, 1.0)        # keep 30-100% of the axis span
                c = rng.uniform(lo + span * frac / 2, hi - span * frac / 2) if frac < 1 else (lo + hi) / 2
                sel = np.abs(X[:, ax] - c) <= span * frac / 2
                if sel.sum() >= self.min_points:
                    X, y = X[sel], y[sel]
        # random count 100-300
        cnt = int(rng.integers(self.min_points, self.max_points + 1)) if self.jitter else min(len(y), self.max_points)
        cnt = min(cnt, len(y))
        sub = rng.choice(len(y), cnt, replace=False)
        X, y = X[sub], y[sub]
        if np.std(y) < 1e-9 * (abs(np.mean(y)) + 1e-9): return None   # degenerate
        if self.jitter and rng.random() < 0.5:                        # 50% noisy 0-2% rel
            y = y * (1 + rng.normal(0, rng.uniform(0, NOISE_REL), len(y))).astype(np.float32)
        seq, consts = self._seq_for(m['expr'])
        return {'X': X.astype(np.float32), 'y': y.astype(np.float32),
                'seq': seq, 'consts': consts, 'dims': self._dims_for(m)}

    def collate(self, idxs):
        """Build a training batch (PairDataset format + dims). None rows (degenerate)
        are replaced by a random valid one so the batch stays full."""
        import torch
        mp, ms = self.max_points, self.max_seq
        rng = np.random.default_rng()
        rows = []
        for idx in idxs:
            r = self.make_row(idx, rng); tries = 0
            while r is None and tries < 8:
                r = self.make_row(int(rng.integers(len(self.recs))), rng); tries += 1
            if r is not None: rows.append(r)
        B = len(rows)
        points = torch.zeros(B, mp, MAX_VARS + 1)
        var_mask = torch.zeros(B, MAX_VARS); point_mask = torch.zeros(B, mp)
        dims = torch.zeros(B, MAX_VARS + 1, DIM_LEN)
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
