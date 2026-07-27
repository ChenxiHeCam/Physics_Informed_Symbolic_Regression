"""
Convert data_pairs_*.jsonl.gz  ->  memmap cache  (memory-safe full-data training).

Why: PairDataset loads every row into RAM as Python lists. At 2.68M rows that is
~100GB and the process OOM-kills silently. Training collate only ever consumes the
first `max_points` (128) points of each row, so the cache stores just those, as a
single flat float32 file we memmap — only touched rows ever enter RAM.

Cache layout (one directory):
  flat.f32   raw float32, each row's (npts, nv+1) matrix row-major, concatenated
  ptr.npy    int64 (N+1,)  element offsets into flat   (row i = flat[ptr[i]:ptr[i+1]])
  npts.npy   int32 (N,)
  nv.npy     int16 (N,)
  seq.npy    int16 (N, MS) node-id sequence, padded with -1
  slen.npy   int16 (N,)
  consts.npy float32 (N, MS)
  exprs.txt  N lines, the source expr (for building the z_f retrieval index)
"""
import sys, os, gzip, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data.ast_grammar import NT2ID, ID2NT
from data.dimensions import class_to_dim, DIM_LEN
from data.dim_infer import infer_dims
from data.dim_analysis import formula_dim
from data.ast_grammar import parse_to_sympy

_DIM_CACHE = {}   # expr -> validated inferred {var: 6-list} (consistent inference only, else None->tags)

MAX_POINTS = 128
MAX_SEQ = 80
MAX_VARS = 16
TAGS = 'data/symbol_semantic_tags_20260522.jsonl'

_SYM2CLASS = {}
def load_tags():
    if _SYM2CLASS: return
    try:
        with open(TAGS, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                _SYM2CLASS[r['symbol']] = r.get('semantic_class', 'other')
    except Exception:
        pass

# precompute: old VAR slot id -> the integer i in "VAR_i"
_VARID2SLOT = {NT2ID[f"VAR_{i}"]: i for i in range(MAX_VARS)}
_SLOT2VARID = {i: NT2ID[f"VAR_{i}"] for i in range(MAX_VARS)}


def remap_seq_target_last(seq, formula_vars, var_names, target):
    """Reorder VAR slots so VAR_0..VAR_{k-1} = the X-column variables (var_names,
    in column order) and VAR_k = the target (the y-channel). This makes the
    decoder's VAR slots consistent with the data-encoder input, which the
    original sorted(all-symbols) convention did not (the target's slot depended
    on its NAME's sort position, unlearnable from numeric y). Returns new seq, or
    None if the row is inconsistent."""
    new_order = list(var_names) + [target]
    if len(formula_vars) != len(new_order) or set(formula_vars) != set(new_order):
        return None
    pos = {v: j for j, v in enumerate(new_order)}
    out = []
    for tid in seq:
        if tid in _VARID2SLOT:
            sym = formula_vars[_VARID2SLOT[tid]]
            out.append(_SLOT2VARID[pos[sym]])
        else:
            out.append(tid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='dataset_20260531/data_pairs_v4_full.jsonl.gz')
    ap.add_argument('--out', default='dataset_20260531/cache_v4_full')
    ap.add_argument('--max-points', type=int, default=MAX_POINTS)
    ap.add_argument('--max-seq', type=int, default=MAX_SEQ)
    ap.add_argument('--limit', type=int, default=0)  # 0 = all
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    load_tags()
    mp, ms = args.max_points, args.max_seq
    flat_path = os.path.join(args.out, 'flat.f32')

    ptr = [0]
    npts_l, nv_l, slen_l = [], [], []
    seq_arr_rows, const_arr_rows, dim_arr_rows = [], [], []
    exprs_f = open(os.path.join(args.out, 'exprs.txt'), 'w', encoding='utf-8')

    t0 = time.time(); n = 0; skipped = 0
    sources = [p for p in args.pairs.split(',') if p.strip()]
    print(f"merging {len(sources)} source(s): {sources}")
    with open(flat_path, 'wb') as fb:
      for _src in sources:
        try:
            _fh = gzip.open(_src, 'rt', encoding='utf-8')
        except Exception as _e:
            print(f"  skip source {_src}: {_e}"); continue
        for line in _fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                seq = r['seq']
                if len(seq) > ms:
                    skipped += 1; continue
                vn = r['var_names']
                if len(vn) > MAX_VARS or len(vn) < 1:
                    skipped += 1; continue
                # remap VAR slots to target-last convention (consistent with z_d)
                seq2 = remap_seq_target_last(seq, r.get('formula_vars', []),
                                             vn, r.get('target', ''))
                if seq2 is None or len(vn) + 1 > MAX_VARS:
                    skipped += 1; continue
                seq = seq2
                X = np.asarray(r['X'], dtype=np.float32)
                y = np.asarray(r['y'], dtype=np.float32)
                if X.ndim != 2 or X.shape[0] != y.shape[0]:
                    skipped += 1; continue
                npt = min(X.shape[0], mp)
                nv = X.shape[1]
                if nv != len(vn):
                    skipped += 1; continue
                # pack (npt, nv+1): X | y
                row = np.empty((npt, nv + 1), dtype=np.float32)
                row[:, :nv] = X[:npt]
                row[:, nv] = y[:npt]
                row = np.nan_to_num(row, nan=0.0, posinf=1e6, neginf=-1e6)
                fb.write(row.tobytes())
                cnt = npt * (nv + 1)
                ptr.append(ptr[-1] + cnt)
                npts_l.append(npt); nv_l.append(nv)
                # seq + consts padded
                srow = np.full(ms, -1, dtype=np.int16)
                crow = np.zeros(ms, dtype=np.float32)
                L = min(len(seq), ms)
                srow[:L] = np.asarray(seq[:L], dtype=np.int16)
                cv = np.asarray(r['consts'][:L], dtype=np.float64)
                cv = np.nan_to_num(cv, nan=0.0, posinf=1e6, neginf=-1e6)
                crow[:L] = np.clip(cv, -1e6, 1e6).astype(np.float32)
                seq_arr_rows.append(srow); const_arr_rows.append(crow)
                slen_l.append(L)
                # per-channel SI dimension vectors: channels 0..nv-1 = others
                # (var_names order), channel MAX_VARS = target (the y channel)
                drow = np.zeros((MAX_VARS + 1, DIM_LEN), dtype=np.int8)
                # FULL SI dims via dimensional inference (anchored on tag-known vars +
                # fundamental constants), cached per-formula. Falls back to tags per-var.
                _ex = r.get('expr', ''); _tgt = r.get('target', '')
                if _ex not in _DIM_CACHE:
                    _allv = list(vn) + [_tgt]
                    _anch = {v: class_to_dim(_SYM2CLASS[v])[:6] for v in _allv
                             if v in _SYM2CLASS and class_to_dim(_SYM2CLASS[v])[6] == 0}
                    try:
                        _ii = infer_dims(_ex, _anch)
                        # VALIDATE: only trust inference if it makes the formula
                        # dimensionally consistent (rejects gauge-floated wrong solves)
                        if _ii is not None:
                            _vd = {k: tuple(v) for k, v in _ii.items()}
                            if not formula_dim(parse_to_sympy(_ex), _vd).consistent:
                                _ii = None
                        _DIM_CACHE[_ex] = _ii
                    except Exception: _DIM_CACHE[_ex] = None
                _inf = _DIM_CACHE[_ex]
                for ci, vname in enumerate(vn):
                    if _inf and vname in _inf:
                        drow[ci, :6] = _inf[vname]
                    else:
                        drow[ci] = class_to_dim(_SYM2CLASS.get(vname, 'other'))
                if _inf and _tgt in _inf:
                    drow[MAX_VARS, :6] = _inf[_tgt]
                else:
                    drow[MAX_VARS] = class_to_dim(_SYM2CLASS.get(_tgt, 'other'))
                dim_arr_rows.append(drow)
                exprs_f.write(r.get('expr', '') + '\n')
                n += 1
                if args.limit and n >= args.limit:
                    break
                if n % 100000 == 0:
                    gb = ptr[-1] * 4 / 1e9
                    print(f"  {n} rows | {gb:.2f}GB flat | {skipped} skip | "
                          f"{n/(time.time()-t0):.0f}/s | {time.time()-t0:.0f}s",
                          flush=True)
            except Exception:
                skipped += 1; continue
    exprs_f.close()

    np.save(os.path.join(args.out, 'ptr.npy'), np.asarray(ptr, dtype=np.int64))
    np.save(os.path.join(args.out, 'npts.npy'), np.asarray(npts_l, dtype=np.int32))
    np.save(os.path.join(args.out, 'nv.npy'), np.asarray(nv_l, dtype=np.int16))
    np.save(os.path.join(args.out, 'seq.npy'), np.stack(seq_arr_rows))
    np.save(os.path.join(args.out, 'slen.npy'), np.asarray(slen_l, dtype=np.int16))
    np.save(os.path.join(args.out, 'consts.npy'), np.stack(const_arr_rows))
    np.save(os.path.join(args.out, 'dims.npy'), np.stack(dim_arr_rows))
    meta = {'n': n, 'max_points': mp, 'max_seq': ms, 'max_vars': MAX_VARS,
            'flat_dtype': 'float32', 'skipped': skipped, 'dim_len': DIM_LEN}
    with open(os.path.join(args.out, 'meta.json'), 'w') as mf:
        json.dump(meta, mf)
    print(f"\nDone: {n} rows | {skipped} skipped | flat {ptr[-1]*4/1e9:.2f}GB | "
          f"{time.time()-t0:.0f}s\nCache -> {args.out}")


if __name__ == '__main__':
    main()
