"""Convert gen_fpool's fpool.jsonl ({seq,consts} per line) -> the seq.npy/consts.npy/slen.npy
directory FormulaPoolDataset expects. Pads seq to MS (matches the with-data cache width 48),
drops formulas longer than MS. Pre-allocates arrays for speed on ~12M rows.
Usage: python fpool_to_cache.py fpool.jsonl fpool_cache [MS]
"""
import sys, os, json
import numpy as np
inp, out = sys.argv[1], sys.argv[2]
MS = int(sys.argv[3]) if len(sys.argv) > 3 else 80

N = 0
with open(inp, encoding='utf-8') as f:
    for _ in f:
        N += 1
print(f'{N} lines; allocating (N,{MS})', flush=True)
seq = np.full((N, MS), -1, dtype=np.int16)
consts = np.zeros((N, MS), dtype=np.float32)
slen = np.zeros(N, dtype=np.int16)
i = 0
with open(inp, encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        s = r.get('seq'); c = r.get('consts') or []
        if not s or len(s) > MS:
            continue
        L = len(s)
        seq[i, :L] = s
        cl = min(len(c), MS); consts[i, :cl] = c[:cl]
        slen[i] = L; i += 1
os.makedirs(out, exist_ok=True)
np.save(os.path.join(out, 'seq.npy'), seq[:i])
np.save(os.path.join(out, 'consts.npy'), consts[:i])
np.save(os.path.join(out, 'slen.npy'), slen[:i])
print(f'DONE: {i} formulas (<= {MS} tokens) -> {out}', flush=True)
