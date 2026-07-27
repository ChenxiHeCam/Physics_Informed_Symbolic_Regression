"""Characterize which formulas FAILED data generation (not in usable set), by type,
to confirm whether complex/transcendental formulas are over-represented in failures."""
import gzip, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from collections import Counter
from data.ast_grammar import expr_to_nodes

usable = set()
for line in open('dataset_20260531/cache_v4_dims/exprs.txt', encoding='utf-8'):
    e = line.strip()
    if e: usable.add(e)
print('usable unique exprs:', len(usable))


def feats(e):
    return e.count('exp'), ('sin' in e or 'cos' in e or 'tan' in e), 'log' in e, \
           ('Abs' in e or 'abs' in e), 'sqrt' in e


fail_c, ok_c = Counter(), Counter()
fail_len, ok_len = [], []
tot = fail = 0
with gzip.open('dataset_20260531/unified_nodes_precon_cleaned.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line); ex = r.get('expr', '')
        if not ex: continue
        tot += 1
        res = expr_to_nodes(ex); sl = len(res[0]) if res else 999
        isfail = ex not in usable
        ne, trig, lg, ab, sq = feats(ex)
        tags = []
        if ne >= 1 and trig: tags.append('exp*trig')
        if ne >= 2: tags.append('multi_exp')
        if ab: tags.append('abs')
        if sl > 48: tags.append('len>48')
        if lg: tags.append('log')
        if not tags: tags.append('plain')
        if isfail:
            fail += 1; fail_len.append(sl)
            for t in tags: fail_c[t] += 1
        else:
            ok_len.append(sl)
            for t in tags: ok_c[t] += 1
        if tot >= 300000: break

print('scanned %d | failed(not in usable) %d (%.0f%%)' % (tot, fail, 100*fail/tot))
print('fail median seqlen %d vs ok median %d' %
      (int(np.median(fail_len)), int(np.median(ok_len))))
print('\ntype          fail_count  fail_rate_within_type')
for t in ['exp*trig', 'multi_exp', 'abs', 'len>48', 'log', 'plain']:
    ft = fail_c[t]; ot = ok_c[t]; tt = ft + ot
    print('%-12s  %8d   %.0f%%' % (t, ft, 100*ft/max(tt, 1)))
