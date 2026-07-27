import sys, os, json, re
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))
import numpy as np
import gen_dataside as GD
from itertools import islice

rng = np.random.default_rng(5)

def repair(e):
    e = e.replace('\\approx', '=').replace('\\left', '').replace('\\right', '').replace('\\,', ' ')
    e = re.sub(r'\^', '**', e)          # ^ power -> ** ; leaves existing ** alone
    return e

fail = rep = 0
samp = []
with open('data/data/augmented/round1_ds_dedup_0050.jsonl') as f:
    for l in islice(f, 0, 3000):
        e = json.loads(l).get('new_expr')
        if not e or GD.make_one(e, rng, None):
            continue
        fail += 1
        try:
            if GD.make_one(repair(e), rng, None):
                rep += 1
                if len(samp) < 5:
                    samp.append(e[:60])
        except Exception:
            pass
print(f'failing {fail}, repaired by ^->**/approx: {rep} = {rep*100//max(fail,1)}%')
for s in samp:
    print('  recovered:', repr(s))
