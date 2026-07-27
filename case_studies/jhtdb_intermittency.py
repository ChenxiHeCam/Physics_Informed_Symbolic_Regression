"""OPEN PROBLEM: anomalous scaling / intermittency exponents zeta_p.
Higher-order longitudinal structure functions S_p(r)=<|du_L|^p> ~ r^zeta_p.
Kolmogorov 1941 (non-intermittent) says zeta_p = p/3. DNS shows DEVIATION (intermittency)
with NO first-principles formula; best models are heuristic (She-Leveque 1994:
zeta_p = p/9 + 2(1-(2/3)^(p/3))). SR task: discover zeta_p as a function of order p.
"""
import numpy as np, json
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData

TOKEN='uk.ac.cam.ch2067-c731b129'
ds = turb_dataset(dataset_title='isotropic1024coarse', output_path='./_gv_out', auth_token=TOKEN)
L=2*np.pi; ETA=0.00280
rng=np.random.RandomState(11)
M=4000
base=rng.rand(M,3)*L
r_vals=np.logspace(np.log10(0.12),np.log10(0.52),12)   # inertial range (from prior compensated-plateau analysis)
axes=np.eye(3)
orders=np.arange(1,9)                                   # p = 1..8
import time as _t
def query(pts):
    for a in range(5):
        try:
            r=getData(ds,'velocity',0.0,'none','none','field',pts.astype(np.float64)); return np.array(r)[0]
        except Exception as e:
            print('  retry',a,str(e)[:40],flush=True); _t.sleep(4)
    raise RuntimeError('fail')

ub=query(base)
Sp=np.zeros((len(r_vals),len(orders)))
for j,r in enumerate(r_vals):
    incs=[]
    for ax in range(3):
        e=axes[ax]; sh=(base+r*e)%L
        d=(query(sh)-ub)@e
        incs.append(d)
    d=np.concatenate(incs)
    for k,p in enumerate(orders):
        Sp[j,k]=np.mean(np.abs(d)**p)
    print('r=%.3f done'%r,flush=True)

# fit zeta_p = slope of log S_p vs log r
zeta=np.array([np.polyfit(np.log(r_vals),np.log(Sp[:,k]),1)[0] for k in range(len(orders))])
np.save('_interm_p.npy',orders); np.save('_interm_zeta.npy',zeta); np.save('_interm_Sp.npy',Sp)
for p,z in zip(orders,zeta):
    sl = p/9 + 2*(1-(2/3)**(p/3))
    print('p=%d  zeta_p(DNS)=%.3f  K41 p/3=%.3f  She-Leveque=%.3f'%(p,z,p/3,sl))

# SR task: discover zeta_p(p). truth ref = K41 p/3 (which is WRONG at high p -> open).
row=json.dumps({'law_id':'anomalous_scaling_zeta_p','symbols':['zeta','p'],
                'eval_truth_surface':'zeta - p/3 = 0',   # K41 reference (fails at high p) - judged by fit
                'train_values':{'zeta':list(map(float,zeta)),'p':list(map(float,orders))},
                'observation_note':'JHTDB isotropic1024 DNS intermittency exponents; K41 p/3 fails, She-Leveque heuristic'})
open('_jhtdb_intermittency.jsonl','w').write(row+'\n')
print('WROTE _jhtdb_intermittency.jsonl')
