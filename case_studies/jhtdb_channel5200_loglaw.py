"""JHTDB channel5200 (Re_tau=5185.9) -> mean U(y) over a LONG log region (y+ 30-2000).
Longer log layer than Re_tau=1000 -> log clearly separable from a weak power law.
"""
import numpy as np, json
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData

TOKEN='uk.ac.cam.ch2067-c731b129'
RETAU=5185.897; UTAU=0.0414872; NU=8e-6
Lx=8*np.pi; Lz=3*np.pi
ds = turb_dataset(dataset_title='channel5200', output_path='./_gv_out', auth_token=TOKEN)

yplus = np.logspace(np.log10(30.0), np.log10(2000.0), 40)   # long log region
yq = -1.0 + yplus/RETAU
NX,NZ=20,20
xg=(np.arange(NX)+0.5)/NX*Lx; zg=(np.arange(NZ)+0.5)/NZ*Lz
XX,ZZ=np.meshgrid(xg,zg); XX=XX.ravel(); ZZ=ZZ.ravel()
times=[1,4,7]   # channel5200 valid times integers [1,11]; average a few snapshots + x,z

import time as _time
def query(pts,t):
    for attempt in range(5):
        try:
            r=getData(ds,'velocity',t,'none','none','field',pts.astype(np.float64)); return np.array(r)[0]
        except Exception as e:
            print('  retry',attempt,str(e)[:60],flush=True); _time.sleep(4)
    raise RuntimeError('query failed after retries')

U=np.zeros(len(yq))
for k,y in enumerate(yq):
    pts=np.stack([XX,np.full_like(XX,y),ZZ],axis=1)
    acc=[query(pts,t)[:,0] for t in times]
    U[k]=np.concatenate(acc).mean()
    print(f'y+={yplus[k]:8.2f}  U+={U[k]/UTAU:7.3f}',flush=True)

Up=U/UTAU
np.save('_ch5200_yplus.npy',yplus); np.save('_ch5200_Up.npy',Up)
c=np.polyfit(np.log(yplus),Up,1)
print('log-law fit over y+ 30-2000: U+ = %.3f ln(y+) + %.3f  kappa=%.3f (textbook 0.41)'%(c[0],c[1],1/c[0]))

row=json.dumps({'law_id':'law_of_the_wall_Re5200','symbols':['Uplus','yplus'],
                'eval_truth_surface':'Uplus - (2.44*log(yplus) + 5.2) = 0',
                'train_values':{'Uplus':list(map(float,Up)),'yplus':list(map(float,yplus))},
                'observation_note':'JHTDB channel5200 Re_tau=5186 DNS, long log region'})
open('_jhtdb_ch5200.jsonl','w').write(row+'\n')
print('WROTE _jhtdb_ch5200.jsonl')
