"""JHTDB isotropic1024coarse -> Kolmogorov structure functions S2(r), S3(r).
Longitudinal structure functions averaged over base points + 3 axis directions.
Targets:  S2(r) = C2 * (eps*r)^(2/3)   [Kolmogorov 2/3 law]
          S3(r) = -(4/5) * eps * r      [Kolmogorov 4/5 exact law]
"""
import numpy as np, json
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData

TOKEN='uk.ac.cam.ch2067-c731b129'
EPS = 0.0928   # known mean dissipation rate of isotropic1024 (JHTDB)
ETA = 0.00280  # Kolmogorov length
L   = 2*np.pi
ds = turb_dataset(dataset_title='isotropic1024coarse', output_path='./_gv_out', auth_token=TOKEN)

rng = np.random.RandomState(7)
M = 1500                       # base points
base = rng.rand(M,3)*L
r_vals = np.logspace(np.log10(6*ETA), np.log10(1.2), 44)  # inertial-ish range
axes = np.eye(3)

def query(pts, t=0.0):
    r = getData(ds, 'velocity', t, 'none', 'none', 'field', pts.astype(np.float64))
    return np.array(r)[0]   # (N,3)

u_base = query(base)          # (M,3)
S2 = np.zeros(len(r_vals)); S3 = np.zeros(len(r_vals))
for j,r in enumerate(r_vals):
    dul = []
    for a in range(3):
        e = axes[a]
        sh = (base + r*e) % L
        u_sh = query(sh)
        d = (u_sh - u_base) @ e     # longitudinal increment
        dul.append(d)
    dul = np.concatenate(dul)
    S2[j] = np.mean(dul**2)
    S3[j] = np.mean(dul**3)
    print(f'r={r:.4f}  S2={S2[j]:.5e}  S3={S3[j]:.5e}  -(4/5)eps r={-0.8*EPS*r:.5e}', flush=True)

np.save('_iso_r.npy', r_vals); np.save('_iso_S2.npy', S2); np.save('_iso_S3.npy', S3)

# Build task jsons (our + PSE format).  inputs: r (and eps as constant column)
def task(name, target, yv, truth):
    X = np.stack([r_vals, np.full_like(r_vals, EPS)], axis=1)  # cols: r, eps
    return {'i':0,'case':name,'target':target,'others':['r','eps'],'truth':truth,
            'X':X.tolist(),'y':list(map(float,yv)),'n_points':len(yv),'n_vars':2}

tasks=[task('iso_S2_kolmogorov23','S2','S2', 'S2 = C*(eps*r)**(2/3)'),
       task('iso_S3_fourfifths','S3','S3',  'S3 = -(4/5)*eps*r')]
json.dump(tasks, open('_tasks_jhtdb_iso.json','w'))
print('WROTE _tasks_jhtdb_iso.json  (2 tasks)  npts=',len(r_vals))
