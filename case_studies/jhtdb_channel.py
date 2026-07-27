"""JHTDB channel (Re_tau=1000) -> mean profile U(y), Reynolds shear stress -<u'v'>(y).
Average over homogeneous x,z and several times.
Targets: (A) log law of the wall  U+ = (1/kappa) ln(y+) + B,  kappa~0.41, B~5.2
         (B) mixing-length closure -<u'v'> = (kappa*d)^2 * (dU/dy)^2   [Reynolds stress closure]
"""
import numpy as np, json
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData

TOKEN='uk.ac.cam.ch2067-c731b129'
NU=5e-5; RETAU=1000.0; UTAU=0.05      # channel constants
Lx=8*np.pi; Lz=3*np.pi
ds = turb_dataset(dataset_title='channel', output_path='./_gv_out', auth_token=TOKEN)

# y-planes: log-spaced in y+ over lower half (wall at y=-1)
yplus = np.logspace(np.log10(1.0), np.log10(950), 55)
yq = -1.0 + yplus/RETAU
NX, NZ = 32, 32
xg = (np.arange(NX)+0.5)/NX*Lx
zg = (np.arange(NZ)+0.5)/NZ*Lz
XX,ZZ = np.meshgrid(xg,zg); XX=XX.ravel(); ZZ=ZZ.ravel()
times=[0.0,4.0,8.0,12.0]

def query(pts,t):
    r=getData(ds,'velocity',t,'none','none','field',pts.astype(np.float64))
    return np.array(r)[0]

U=np.zeros(len(yq)); uv=np.zeros(len(yq))
for k,y in enumerate(yq):
    ux_acc=[]; uv_acc=[]
    for t in times:
        pts=np.stack([XX, np.full_like(XX,y), ZZ],axis=1)
        v=query(pts,t)                 # (N,3): ux,uy,uz
        ux=v[:,0]; uy=v[:,1]
        Um=ux.mean()
        ux_acc.append(ux); uv_acc.append((ux-Um)*(uy-uy.mean()))
    ux_all=np.concatenate(ux_acc)
    U[k]=ux_all.mean()
    uv[k]=np.concatenate(uv_acc).mean()
    print(f'y+={yplus[k]:7.2f}  U+={U[k]/UTAU:7.3f}  -uv={-uv[k]:.4e}',flush=True)

np.save('_ch_yplus.npy',yplus); np.save('_ch_U.npy',U); np.save('_ch_uv.npy',uv)
dUdy=np.gradient(U,yq)
d_wall=1.0+yq                          # distance from lower wall (outer units)

def dump(name,target,cols,colvals,yv,truth):
    X=np.stack(colvals,axis=1)
    return {'i':0,'case':name,'target':target,'others':cols,'truth':truth,
            'X':X.tolist(),'y':list(map(float,yv)),'n_points':len(yv),'n_vars':len(cols)}

log_m = yplus>=30                      # log-law region
Up=U/UTAU
tA=dump('channel_loglaw','Uplus',['yplus'],[yplus[log_m]],Up[log_m],
        'Uplus = (1/0.41)*log(yplus) + 5.2')
clos_m=(yplus>=30)&(yplus<=350)        # log layer where mixing length valid
tB=dump('channel_reystress_closure','neguv',['d','dUdy'],
        [d_wall[clos_m], dUdy[clos_m]], (-uv)[clos_m],
        'neguv = (0.41*d)**2 * dUdy**2')
json.dump([tA,tB],open('_tasks_jhtdb_channel.json','w'))
print('WROTE _tasks_jhtdb_channel.json  loglaw npts=',log_m.sum(),' closure npts=',clos_m.sum())
