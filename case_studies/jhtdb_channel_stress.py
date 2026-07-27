"""JHTDB channel -> full Reynolds stress tensor -> k, structure parameter -<u'v'>/k, C_mu.
OPEN PROBLEM: standard k-epsilon assumes C_mu=0.09 (=> -uv/k=0.30) CONSTANT.
DNS shows it varies strongly across the channel -> no universal closed form.
Discover -uv/k (and C_mu) as function of y+ ; compare a-priori vs constant baseline.
"""
import numpy as np, json
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData

TOKEN='uk.ac.cam.ch2067-c731b129'
RETAU=1000.0; UTAU=0.05
Lx=8*np.pi; Lz=3*np.pi
ds = turb_dataset(dataset_title='channel', output_path='./_gv_out', auth_token=TOKEN)

yplus = np.logspace(np.log10(2.0), np.log10(1000.0), 48)
yq = -1.0 + yplus/RETAU
NX,NZ=40,40
xg=(np.arange(NX)+0.5)/NX*Lx; zg=(np.arange(NZ)+0.5)/NZ*Lz
XX,ZZ=np.meshgrid(xg,zg); XX=XX.ravel(); ZZ=ZZ.ravel()
times=[0.0,3.0,6.0,9.0,12.0]

def query(pts,t):
    r=getData(ds,'velocity',t,'none','none','field',pts.astype(np.float64)); return np.array(r)[0]

U=np.zeros(len(yq)); uu=np.zeros(len(yq)); vv=np.zeros(len(yq)); ww=np.zeros(len(yq)); uv=np.zeros(len(yq))
for k,y in enumerate(yq):
    ac=[]
    for t in times:
        pts=np.stack([XX,np.full_like(XX,y),ZZ],axis=1)
        ac.append(query(pts,t))
    v=np.concatenate(ac,axis=0)
    ux,uy,uz=v[:,0],v[:,1],v[:,2]
    U[k]=ux.mean()
    up=ux-ux.mean(); vp=uy-uy.mean(); wp=uz-uz.mean()
    uu[k]=np.mean(up*up); vv[k]=np.mean(vp*vp); ww[k]=np.mean(wp*wp); uv[k]=np.mean(up*vp)
    tke=0.5*(uu[k]+vv[k]+ww[k])
    print(f'y+={yplus[k]:7.2f}  k={tke:.4e}  -uv/k={-uv[k]/tke:.4f}  Cmu_eq={(uv[k]/tke)**2:.4f}',flush=True)

kk=0.5*(uu+vv+ww)
np.save('_chs_yplus.npy',yplus); np.save('_chs_k.npy',kk); np.save('_chs_uv.npy',uv); np.save('_chs_U.npy',U)
print('DONE stresses')
