"""OPEN PROBLEM (constraint-verifiable, 'several unknowns, known sum'):
TKE budget in channel. Production P = -<u'v'> dU/dy (closed/measurable).
Dissipation eps = 2 nu <s'_ij s'_ij> (fluctuating strain, from velocity gradient).
Steady-state budget: P - eps + Transport = 0  ->  Transport = eps - P (the unclosed residual).
KNOWN SUM (=0) constrains the unknown transport closure. In log layer P~=eps (equilibrium).
"""
import numpy as np, json
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData

TOKEN='uk.ac.cam.ch2067-c731b129'
NU=5e-5; RETAU=1000.0; UTAU=0.05
Lx=8*np.pi; Lz=3*np.pi
ds=turb_dataset(dataset_title='channel', output_path='./_gv_out', auth_token=TOKEN)
yplus=np.logspace(np.log10(10.0), np.log10(700.0), 30)
yq=-1.0+yplus/RETAU
NX,NZ=24,24
xg=(np.arange(NX)+0.5)/NX*Lx; zg=(np.arange(NZ)+0.5)/NZ*Lz
XX,ZZ=np.meshgrid(xg,zg); XX=XX.ravel(); ZZ=ZZ.ravel()
times=[0.0,6.0]
import time as _t
def q(pts,t,op='field',sm='none'):
    for a in range(5):
        try:
            r=getData(ds,'velocity',t,'none',sm,op,pts.astype(np.float64))
            return np.array(r)[0]
        except Exception as e:
            print('  retry',a,str(e)[:50],flush=True); _t.sleep(4)
    raise RuntimeError('fail')

U=np.zeros(len(yq)); uv=np.zeros(len(yq)); eps=np.zeros(len(yq)); dUdy=np.zeros(len(yq))
for k,y in enumerate(yq):
    ux_a=[]; uv_a=[]; eps_a=[]; dudy_a=[]
    for t in times:
        pts=np.stack([XX,np.full_like(XX,y),ZZ],axis=1)
        v=q(pts,t,'field','none')                # (N,3)
        g=q(pts,t,'gradient','fd4noint')         # (N,9): dudx,dudy,dudz,dvdx,...
        ux=v[:,0]; uy=v[:,1]
        ux_a.append(ux); uv_a.append((ux-ux.mean())*(uy-uy.mean()))
        dudy_a.append(g[:,1].mean())             # d<u>/dy
        G=g.reshape(-1,3,3)
        Gm=G.mean(axis=0)
        Gp=G-Gm                                  # fluctuating velocity gradient
        s=0.5*(Gp+np.transpose(Gp,(0,2,1)))
        eps_a.append(2*NU*np.mean(np.einsum('nij,nij->n',s,s)))
    U[k]=np.concatenate(ux_a).mean()
    uv[k]=np.concatenate(uv_a).mean()
    dUdy[k]=np.mean(dudy_a); eps[k]=np.mean(eps_a)
    print('y+=%7.2f  -uv+=%.4f  eps+=%.4e  dUdy+=%.4f'%(yplus[k],-uv[k]/UTAU**2,eps[k]*NU/UTAU**4,dUdy[k]/(UTAU*RETAU)),flush=True)

P=(-uv)*dUdy                                     # production
np.save('_tke_yplus.npy',yplus); np.save('_tke_P.npy',P); np.save('_tke_eps.npy',eps)
np.save('_tke_uv.npy',uv); np.save('_tke_dUdy.npy',dUdy)
Pp=P*NU/UTAU**4; epsp=eps*NU/UTAU**4
print('\nTKE budget (wall units): Production vs Dissipation (log layer should ~balance)')
for i in range(0,len(yplus),3):
    print(' y+=%7.2f  P+=%.4f  eps+=%.4f  P-eps=%+.4f (=transport residual)'%(yplus[i],Pp[i],epsp[i],Pp[i]-epsp[i]))
print('WROTE _tke_*.npy')
