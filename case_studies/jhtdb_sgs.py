"""OPEN PROBLEM A: LES subgrid-scale (SGS) stress closure, a-priori from isotropic DNS.
Box-filter DNS at scale Delta -> exact SGS stress tau_ij = (u_i u_j)_bar - ubar_i ubar_j.
Target: SGS dissipation eps_sgs = -tau_ij^dev Sbar_ij.
Baseline: Smagorinsky eps_sgs = 2(Cs*Delta)^2 |Sbar|^3  (famously poor a-priori correlation ~0.3).
Discover a better closure from resolved invariants |Sbar|, |Wbar|, gradient-model term.
"""
import numpy as np, json
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData

TOKEN='uk.ac.cam.ch2067-c731b129'
ds = turb_dataset(dataset_title='isotropic1024coarse', output_path='./_gv_out', auth_token=TOKEN)
L=2*np.pi; NG=1024; dx=L/NG
N=64; F=8                        # fine grid N^3=262k (robust), box filter F -> 8^3=512 filtered
Delta=F*dx
x0=np.array([1.0,1.0,1.0])       # subvolume origin
# fine grid points
ax=[x0[i]+np.arange(N)*dx for i in range(3)]
GX,GY,GZ=np.meshgrid(ax[0],ax[1],ax[2],indexing='ij')
pts=np.stack([GX.ravel(),GY.ravel(),GZ.ravel()],axis=1)
print('querying',len(pts),'points...',flush=True)
import time as _t
CH=100000
vel=np.zeros((len(pts),3))
for s in range(0,len(pts),CH):
    for attempt in range(6):
        try:
            r=getData(ds,'velocity',0.0,'none','none','field',pts[s:s+CH].astype(np.float64))
            vel[s:s+CH]=np.array(r)[0]; break
        except Exception as e:
            print('  retry',attempt,str(e)[:50],flush=True); _t.sleep(5)
    else:
        raise RuntimeError('chunk failed')
    print('  got',s+min(CH,len(pts)-s),flush=True)
u=vel.reshape(N,N,N,3)

def boxfilter(a):   # average over F^3 blocks -> (N/F)^3
    M=N//F
    return a.reshape(M,F,M,F,M,F,*a.shape[3:]).mean(axis=(1,3,5))

ubar=boxfilter(u)                                   # (M,M,M,3)
uu=np.stack([u[...,i]*u[...,j] for i in range(3) for j in range(3)],axis=-1).reshape(N,N,N,3,3)
uubar=boxfilter(uu)                                 # (M,M,M,3,3)
tau=uubar-np.einsum('...i,...j->...ij',ubar,ubar)   # SGS stress
M=N//F
# resolved velocity gradient on filtered grid
g=np.zeros((M,M,M,3,3))
for i in range(3):
    for a in range(3):
        g[...,i,a]=np.gradient(ubar[...,i],Delta,axis=a)
S=0.5*(g+np.transpose(g,(0,1,2,4,3)))               # strain
W=0.5*(g-np.transpose(g,(0,1,2,4,3)))               # rotation
tau_dev=tau-np.eye(3)*(np.trace(tau,axis1=3,axis2=4)[...,None,None]/3)
eps_sgs=-np.einsum('...ij,...ij->...',tau_dev,S)    # SGS dissipation (target)
Smag=np.sqrt(2*np.einsum('...ij,...ij->...',S,S))   # |Sbar|
Wmag=np.sqrt(2*np.einsum('...ij,...ij->...',W,W))   # |Wbar|
gg=np.einsum('...ia,...ja->...ij',g,g)              # gradient-model tensor g g^T
grad_inv=-np.einsum('...ij,...ij->...',gg-np.eye(3)*np.trace(gg,axis1=3,axis2=4)[...,None,None]/3,S)

es=eps_sgs.ravel(); sm=Smag.ravel(); wm=Wmag.ravel(); gi=grad_inv.ravel()
np.save('_sgs_eps.npy',es); np.save('_sgs_S.npy',sm); np.save('_sgs_W.npy',wm); np.save('_sgs_grad.npy',gi)

def r2(p,t): return 1-np.sum((p-t)**2)/np.sum((t-t.mean())**2)
def corr(p,t): return float(np.corrcoef(p,t)[0,1])
# Smagorinsky a-priori (fit Cs^2 by least squares): eps ~ c*|S|^3
smag_pred=sm**3; c=np.sum(smag_pred*es)/np.sum(smag_pred**2)
print('N filtered pts=',len(es),' Delta=%.4f (%.0f eta)'%(Delta,Delta/0.0028))
print('Smagorinsky |S|^3  : corr=%.3f  R2=%.3f'%(corr(smag_pred,es),r2(c*smag_pred,es)))
print('gradient-model     : corr=%.3f'%corr(gi,es))

# build jsonl: target eps_sgs from resolved invariants
tv={'eps':list(map(float,es)),'Smag':list(map(float,sm)),'Wmag':list(map(float,wm)),'grad':list(map(float,gi))}
row=json.dumps({'law_id':'sgs_closure_open','symbols':['eps','Smag','Wmag','grad'],
                'eval_truth_surface':'eps - Smag**3 = 0',  # Smagorinsky reference (weak); judged by a-priori corr
                'train_values':tv,'observation_note':'JHTDB isotropic1024 filtered DNS; OPEN SGS closure, Smagorinsky baseline'})
open('_jhtdb_sgs.jsonl','w').write(row+'\n')
print('WROTE _jhtdb_sgs.jsonl')
