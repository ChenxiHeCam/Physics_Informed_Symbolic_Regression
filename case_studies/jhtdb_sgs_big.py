"""SGS closure, STRENGTHENED: multiple 64^3 filter boxes -> thousands of filtered points.
Each 64^3 box = proven-stable 262k query. Sample several regions for robust a-priori stats.
Target eps_sgs = -tau_ij^dev Sbar_ij ; baseline Smagorinsky |Sbar|^3 (lit corr~0.24).
"""
import numpy as np, json, time as _t
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData
TOKEN='uk.ac.cam.ch2067-c731b129'
ds=turb_dataset(dataset_title='isotropic1024coarse', output_path='./_gv_out', auth_token=TOKEN)
L=2*np.pi; NG=1024; dx=L/NG; N=64; F=8; Delta=F*dx; M=N//F
origins=[np.array([ox,oy,oz]) for ox in (0.5,3.5) for oy in (0.5,3.5) for oz in (0.5,3.5)]  # 8 boxes
def query(pts):
    for a in range(6):
        try:
            r=getData(ds,'velocity',0.0,'none','none','field',pts.astype(np.float64)); return np.array(r)[0]
        except Exception as e:
            print('  retry',a,str(e)[:45],flush=True); _t.sleep(4)
    raise RuntimeError('fail')
def boxfilter(a):
    return a.reshape(M,F,M,F,M,F,*a.shape[3:]).mean(axis=(1,3,5))
EPS=[];SM=[];WM=[];GR=[]
for bi,x0 in enumerate(origins):
    ax=[x0[i]+np.arange(N)*dx for i in range(3)]
    GX,GY,GZ=np.meshgrid(ax[0],ax[1],ax[2],indexing='ij')
    pts=np.stack([GX.ravel(),GY.ravel(),GZ.ravel()],axis=1)%L
    vel=np.zeros((len(pts),3))
    for s in range(0,len(pts),100000):
        vel[s:s+100000]=query(pts[s:s+100000])
    u=vel.reshape(N,N,N,3)
    ubar=boxfilter(u)
    uu=np.stack([u[...,i]*u[...,j] for i in range(3) for j in range(3)],-1).reshape(N,N,N,3,3)
    tau=boxfilter(uu)-np.einsum('...i,...j->...ij',ubar,ubar)
    g=np.zeros((M,M,M,3,3))
    for i in range(3):
        for aa in range(3): g[...,i,aa]=np.gradient(ubar[...,i],Delta,axis=aa)
    S=0.5*(g+np.transpose(g,(0,1,2,4,3))); W=0.5*(g-np.transpose(g,(0,1,2,4,3)))
    td=tau-np.eye(3)*(np.trace(tau,axis1=3,axis2=4)[...,None,None]/3)
    eps=-np.einsum('...ij,...ij->...',td,S)
    Smag=np.sqrt(2*np.einsum('...ij,...ij->...',S,S)); Wmag=np.sqrt(2*np.einsum('...ij,...ij->...',W,W))
    gg=np.einsum('...ia,...ja->...ij',g,g)
    gi=-np.einsum('...ij,...ij->...',gg-np.eye(3)*np.trace(gg,axis1=3,axis2=4)[...,None,None]/3,S)
    EPS+=list(eps.ravel());SM+=list(Smag.ravel());WM+=list(Wmag.ravel());GR+=list(gi.ravel())
    print('box',bi,'done, total filtered pts=',len(EPS),flush=True)
eps=np.array(EPS);sm=np.array(SM);wm=np.array(WM);gr=np.array(GR)
np.save('_sgs_eps.npy',eps);np.save('_sgs_S.npy',sm);np.save('_sgs_W.npy',wm);np.save('_sgs_grad.npy',gr)
def corr(p,t): return float(np.corrcoef(p,t)[0,1])
c=np.sum(sm**3*eps)/np.sum(sm**6)
print('N=%d filtered pts'%len(eps))
print('Smagorinsky |S|^3 corr=%.3f'%corr(c*sm**3,eps))
print('gradient-model corr=%.3f'%corr(gr,eps))
tv={'eps':list(map(float,eps)),'Smag':list(map(float,sm)),'Wmag':list(map(float,wm)),'grad':list(map(float,gr))}
open('_jhtdb_sgs.jsonl','w').write(json.dumps({'law_id':'sgs_closure_open','symbols':['eps','Smag','Wmag','grad'],
  'eval_truth_surface':'eps - Smag**3 = 0','train_values':tv,
  'observation_note':'JHTDB isotropic1024 filtered DNS, 8 boxes; OPEN SGS closure vs Smagorinsky'})+'\n')
print('WROTE _jhtdb_sgs.jsonl (strengthened)')
