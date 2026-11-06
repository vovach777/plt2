# error-axis neighbours for the 4-D context: left+top (now) vs top-row only; zigzag sum, 3 levels, 1029 cells -> 8/16 groups
import sys, glob, numpy as np
sys.path.insert(0, '/tmp/rct'); from llcomp_q7e12 import agglo, Hrows
from PIL import Image
S='/tmp/claude-0/-home-claude/739fa5c5-1bdc-534c-8b96-8baaa717ae9c/scratchpad'
files=sorted(glob.glob(S+'/dset/*.ppm')); M=256; EG={0:(1,3,10),1:(1,3,7)}
def qe(x,E): a=np.abs(x); return np.sign(x)*sum((a>=t).astype(np.int32) for t in E)
VAR=['left+top','max(topleft,top)','median(topleft,top,topright)','max(topleft,top,topright)']
T1=(1,2,3,4,6,8,12,16); T2=(2,3,4,6,8,12,16,24,32,48,64)
J={(v,sp,t):np.zeros((343,130,M),np.float32) for v in VAR for sp in (0,1) for t in (0,1)}
for i,f in enumerate(files):
    A=np.asarray(Image.open(f).convert('RGB')).astype(np.int32); R,G,B=A[...,0],A[...,1],A[...,2]
    for P,t,nb in (((R+2*G+B)>>2,0,8),(B-G,1,9),(R-G,1,9)):
        H,W=P.shape; Z=np.zeros((H+1,W+2),np.int32); Z[1:,1:-1]=P
        a=Z[1:,:-2].copy(); b=Z[:-1,1:-1].copy(); c=Z[:-1,:-2].copy(); d=Z[:-1,2:].copy()
        a[:,0]=b[:,0]; c[:,0]=b[:,0]; d[:,-1]=b[:,-1]
        m=1<<nb; e=((P-np.clip(a+b-c,np.minimum(a,b),np.maximum(a,b))+m//2)&(m-1))-m//2
        z=np.where(e>=0,2*e,-2*e-1); zp=np.zeros((H+1,W+2),np.int32); zp[1:,1:-1]=z
        zl=zp[1:,:-2]; zt=zp[:-1,1:-1]; ztl=zp[:-1,:-2]; ztr=zp[:-1,2:]
        ga,gb,gd=c-a,c-b,d-b; sec=(3-qe(ga,EG[t]))*7+qe(gb,EG[t])+3; dc=qe(gd,EG[t])+3; cell=(sec*7+dc).ravel()
        ae=np.minimum(np.abs(e),M-1).ravel()
        vs={'left+top':zl+zt,'max(topleft,top)':np.maximum(ztl,zt),'median(topleft,top,topright)':np.maximum(np.minimum(ztl,zt),np.minimum(np.maximum(ztl,zt),ztr)),'max(topleft,top,topright)':np.maximum(np.maximum(ztl,zt),ztr)}
        for vn,v in vs.items():
            vv=np.minimum(v,129).ravel(); J[vn,i&1,t]+=np.bincount((cell*130+vv)*M+ae,minlength=343*130*M).reshape(343,130,M)
    print(i+1,flush=True)
def ev(tr,te,sizes=(16,8)):
    idx=np.where(tr.sum(1)>0)[0]; r=agglo(tr[idx].astype(np.float64),list(sizes)); out={}
    for g in sizes:
        lab=np.zeros(len(tr),np.int64)
        for k,gr in enumerate(r[g][1]): lab[idx[gr]]=k
        h=np.zeros((g,M)); np.add.at(h,lab,te.astype(np.float64)); out[g]=Hrows(h).sum()
    return out
for t,nm in ((0,'Y'),(1,'U/V')):
    tr2=J['left+top',0,t].sum(1).reshape(49,7,M).sum(1); te2=J['left+top',1,t].sum(1).reshape(49,7,M).sum(1); N=te2.sum(); b2=ev(tr2,te2)
    print(f'== {nm}: 2-D 8 groups {b2[8]/N:.4f}')
    for vn in VAR:
        best=None
        for t1 in T1:
            for t2 in T2:
                if t2<=t1: continue
                lv=np.minimum((np.arange(130)>=t1).astype(int)+(np.arange(130)>=t2),2)
                tr=np.stack([J[vn,0,t][:,lv==l].sum(1) for l in range(3)],1).reshape(-1,M); te=np.stack([J[vn,1,t][:,lv==l].sum(1) for l in range(3)],1).reshape(-1,M)
                b=ev(tr,te)
                if best is None or b[8]<best[2][8]: best=(t1,t2,b)
        t1,t2,b=best; print(f'   {vn:30s} best v<{t1},<{t2}:  8 groups {100*(b[8]/b2[8]-1):+.2f}%   16 groups {100*(b[16]/b2[16]-1):+.2f}%  (vs 2-D)',flush=True)
