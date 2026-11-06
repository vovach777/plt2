import os; os.environ['AX']='d'
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ctx_axis_d.py')).read().split('npx=sum')[0]; exec(src)        # data with d axis: (z, sector, d)
import numpy as np
# L for the same planes (recompute residual neighbours)
def Lax(P,nb):
    H,W=P.shape; Z=np.zeros((H+1,W+2),np.int32); Z[1:,1:-1]=P
    a=Z[1:,:-2].copy(); b=Z[:-1,1:-1].copy(); c=Z[:-1,:-2].copy(); a[:,0]=b[:,0]; c[:,0]=b[:,0]
    p=np.clip(a+b-c,np.minimum(a,b),np.maximum(a,b)); m=1<<nb; e0=((P-p+m//2)&(m-1))-m//2
    E1=np.zeros_like(e0); E2=np.zeros_like(e0); E1[:,1:]=e0[:,:-1]; E2[1:,:]=e0[:-1,:]; v=np.abs(E1)+np.abs(E2)
    return np.minimum(np.where(v>0,np.floor(np.log2(np.maximum(v,1))).astype(int)+1,0),8).ravel()
LL={0:[],1:[]}
for f in files:
    A=np.asarray(Image.open(f).convert('RGB')).astype(np.int32); R,G,B=A[...,0],A[...,1],A[...,2]
    LL[0].append(Lax((R+2*G+B)>>2,8)); LL[1].append(Lax(B-G,9)); LL[1].append(Lax(R-G,9))
npx=sum(np.prod(Image.open(f).size) for f in files); res={}
for G in (8,16):
    tot=0
    for pt,A_ in ((0,257),(1,513)):
        c1=[s*7+dd for z,s,dd in data[pt]]; pooled=np.zeros((343,A_))
        for (z,s,dd),c in zip(data[pt],c1): np.add.at(pooled,(c,z),1)
        l1=greedy(pooled,16)                                              # step 1: (sector, d) -> 16
        c2=[np.maximum(l1[c],0)*9+L for c,L in zip(c1,LL[pt])]; pooled=np.zeros((144,A_))
        for (z,s,dd),c in zip(data[pt],c2): np.add.at(pooled,(c,z),1)
        l2=greedy(pooled,G)                                               # step 2: x L -> G
        per=len(data[pt])//len(files)
        for i in range(len(files)):
            for (z,s,dd),c in list(zip(data[pt],c2))[i*per:(i+1)*per]:
                g=l2[c]
                for k in range(G):
                    m=g==k
                    if m.any(): tot+=Hc(np.bincount(z[m],minlength=A_).astype(float))+400
    print(f'+ d + L  {G:2d} groups: {tot/npx:.4f} bpp  {100*(tot/npx/7.8601-1):+.2f}% vs 2 axes / 8 groups',flush=True)
