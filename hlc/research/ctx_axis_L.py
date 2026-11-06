# Does the up-right neighbour d pay off in HLC5 contexts? MED residual of RCT planes, sign folded like HLC5 (s from c-a, c-b),
# contexts q7(c-a) x q7(c-b) [current] vs + q7(d-b) third axis; contexts greedily merged (pooled histograms) into G groups
# per plane type; cost = sum over images of static entropy per (image, group) + ~50 B per used table.
import numpy as np, glob, sys, os
AX=os.environ.get('AX','d')
from PIL import Image
S='/tmp/claude-0/-home-claude/739fa5c5-1bdc-534c-8b96-8baaa717ae9c/scratchpad'
files=sorted(glob.glob(S+'/photos16h/*.ppm'))+[S+'/cat_h.ppm']
E={0:(1,3,10),1:(1,3,7)}
def q7(x,ed):
    a=np.minimum(np.abs(x),15); return np.sign(x)*((a>=ed[0]).astype(int)+(a>=ed[1])+(a>=ed[2]))
def plane_ctx(P,nb,pt):
    H,W=P.shape; Z=np.zeros((H+1,W+2),np.int32); Z[1:,1:-1]=P
    a=Z[1:,:-2].copy(); b=Z[:-1,1:-1].copy(); c=Z[:-1,:-2].copy(); d=Z[:-1,2:].copy()
    a[:,0]=b[:,0]; c[:,0]=b[:,0]; d[:,-1]=b[:,-1]                       # x=0: a=c=b ; right edge: d=b ; y=0 row: b=c=d=0
    p=np.clip(a+b-c,np.minimum(a,b),np.maximum(a,b)); m=1<<nb; e=((P-p+m//2)&(m-1))-m//2
    s=np.where(c-a!=0,np.sign(c-a),np.where(c-b!=0,np.sign(c-b),1)); e=s*e; z=np.where(e>=0,2*e,-2*e-1)
    ed=E[pt]; A=q7(c-a,ed)+3; B=q7(c-b,ed)+3
    if AX=='d': Dd=q7(d-b,ed)+3
    else:
        e0=((P-p+m//2)&(m-1))-m//2; E1=np.zeros_like(e0); E2=np.zeros_like(e0); E1[:,1:]=e0[:,:-1]; E2[1:,:]=e0[:-1,:]
        v=np.abs(E1)+np.abs(E2); Dd=np.minimum(np.where(v>0,np.floor(np.log2(np.maximum(v,1))).astype(int)+1,0),8)
    return z.ravel(), ((3-(A-3))*7+B).ravel(), Dd.ravel()                 # sector 0..48 as in HLC5 numbering, third axis 0..6
data={0:[],1:[]}
for f in files:
    A=np.asarray(Image.open(f).convert('RGB')).astype(np.int32); R,G,B=A[...,0],A[...,1],A[...,2]
    Y=(R+2*G+B)>>2; U=B-G; V=R-G
    data[0].append(plane_ctx(Y,8,0)); data[1].append(plane_ctx(U,9,1)); data[1].append(plane_ctx(V,9,1))
def Hc(c): c=c[c>0]; return float(-(c*np.log2(c/c.sum())).sum())
def xlx(x): return np.where(x>0,x*np.log2(np.maximum(x,1)),0.)
def Fh(h): n=h.sum(-1); return xlx(n)-xlx(h).sum(-1)     # total bits of a histogram (static entropy)
def greedy(Hs,G):          # merge pairs with min entropy increase down to G groups; vectorised row updates
    idx=[i for i in range(len(Hs)) if Hs[i].sum()>0]; h=Hs[idx].copy(); n=len(idx); f=Fh(h)
    M=np.full((n,n),np.inf)
    for i in range(n): M[i,i+1:]=Fh(h[i]+h[i+1:])-f[i]-f[i+1:]
    alive=np.ones(n,bool); members=[[idx[i]] for i in range(n)]; cnt=n
    while cnt>G:
        k=int(np.argmin(M)); i,j=divmod(k,n)
        h[i]+=h[j]; f[i]=Fh(h[i]); members[i]+=members[j]; alive[j]=False; M[j,:]=np.inf; M[:,j]=np.inf; cnt-=1
        row=Fh(h[i]+h)-f[i]-f; row[~alive]=np.inf; row[i]=np.inf
        M[i,:]=np.where(np.arange(n)>i,row,np.inf); M[:i,i]=np.where(alive[:i],row[:i],np.inf)
    lab=np.full(len(Hs),-1)
    for g,i in enumerate(np.where(alive)[0]): lab[members[i]]=g
    return lab
def run(use_d,G):
    tot=0
    for pt,A_ in ((0,257),(1,513)):
        nctx=49*9 if use_d else 49
        ctx=[ (s*9+dd if use_d else s) for z,s,dd in data[pt]]
        pooled=np.zeros((nctx,A_))
        for (z,s,dd),c in zip(data[pt],ctx): np.add.at(pooled,(c,z),1)
        # pre-merge: contexts with < 200 samples go to their 2-axis sector's most populous sibling (keeps greedy tractable)
        lab=greedy(pooled,G)
        per=len(data[pt])//len(files)
        for i in range(len(files)):
            for (z,s,dd),c in list(zip(data[pt],ctx))[i*per:(i+1)*per]:
                g=lab[c]; 
                for k in range(G):
                    m=g==k
                    if m.any(): tot+=Hc(np.bincount(z[m],minlength=A_).astype(float))+400
    return tot
npx=sum(np.prod(Image.open(f).size) for f in files)
base=None
for use_d,G in [(0,8)]+[(1,8),(1,16)]:
    t=run(use_d,G); base=base or t
    print(f'{("+ axis "+AX) if use_d else "2 axes   "}  {G:2d} groups: {t/npx:.4f} bpp  {100*(t/base-1):+.2f}%',flush=True)
