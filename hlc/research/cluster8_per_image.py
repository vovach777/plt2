import numpy as np, glob, time, pickle, os
from PIL import Image
S='/tmp/claude-0/-home-claude/739fa5c5-1bdc-534c-8b96-8baaa717ae9c/scratchpad'
def cnt(z,A=256): return np.bincount(z.ravel(),minlength=A).astype(np.float64)
def Hc(c): c=c[c>0]; return float(-(c*np.log2(c/c.sum())).sum())
def mede(p,n9=0):
    Z=np.zeros((p.shape[0]+1,p.shape[1]+1),np.int32); Z[1:,1:]=p
    a=Z[1:,:-1].copy(); b=Z[:-1,1:].copy(); c=Z[:-1,:-1].copy(); a[:,0]=b[:,0]; c[:,0]=b[:,0]
    e=p-np.clip(a+b-c,np.minimum(a,b),np.maximum(a,b)); return ((e+256)&511)-256 if n9 else ((e+128)&255)-128
def zz(e): return np.where(e>=0,2*e,-2*e-1)
pk=S+'/clu/data.pkl'
if os.path.exists(pk): D=pickle.load(open(pk,'rb'))
else:
    D=[]
    for f in sorted(glob.glob(S+'/photos16h/*.ppm'))+[S+'/cat_h.ppm']:
        A=np.asarray(Image.open(f).convert('RGB')).astype(np.int32); R,G,B=A[...,0],A[...,1],A[...,2]; Hh,Ww=G.shape
        zg=zz(mede(G)); Zt=[zz(mede((R+2*G+B)>>2)),zz(mede(R-G,1)),zz(mede(B-G,1))]
        tx=(Ww+47)//48; ty=(Hh+47)//48; F=[]; T=[]
        for j in range(ty):
            for i in range(tx):
                x0=i*48 if i*48+48<=Ww else Ww-48; y0=j*48 if j*48+48<=Hh else Hh-48
                F.append(Hc(cnt(zg[y0:y0+48,x0:x0+48]))/2304)
                s=np.s_[j*48:j*48+48,i*48:i*48+48]; T.append(np.concatenate([cnt(Zt[0][s]),cnt(Zt[1][s],512),cnt(Zt[2][s],512)]))
        D.append((f.split('/')[-1][:-4],np.array(F),np.array(T),Hh*Ww))
    pickle.dump(D,open(pk,'wb'))
OV=400*3  # ~50 B/table *3 planes, bits
def cost_hist(h): return Hc(h[:256])+Hc(h[256:768])+Hc(h[768:])
def bits_of(lab,T):
    return sum(cost_hist(T[lab==k].sum(0))+OV for k in np.unique(lab))
K=8
X=np.concatenate([d[1] for d in D])
def km1d(x,K,it=100):
    c=np.quantile(x,(np.arange(K)+.5)/K)
    for _ in range(it):
        th=(c[1:]+c[:-1])/2; l=np.searchsorted(th,x); n=np.array([x[l==k].mean() if (l==k).any() else c[k] for k in range(K)])
        if np.allclose(n,c): break
        c=n
    return (c[1:]+c[:-1])/2
def km_hist(x,K,B=256,it=50):   # Lloyd on 256-bin histogram of H_G in [0,8)
    h=np.bincount(np.clip((x*B/8).astype(int),0,B-1),minlength=B).astype(float); v=(np.arange(B)+.5)*8/B
    cs=np.cumsum(h); c=np.array([v[np.searchsorted(cs,cs[-1]*(k+.5)/K)] for k in range(K)])
    for _ in range(it):
        th=(c[1:]+c[:-1])/2; l=np.searchsorted(th,v); n=np.array([ (h[l==k]*v[l==k]).sum()/h[l==k].sum() if h[l==k].sum()>0 else c[k] for k in range(K)])
        if np.allclose(n,c): break
        c=n
    return (c[1:]+c[:-1])/2
def dp_bits(F,T,K,NB=64):
    o=np.argsort(F,kind='stable'); n=len(F); NB=min(NB,n); edges=np.linspace(0,n,NB+1).astype(int)
    P=np.zeros((NB+1,T.shape[1])); 
    for b in range(NB): P[b+1]=P[b]+T[o[edges[b]:edges[b+1]]].sum(0)
    C=np.full((NB+1,NB+1),np.inf)
    for i in range(NB):
        for j in range(i+1,NB+1): C[i,j]=cost_hist(P[j]-P[i])+OV
    best=np.full((K+1,NB+1),np.inf); arg=np.zeros((K+1,NB+1),int); best[0,0]=0
    for k in range(1,K+1):
        for j in range(1,NB+1):
            v=best[k-1,:j]+C[:j,j]; i=int(np.argmin(v)); best[k,j]=v[i]; arg[k,j]=i
    kb=int(np.argmin(best[1:,NB]))+1; lab=np.empty(n,int); j=NB
    for k in range(kb,0,-1): i=arg[k,j]; lab[o[edges[i]:edges[j]]]=k-1; j=i
    return lab,kb
gth=km1d(X,K)
res={m:[0,0] for m in ['1 table','global thr','quantile','kmeans','kmeans_hist','DP bits']}; tot=0; per=[]
for name,F,T,npx in D:
    tot+=npx; r={}
    r['1 table']=cost_hist(T.sum(0))+OV
    r['global thr']=bits_of(np.searchsorted(gth,F),T)
    r['quantile']=bits_of(np.searchsorted(np.quantile(F,np.arange(1,K)/K),F),T)
    r['kmeans']=bits_of(np.searchsorted(km1d(F,K),F),T)
    r['kmeans_hist']=bits_of(np.searchsorted(km_hist(F,K),F),T)
    lab,kb=dp_bits(F,T,K); r['DP bits']=bits_of(lab,T)
    for m in r: res[m][0]+=r[m]
    per.append((name,len(F),npx,r,kb))
b0=res['1 table'][0]
print('method        bpp      vs 1 table')
for m in res: print(f'{m:12s} {res[m][0]/tot:7.4f}  {100*(res[m][0]/b0-1):+6.2f}%')
print('\nper image vs 1 table:  tiles  global  quant  kmeans kmhist  DP(K used)')
for name,nt,npx,r,kb in per:
    b=r['1 table']; print(f'{name:6s} {nt:5d} '+' '.join(f'{100*(r[m]/b-1):+6.2f}' for m in ['global thr','quantile','kmeans','kmeans_hist','DP bits'])+f' ({kb})')
