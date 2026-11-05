# Givens rotation of the (c-a, c-b) plane by delta before q7 (edges at half-integers 0.5 3.5 15.5 so delta=0 equals the integer grid).
# Prints covariance eigvector angle, diagonal (swap) asymmetry of shares and entropy change per delta. Needs /tmp/rct/llcomp_q7calib.py (llcomp rc).
# usage: python givens_rot.py <folder>
import sys, glob, os, numpy as np
sys.path.insert(0,'/tmp/rct'); import llcomp_q7calib as C
from PIL import Image
files=sorted(glob.glob(os.path.join(sys.argv[1],'*'))); rng=np.random.default_rng(1); D={0:[],1:[]}
for f in files:
    a0=np.asarray(Image.open(f).convert('RGB')).astype(np.int32); G=a0[...,1]; V=a0[...,0]-G; U=a0[...,2]-G; Y=G+((U+V)>>2)
    for P,t in ((Y,0),(V,1),(U,1)):
        a=np.zeros_like(P); a[:,1:]=P[:,:-1]; b=np.zeros_like(P); b[1:]=P[:-1]; c=np.zeros_like(P); c[1:,1:]=P[:-1,:-1]; a[:,0]=b[:,0]; c[:,0]=b[:,0]
        e=P-np.clip(a+b-c,np.minimum(a,b),np.maximum(a,b)); e=((e+256)&511)-256 if t else ((e+128)&255)-128
        D[t].append(((c-a).ravel(),(c-b).ravel(),e.ravel()))
def qf(x): a=np.abs(x); return (np.sign(x)*((a>=0.5).astype(np.int64)+(a>=3.5)+(a>=15.5))).astype(np.int64)
def H(h): c=h[h>0]; return float(-(c*np.log2(c/c.sum())).sum())
px=sum(x[0].size for x in D[0])
for t,nm in ((0,'Y'),(1,'UV')):
    ga=np.concatenate([x[0] for x in D[t]]).astype(float); gb=np.concatenate([x[1] for x in D[t]]).astype(float)
    Cv=np.cov(ga,gb); th=0.5*np.degrees(np.arctan2(2*Cv[0,1],Cv[0,0]-Cv[1,1]))
    w,v=np.linalg.eigh(Cv)
    print(f'{nm}: cov sa^2={Cv[0,0]:.1f} sb^2={Cv[1,1]:.1f} rho={Cv[0,1]:.1f}  major eigvec angle {th:.2f} deg (45 = on the diagonal)')
    for dg in (0,-8,-6,-4,-3,-2,-1,-0.5,0.5,1,2,3,4,6,8):
        r=np.radians(dg); cs,sn=np.cos(r),np.sin(r); tot=0.0; G=np.zeros(49)
        for ga_,gb_,e in D[t]:
            x=cs*ga_-sn*gb_; y=sn*ga_+cs*gb_; cls=(qf(x)+3)*7+qf(y)+3; G+=np.bincount(cls,minlength=49)
            z=np.where(e>=0,2*e,-2*e-1); h=np.bincount(cls*512+z,minlength=49*512).reshape(49,512); tot+=sum(H(row) for row in h)
        S_=100*G.reshape(7,7)/G.sum(); 
        if dg==0: b0=tot
        print(f'   rotate {dg:+5.1f} deg: diagonal asymmetry {C.asym(S_):.3f} pp   entropy {100*tot/b0-100:+.3f}%')
        if dg==0: pass
