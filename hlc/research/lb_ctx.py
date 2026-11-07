# lb_ctx.py - cost of the PLN1 noise stream L (per 16-block, off 2) and its conditional entropy given the left / upper
# block L, on 288 tiles of 8-bit P5 files.  usage: python lb_ctx.py dir
import sys, glob, numpy as np
def pgm(p):
    b=open(p,'rb').read(); t=b.split(maxsplit=4); w,h=int(t[1]),int(t[2]); return np.frombuffer(t[4][:w*h],np.uint8).reshape(h,w).astype(np.int32)
def H(c): c=c[c>0]; p=c/c.sum(); return -(c*np.log2(p)).sum()
def Hc(x,ctx):  # conditional entropy bits
    tot=0
    for v in np.unique(ctx): tot+=H(np.bincount(x[ctx==v],minlength=8).astype(float))
    return tot
b0=b1=b2=b3=0; npx=0
for f in sorted(glob.glob(sys.argv[1]+'/*.pgm')):
    im=pgm(f)
    for ty in range(0,im.shape[0]-287,288):
        for tx in range(0,im.shape[1]-287,288):
            t=im[ty:ty+288,tx:tx+288]; z=np.zeros_like(t)
            a=t[1:,:-1]; b=t[:-1,1:]; c=t[:-1,:-1]; x=t[1:,1:]
            p=np.clip(a+b-c,np.minimum(a,b),np.maximum(a,b)); e=((x-p+128)&255)-128; z[1:,1:]=np.where(e>=0,2*e,-2*e-1)
            m=z.reshape(288,18,16).sum(2)>>4; L=np.clip(np.where(m>0,np.floor(np.log2(np.maximum(m,1)))+1,0)-2,0,7).astype(int)
            left=np.concatenate([np.zeros((288,1),int),L[:,:-1]],1); up=np.concatenate([np.zeros((1,18),int),L[:-1]],0)
            Lf=L.ravel(); b0+=H(np.bincount(Lf,minlength=8).astype(float)); b1+=Hc(Lf,left.ravel()); b2+=Hc(Lf,up.ravel()); b3+=Hc(Lf,(left*8+up).ravel()); npx+=288*288
print('L stream bits/px: H0 %.4f  |left %.4f  |up %.4f  |left,up %.4f' % (b0/npx,b1/npx,b2/npx,b3/npx))
