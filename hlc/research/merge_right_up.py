# greedy region merge of 48x48 tiles by H_G: (1) extend run to the right while |H - run mean| <= dh,
# (2) attach a run to the region of the run above if spans nearly equal (|dx0|,|dx1| <= tol) [and |mean diff| <= dh if ent]
import numpy as np, sys, colorsys
from PIL import Image, ImageDraw
exec(open(sys.path[0]+'/cluster8_per_image.py').read().split('gth=km1d')[0])
S='/tmp/claude-0/-home-claude/739fa5c5-1bdc-534c-8b96-8baaa717ae9c/scratchpad'
gth=km1d(np.concatenate([d[1] for d in D]),K)
def regions(H,dh,tol,ent):
    ty,tx=H.shape; rid=np.zeros((ty,tx),int); R=[]   # R: [sum,count,x0,x1(last row)]
    prev=[]
    for y in range(ty):
        runs=[]; x=0
        while x<tx:
            s=H[y,x]; n=1
            while x+n<tx and abs(H[y,x+n]-s/n)<=dh: s+=H[y,x+n]; n+=1
            runs.append((x,x+n,s,n)); x+=n
        cur=[]
        for (x0,x1,s,n) in runs:
            r=None
            for (px0,px1,pr) in prev:
                if abs(px0-x0)<=tol and abs(px1-x1)<=tol and (not ent or abs(R[pr][0]/R[pr][1]-s/n)<=dh): r=pr; break
            if r is None: r=len(R); R.append([0.,0,x0,x1])
            R[r][0]+=s; R[r][1]+=n; rid[y,x0:x1]=r; cur.append((x0,x1,r))
        prev=cur
    return rid,np.array([a/b for a,b,_,_ in R])
COLS=[tuple(int(255*v) for v in colorsys.hsv_to_rgb(0.66*(1-k/(K-1)),0.85,0.95)) for k in range(K)]
cfgs=[('tiles only (global thr)',None)]+[(f'dh={dh} tol={tol} {"len+ent" if e else "len only"}',(dh,tol,e)) for dh in (0.3,0.5,0.8) for tol in (0,1) for e in (0,1)]
print(f'{"config":34s} regions  tiles/reg  bits vs 1 table')
for name_c,cfg in cfgs:
    nreg=0; nt=0; b=0; b1=0
    for name,F,T,npx in D:
        img=f'{S}/photos16h/{name}.ppm' if name!='cat_h' else f'{S}/cat_h.ppm'
        Hh,Ww=Image.open(img).size[::-1]; ty=(Hh+47)//48; tx=(Ww+47)//48; H=F.reshape(ty,tx)
        if cfg is None: lab=np.searchsorted(gth,F); rid=None
        else:
            rid,mean=regions(H,*cfg); lab=np.searchsorted(gth,mean)[rid].ravel()
        n= (rid.max()+1) if rid is not None else len(F)
        nreg+=n; nt+=len(F); b+=bits_of(lab,T); b1+=cost_hist(T.sum(0))+OV
        if cfg==(0.5,1,1) and name in ('p15','p12','p11','p05'):
            A=np.asarray(Image.open(img).convert('RGB')).astype(np.float32); c=lab.reshape(ty,tx)
            pale=A*0.5+127.5; out=A.copy(); cm=np.repeat(np.repeat(c,48,0),48,1)[:Hh,:Ww]
            for k in range(K): m=cm==k; out[m]=pale[m]*0.5+np.array(COLS[k],np.float32)*0.5
            rm=np.repeat(np.repeat(rid,48,0),48,1)[:Hh,:Ww]; edge=np.zeros((Hh,Ww),bool)
            edge[:,1:]|=rm[:,1:]!=rm[:,:-1]; edge[1:]|=rm[1:]!=rm[:-1]
            from scipy.ndimage import binary_dilation; edge=binary_dilation(edge,iterations=1); out[edge]=255
            im=Image.fromarray(out.clip(0,255).astype(np.uint8)); d=ImageDraw.Draw(im)
            d.rectangle([0,0,300,20],fill=(255,255,255)); d.text((6,4),f'{name}: {n} regions, dh 0.5 tol 1 len+ent',fill=(0,0,0))
            im.save(f'{S}/vis/c8/{name}_merge.png')
    print(f'{name_c:34s} {nreg:6d}  {nt/nreg:7.1f}    {100*(b/b1-1):+.2f}%')
