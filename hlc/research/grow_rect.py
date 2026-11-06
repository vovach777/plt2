# rectangle region growing on per-tile H_G: seed = first free tile in raster order; each step expands the rectangle
# by one full strip (R/D/L/U, free tiles only) whose mean is closest to the region mean; stop when all > thr
import numpy as np, sys, colorsys
from PIL import Image, ImageDraw
exec(open(sys.path[0]+'/cluster8_per_image.py').read().split('gth=km1d')[0])
S='/tmp/claude-0/-home-claude/739fa5c5-1bdc-534c-8b96-8baaa717ae9c/scratchpad'
gth=km1d(np.concatenate([d[1] for d in D]),K)
def grow(H,thr,rel,cap):
    ty,tx=H.shape; rid=-np.ones((ty,tx),int); out=[]
    for sy in range(ty):
        for sx in range(tx):
            if rid[sy,sx]>=0: continue
            x0,x1,y0,y1=sx,sx+1,sy,sy+1; s=H[sy,sx]; n=1; r=len(out); rid[sy,sx]=r
            while True:
                m=s/n; t=thr*m if rel else thr; best=None
                for d,(a,b,c,e) in enumerate(((x1,x1+1,y0,y1),(x0,x1,y1,y1+1),(x0-1,x0,y0,y1),(x0,x1,y0-1,y0))):
                    if a<0 or c<0 or b>tx or e>ty: continue
                    blk=rid[c:e,a:b]
                    if (blk>=0).any(): continue
                    if cap and n+blk.size>cap: continue
                    dm=abs(H[c:e,a:b]-m).max() if MAXD else abs(H[c:e,a:b].mean()-m)
                    if (dm<=t or n==1) and (best is None or dm<best[0]): best=(dm,d,a,b,c,e)
                if best is None: break
                _,d,a,b,c,e=best; rid[c:e,a:b]=r; s+=H[c:e,a:b].sum(); n+=(b-a)*(e-c)
                x0,x1,y0,y1=min(x0,a),max(x1,b),min(y0,c),max(y1,e)
            out.append((x0,y0,x1,y1,s/n))
    return rid,out
COLS=[tuple(int(255*v) for v in colorsys.hsv_to_rgb(0.66*(1-k/(K-1)),0.85,0.95)) for k in range(K)]
MAXD=1; cfgs=[(t,0,0) for t in (0.3,0.5,0.8,1.0)]+[(0.5,0,14)]
print(f'{"threshold":22s} rects  tiles/rect  bits vs 1 table   (ref: tiles -5.88%, opt rects 2480 -5.88%, right/up dh0.3 1940 -5.85%)')
for thr,rel,cap in cfgs:
    nreg=nt=b=b1=0; per={}
    for name,F,T,npx in D:
        img=f'{S}/photos16h/{name}.ppm' if name!='cat_h' else f'{S}/cat_h.ppm'
        Hh,Ww=Image.open(img).size[::-1]; ty=(Hh+47)//48; tx=(Ww+47)//48; H=F.reshape(ty,tx)
        rid,R=grow(H,thr,rel,cap); mean=np.array([r[4] for r in R]); lab=np.searchsorted(gth,mean)[rid].ravel()
        nreg+=len(R); nt+=len(F); b+=bits_of(lab,T); b1+=cost_hist(T.sum(0))+OV; per[name]=len(R)
        tag=f'{"rel" if rel else "abs"}{thr}{"_c" if cap else ""}'
        if name in ('p15','p05') and (thr,rel,cap) in ((0.1,1,0),(0.5,0,0)):
            A=np.asarray(Image.open(img).convert('RGB')).astype(np.float32); c=lab.reshape(ty,tx)
            o=A*0.5+127.5; cm=np.repeat(np.repeat(c,48,0),48,1)[:Hh,:Ww]; out=A.copy()
            for k in range(K): mm=cm==k; out[mm]=o[mm]*0.5+np.array(COLS[k],np.float32)*0.5
            im=Image.fromarray(out.clip(0,255).astype(np.uint8)).resize((Ww*2,Hh*2),Image.NEAREST); d=ImageDraw.Draw(im)
            for (x0,y0,x1,y1,m) in R:
                d.rectangle([x0*96+1,y0*96+1,min(x1*96,2*Ww)-2,min(y1*96,2*Hh)-2],outline=(255,255,255),width=3)
                d.text((x0*96+8,y0*96+6),f'{m:.1f}',fill=(0,0,0)); d.text((x0*96+7,y0*96+5),f'{m:.1f}',fill=(255,255,255))
            d.rectangle([0,0,420,26],fill=(255,255,255)); d.text((8,8),f'{name}: {len(R)} rects, thr {thr}{"*H" if rel else ""}',fill=(0,0,0))
            im.save(f'{S}/vis/c8/{name}_growf_{tag}.png')
    print(f'{(str(thr)+("*H" if rel else " abs")+(" cap14" if cap else "")):22s} {nreg:5d}  {nt/nreg:6.1f}     {100*(b/b1-1):+.2f}%    p05 {per["p05"]} p15 {per["p15"]} p12 {per["p12"]}')
