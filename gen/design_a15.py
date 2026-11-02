# design the 15-letter alphabet: 0, +-7 classes [lo, lo+2^raw), classes contiguous from 1, same for both signs.
# cost on corpus counts (per cluster, folded sign), uniform inside a corpus bucket and inside a class.
import numpy as np, sys, itertools
sys.path.insert(0,'/home/claude/plf2/src/palic2py/corpus_tools')
from invcommon import NSYM, SYM_SIGN, SYM_LO, SYM_HI
SP='/tmp/claude-0/-home-claude/739fa5c5-1bdc-534c-8b96-8baaa717ae9c/scratchpad/clic/'
z=np.load(SP+'jointstats.npz')
def values(split, ci, maxv):
    J=z[f'J{ci}_{split}'].reshape(-1,NSYM).astype(float)
    if ci==1: J=J.reshape(576,17,NSYM).sum(1)
    if ci==2: J=J.reshape(576,289,NSYM).sum(1)
    J=J.reshape(64,9,NSYM).sum(1)                          # per cluster
    V=np.zeros((64,2,maxv+1))                              # [cluster][sign 0:+ 1:-][|e|]
    zero=J[:,0]
    for s in range(1,NSYM):
        lo,hi=SYM_LO[s],min(SYM_HI[s],maxv); 
        if lo>maxv: continue
        sg=0 if SYM_SIGN[s]>0 else 1
        V[:,sg,lo:hi+1]+=J[:,s:s+1]/(SYM_HI[s]-SYM_LO[s]+1)
    return zero,V
def cost(zero,V,widths):
    # classes: [1, 1+w0), [1+w0, ...) ...
    P=np.concatenate([np.zeros((64,2,1)),np.cumsum(V[:,:,1:],axis=2)],axis=2)   # P[...,k] = sum |e| in 1..k
    maxv=V.shape[2]-1; edges=[0]
    for w in widths: edges.append(min(edges[-1]+w,maxv))
    cnt=[zero[:,None]]; bits=0
    for k in range(len(widths)):
        c=P[:,:,edges[k+1]]-P[:,:,edges[k]]               # (64,2)
        cnt.append(c); bits+= c.sum()*np.log2(widths[k])
    C=np.concatenate([cnt[0]]+[c for c in cnt[1:]],axis=1)
    n=C.sum(1,keepdims=True); p=np.where(C>0,C/n,1); H=-(C*np.log2(p)).sum()
    return (H+bits)/n.sum()
def current31(zero,V,g):
    LOs=[[1,2,3,4,5,6,7,8,12,16,24,32,48,64,128],[1,2,3,4,5,6,7,8,10,12,14,16,24,32,64]][g]
    RAWs=[[0,0,0,0,0,0,0,2,2,3,3,4,4,6,0],[0,0,0,0,0,0,0,1,1,1,1,3,3,5,8]][g]
    return cost(zero,V,[2**r for r in RAWs])
pw=[1,2,4,8,16,32,64,128]
for g,(nm,comps,maxv) in enumerate((('Y',[0],128),('UV',[1,2],256))):
    tr=[values('train',c,maxv) for c in comps]; te=[values('test',c,maxv) for c in comps]
    ztr=sum(t[0] for t in tr); Vtr=sum(t[1] for t in tr); zte=sum(t[0] for t in te); Vte=sum(t[1] for t in te)
    best=[]
    for ws in itertools.combinations_with_replacement(pw,7):
        if sum(ws)<maxv: continue
        best.append((cost(ztr,Vtr,ws),ws))
    best.sort(); 
    c31tr=current31(ztr,Vtr,g); c31te=current31(zte,Vte,g)
    print(f'== {nm}: 31 letters train {c31tr:.4f} test {c31te:.4f} b/sym')
    for c,ws in best[:5]:
        cte=cost(zte,Vte,ws); print(f'   widths {ws}  train {c:.4f} ({100*(c/c31tr-1):+.2f}%)  test {cte:.4f} ({100*(cte/c31te-1):+.2f}%)')
