import glob,os,sys
G={}
for f in glob.glob(sys.argv[1]+'/holdout_*.txt'):
    g=os.path.basename(f)[8:-4]
    for l in open(f):
        t=l.split()
        if len(t)<11: continue
        n,zy,zu,zv,zs=map(int,t[2:7]); a=list(map(int,t[7:11])); G.setdefault(g,[]).append((zy/n,zs/n,a))
def ev(R,dec):
    o=sum(min(r[2]) for r in R); c=0; tot=0
    for r in R:
        y,s,cy,cs=dec(r); c+=cy+cs; tot+=r[2][(1 if y else 0)|(2 if s else 0)]
    return 100*(tot/o-1), 100*c/(2*len(R))
def mk(loy,hiy,los,his):
    def d(r):
        a=r[2]; fy,fs=r[0],r[1]
        # costing = per-plane oracle given other plane at PLR (additive approx)
        cy= loy<fy<hiy; cs= los<fs<his
        y = (a[1]<a[0]) if cy else fy>=hiy
        s = (a[2]<a[0]) if cs else fs>=his
        return y,s,cy,cs
    return d
groups=sorted(G); print('groups',[(g,len(G[g])) for g in groups])
def row(name,d):
    L=[ev(G[g],d) for g in groups]; print('%-28s'%name+''.join(' %s %+.2f%%/%2.0f%%'%(g[:4],l,c) for g,(l,c) in zip(groups,L))+'  | mean loss %.3f costed %.1f%%'%(sum(x[0] for x in L)/len(L),sum(x[1] for x in L)/len(L)))
row('probe2 45/50 (no costing)',mk(.45,.45,.5,.5))
for w in (0.05,0.10,0.15,0.20,0.30):
    row('hybrid Y45±%d S50±%d'%(w*100,w*100),mk(.45-w,.45+w,.5-w,.5+w))
row('all costed',mk(-1,2,-1,2))
