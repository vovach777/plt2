import glob,os,sys
D=os.path.dirname(os.path.abspath(__file__))
rows=[]
for f in sorted(glob.glob(D+'/*.txt')):
    s=os.path.basename(f)[:-4]
    for l in open(f):
        t=l.split(); n,zy,zu,zv,zs=map(int,t[2:7]); a=list(map(int,t[7:11])); rows.append((s,zy/n,(zy+zu+zv)/(3*n),zs/n,a))
SCR=('full12','iphone','web8')
for grp,name in [(SCR,'screens'),(('valid41','kodak'),'photo')]:
    R=[r for r in rows if r[0] in grp]
    ora=sum(min(r[4]) for r in R)
    tile_ora=sum(min(r[4][0],r[4][3]) for r in R)
    dis=sum(1 for r in R if min(range(4),key=lambda k:r[4][k]) in (1,2))
    dis_gain=sum(min(r[4][0],r[4][3])-min(r[4]) for r in R)
    print(f'{name}: tiles {len(R)}, oracle split Y!=S in {dis} ({100*dis/len(R):.1f}%), per-plane oracle vs tile oracle {100*dis_gain/ora:.2f}% bytes')
    # how good is the best zero threshold, per plane, vs per-plane oracle
    best=min(((sum(r[4][(r[1]>=ty/100)|((r[3]>=ts/100)<<1)] for r in R)/ora-1)*100,ty,ts) for ty in range(30,100,5) for ts in range(30,100,5))
    tbest=min(((sum(r[4][3 if r[2]>=t/100 else 0] for r in R)/ora-1)*100,t) for t in range(30,100,5))
    print(f'  best per-plane zero thresholds Y>={best[1]} S>={best[2]}: +{best[0]:.2f}% vs oracle;  best tile threshold {tbest[1]}: +{tbest[0]:.2f}%;  tile oracle itself +{100*(tile_ora/ora-1):.2f}%')
    # error rate of per-plane threshold decisions (planes, weighted by stake)
    for pl,idx,fe in [('Y',1,1),('S',2,3)]:
        wrong=0;stake=0;cnt=0
        for r in R:
            d=(r[4][idx]-r[4][0]) if True else 0  # PLS-PLR delta for this plane (others PLR)
            want=d<0; got=r[fe]>=(best[1] if pl=='Y' else best[2])/100
            if want!=got: wrong+=1; stake+=abs(d)
        print(f'  plane {pl}: wrong decisions {wrong}/{len(R)} ({100*wrong/len(R):.1f}%), their cost {100*stake/ora:.2f}% bytes')
