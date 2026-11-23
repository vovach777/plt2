import glob,os,sys
from collections import defaultdict
G={}
for f in glob.glob(sys.argv[1]+'/*.txt'):
    g=os.path.basename(f)[:-4]
    for l in open(f):
        t=l.split(); v=list(map(int,t[1:16])); d=list(map(int,t[17:21]))
        G.setdefault(g,[]).append((v,d))
gs=sorted(G)
print('%-8s %6s | %s'%('set','tiles','per plane: probe mode share (sure-PLR / unsure / sure-PLS); unsure planes: chose PLS, wrong->PLR-should-be-PLS, wrong->PLS-should-be-PLR, cost of each, % of set bytes'))
for g in gs:
    R=G[g]; ora=sum(min(v[5:9]) for v,d in R); tot_old=0
    st=defaultdict(int); cost=defaultdict(int)
    for v,d in R:
        a=v[5:9]
        fin=(d[1]|(d[3]<<1)); tot_old+=a[fin]
        for pl,(sp,fn,idx) in enumerate(((d[0],d[1],1),(d[2],d[3],2))):
            want = a[idx]<a[0]   # PLS better for this plane (other plane PLR, additive approx)
            st['sp%d'%sp]+=1
            if sp==1:
                st['u_pls']+=fn
                if fn==0 and want: st['u_missPLS']+=1; cost['u_missPLS']+=a[0]-a[idx]
                if fn==1 and not want: st['u_missPLR']+=1; cost['u_missPLR']+=a[idx]-a[0]
            else:
                if fn==0 and want: st['s_missPLS']+=1; cost['s_missPLS']+=a[0]-a[idx]
                if fn==1 and not want: st['s_missPLR']+=1; cost['s_missPLR']+=a[idx]-a[0]
    n=2*len(R)
    print('%-8s %6d | modes %4.1f/%4.1f/%4.1f%% | unsure->PLS %d of %d | unsure wrong: PLR-instead-of-PLS %d (%.2f%%), PLS-instead-of-PLR %d (%.2f%%) | sure wrong: %d (%.2f%%) / %d (%.2f%%) | old -s vs oracle +%.2f%%'%(
        g,len(R),100*st['sp0']/n,100*st['sp1']/n,100*st['sp2']/n,st['u_pls'],st['sp1'],st['u_missPLS'],100*cost['u_missPLS']/ora,st['u_missPLR'],100*cost['u_missPLR']/ora,
        st['s_missPLS'],100*cost['s_missPLS']/ora,st['s_missPLR'],100*cost['s_missPLR']/ora,100*(tot_old/ora-1)))
