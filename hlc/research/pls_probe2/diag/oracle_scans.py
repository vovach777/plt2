import os,sys
D=os.path.dirname(os.path.abspath(__file__))
R=[]
for l in open(D+'/scans.txt'):
    t=l.split(); n,zy,zu,zv,zs=map(int,t[2:7]); a=list(map(int,t[7:11])); R.append((t[0],zy/n,(zy+zu+zv)/(3*n),zs/n,a))
ora=sum(min(r[4]) for r in R)
def L(f): return 100*(sum(r[4][f(r)] for r in R)/ora-1)
k=lambda y,s:(1 if y else 0)|(2 if s else 0)
print('scans: tiles',len(R),'images',len(set(r[0] for r in R)))
cnt=[0]*4
for r in R: cnt[min(range(4),key=lambda i:r[4][i])]+=1
print('oracle choice (Y,S): PLR/PLR %d  PLS/PLR %d  PLR/PLS %d  PLS/PLS %d'%tuple(cnt))
print('tile oracle (same for both) +%.2f%%'%(100*(sum(min(r[4][0],r[4][3]) for r in R)/ora-1)))
print('probe1 tile65 +%.2f%%  plane65 +%.2f%%  plane50 +%.2f%%  allPLR +%.2f%%  allPLS +%.2f%%'%(L(lambda r:3 if r[2]>=.65 else 0),L(lambda r:k(r[1]>=.65,r[3]>=.65)),L(lambda r:k(r[1]>=.5,r[3]>=.5)),L(lambda r:0),L(lambda r:3)))
b=min((L(lambda r,ty=ty,ts=ts:k(r[1]>=ty/100,r[3]>=ts/100)),ty,ts) for ty in range(20,100,5) for ts in range(20,100,5))
bt=min((L(lambda r,t=t:3 if r[2]>=t/100 else 0),t) for t in range(20,100,5))
print('best per-plane Y>=%d S>=%d +%.2f%%;  best tile thr %d +%.2f%%'%(b[1],b[2],b[0],bt[1],bt[0]))
