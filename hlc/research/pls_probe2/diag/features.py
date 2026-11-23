import glob,os,sys
G={}
for f in glob.glob(sys.argv[1]+'/*.txt'):
    g=os.path.basename(f)[:-4]
    for l in open(f):
        t=l.split(); v=list(map(int,t[1:])); n,zy,zu,zv,zs=v[:5]; a=v[5:9]; hy=v[9:12]; hs=v[12:15]
        G.setdefault(g,[]).append(((zy/n,hy[0]/n,hy[1]/n,hy[2]/n),(zs/n,hs[0]/n,hs[1]/n,hs[2]/n),a))
gs=sorted(G)
def loss(R,fy,fs):
    o=sum(min(r[2]) for r in R); return 100*(sum(r[2][(1 if fy(r[0]) else 0)|(2 if fs(r[1]) else 0)] for r in R)/o-1)
def show(name,fy,fs):
    L=[loss(G[g],fy,fs) for g in gs]; print('%-40s'%name+''.join('%8.2f'%x for x in L)+'  mean %.3f'%(sum(L)/len(L))); return sum(L)/len(L)
print('%-40s'%'loss vs pair oracle %'+''.join('%8s'%g for g in gs))
show('zeros only Y45 S50',lambda p:p[0]>=.45,lambda p:p[0]>=.50)
# score = zeros - a*big(>=32) - b*mid(8..31)
best=[]
for a in (0,0.5,1,2,3,5):
  for b in (0,0.25,0.5,1):
    for ty in range(20,80,5):
      for ts in range(20,80,5):
        fy=lambda p,a=a,b=b,t=ty/100:p[0]-a*p[3]-b*p[2]>=t; fs=lambda p,a=a,b=b,t=ts/100:p[0]-a*p[3]-b*p[2]>=t
        best.append((sum(loss(G[g],fy,fs) for g in gs)/len(gs),a,b,ty,ts))
best.sort()
for m,a,b,ty,ts in best[:3]:
    show('zeros-%.1f*big-%.2f*mid  Y%d S%d'%(a,b,ty,ts),lambda p:p[0]-a*p[3]-b*p[2]>=ty/100,lambda p:p[0]-a*p[3]-b*p[2]>=ts/100)
# old-probe-like: zeros vs small
for k in (1,2,3):
    show('zeros >= %d*small (old-probe style)'%k,lambda p,k=k:p[0]>=k*p[1],lambda p,k=k:p[0]>=k*p[1])
print('\nplanes in hybrid band (zeros within +-15 of 45/50): can "big >= q" send them to PLR without costing?')
band=[]
for g in gs:
    for r in G[g]:
        a=r[2]
        for pl,(p,idx,t) in enumerate(((r[0],1,.45),(r[1],2,.50))):
            if abs(p[0]-t)<=.15: band.append((g,p,a[idx]-a[0]))   # delta>0 -> PLR better
tot=sum(abs(d) for _,_,d in band)
print('band planes',len(band))
for q in (0.02,0.05,0.10,0.15,0.20,0.30):
    sel=[b for b in band if b[1][3]>=q]; wrong=[b for b in sel if b[2]<0]
    print('big>=%.2f: %5d planes (%4.1f%% of band) sent to PLR, wrong %d, cost of wrong %.0f bytes = %.2f%% of band stake'%(q,len(sel),100*len(sel)/len(band),len(wrong),-sum(b[2] for b in wrong),100*-sum(b[2] for b in wrong)/tot))
