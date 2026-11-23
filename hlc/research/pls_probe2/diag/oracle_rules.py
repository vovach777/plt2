import sys,glob,os
D=os.path.dirname(os.path.abspath(__file__))
rows=[]
for f in sorted(glob.glob(D+'/*.txt')):
    s=os.path.basename(f)[:-4]
    for l in open(f):
        t=l.split(); n,zy,zu,zv,zs=map(int,t[2:7]); a=list(map(int,t[7:11]))
        rows.append((s,n,zy/n,(zu+zv+zy)/(3*n),zs/n,a))
sets=sorted(set(r[0] for r in rows))
def cost(r,y,sv):  # y,sv: True=PLS
    return r[5][(1 if y else 0)|(2 if sv else 0)]
# additivity check
err=[abs(r[5][3]-(r[5][1]+r[5][2]-r[5][0]))/max(1,r[5][3]) for r in rows]; print('additivity max rel err %.4f mean %.5f'%(max(err),sum(err)/len(err)))
rules={
 'oracle_pair': lambda r: min(((y,s) for y in (0,1) for s in (0,1)), key=lambda p: cost(r,*p)),
 'tile65(probe1)': lambda r: (r[3]>=.65,)*2,
 'plane65': lambda r:(r[2]>=.65, r[4]>=.65),
 'plane50': lambda r:(r[2]>=.50, r[4]>=.50),
 'allPLS': lambda r:(1,1), 'allPLR': lambda r:(0,0),
}
print('%-16s'%'rule'+''.join('%11s'%s for s in sets))
base={s:sum(cost(r,*rules['oracle_pair'](r)) for r in rows if r[0]==s) for s in sets}
for k,f in rules.items():
    print('%-16s'%k+''.join('%+10.2f%%'%(100*(sum(cost(r,*f(r)) for r in rows if r[0]==s)/base[s]-1)) for s in sets))
# disagreement analysis
print('\ncategory by oracle per plane (Y,S): count, share of bytes, loss of rules vs oracle (bytes)')
cats={}
for r in rows:
    o=rules['oracle_pair'](r); cats.setdefault((r[0],o),[]).append(r)
for s in sets:
    tot=sum(cost(r,*rules['oracle_pair'](r)) for r in rows if r[0]==s)
    for o in [(0,0),(1,1),(1,0),(0,1)]:
        L=cats.get((s,o),[]);
        if not L: continue
        line='%-8s oracle Y=%s S=%s  n=%4d '%(s,'PLS' if o[0] else 'PLR','PLS' if o[1] else 'PLR',len(L))
        for k in ['tile65(probe1)','plane65','plane50']:
            line+=' %s %+.2f%%'%(k,100*sum(cost(r,*rules[k](r))-cost(r,*o) for r in L)/tot)
        fy=sum(r[2] for r in L)/len(L); fs=sum(r[4] for r in L)/len(L)
        print(line+'   mean zY %.2f zS %.2f'%(fy,fs))

print('\nper-plane threshold grid, sum over full12+iphone+web8 (screens) and valid41+kodak (photo), loss vs oracle')
scr=[r for r in rows if r[0] in ('full12','iphone','web8')]; pho=[r for r in rows if r[0] in ('valid41','kodak')]
def loss(R,f): o=sum(cost(r,*rules['oracle_pair'](r)) for r in R); return 100*(sum(cost(r,*f(r)) for r in R)/o-1)
best=[]
for ty in range(40,100,5):
    for ts in range(40,100,5):
        f=lambda r,ty=ty,ts=ts:(r[2]>=ty/100, r[4]>=ts/100)
        best.append((loss(scr,f)+loss(pho,f),ty,ts,loss(scr,f),loss(pho,f)))
best.sort()
for b in best[:6]: print('Y>=%d S>=%d  screens %+.3f%% photo %+.3f%%'%(b[1],b[2],b[3],b[4]))
for ty,ts in [(65,65),(50,50)]:
    f=lambda r:(r[2]>=ty/100, r[4]>=ts/100); print('Y>=%d S>=%d  screens %+.3f%% photo %+.3f%%'%(ty,ts,loss(scr,f),loss(pho,f)))
f=rules['tile65(probe1)']; print('tile65  screens %+.3f%% photo %+.3f%%'%(loss(scr,f),loss(pho,f)))
# S misrouted: oracle S=PLR but zS high
mis=[r for r in scr if r[4]>=.65 and not rules['oracle_pair'](r)[1]]
print('\nS tiles with zS>=65%% but oracle PLR: %d of %d screen tiles, excess bytes %.0f (%.2f%% of screen bytes)'%(len(mis),len(scr),
   sum(cost(r,rules['oracle_pair'](r)[0],1)-cost(r,*rules['oracle_pair'](r)) for r in mis),
   100*sum(cost(r,rules['oracle_pair'](r)[0],1)-cost(r,*rules['oracle_pair'](r)) for r in mis)/sum(cost(r,*rules['oracle_pair'](r)) for r in scr)))
