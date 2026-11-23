#!/usr/bin/env python3
"""Extended oracle analysis on the held-out parts (per group threshold grid).
Reads the DIAG logs produced by collect_diag.py.
"""
import sys, glob, os, collections

D = sys.argv[1] if len(sys.argv) > 1 else '/home/devin/train/bench-independent/p2/diag'
rows = collections.defaultdict(list)
for f in sorted(glob.glob(D + '/holdout_*.txt')):
    s = os.path.basename(f)[len('holdout_'):-4]
    for l in open(f):
        t = l.split()
        n, zy, zu, zv, zs = map(int, t[2:7]); a = list(map(int, t[7:11]))
        rows[s].append((n, zy / n, (zy + zu + zv) / (3 * n), zs / n, a))

def cost(r, y, sv): return r[4][(1 if y else 0) | (2 if sv else 0)]
def oracle(r): return min(((y, s) for y in (0, 1) for s in (0, 1)), key=lambda p: cost(r, *p))
def base(R): return sum(cost(r, *oracle(r)) for r in R)
def loss(R, f): return 100 * (sum(cost(r, *f(r)) for r in R) / base(R) - 1)

groups = sorted(rows)
print('tiles per group:', {g: len(rows[g]) for g in groups})
print('\n== loss vs oracle, per group (%), named rules ==')
rules = {
    'tile65(probe1)': lambda r: (r[2] >= .65, r[2] >= .65),
    'plane65': lambda r: (r[1] >= .65, r[3] >= .65),
    'plane50': lambda r: (r[1] >= .50, r[3] >= .50),
    'plane55': lambda r: (r[1] >= .55, r[3] >= .55),
    'plane45': lambda r: (r[1] >= .45, r[3] >= .45),
}
print('%-16s' % 'rule' + ''.join('%12s' % g for g in groups) + '%10s' % 'sum')
allr = [r for g in groups for r in rows[g]]
b = base(allr)
for k, f in rules.items():
    print('%-16s' % k + ''.join('%+11.2f%%' % loss(rows[g], f) for g in groups) +
          '%9.2f%%' % (100 * (sum(cost(r, *f(r)) for r in allr) / b - 1)))

print('\n== best (tY,tS) per group, grid 40..90 step 5 ==')
best_by_group = {}
for g in groups:
    R = rows[g]
    cand = []
    for ty in range(40, 95, 5):
        for ts in range(40, 95, 5):
            f = lambda r, ty=ty, ts=ts: (r[1] >= ty / 100, r[3] >= ts / 100)
            cand.append((loss(R, f), ty, ts))
    cand.sort()
    best_by_group[g] = cand[:3]
    print('%-12s ' % g + '  '.join('Y>=%d S>=%d %+.2f%%' % (b[1], b[2], b[0]) for b in cand[:3]))

print('\n== scan-friendly: sum of losses over screen groups (all but photo), grid ==')
scr = [r for g in groups if g != 'photo' for r in rows[g]]
pho = rows.get('photo', [])
cand = []
for ty in range(40, 95, 5):
    for ts in range(40, 95, 5):
        f = lambda r, ty=ty, ts=ts: (r[1] >= ty / 100, r[3] >= ts / 100)
        cand.append((loss(scr, f), ty, ts))
cand.sort()
for c in cand[:6]:
    print('  Y>=%d S>=%d  screens %+.3f%%' % (c[1], c[2], c[0]))

print('\n== fork rule: S->PLS only if zS>=tS and (Y PLS or zS>=tS_hi) ==')
for tY, tS, tHi in [(50, 50, 75), (50, 50, 80), (50, 55, 75), (55, 50, 75), (50, 45, 70), (45, 50, 70), (50, 50, 70)]:
    def f(r, tY=tY, tS=tS, tHi=tHi):
        y = r[1] >= tY / 100
        s = r[3] >= tS / 100 and (y or r[3] >= tHi / 100)
        return (y, s)
    tot = 100 * (sum(cost(r, *f(r)) for r in allr) / b - 1)
    per = ' '.join('%s %+.2f' % (g[:6], loss(rows[g], f)) for g in groups)
    print('  Y>=%d S>=%d hi %d   sum %+.3f%%   %s' % (tY, tS, tHi, tot, per))