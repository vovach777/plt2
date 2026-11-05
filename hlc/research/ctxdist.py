import numpy as np, sys, glob
from PIL import Image
exec(open('ctxest.py').read().split('def run')[0])
files = sorted(glob.glob(sys.argv[1]))
cnt = np.zeros(11); bits = np.zeros(11); absum = np.zeros(11); maxz = np.zeros(11)
hist = [dict() for _ in range(11)]
for f in files:
    for ci, (P, n9) in enumerate(planes(f)):
        e, a, b, c, d, aa = resid(P, n9); z = np.where(e >= 0, 2 * e, -2 * e - 1)
        k = cls(np.abs(a - c) + np.abs(c - b) + np.abs(a - aa), THC if n9 else THY)
        for j in range(6):
            sid = 0 if j == 0 else (j if ci == 2 else 5 + j); zz = z[k == j]
            cnt[sid] += zz.size; absum[sid] += np.abs(e[k == j]).sum(); maxz[sid] = max(maxz[sid], zz.max() if zz.size else 0)
            v, cc = np.unique(zz, return_counts=True)
            for x, y in zip(v, cc): hist[sid][x] = hist[sid].get(x, 0) + y
for s in range(11):
    c = np.array(list(hist[s].values()), float); p = c / c.sum(); bits[s] = -(c * np.log2(p)).sum()
names = ['0 shared'] + [f'Y{j}' for j in range(1, 6)] + [f'UV{j}' for j in range(1, 6)]
print(f'{"stream":9s} {"symbols":>8s} {"bits":>7s} {"bits/sym":>8s} {"mean|e|":>8s} {"max zz":>6s}')
for s in range(11):
    print(f'{names[s]:9s} {100*cnt[s]/cnt.sum():7.1f}% {100*bits[s]/bits.sum():6.1f}% {bits[s]/cnt[s]:8.2f} {absum[s]/cnt[s]:8.2f} {int(maxz[s]):6d}')
print(f'total {cnt.sum():.0f} symbols, {bits.sum()/cnt.sum():.3f} bits/sym (corpus-wide tables)')
