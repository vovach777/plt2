import numpy as np, sys, glob, os
from PIL import Image
exec(open('ctxest.py').read().split('def run')[0])
files = sorted(glob.glob(sys.argv[1]))
print('photo   size      ' + ' '.join(f'{n:>5s}' for n in ['0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'UV1', 'UV2', 'UV3', 'UV4', 'UV5']) + '  | bits: 0   Y5  | bpp')
for f in files:
    cnt = np.zeros(11); bits = np.zeros(11); H = [dict() for _ in range(11)]
    for ci, (P, n9) in enumerate(planes(f)):
        e, a, b, c, d, aa = resid(P, n9); z = np.where(e >= 0, 2 * e, -2 * e - 1)
        k = cls(np.abs(a - c) + np.abs(c - b) + np.abs(a - aa), THC if n9 else THY)
        for j in range(6):
            sid = 0 if j == 0 else (j if ci == 2 else 5 + j); zz = z[k == j]; cnt[sid] += zz.size
            v, cc = np.unique(zz, return_counts=True)
            for x, y in zip(v, cc): H[sid][x] = H[sid].get(x, 0) + y
    for s in range(11):
        if H[s]: c = np.array(list(H[s].values()), float); bits[s] = -(c * np.log2(c / c.sum())).sum()
    W, Hh = Image.open(f).size
    print(f'{os.path.basename(f)[:5]:5s} {W:4d}x{Hh:<4d} ' + ' '.join(f'{100*x/cnt.sum():5.1f}' for x in cnt) + f'  | {100*bits[0]/bits.sum():4.1f} {100*bits[5]/bits.sum():4.1f} | {bits.sum()/(W*Hh):.2f}')
