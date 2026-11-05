import numpy as np, glob
from PIL import Image
def bw(x): x = np.abs(x); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0)
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
files = sorted(glob.glob('photos16h/*.ppm')) + ['cat_h.ppm']
# radius bw(L1) 0..6 -> quantised level (0 stays 0)
Q = {'6 levels (1,2,3,4,5,6+)': [0, 1, 2, 3, 4, 5, 6], '5: 1,2,3,4,5+': [0, 1, 2, 3, 4, 5, 5], '4: 1,2,3-4,5+': [0, 1, 2, 3, 3, 4, 4],
     '4: 1-2,3,4,5+': [0, 1, 1, 2, 3, 4, 4], '3: 1-2,3-4,5+': [0, 1, 1, 2, 2, 3, 3], '2: 1-3,4+': [0, 1, 1, 1, 2, 2, 2]}
tot = {k: 0.0 for k in Q}; tot0 = 0.0; px = 0; nst = {k: set() for k in Q}
for f in files:
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    S = {k: dict() for k in Q}; S0 = dict()
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        ga, gb = tl - l, tl - t; sa, sb = np.sign(ga), np.sign(gb); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
        x = s * gb; y = s * ga
        ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
        r = np.minimum(bw(np.abs(x) + y), 6); sec = np.where(x == 0, 3, np.where(x > y, 0, np.where(-x > y, 1, 2)))
        k0 = np.where(r == 0, 0, r + (1000 if n9 else 0))
        for v in np.unique(k0): S0.setdefault(int(v), []).append(z[k0 == v])
        for nm, q in Q.items():
            rq = np.array(q)[r]; k = np.where(rq == 0, 0, rq * 4 + sec + (1000 if n9 else 0))
            for v in np.unique(k): S[nm].setdefault(int(v), []).append(z[k == v]); nst[nm].add(int(v))
    tot0 += sum(H(np.concatenate(v)) for v in S0.values())
    for nm in Q: tot[nm] += sum(H(np.concatenate(v)) for v in S[nm].values())
print(f'{"L1 radius only, 6 levels":36s} {tot0 / px:.4f}')
for nm in Q: print(f'triangles+ridge, radius {nm:24s} {tot[nm] / px:.4f}  {100 * tot[nm] / tot0 - 100:+.2f}% vs L1   streams {len(nst[nm])}')
