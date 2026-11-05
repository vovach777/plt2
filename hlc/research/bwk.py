import numpy as np, glob, sys
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def bw(x): x = np.abs(x); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0)
KS = [2, 3, 4, 5, 6, 8, 10, 12]                       # coefficient = k/4
files = sorted(glob.glob(sys.argv[1]))
# per file: per plane-type: list of (z, act); entropy computed per (ky, kc) with shared class 0
data = []; px = 0
for f in files:
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    d = {True: [], False: []}
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        ll = l.copy(); ll[:, 2:] = P[:, :-2]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        ga, gb = tl - l, tl - t; sa, sb = np.sign(ga), np.sign(gb); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
        ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
        d[n9].append((z.ravel(), (np.abs(ga) + np.abs(gb) + np.abs(l - ll)).ravel()))
    data.append({k: (np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v])) for k, v in d.items()})
def cls(act, k): return np.minimum(bw((act * k) >> 2), 6)
# class-0 stream shared: compute per (ky,kc); other streams separable per plane type
def total(ky, kc):
    t = 0.0
    for d in data:
        zy, ay = d[False]; zc, ac = d[True]; cy, cc = cls(ay, ky), cls(ac, kc)
        t += H(np.concatenate([zy[cy == 0], zc[cc == 0]]))
        for j in range(1, 7): t += H(zy[cy == j]) + H(zc[cc == j])
    return t
base = total(4, 4)
print('same k for Y and UV:')
for k in KS: print(f'  k={k / 4:4.2f}  {total(k, k) / px:.4f}  {100 * total(k, k) / base - 100:+.2f}%')
best = min(((total(ky, kc), ky, kc) for ky in KS for kc in KS))
print(f'best per plane: kY={best[1] / 4} kUV={best[2] / 4}  {best[0] / px:.4f}  {100 * best[0] / base - 100:+.2f}% vs k=1')
