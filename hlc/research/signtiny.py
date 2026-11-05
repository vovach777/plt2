import numpy as np, glob, sys
from PIL import Image
THY, THC = [1, 3, 10, 18, 35], [1, 4, 8, 16, 30]
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def q3(x, dz): return (x > dz).astype(np.int32) - (x < -dz).astype(np.int32)
def first(*qs):
    s = np.zeros_like(qs[0])
    for q in qs: s = np.where(s == 0, q, s)
    return np.where(s == 0, 1, s)
V_ = ['no sign', 'now: sign(a-c)', 'tiny: q3(c-a), q3(c-b), dz=4', 'tiny dz=2', 'tiny dz=1', 'tiny dz=0', 'Standard (needs d): b-d, c-b, a-c']
tot = np.zeros(len(V_)); px = 0
for f in sorted(glob.glob(sys.argv[1])):
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    S = [[[] for _ in range(11)] for _ in V_]
    for ci, (P, n9) in enumerate([(V, True), (U, True), (Y, False)]):
        Hh, W = P.shape; Z = np.zeros((Hh + 1, W + 2), np.int32); Z[1:, 1:-1] = P
        l = Z[1:, :-2].copy(); t = Z[:-1, 1:-1].copy(); tl = Z[:-1, :-2].copy(); tr = Z[:-1, 2:].copy()
        l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]; tr[:, -1] = t[:, -1]
        ll = l.copy(); ll[:, 2:] = P[:, :-2]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        k = sum(((np.abs(l - tl) + np.abs(tl - t) + np.abs(l - ll)) >= th).astype(np.int32) for th in (THC if n9 else THY))
        sid = np.where(k == 0, 0, np.where(ci == 2, k, 5 + k))
        sg = [np.ones_like(P), first(np.sign(l - tl)), first(q3(tl - l, 4), q3(tl - t, 4)), first(q3(tl - l, 2), q3(tl - t, 2)),
              first(q3(tl - l, 1), q3(tl - t, 1)), first(q3(tl - l, 0), q3(tl - t, 0)), first(np.sign(t - tr), np.sign(tl - t), np.sign(l - tl))]
        for vi, s in enumerate(sg):
            ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128
            z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
            for j in range(11): S[vi][j].append(z[sid == j])
    for vi in range(len(V_)): tot[vi] += sum(H(np.concatenate(S[vi][j])) for j in range(11))
for vi, v in enumerate(V_): print(f'{v:36s} {tot[vi] / px:.4f} bpp  {100 * tot[vi] / tot[0] - 100:+.2f}%')
