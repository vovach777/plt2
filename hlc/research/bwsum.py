import numpy as np, glob, sys
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def bw(x): x = np.abs(x); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0)
TY, TC = [1, 3, 10, 18, 35], [1, 4, 8, 16, 30]
V_ = ['act11 + sign (HLC5 default)', 'bw(c-a)+bw(c-b)  (0..18)', 'bw(c-a)+bw(c-b) cap 8', 'bw(c-a)+bw(c-b)+bw(a-aa) (0..27)', '... cap 10',
      'bw(act) (0..11)', 'bw(act) cap 6 (=thr 1 2 4 8 16 32)']
tot = np.zeros(len(V_)); px = 0; ns = [0] * len(V_)
for f in sorted(glob.glob(sys.argv[1])):
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    S = [dict() for _ in V_]
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        ll = l.copy(); ll[:, 2:] = P[:, :-2]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        ga, gb = tl - l, tl - t; sa, sb = np.sign(ga), np.sign(gb); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
        ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
        act = np.abs(ga) + np.abs(gb) + np.abs(l - ll)
        k0 = sum((act >= th).astype(np.int32) for th in (TC if n9 else TY))
        s2 = bw(ga) + bw(gb); s3 = s2 + bw(l - ll)
        ks = [k0, s2, np.minimum(s2, 8), s3, np.minimum(s3, 10), bw(act), np.minimum(bw(act), 6)]
        for vi, kk in enumerate(ks):
            kk = np.where(kk == 0, 0, kk + (100 if n9 else 0))
            for v in np.unique(kk): S[vi].setdefault(int(v), []).append(z[kk == v])
    for vi in range(len(V_)): tot[vi] += sum(H(np.concatenate(x)) for x in S[vi].values()); ns[vi] = max(ns[vi], len(S[vi]))
for vi, v in enumerate(V_): print(f'{v:36s} {tot[vi] / px:.4f} bpp  {100 * tot[vi] / tot[0] - 100:+.2f}%   streams {ns[vi]}')
