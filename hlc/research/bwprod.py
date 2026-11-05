import numpy as np, glob, sys
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def bw(x): x = np.abs(x); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0)
TY, TC = [1, 3, 10, 18, 35], [1, 4, 8, 16, 30]
V_ = ['act11 + sign (HLC5 default)', '(bw*bw+4)>>2 max16, Y|UV', '(bw*bw+4)>>2 max16, shared Y+UV', 'mine: ((bw+1)*(bw+1)+4)>>2 max16, Y|UV']
tot = np.zeros(len(V_)); px = 0; cnt = np.zeros((2, 32))
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
        k0 = sum(((np.abs(l - tl) + np.abs(tl - t) + np.abs(l - ll)) >= th).astype(np.int32) for th in (TC if n9 else TY))
        k0 = np.where(k0 == 0, 0, k0 + (100 if n9 else 0))
        ba, bb = bw(ga), bw(gb)
        k1 = np.minimum((ba * bb + 4) >> 2, 16); k3 = np.minimum(((ba + 1) * (bb + 1) + 4) >> 2, 16)
        for i, x in enumerate(np.bincount(k1.ravel(), minlength=32)[:32]): cnt[int(n9), i] += x
        keys = [k0, k1 + (100 if n9 else 0), k1, k3 + (100 if n9 else 0)]
        for vi, kk in enumerate(keys):
            for v in np.unique(kk): S[vi].setdefault(int(v), []).append(z[kk == v])
    for vi in range(len(V_)): tot[vi] += sum(H(np.concatenate(x)) for x in S[vi].values())
for vi, v in enumerate(V_): print(f'{v:40s} {tot[vi] / px:.4f} bpp  {100 * tot[vi] / tot[0] - 100:+.2f}%')
for t, nm in ((0, 'Y'), (1, 'UV')):
    c = cnt[t][:17]; print(f'{nm} class share %: ' + ' '.join(f'{i}:{100 * x / c.sum():.1f}' for i, x in enumerate(c) if x))
