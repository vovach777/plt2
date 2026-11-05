import numpy as np, glob, sys
from PIL import Image
THY, THC = [1, 3, 10, 18, 35], [1, 4, 8, 16, 30]
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def wrapzz(ee, n9):
    ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; return np.where(ee >= 0, 2 * ee, -2 * ee - 1)
V_ = ['no sign', 'flip by a-c', 'flip + 5 sign cells (a-c, c-b)', 'flip + cells + bias per cell', 'flip + bias per class']
tot = np.zeros(len(V_)); px = 0; nstreams = [11, 11, 55, 55, 11]
for f in sorted(glob.glob(sys.argv[1])):
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    S = [dict() for _ in V_]
    for ci, (P, n9) in enumerate([(V, True), (U, True), (Y, False)]):
        Hh, W = P.shape; Z = np.zeros((Hh + 1, W + 1), np.int32); Z[1:, 1:] = P
        a_ = Z[1:, :-1].copy(); b = Z[:-1, 1:].copy(); c = Z[:-1, :-1].copy(); a_[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
        aa = a_.copy(); aa[:, 2:] = P[:, :-2]
        e = P - np.clip(a_ + b - c, np.minimum(a_, b), np.maximum(a_, b))
        k = sum(((np.abs(a_ - c) + np.abs(c - b) + np.abs(a_ - aa)) >= t).astype(np.int32) for t in (THC if n9 else THY))
        g1, g2 = np.sign(a_ - c), np.sign(c - b)
        s = np.where(g1 != 0, g1, np.where(g2 != 0, g2, 1))                 # mirror: first nonzero of (g1, g2) -> +
        cell = (g1 * s + 1) * 3 + (g2 * s + 1)                               # after mirror g1 in {0,+}: cells (0,0),(0,+),(+,-),(+,0),(+,+)
        sid = np.where(k == 0, 0, np.where(ci == 2, k, 5 + k))
        s1 = np.where(g1 != 0, g1, 1)
        for vi in range(len(V_)):
            if vi == 0: zz, key = wrapzz(e, n9), sid
            elif vi == 1: zz, key = wrapzz(s1 * e, n9), sid
            elif vi in (2, 3): ee = s * e; key = sid * 16 + cell
            else: ee = s1 * e; key = sid
            if vi == 3 or vi == 4:                                          # bias: rounded mean of the flipped error per key (sent per file)
                ee = ee.copy()
                for kk in np.unique(key):
                    m = key == kk; ee[m] -= int(np.round(ee[m].mean()))
            if vi >= 2: zz = wrapzz(ee, n9)
            for kk in np.unique(key):
                S[vi].setdefault(int(kk) if vi in (2, 3) or True else 0, []).append(zz[key == kk])
    for vi in range(len(V_)): tot[vi] += sum(H(np.concatenate(v)) for v in S[vi].values())
for vi, v in enumerate(V_): print(f'{v:34s} {tot[vi] / px:.4f} bpp  {100 * tot[vi] / tot[0] - 100:+.2f}%   ({nstreams[vi]} streams)')
