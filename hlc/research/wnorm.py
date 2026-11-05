import numpy as np, glob
from PIL import Image
def bw(x): x = np.abs(x); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0)
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
files = sorted(glob.glob('photos16h/*.ppm')) + ['cat_h.ppm']
V_ = {'|x|+|y| (L1)': (2, 2), '2|x|+|y|': (4, 2), '3|x|+2|y| ': (3, 2), '|x|+2|y|': (2, 4)}   # (wx, wy) in halves: r = (wx|x| + wy|y|)/2
tot = {k: 0.0 for k in V_}; px = 0
for f in files:
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    S = {k: dict() for k in V_}
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        ga, gb = tl - l, tl - t; sa, sb = np.sign(ga), np.sign(gb); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
        ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
        for k, (wx, wy) in V_.items():
            r = (wx * np.abs(gb) + wy * np.abs(ga)) >> 1; c = np.minimum(bw(r), 6); c = np.where(c == 0, 0, c + (100 if n9 else 0))
            for v in np.unique(c): S[k].setdefault(int(v), []).append(z[c == v])
    for k in V_: tot[k] += sum(H(np.concatenate(v)) for v in S[k].values())
b = tot['|x|+|y| (L1)']
for k in V_: print(f'bw({k}) cap 6: {tot[k] / px:.4f} bpp  {100 * tot[k] / b - 100:+.2f}% vs L1')
