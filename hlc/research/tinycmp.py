import numpy as np, glob
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
files = sorted(glob.glob('photos16h/*.ppm')) + ['cat_h.ppm']
def q3(x, dz): return (x > dz).astype(np.int32) - (x < -dz).astype(np.int32)
V_ = ['llcomp tiny as committed: dz 4, sign from hash', 'tiny dz 4, sign raw (HLC5 -ctx 9)', 'tiny dz 0 (HLC5 -ctx 9 best)']
tot = np.zeros(3); px = 0
for f in files:
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    S = [dict() for _ in V_]
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        ga, gb = tl - l, tl - t; sa, sb = np.sign(ga), np.sign(gb); sraw = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
        for vi, (dz, hashsign) in enumerate([(4, True), (4, False), (0, False)]):
            h = q3(gb, dz) + 3 * q3(ga, dz); s = np.where(h < 0, -1, 1) if hashsign else sraw
            ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
            cx = np.abs(h); k = np.where(cx == 0, 0, cx + (4 if n9 else 0))
            for v in np.unique(k): S[vi].setdefault(int(v), []).append(z[k == v])
    for vi in range(3): tot[vi] += sum(H(np.concatenate(v)) for v in S[vi].values())
for vi, v in enumerate(V_): print(f'{v:46s} {tot[vi] / px:.4f}  {100 * tot[vi] / 8.1294 - 100:+.2f}% vs L1 radius 6 levels   streams 9')
