import numpy as np, glob
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def qsym(x, edges):                       # llcomp-style: q(x) = sign(x) * #(edges <= |x|)
    a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in edges)
Q = {'q3 dz4 (tiny)': (3, [5]), 'q3 dz0': (3, [1]), 'q5': (5, [1, 4]), 'q7': (7, [1, 5, 16]), 'q11': (11, [1, 2, 5, 12, 35])}
files = sorted(glob.glob('photos16h/*.ppm')) + ['cat_h.ppm']
tot = {k: [0.0, 0.0] for k in Q}; px = 0; nst = {k: set() for k in Q}
for f in files:
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    S = {k: [dict(), dict()] for k in Q}
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        ga, gb = tl - l, tl - t; sa, sb = np.sign(ga), np.sign(gb); sraw = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
        for k, (n, edges) in Q.items():
            h = qsym(gb, edges) + n * qsym(ga, edges); cx = np.abs(h); kk = np.where(cx == 0, 0, cx + (1000 if n9 else 0))
            for si, s in enumerate([np.where(h < 0, -1, 1), sraw]):
                ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
                for v in np.unique(kk): S[k][si].setdefault(int(v), []).append(z[kk == v]); nst[k].add(int(v))
    for k in Q:
        for si in range(2): tot[k][si] += sum(H(np.concatenate(v)) for v in S[k][si].values())
for k in Q: print(f'{k:16s} sign from hash {tot[k][0] / px:.4f} ({100 * tot[k][0] / px / 8.1294 - 100:+.2f}%)   raw sign {tot[k][1] / px:.4f} ({100 * tot[k][1] / px / 8.1294 - 100:+.2f}%)   streams {len(nst[k])}')
