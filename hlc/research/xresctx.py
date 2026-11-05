import numpy as np, glob, sys
from PIL import Image
TY, TC = [1, 4, 9, 20, 52], [1, 3, 7, 12, 24]
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def zz(e): return np.where(e >= 0, 2 * e, -2 * e - 1)
w9 = lambda e: ((e + 256) & 511) - 256
def plane(P, n9):
    Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
    l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
    ll = l.copy(); ll[:, 2:] = P[:, :-2]
    e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t)); e = w9(e) if n9 else ((e + 128) & 255) - 128
    k = sum(((np.abs(l - tl) + np.abs(tl - t) + np.abs(l - ll)) >= th).astype(np.int32) for th in (TC if n9 else TY))
    sa, sb = np.sign(tl - l), np.sign(tl - t); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
    return e, k, s
def xres(eV, eU, both, TW=256, TH=128):
    eV, eU = eV.copy(), eU.copy(); st = [0, 0, 0]
    for y0 in range(0, eV.shape[0], TH):
        for x0 in range(0, eV.shape[1], TW):
            v = eV[y0:y0 + TH, x0:x0 + TW]; u = eU[y0:y0 + TH, x0:x0 + TW]
            best = (H(zz(u)) + H(zz(v)), 0, 0)
            for d in ((1, 2) if both else (1,)):
                src, tgt, oth = (v, u, v) if d == 1 else (u, v, u)
                for kk in range(-8, 9):
                    if kk == 0: continue
                    c = H(zz(w9(tgt - ((kk * src + 4) >> 3)))) + H(zz(oth))
                    if c < best[0]: best = (c, d, kk)
            _, d, kk = best; st[d] += 1
            if d == 1: eU[y0:y0 + TH, x0:x0 + TW] = w9(u - ((kk * v + 4) >> 3))
            elif d == 2: eV[y0:y0 + TH, x0:x0 + TW] = w9(v - ((kk * u + 4) >> 3))
    return eV, eU, st
V_ = ['contexts + sign (no xres)', '+ xres U<-V only', '+ xres both directions']
tot = np.zeros(3); px = 0; ST = [[0, 0, 0], [0, 0, 0]]
for f in sorted(glob.glob(sys.argv[1])):
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    (eV, kV, sV), (eU, kU, sU), (eY, kY, sY) = plane(V, True), plane(U, True), plane(Y, False)
    for vi in range(3):
        if vi == 0: v, u = eV, eU
        else: v, u, st = xres(eV, eU, vi == 2); ST[vi - 1] = [x + y for x, y in zip(ST[vi - 1], st)]
        S = [[] for _ in range(11)]
        for e, k, s, n9 in ((v, kV, sV, True), (u, kU, sU, True), (eY, kY, sY, False)):
            ee = s * e; ee = w9(ee) if n9 else ((ee + 128) & 255) - 128; z = zz(ee)
            for j in range(6): S[0 if j == 0 else (5 + j if n9 else j)].append(z[k == j])
        tot[vi] += sum(H(np.concatenate(x)) for x in S)
for vi, v in enumerate(V_): print(f'{v:28s} {tot[vi] / px:.4f} bpp  {100 * tot[vi] / tot[0] - 100:+.2f}%')
print('tiles off / U<-V / V<-U:  U<-V only', ST[0], '  both', ST[1])
