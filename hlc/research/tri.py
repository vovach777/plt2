import numpy as np, glob
from PIL import Image
def bw(x): x = np.abs(x); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0)
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
files = sorted(glob.glob('photos16h/*.ppm')) + ['cat_h.ppm']
VARS = ['L1 only', 'L1 x 3 triangles', 'L1 x 3 triangles + ridge x=0', 'L1 x MED branch (x>0 / x<=0)']
tot = np.zeros(len(VARS)); px = 0
st = {t: np.zeros((4, 7, 3)) for t in (0, 1)}          # [plane type][sector][radius class] -> n, sum|e|
for f in files:
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    S = [dict() for _ in VARS]
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        ga, gb = tl - l, tl - t; sa, sb = np.sign(ga), np.sign(gb); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
        x = s * gb; y = s * ga
        ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
        r = np.minimum(bw(np.abs(x) + y), 6)
        sec = np.where(x > y, 0, np.where(-x > y, 1, 2))             # T1 extremum, T2 ramp, T3 c-a dominant
        sec_r = np.where(x == 0, 3, sec)                             # ridge c = b
        for sc in range(4):
            for rc in range(7):
                m = (sec_r == sc) & (r == rc); st[int(n9)][sc, rc] += [m.sum(), np.abs(ee[m]).sum(), 0]
        keys = [r, r * 4 + np.where(r == 0, 0, sec), r * 4 + np.where(r == 0, 0, sec_r), r * 2 + np.where(r == 0, 0, x > 0)]
        for vi, k in enumerate(keys):
            k = np.where(r == 0, 0, k + (1000 if n9 else 0))
            for v in np.unique(k): S[vi].setdefault(int(v), []).append(z[k == v])
    for vi in range(len(VARS)): tot[vi] += sum(H(np.concatenate(v)) for v in S[vi].values()); 
for vi, v in enumerate(VARS): print(f'{v:32s} {tot[vi] / px:.4f} bpp  {100 * tot[vi] / tot[0] - 100:+.2f}%')
names = ['T1 c extremum (x>y)', 'T2 ramp (-x>y)', 'T3 |c-a| dominant', 'ridge c=b (x=0)']
for t, nm in ((0, 'Y'), (1, 'UV')):
    N = st[t][:, :, 0].sum(); print(f'== {nm}: share % / mean|e| per sector x radius class bw(L1) 1..6')
    for sc in range(4):
        print(f'  {names[sc]:22s} ' + '  '.join(f'{100 * st[t][sc, rc, 0] / N:5.1f}/{st[t][sc, rc, 1] / max(1, st[t][sc, rc, 0]):5.1f}' for rc in range(1, 7)))
