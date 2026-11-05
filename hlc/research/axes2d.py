import numpy as np, glob, sys, pickle
from PIL import Image
def bw(x): x = np.abs(x); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0)
files = sorted(glob.glob(sys.argv[1])); out = sys.argv[2]
# per plane type t (0 Y, 1 UV): cell (l1 0..9, l2 -9..9) -> histogram of zz (512)
Hc = np.zeros((2, 10, 19, 512), np.int64)
for f in files:
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2)
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        ga, gb = tl - l, tl - t; sa, sb = np.sign(ga), np.sign(gb); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
        ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
        l1 = bw(ga); l2 = bw(gb) * np.sign(s * gb)
        idx = ((l1 * 19 + (l2 + 9)) * 512 + z).ravel()
        Hc[int(n9)] += np.bincount(idx, minlength=10 * 19 * 512).reshape(10, 19, 512)
pickle.dump(Hc, open(out, 'wb'))
for t, nm in ((0, 'Y'), (1, 'UV')):
    n = Hc[t].sum(-1); N = n.sum(); zv = np.arange(512); ev = np.where(zv & 1, (zv + 1) // 2, zv // 2)
    me = (Hc[t] * ev).sum(-1) / np.maximum(n, 1)
    print(f'== {nm}: share % (rows l1 = bw|c-a| 0..9, cols l2 = signed bw(c-b) after mirror, -9..9)')
    print('l1\\l2 ' + ''.join(f'{v:6d}' for v in range(-9, 10)))
    for i in range(10): print(f'{i:5d} ' + ''.join(f'{100 * n[i, j] / N:6.2f}' if n[i, j] else '     .' for j in range(19)))
    print(f'== {nm}: mean |e|')
    for i in range(10): print(f'{i:5d} ' + ''.join(f'{me[i, j]:6.1f}' if n[i, j] > 100 else '     .' for j in range(19)))
