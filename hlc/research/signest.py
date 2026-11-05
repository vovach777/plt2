import numpy as np, glob, sys
from PIL import Image
THY, THC = [1, 3, 10, 18, 35], [1, 4, 8, 16, 30]
def first_sign(*gs):
    s = np.zeros_like(gs[0])
    for g in gs: s = np.where(s == 0, np.sign(g), s)
    return np.where(s == 0, 1, s)
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
variants = ['no sign', 'c-b, a-c, a-aa', 'a-c, c-b, a-aa', 'c-b, a-c', 'a-c only', 'ref: b-d, c-b, a-c (d!)']
tot = np.zeros(len(variants)); px = 0; perb = np.zeros((len(variants), 11))
for f in sorted(glob.glob(sys.argv[1])):
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    S = [[[] for _ in range(11)] for _ in variants]
    for ci, (P, n9) in enumerate([(V, True), (U, True), (Y, False)]):
        Hh, W = P.shape; Z = np.zeros((Hh + 1, W + 2), np.int32); Z[1:, 1:-1] = P
        a_ = Z[1:, :-2].copy(); b = Z[:-1, 1:-1].copy(); c = Z[:-1, :-2].copy(); d = Z[:-1, 2:].copy()
        a_[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]; d[:, -1] = b[:, -1]
        aa = a_.copy(); aa[:, 2:] = P[:, :-2]
        e = P - np.clip(a_ + b - c, np.minimum(a_, b), np.maximum(a_, b))
        k = sum(((np.abs(a_ - c) + np.abs(c - b) + np.abs(a_ - aa)) >= t).astype(np.int32) for t in (THC if n9 else THY))
        sg = [np.ones_like(P), first_sign(c - b, a_ - c, a_ - aa), first_sign(a_ - c, c - b, a_ - aa), first_sign(c - b, a_ - c), first_sign(a_ - c), first_sign(b - d, c - b, a_ - c)]
        for vi, s in enumerate(sg):
            ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128
            z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
            for j in range(6):
                sid = 0 if j == 0 else (j if ci == 2 else 5 + j); S[vi][sid].append(z[k == j])
    for vi in range(len(variants)):
        for sid in range(11):
            h = H(np.concatenate(S[vi][sid])); perb[vi, sid] += h; tot[vi] += h
names = ['0'] + [f'Y{j}' for j in range(1, 6)] + [f'UV{j}' for j in range(1, 6)]
for vi, v in enumerate(variants): print(f'{v:26s} {tot[vi] / px:.4f} bpp  {100 * tot[vi] / tot[0] - 100:+.2f}%')
b = int(np.argmin(tot[1:5])) + 1
print(f'per bucket, best wave-legal variant "{variants[b]}":  ' + '  '.join(f'{n} {100 * perb[b, i] / perb[0, i] - 100:+.1f}%' for i, n in enumerate(names)))
