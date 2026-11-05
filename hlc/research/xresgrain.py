import numpy as np, glob, sys
from PIL import Image
exec(open('xresctx.py').read().split("V_ = [")[0])
G = [('tiles 256x128', 256, 128), ('band 8 rows x full W', 100000, 8), ('strip 32 rows x W', 100000, 32), ('strip 128 rows x W', 100000, 128), ('whole image', 100000, 100000)]
tot = np.zeros(len(G) + 1); px = 0; nsel = np.zeros(len(G))
for f in sorted(glob.glob(sys.argv[1])):
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, Gr, B = a[..., 0], a[..., 1], a[..., 2]; V = R - Gr; U = B - Gr; Y = Gr + ((U + V) >> 2); px += Y.size
    (eV, kV, sV), (eU, kU, sU), (eY, kY, sY) = plane(V, True), plane(U, True), plane(Y, False)
    def total(v, u):
        S = [[] for _ in range(11)]
        for e, k, s, n9 in ((v, kV, sV, True), (u, kU, sU, True), (eY, kY, sY, False)):
            ee = s * e; ee = w9(ee) if n9 else ((ee + 128) & 255) - 128; z = zz(ee)
            for j in range(6): S[0 if j == 0 else (5 + j if n9 else j)].append(z[k == j])
        return sum(H(np.concatenate(x)) for x in S)
    tot[0] += total(eV, eU)
    for gi, (_, TW, TH) in enumerate(G):
        v, u, st = xres(eV, eU, False, TW, TH); tot[gi + 1] += total(v, u); nsel[gi] += sum(st)
print(f'{"no xres":24s} {tot[0] / px:.4f}')
for gi, (nm, _, _) in enumerate(G): print(f'{nm:24s} {tot[gi + 1] / px:.4f}  {100 * tot[gi + 1] / tot[0] - 100:+.2f}%   side info ~{nsel[gi]:.0f} bytes total ({8 * nsel[gi] / px * 100 / (tot[0] / px):.3f}% of size)')
