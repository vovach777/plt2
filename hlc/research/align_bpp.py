#!/usr/bin/env python3
"""bpp of the q7 contexts as is vs with c-b stretched x1.125 (x + (x>>3)) to even the grid about its diagonal.
Variants: 49 sectors, tiny-sign mirror 25, mirror + swap 16, (|u|,|v|) 1 3 8 24 (25). Per-file tables, Y and U/V separate.
usage: python align_bpp.py <ppm/png files or globs...>"""
import numpy as np, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uv_axes import H, qs, qa, shift, E7
from PIL import Image
def st(x): return np.sign(x) * (np.abs(x) + (np.abs(x) >> 3))
def tiny(ga, gb): return np.where(ga != 0, np.sign(ga), np.where(gb != 0, np.sign(gb), 1))
def v49(ga, gb): return (qs(ga, E7) + 3) * 7 + qs(gb, E7) + 3, 1
def v25(ga, gb): s = tiny(ga, gb); return (qs(s * ga, E7) + 3) * 7 + qs(s * gb, E7) + 3, s
def v16(ga, gb):
    s = tiny(ga, gb); p, q = qs(s * ga, E7), qs(s * gb, E7); return (np.minimum(p, q) + 3) * 7 + np.maximum(p, q) + 3, s
def vuv(ga, gb): u = ga + gb; v = ga - gb; s = np.where(u < 0, -1, 1); return qa(u, [1, 3, 8, 24]) * 16 + qa(v, [1, 3, 8, 24]), s
V = {'49 sectors': v49, 'tiny mirror 25': v25, 'mirror + swap 16': v16, '(|u|,|v|) 25': vuv}
def main(files):
    tot = {(k, al): np.zeros(2) for k in V for al in (0, 1)}; px = 0
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        G = a0[..., 1]; Vv = a0[..., 0] - G; U = a0[..., 2] - G; Y = G + ((U + Vv) >> 2); px += Y.size
        acc = {key: [{}, {}] for key in tot}
        for P, n9 in [(Vv, True), (U, True), (Y, False)]:
            t = int(n9); a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); ga, gb0 = c - a, c - b
            for al in (0, 1):
                gb = st(gb0) if al else gb0
                for k, fn in V.items():
                    cls, fl = fn(ga, gb); ee = fl * e
                    ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
                    for v in np.unique(cls): acc[k, al][t].setdefault(int(v), []).append(z[cls == v])
        for key in tot:
            for t in (0, 1): tot[key][t] += sum(H(np.concatenate(x)) for x in acc[key][t].values())
    b = tot['49 sectors', 0].sum()
    for k in V:
        r0, r1 = tot[k, 0], tot[k, 1]
        print(f'{k:18s} as is {r0.sum() / px:.4f} ({100 * r0.sum() / b - 100:+.2f}%)  aligned {r1.sum() / px:.4f} ({100 * r1.sum() / b - 100:+.2f}%)'
              f'  Y {r0[0] / px:.4f}->{r1[0] / px:.4f}  UV {r0[1] / px:.4f}->{r1[1] / px:.4f}')
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
