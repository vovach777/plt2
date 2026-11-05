#!/usr/bin/env python3
"""Diagonal coordinates for the MED context: u = (c-a)+(c-b) = 2c-a-b, v = (c-a)-(c-b) = b-a.
Central mirror + axis swap = (u,v) -> (|u|,|v|) with the residual flipped by sign(u). Context = q(|u|) x q(|v|).

usage: python uv_axes.py <ppm/png files or globs...>
Compares with the q7 x q7 sectors (49), the central mirror (25) and mirror + swap (16) in the (c-a, c-b) axes.
Entropy with per-file tables, Y and U/V separate, class count per plane type printed.
"""
import numpy as np, glob, sys
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def qs(x, edges): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in edges)
def qa(x, edges): return sum((np.abs(x) >= t).astype(np.int32) for t in edges)
def shift(A, dy, dx):
    O = np.zeros_like(A); h, w = A.shape; O[dy:, dx:] = A[:h - dy, :w - dx]; return O
E7 = [1, 4, 16]
def variants():
    V = {}
    V['q7 x q7, 49 sectors'] = lambda d: ((qs(d['ga'], E7) + 3) * 7 + qs(d['gb'], E7) + 3, 1)
    def mirror(d):
        h = qs(d['gb'], E7) + 7 * qs(d['ga'], E7); return np.abs(h), np.where(h < 0, -1, 1)
    V['q7 central mirror, 25'] = mirror
    def ms(d):
        p, q = qs(d['ga'], E7), qs(d['gb'], E7); s = np.where((p + q > 0) | ((p + q == 0) & (p > 0)), 1, np.where(p + q < 0, -1, np.where(p < 0, -1, 1)))
        p2, q2 = s * p, s * q; lo, hi = np.minimum(p2, q2), np.maximum(p2, q2)        # swap -> unordered pair
        return (lo + 3) * 7 + hi + 3, s
    V['q7 mirror + swap, 16'] = ms
    for eu, ev, nm in ((E7, E7, 'u,v edges 1 4 16'), ([1, 3, 8, 24], [1, 3, 8, 24], 'u,v edges 1 3 8 24'), ([1, 2, 6, 16, 40], [1, 2, 6, 16, 40], 'u,v edges 1 2 6 16 40'),
                       ([1, 3, 8, 24], [1, 4, 16], '|u| 1 3 8 24, |v| 1 4 16'), ([1, 4, 16], [1, 3, 8, 24], '|u| 1 4 16, |v| 1 3 8 24'),
                       ([1, 2, 6, 16, 40], [1, 4, 16], '|u| 1 2 6 16 40, |v| 1 4 16'), ([2, 8, 32], [1, 4, 16], '|u| 2 8 32, |v| 1 4 16 (u scaled 1/2)')):
        nU = len(eu) + 1
        for fb in ('+1', 'sign(v)'):
            def f(d, eu=eu, ev=ev, nU=nU, fb=fb):
                u, v = d['ga'] + d['gb'], d['ga'] - d['gb']
                s = np.sign(u); s = np.where(s == 0, (np.where(v < 0, -1, 1) if fb == 'sign(v)' else 1), s)
                return qa(u, eu) * 16 + qa(v, ev), s
            V[f'(|u|,|v|) {nm}, u=0 flip {fb}'] = f
    return V
def main(files):
    V = variants(); tot = {k: np.zeros(2) for k in V}; ncls = {k: [set(), set()] for k in V}; px = 0
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        R_, G, B = a0[..., 0], a0[..., 1], a0[..., 2]; Vv = R_ - G; U = B - G; Y = G + ((U + Vv) >> 2); px += Y.size
        acc = {k: [{}, {}] for k in V}
        for P, n9 in [(Vv, True), (U, True), (Y, False)]:
            t = int(n9)
            a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); d = {'ga': c - a, 'gb': c - b}
            for k, fn in V.items():
                cls, fl = fn(d); ee = fl * e
                ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
                for v in np.unique(cls): acc[k][t].setdefault(int(v), []).append(z[cls == v]); ncls[k][t].add(int(v))
        for k in V:
            for t in (0, 1): tot[k][t] += sum(H(np.concatenate(x)) for x in acc[k][t].values())
    b = tot['q7 x q7, 49 sectors']
    for k in V:
        print(f'{k:52s} classes Y {len(ncls[k][0]):2d} UV {len(ncls[k][1]):2d}   {tot[k].sum() / px:.4f} bpp  {100 * tot[k].sum() / b.sum() - 100:+.2f}% vs 49 sectors')
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
