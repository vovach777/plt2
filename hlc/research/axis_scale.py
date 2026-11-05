#!/usr/bin/env python3
"""Axis scaling suggested by the CLIC sector table: c-b x k (q axes) and v x k (u,v frame), integer |x|*k rounded down.
usage: python axis_scale.py <ppm/png files or globs...>   (reuses uv_axes.py helpers)"""
import numpy as np, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uv_axes import H, qs, qa, shift, E7
from PIL import Image
def sc(x, k): return np.sign(x) * np.floor(np.abs(x) * k).astype(np.int32)
def variants():
    V = {}
    for k in (1, 1.125, 1.25, 1.5):
        V[f'q7 49, c-b x{k}'] = lambda d, k=k: ((qs(d['ga'], E7) + 3) * 7 + qs(sc(d['gb'], k), E7) + 3, 1)
        def ms(d, k=k):
            p, q = qs(d['ga'], E7), qs(sc(d['gb'], k), E7); s = np.where(p + q > 0, 1, np.where(p + q < 0, -1, np.where(p < 0, -1, 1)))
            p2, q2 = s * p, s * q; return (np.minimum(p2, q2) + 3) * 7 + np.maximum(p2, q2) + 3, s
        V[f'q7 mirror+swap 16, c-b x{k}'] = ms
    for eg, nm in ((E7, '1 4 16 (16)'), ([1, 3, 8, 24], '1 3 8 24 (25)')):
        for k in (1, 1.25, 1.5, 2):
            def f(d, eg=eg, k=k):
                u, v = d['ga'] + d['gb'], sc(d['ga'] - d['gb'], k); s = np.where(u < 0, -1, 1)
                return qa(u, eg) * 16 + qa(v, eg), s
            V[f'(|u|,|v|) {nm}, v x{k}'] = f
    return V
def main(files):
    V = variants(); tot = {k: 0.0 for k in V}; px = 0
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        G = a0[..., 1]; Vv = a0[..., 0] - G; U = a0[..., 2] - G; Y = G + ((U + Vv) >> 2); px += Y.size
        acc = {k: [{}, {}] for k in V}
        for P, n9 in [(Vv, True), (U, True), (Y, False)]:
            t = int(n9); a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); d = {'ga': c - a, 'gb': c - b}
            for k, fn in V.items():
                cls, fl = fn(d); ee = fl * e
                ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
                for v in np.unique(cls): acc[k][t].setdefault(int(v), []).append(z[cls == v])
        for k in V: tot[k] += sum(H(np.concatenate(x)) for t in (0, 1) for x in acc[k][t].values())
    base = {}
    for k in V:
        g = k.rsplit(' x', 1)[0]; base.setdefault(g, tot[k])
        print(f'{k:40s} {tot[k] / px:.4f} bpp  {100 * tot[k] / base[g] - 100:+.2f}% vs x1')
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
