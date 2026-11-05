#!/usr/bin/env python3
"""HLC5 context study: two-axis direction context x 2-bit disagreement MED vs a second predictor.

usage: python predctx.py <ppm/png files or globs...>
Entropy (zero order, tables per file, FSE headers ignored) of MED residuals in context streams; sign = llcomp tiny
(sign(c-a), else sign(c-b)); Y and U/V separate, class 0 shared. Third axis t = |P_med - P_alt| -> 0 | 1-3 | 4-15 | 16+.
"""
import numpy as np, glob, sys
from PIL import Image

def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def qsym(x, edges): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in edges)
def bw(x): x = np.abs(x); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0)

def planes(f):
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2)
    return [(V, True), (U, True), (Y, False)]

def neigh(P):
    Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
    l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
    ll = l.copy(); ll[:, 2:] = P[:, :-2]
    return l, t, tl, ll

ALT = {'planar a+b-c': lambda a, b, c: a + b - c,
       'HALIC /3': lambda a, b, c: np.trunc((2 * (a + b) - c) / 3).astype(np.int32),
       'HALIC avg': lambda a, b, c: (a + b + 1) >> 1,
       'HALIC none': lambda a, b, c: np.zeros_like(a)}
DIR = {'q3 dz0 (direction only)': (3, [1]), 'q5 (1-3, 4+)': (5, [1, 4]), 'q7 (1-3, 4-15, 16+)': (7, [1, 4, 16])}
T3 = [1, 4, 16]

def main(files):
    keys = ['bw(act3) cap 6 (HLC5-like, reference)'] + [f'{d}' for d in DIR] + [f'{d} x t[{p}]' for d in DIR for p in ALT]
    tot = {k: 0.0 for k in keys}; ns = {k: set() for k in keys}; px = 0
    for f in files:
        S = {k: dict() for k in keys}
        for P, n9 in planes(f):
            if not n9: px += P.size
            a, b, c, aa = neigh(P)
            med = np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); e = P - med
            ga, gb = c - a, c - b; sa, sb = np.sign(ga), np.sign(gb); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
            ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
            off = 100000 if n9 else 0
            def put(k, cls):
                cls = np.where(cls == 0, 0, cls + off)
                for v in np.unique(cls): S[k].setdefault(int(v), []).append(z[cls == v]); ns[k].add(int(v))
            put(keys[0], np.minimum(bw(np.abs(ga) + np.abs(gb) + np.abs(a - aa)), 6))
            for d, (n, edges) in DIR.items():
                cx = np.abs(qsym(gb, edges) + n * qsym(ga, edges)); put(d, cx)
                for p, fn in ALT.items():
                    t = sum((np.abs(med - fn(a, b, c)) >= th).astype(np.int32) for th in T3)
                    put(f'{d} x t[{p}]', np.where((cx == 0) & (t == 0), 0, cx * 4 + t + 1))
        for k in keys: tot[k] += sum(H(np.concatenate(v)) for v in S[k].values())
    base = tot[keys[0]]
    for k in keys: print(f'{k:52s} {tot[k] / px:.4f} bpp  {100 * tot[k] / base - 100:+.2f}%  streams {len(ns[k])}')

if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
