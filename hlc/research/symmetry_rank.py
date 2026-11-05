#!/usr/bin/env python3
"""Rank symmetry groups on the q7(c-a) x q7(c-b) sectors by the entropy cost of folding them.

usage: python symmetry_rank.py <ppm/png files or globs...>
Each generator maps (q1, q2) -> (q1', q2') and multiplies the MED residual by a flip (+1/-1). A group = generators;
each orbit becomes one context, the residual is multiplied by the flip of the path to the orbit's canonical member.
Cost = zero-order entropy with per-file tables vs the 49 separate sectors; Y and U/V separate, never merged.
"""
import numpy as np, glob, sys, itertools
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def q7(x): a = np.abs(x); return np.sign(x) * ((a >= 1).astype(np.int32) + (a >= 4) + (a >= 16))
def shift(A, dy, dx):
    O = np.zeros_like(A); h, w = A.shape; O[dy:, dx:] = A[:h - dy, :w - dx]; return O
GEN = {'M': (lambda p, q: (-p, -q), -1),                      # central mirror (the HLC5 / llcomp one)
       'X2+': (lambda p, q: (p, -q), +1), 'X2-': (lambda p, q: (p, -q), -1),     # drop the sign of q(c-b)
       'X1+': (lambda p, q: (-p, q), +1), 'X1-': (lambda p, q: (-p, q), -1),     # drop the sign of q(c-a)
       'T': (lambda p, q: (q, p), +1)}                         # swap the axes (a <-> b, image transpose)
GROUPS = [[], ['M'], ['M', 'X2+'], ['M', 'X2-'], ['M', 'X1+'], ['M', 'X1-'], ['M', 'T'], ['M', 'T', 'X2+'], ['M', 'T', 'X2-']]
def orbits(gens):
    """map (q1,q2) -> (orbit id, flip to canonical)"""
    out = {}
    for p in range(-3, 4):
        for q in range(-3, 4):
            seen = {(p, q): 1}; todo = [(p, q)]
            while todo:
                x = todo.pop()
                for g in gens:
                    f, fl = GEN[g]; y = f(*x)
                    if y not in seen: seen[y] = seen[x] * fl; todo.append(y)
            can = min(seen); out[(p, q)] = (can, seen[(p, q)] * 1)     # flip from (p,q) to canonical: path sign
    # path sign: seen[y] = flip from start to y; flip from (p,q) to canonical = seen[can] (start = (p,q))
    return out
def tables(gens):
    m = {}
    for p in range(-3, 4):
        for q in range(-3, 4):
            seen = {(p, q): 1}; todo = [(p, q)]
            while todo:
                x = todo.pop()
                for g in gens:
                    f, fl = GEN[g]; y = f(*x)
                    if y not in seen: seen[y] = seen[x] * fl; todo.append(y)
            can = min(seen); m[(p, q)] = (can, seen[can])
    ids = {c: i for i, c in enumerate(sorted({v[0] for v in m.values()}))}
    oid = np.zeros((7, 7), np.int32); flp = np.ones((7, 7), np.int32)
    for (p, q), (c, f) in m.items(): oid[p + 3, q + 3] = ids[c]; flp[p + 3, q + 3] = f
    return oid, flp, len(ids)
def main(files):
    TB = [tables(g) for g in GROUPS]; tot = np.zeros((len(GROUPS), 2)); px = 0
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        R_, G, B = a0[..., 0], a0[..., 1], a0[..., 2]; V = R_ - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
        acc = [[{}, {}] for _ in GROUPS]
        for P, n9 in [(V, True), (U, True), (Y, False)]:
            t = int(n9)
            a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b))
            q1, q2 = q7(c - a) + 3, q7(c - b) + 3
            for gi, (oid, flp, n) in enumerate(TB):
                k = oid[q1, q2]; ee = flp[q1, q2] * e
                ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
                for v in np.unique(k): acc[gi][t].setdefault(int(v), []).append(z[k == v])
        for gi in range(len(GROUPS)):
            for t in (0, 1): tot[gi, t] += sum(H(np.concatenate(x)) for x in acc[gi][t].values())
    for gi, g in enumerate(GROUPS):
        n = TB[gi][2]; c = [100 * tot[gi, t] / tot[0, t] - 100 for t in (0, 1)]; call = 100 * tot[gi].sum() / tot[0].sum() - 100
        print(f'{"+".join(g) or "none (49 sectors)":12s} contexts {n:2d} per plane type   cost Y {c[0]:+.2f}%  U/V {c[1]:+.2f}%  all {call:+.2f}%   ({tot[gi].sum() / px:.4f} bpp)')
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
