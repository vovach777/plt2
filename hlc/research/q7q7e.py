#!/usr/bin/env python3
"""HLC5: q7(c-a) x q7(c-b) (mirrored, 25 cells) x lg2(|e_a| + |e_b|), geometric sign (c-a, else c-b).

usage: python q7q7e.py <ppm/png files or globs...>
lg2 = bit width of |e_a|+|e_b| (0..10), uncapped and capped; reference: 2-bit mean error of a, b, c.
Entropy as predctx.py; streams = non-empty (plane type, cell, error level).
"""
import numpy as np, glob, sys
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def qsym(x, edges): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in edges)
def bw(x): x = np.abs(x); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0)
def shift(A, dy, dx):
    O = np.zeros_like(A); h, w = A.shape
    O[dy:, dx:] = A[:h - dy, :w - dx]; return O
def first(*gs):
    s = np.zeros_like(gs[0])
    for g in gs: s = np.where(s == 0, np.sign(g), s)
    return np.where(s == 0, 1, s)
V_ = ['q7 alone', 'q7 x 2-bit mean(|ea|,|eb|,|ec|) (ref)', 'q7 x lg2(|ea|+|eb|)', 'q7 x lg2 cap 5', 'q7 x lg2 cap 4', 'q7 x lg2 cap 3']
def main(files):
    tot = {v: 0.0 for v in V_}; ns = {v: set() for v in V_}; px = 0
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        R_, G, B = a0[..., 0], a0[..., 1], a0[..., 2]; V = R_ - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
        S = {v: dict() for v in V_}
        for P, n9 in [(V, True), (U, True), (Y, False)]:
            a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
            ea, eb, ec = np.abs(shift(e, 0, 1)), np.abs(shift(e, 1, 0)), np.abs(shift(e, 1, 1))
            s = first(c - a, c - b); ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128
            z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
            cx = np.abs(qsym(c - b, [1, 4, 16]) + 7 * qsym(c - a, [1, 4, 16]))
            lg = bw(ea + eb); m3 = sum(((ea + eb + ec) // 3 >= t).astype(np.int32) for t in (1, 4, 16))
            T = {V_[0]: None, V_[1]: m3, V_[2]: lg, V_[3]: np.minimum(lg, 5), V_[4]: np.minimum(lg, 4), V_[5]: np.minimum(lg, 3)}
            off = 1000000 if n9 else 0
            for v, t in T.items():
                cls = cx if t is None else np.where((cx == 0) & (t == 0), 0, cx * 16 + t + 1)
                cls = np.where(cls == 0, 0, cls + off)
                for u in np.unique(cls): S[v].setdefault(int(u), []).append(z[cls == u]); ns[v].add(int(u))
        for v in V_: tot[v] += sum(H(np.concatenate(x)) for x in S[v].values())
    base = tot[V_[0]]
    for v in V_: print(f'{v:40s} {tot[v] / px:.4f} bpp  {100 * tot[v] / base - 100:+.2f}% vs q7 alone   streams {len(ns[v])}')
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
