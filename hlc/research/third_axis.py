#!/usr/bin/env python3
"""HLC5: what can replace the up-right d as a third context axis (2 bits: 0 | 1-3 | 4-15 | 16+) on top of q5 / q7.

usage: python third_axis.py <ppm/png files or globs...>
Candidates: |d-b| (reference, not wave-legal with skew 1), neighbour errors max(|e_a|,|e_b|) (PLT2 Enb-like),
(|e_a|+|e_b|)/2, (|e_a|+|e_b|+|e_c|)/3, |a-aa|, |b-bb| (bb = two rows up). Entropy as predctx.py.
"""
import numpy as np, glob, sys
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def qsym(x, edges): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in edges)
def q2(x): x = np.abs(x); return sum((x >= t).astype(np.int32) for t in (1, 4, 16))
def shift(A, dy, dx):                       # value at (y-dy, x-dx), 0 outside
    O = np.zeros_like(A); h, w = A.shape
    O[dy:, dx:] = A[:h - dy, :w - dx] if dx >= 0 else 0
    if dx < 0: O[dy:, :w + dx] = A[:h - dy, -dx:]
    return O
DIR = {'q5': (5, [1, 4]), 'q7': (7, [1, 4, 16])}
def main(files):
    names = ['none'] + ['|d-b| (ref, needs d)', 'max(|ea|,|eb|)', '(|ea|+|eb|)/2', '(|ea|+|eb|+|ec|)/3', '|a-aa|', '|b-bb|']
    keys = [(d, n) for d in DIR for n in names]; tot = {k: 0.0 for k in keys}; ns = {k: set() for k in keys}; px = 0
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        R_, G, B = a0[..., 0], a0[..., 1], a0[..., 2]; V = R_ - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
        S = {k: dict() for k in keys}
        for P, n9 in [(V, True), (U, True), (Y, False)]:
            a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); d = shift(P, 1, -1); aa = shift(P, 0, 2); bb = shift(P, 2, 0)
            a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]; d[:, -1] = b[:, -1]; aa[:, :2] = a[:, :2]; bb[:2] = b[:2]
            med = np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); e = P - med
            e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
            ea, eb, ec = np.abs(shift(e, 0, 1)), np.abs(shift(e, 1, 0)), np.abs(shift(e, 1, 1))
            ga, gb = c - a, c - b; sa, sb = np.sign(ga), np.sign(gb); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
            ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
            T = {'none': None, '|d-b| (ref, needs d)': q2(d - b), 'max(|ea|,|eb|)': q2(np.maximum(ea, eb)), '(|ea|+|eb|)/2': q2((ea + eb) >> 1),
                 '(|ea|+|eb|+|ec|)/3': q2((ea + eb + ec) // 3), '|a-aa|': q2(a - aa), '|b-bb|': q2(b - bb)}
            off = 100000 if n9 else 0
            for dn, (n, edges) in DIR.items():
                cx = np.abs(qsym(gb, edges) + n * qsym(ga, edges))
                for tn, t in T.items():
                    cls = cx if t is None else np.where((cx == 0) & (t == 0), 0, cx * 4 + t + 1)
                    cls = np.where(cls == 0, 0, cls + off)
                    for v in np.unique(cls): S[(dn, tn)].setdefault(int(v), []).append(z[cls == v]); ns[(dn, tn)].add(int(v))
        for k in keys: tot[k] += sum(H(np.concatenate(v)) for v in S[k].values())
    for dn in DIR:
        base = tot[(dn, 'none')]
        for tn in names: print(f'{dn} x {tn:24s} {tot[(dn, tn)] / px:.4f} bpp  {100 * tot[(dn, tn)] / base - 100:+.2f}% vs {dn} alone   streams {len(ns[(dn, tn)])}')
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
