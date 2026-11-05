#!/usr/bin/env python3
"""HLC5: sign rule when the neighbour error is already a context axis.

usage: python sign_err.py <ppm/png files or globs...>
Context: two-axis q5 / q7 (|hash| of q(c-b) + n q(c-a)) x 2-bit mean neighbour error (|e_a|+|e_b|+|e_c|)/3.
Sign variants: geometric tiny (c-a, else c-b), error sum (e_a + e_b), geometric with error fallback, error with
geometric fallback, none. Entropy as predctx.py.
"""
import numpy as np, glob, sys
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def qsym(x, edges): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in edges)
def q2(x): x = np.abs(x); return sum((x >= t).astype(np.int32) for t in (1, 4, 16))
def shift(A, dy, dx):
    O = np.zeros_like(A); h, w = A.shape
    if dx >= 0: O[dy:, dx:] = A[:h - dy, :w - dx]
    else: O[dy:, :w + dx] = A[:h - dy, -dx:]
    return O
def first(*gs):
    s = np.zeros_like(gs[0])
    for g in gs: s = np.where(s == 0, np.sign(g), s)
    return np.where(s == 0, 1, s)
DIR = {'q5': (5, [1, 4]), 'q7': (7, [1, 4, 16])}
SG = ['none', 'geometric c-a, c-b', 'error sum e_a+e_b', 'geometric, then error', 'error, then geometric']
def main(files):
    tot = {(d, s): 0.0 for d in DIR for s in SG}; px = 0
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        R_, G, B = a0[..., 0], a0[..., 1], a0[..., 2]; V = R_ - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
        S = {k: dict() for k in tot}
        for P, n9 in [(V, True), (U, True), (Y, False)]:
            a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
            ea, eb, ec = shift(e, 0, 1), shift(e, 1, 0), shift(e, 1, 1)
            t = q2((np.abs(ea) + np.abs(eb) + np.abs(ec)) // 3)
            sg = {SG[0]: np.ones_like(P), SG[1]: first(c - a, c - b), SG[2]: first(ea + eb),
                  SG[3]: first(c - a, c - b, ea + eb), SG[4]: first(ea + eb, c - a, c - b)}
            off = 100000 if n9 else 0
            for dn, (n, edges) in DIR.items():
                cx = np.abs(qsym(c - b, edges) + n * qsym(c - a, edges))
                cls = np.where((cx == 0) & (t == 0), 0, cx * 4 + t + 1); cls = np.where(cls == 0, 0, cls + off)
                for sn, s in sg.items():
                    ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
                    for v in np.unique(cls): S[(dn, sn)].setdefault(int(v), []).append(z[cls == v])
        for k in tot: tot[k] += sum(H(np.concatenate(v)) for v in S[k].values())
    for dn in DIR:
        base = tot[(dn, 'none')]; print(f'== {dn} x mean neighbour error')
        for sn in SG: print(f'  sign {sn:24s} {tot[(dn, sn)] / px:.4f} bpp  {100 * tot[(dn, sn)] / base - 100:+.2f}%')
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
