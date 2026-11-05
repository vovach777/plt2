#!/usr/bin/env python3
"""HLC5 vs llcomp sign rules (mirror sign applied to the MED residual), same unsigned contexts for all.

usage: python sign_cmp.py <ppm/png files or globs...>
Contexts (do not depend on the sign): bw(act) cap 6 and the 11 pctx-threshold buckets, act = |a-c|+|c-b|+|a-aa|.
Sign rules (s = sign of the first nonzero term, else +1; residual coded as s*e):
  HLC5 now (llcomp ModelSizeTiny hash, dz 0):  c-a, c-b
  llcomp_rc tiny (q5 hash idx > 62 -> s = -1):  -(a-c), -(c-b), -(d-b)   = c-a, b-c, b-d      (needs d)
  llcomp_rc without d:                          c-a, b-c
  llcomp Standard (q11 hash, top weight t-tr):  -(t-tr), -(tl-t), -(l-tl) = d-b, b-c, c-a  (needs d)
  extra (not llcomp): sign of the neighbours' error sum e_a + e_b (bias-like sign predictor)
"""
import numpy as np, glob, sys
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def bw(x): x = np.abs(x); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0)
def first(*gs):
    s = np.zeros_like(gs[0])
    for g in gs: s = np.where(s == 0, np.sign(g), s)
    return np.where(s == 0, 1, s)
def shift(A, dy, dx):
    O = np.zeros_like(A); h, w = A.shape
    if dx >= 0: O[dy:, dx:] = A[:h - dy, :w - dx]
    else: O[dy:, :w + dx] = A[:h - dy, -dx:]
    return O
TY, TC = [1, 3, 10, 18, 35], [1, 4, 8, 16, 30]
RULES = ['no sign', 'HLC5 now: c-a, c-b', 'llcomp_rc: c-a, b-c, b-d (d)', 'llcomp_rc w/o d: c-a, b-c', 'llcomp Standard: d-b, b-c, c-a (d)', 'extra: sign(e_a + e_b)']
def main(files):
    tot = {(r, k): 0.0 for r in RULES for k in ('bw', 'pctx')}; px = 0
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        R_, G, B = a0[..., 0], a0[..., 1], a0[..., 2]; V = R_ - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
        S = {k: dict() for k in tot}
        for P, n9 in [(V, True), (U, True), (Y, False)]:
            a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); d = shift(P, 1, -1); aa = shift(P, 0, 2)
            a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]; d[:, -1] = b[:, -1]; aa[:, :2] = a[:, :2]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
            act = np.abs(a - c) + np.abs(c - b) + np.abs(a - aa)
            kb = np.minimum(bw(act), 6); kp = sum((act >= t).astype(np.int32) for t in (TC if n9 else TY))
            sg = {RULES[0]: np.ones_like(P), RULES[1]: first(c - a, c - b), RULES[2]: first(c - a, b - c, b - d),
                  RULES[3]: first(c - a, b - c), RULES[4]: first(d - b, b - c, c - a), RULES[5]: first(shift(e, 0, 1) + shift(e, 1, 0))}
            off = 1000 if n9 else 0
            for r, s in sg.items():
                ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
                for kn, k in (('bw', kb), ('pctx', kp)):
                    k = np.where(k == 0, 0, k + off)
                    for v in np.unique(k): S[(r, kn)].setdefault(int(v), []).append(z[k == v])
        for key in tot: tot[key] += sum(H(np.concatenate(v)) for v in S[key].values())
    for kn, lab in (('bw', 'bw(act) cap 6'), ('pctx', '11 pctx buckets')):
        base = tot[(RULES[0], kn)]; print(f'== contexts: {lab}')
        for r in RULES: print(f'  {r:38s} {tot[(r, kn)] / px:.4f} bpp  {100 * tot[(r, kn)] / base - 100:+.2f}%')
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
