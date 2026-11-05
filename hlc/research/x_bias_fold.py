#!/usr/bin/env python3
"""After the central mirror the second axis x = s*(c-b) is not symmetric. Shift it by a bias beta, then fold its sign:
contexts (q7(y), |q7(x - beta)|), y = s*(c-a) >= 0, residual flipped (or not) on the negative side.

usage: python x_bias_fold.py <ppm/png files or globs...>
beta: constant integers, per-row median of x (row = q7(y)), proportional alpha * y. Cost vs the 49 sectors, entropy
with per-file tables, Y and U/V separate.
"""
import numpy as np, glob, sys
from PIL import Image
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def q7(x): a = np.abs(x); return np.sign(x) * ((a >= 1).astype(np.int32) + (a >= 4) + (a >= 16))
def shift(A, dy, dx):
    O = np.zeros_like(A); h, w = A.shape; O[dy:, dx:] = A[:h - dy, :w - dx]; return O
def load(files):
    D = []
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        R_, G, B = a0[..., 0], a0[..., 1], a0[..., 2]; V = R_ - G; U = B - G; Y = G + ((U + V) >> 2); pl = []
        for P, n9 in [(V, True), (U, True), (Y, False)]:
            a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b))
            sa, sb = np.sign(c - a), np.sign(c - b); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
            pl.append((int(n9), (s * (c - a)).ravel(), (s * (c - b)).ravel(), (s * e).ravel(), n9))
        D.append(pl)
    return D
def cost(D, ctx_fn):
    tot = np.zeros(2)
    for pl in D:
        acc = [{}, {}]
        for t, y, x, e, n9 in pl:
            k, fl = ctx_fn(t, y, x); ee = fl * e
            ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
            for v in np.unique(k): acc[t].setdefault(int(v), []).append(z[k == v])
        for t in (0, 1): tot[t] += sum(H(np.concatenate(a)) for a in acc[t].values())
    return tot
def main(files):
    D = load(files)
    base = cost(D, lambda t, y, x: ((q7(y) + 3) * 7 + q7(x) + 3, 1))                       # 25 mirrored sectors
    full = None
    def rep(name, tot):
        print(f'{name:44s} Y {100 * tot[0] / base[0] - 100:+.2f}%  U/V {100 * tot[1] / base[1] - 100:+.2f}%  all {100 * tot.sum() / base.sum() - 100:+.2f}%  vs the 25 mirrored sectors')
    # medians of x per row
    med = {}
    for t in (0, 1):
        ys = np.concatenate([y for pl in D for (tt, y, x, e, n9) in pl if tt == t]); xs = np.concatenate([x for pl in D for (tt, y, x, e, n9) in pl if tt == t])
        for r in range(4):
            m = q7(ys) == r; med[(t, r)] = float(np.median(xs[m])) if m.any() else 0.0
        print(f'{"Y" if t == 0 else "U/V"}: median of x per row q7(y) 0, 1-3, 4-15, 16+: ' + ', '.join(f'{med[(t, r)]:+.1f}' for r in range(4)) +
              '    mean: ' + ', '.join(f'{xs[q7(ys) == r].mean():+.2f}' for r in range(4)))
    def fold(beta_fn, flip):
        def f(t, y, x):
            xb = x - beta_fn(t, y); q = q7(xb); fl = np.where(q < 0, -1, 1) if flip else 1
            return (q7(y) * 4 + np.abs(q)), fl
        return f
    rep('fold |q7(x)|, no bias (flip)', cost(D, fold(lambda t, y: 0, True)))
    rep('fold |q7(x)|, no bias (no flip)', cost(D, fold(lambda t, y: 0, False)))
    for b in (-4, -2, -1, 1, 2, 4):
        rep(f'fold |q7(x - {b:+d})| (flip)', cost(D, fold(lambda t, y, b=b: b, True)))
    for flip in (True, False):
        rep(f'fold, beta = row median ({"flip" if flip else "no flip"})', cost(D, fold(lambda t, y: np.array([med[(t, r)] for r in range(4)])[q7(y)].round().astype(np.int32), flip)))
    for al in (0.25, 0.5, 1.0):
        for flip in (True, False):
            rep(f'fold, beta = {al} * y ({"flip" if flip else "no flip"})', cost(D, fold(lambda t, y, al=al: np.round(al * y).astype(np.int32), flip)))
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
