#!/usr/bin/env python3
"""Per-direction axis scale: 8 octants of the (c-a, c-b) plane (signs + |c-a| >= |c-b|), 4 after the tiny-sign mirror.
Per octant and plane type (Y / UV) least squares |e| ~ wa|c-a| + wb|c-b| (+ w0) over the whole set (in-sample = upper bound).
Then the axes in that octant are multiplied by wa/W, wb/W (W = global fit, so the mean scale stays ~1) and quantised
with the usual edges 1 4 16 (float compare = per-direction thresholds). Variants: radial (one scale per octant) and
per-axis (two scales per octant). Baseline: 49 sectors and tiny-sign mirror 25.
usage: python oct_scale.py <ppm/png files or globs...>"""
import numpy as np, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uv_axes import H, shift, E7
from PIL import Image
def qsf(x): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in E7)
def load(files):
    D = [];
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        G = a0[..., 1]; Vv = a0[..., 0] - G; U = a0[..., 2] - G; Y = G + ((U + Vv) >> 2); pl = []
        for P, n9 in [(Vv, True), (U, True), (Y, False)]:
            a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b))
            e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
            pl.append((n9, (c - a).ravel(), (c - b).ravel(), e.ravel()))
        D.append((Y.size, pl))
    return D
def octant(ga, gb): return (ga < 0) * 4 + (gb < 0) * 2 + (np.abs(ga) < np.abs(gb))
def fit(D, mirror):
    """weights per (plane type, octant): [radial, wa, wb] normalised by the global fit"""
    A = {}; 
    for t in (0, 1):
        X = []; Yv = []; O = []
        for _, pl in D:
            for n9, ga, gb, e in pl:
                if int(n9) != t: continue
                if mirror: s = np.where(ga != 0, np.sign(ga), np.where(gb != 0, np.sign(gb), 1)); ga, gb = s * ga, s * gb
                X.append(np.stack([np.abs(ga), np.abs(gb), np.ones_like(ga)], 1)); Yv.append(np.abs(e)); O.append(octant(ga, gb))
        X = np.concatenate(X).astype(np.float64); Yv = np.concatenate(Yv).astype(np.float64); O = np.concatenate(O)
        wg = np.linalg.lstsq(X, Yv, rcond=None)[0]; W = (wg[0] + wg[1]) / 2
        r = X[:, 0] + X[:, 1]; rg = np.linalg.lstsq(np.stack([r, X[:, 2]], 1), Yv, rcond=None)[0][0]
        for o in range(8):
            m = O == o
            if m.sum() < 1000: A[t, o] = (1.0, 1.0, 1.0); continue
            w = np.linalg.lstsq(X[m], Yv[m], rcond=None)[0]
            wr = np.linalg.lstsq(np.stack([r[m], X[m, 2]], 1), Yv[m], rcond=None)[0][0]
            A[t, o] = (max(wr / rg, 0.05), max(w[0] / W, 0.05), max(w[1] / W, 0.05))
    return A
def cost(D, A, mirror, mode):
    tot = 0.0; px = 0
    for n, pl in D:
        px += n
        for n9, ga, gb, e in pl:
            t = int(n9); s = 1
            if mirror: s = np.where(ga != 0, np.sign(ga), np.where(gb != 0, np.sign(gb), 1)); ga, gb = s * ga, s * gb
            ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
            fa = np.ones(ga.shape); fb = np.ones(ga.shape)
            if A is not None:
                o = octant(ga, gb); k = {'radial': (0, 0), 'axis': (1, 2)}[mode]
                ta = np.array([A[t, i][k[0]] for i in range(8)]); tb = np.array([A[t, i][k[1]] for i in range(8)]); fa, fb = ta[o], tb[o]
            cls = (qsf(ga * fa) + 3) * 7 + qsf(gb * fb) + 3
            tot += sum(H(z[cls == v]) for v in np.unique(cls))
    return tot / px
def main(files):
    D = load(files)
    for mirror, nm in ((False, '49 sectors, 8 octants'), (True, 'tiny mirror 25, 4 octants')):
        b = cost(D, None, mirror, None); A = fit(D, mirror); print(f'{nm}: baseline {b:.4f} bpp')
        for t, tn in ((0, 'Y'), (1, 'UV')):
            print(f'   {tn} scales radial / (wa, wb) per octant:', '  '.join(f'{o}:{A[t, o][0]:.2f}/({A[t, o][1]:.2f},{A[t, o][2]:.2f})' for o in range(8) if A[t, o] != (1.0, 1.0, 1.0)))
        for mode in ('radial', 'axis'):
            c = cost(D, A, mirror, mode); print(f'   {mode:6s} {c:.4f} bpp  {100 * c / b - 100:+.2f}%')
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
