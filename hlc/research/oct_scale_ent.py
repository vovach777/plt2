#!/usr/bin/env python3
"""Per-direction scale fitted directly on entropy (not on |e|): tiny-sign mirror, 4 octants after the mirror
(c-b >= 0 / < 0 x |c-a| >= |c-b| or not), one radial scale per octant and plane type; coordinate descent over a grid.
Cost = per-file per-plane zero-order entropy of the 25 q7 cells (edges 1 4 16 on the scaled axes). In-sample.
usage: python oct_scale_ent.py <ppm/png files or globs...>"""
import numpy as np, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uv_axes import shift, E7
from PIL import Image
GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 2.0]
def qsf(x): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in E7)
def load(files):
    P_ = {0: [], 1: []}; px = 0
    for fi, f in enumerate(files):
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        G = a0[..., 1]; Vv = a0[..., 0] - G; U = a0[..., 2] - G; Y = G + ((U + Vv) >> 2); px += Y.size
        for pi, (P, n9) in enumerate([(Vv, True), (U, True), (Y, False)]):
            a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); ga, gb = (c - a).ravel(), (c - b).ravel()
            s = np.where(ga != 0, np.sign(ga), np.where(gb != 0, np.sign(gb), 1)); ga, gb = s * ga, s * gb
            ee = s * e.ravel(); ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
            o = (gb < 0) * 2 + (np.abs(ga) < np.abs(gb)); P_[int(n9)].append((fi * 3 + pi, ga, gb, z, o))
    return P_, px
def ent(cnt): c = cnt[cnt > 0]; return float(-(c * np.log2(c / c.sum())).sum())
def cost(L, sc):
    t = 0.0
    for key, ga, gb, z, o in L:
        f = sc[o]; cls = ((qsf(ga * f) + 3) * 7 + qsf(gb * f) + 3).astype(np.int64)
        h = np.bincount(cls * 512 + z, minlength=49 * 512).reshape(49, 512); t += sum(ent(r) for r in h)
    return t
def main(files):
    P_, px = load(files); T0 = T1 = 0.0
    for t, tn in ((0, 'Y'), (1, 'UV')):
        L = P_[t]; sc = np.ones(4); b = cost(L, sc); best = b
        for r in range(3):
            moved = False
            for k in range(4):
                for g in GRID:
                    if g == sc[k]: continue
                    s2 = sc.copy(); s2[k] = g; c = cost(L, s2)
                    if c < best: best, sc, moved = c, s2, True
            print(f'  {tn} round {r + 1}: scales ext|a|>=|b| {sc[0]}, ext|a|<|b| {sc[1]}, ramp|a|>=|b| {sc[2]}, ramp|a|<|b| {sc[3]}  {100 * best / b - 100:+.2f}%', flush=True)
            if not moved: break
        T0 += b; T1 += best
    print(f'total: baseline {T0 / px:.4f} bpp -> fitted {T1 / px:.4f} bpp  ({100 * T1 / T0 - 100:+.2f}%)')
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
