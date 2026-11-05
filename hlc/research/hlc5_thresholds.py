#!/usr/bin/env python3
"""HLC5 context thresholds: fit one common set on a corpus (e.g. CLIC), compare with the pctx set.

usage:  python hlc5_thresholds.py <dir> [-max N] [-perfile]
        needs numpy + Pillow; reads .ppm/.pnm/.png (RGB)

Model = HLC5 exactly: RCT V=R-G, U=B-G, Y=G+((U+V)>>2); MED with raster borders (x=0 from above, y=0 from left);
residual wrapped (Y mod 256, U/V mod 512), zigzag; act = |a-c| + |c-b| + |a-aa| with x=0: a=c=b, x<=1: aa=a,
y=0: b=c=0; class = #thresholds <= act; 11 streams (class 0 shared, Y 1..5, U/V 1..5).
Cost = zero-order entropy per stream with tables per file (FSE headers ignored, ~equal for all sets).
Memory: ~1 MB per file (cumulative histograms over act, int32).
"""
import sys, os, glob
import numpy as np
from PIL import Image

AC = 256
PCTX_Y, PCTX_C = [1, 3, 10, 18, 35], [1, 4, 8, 16, 30]

def planes(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    V = R - G; U = B - G; Y = G + ((U + V) >> 2)
    return [(V, True), (U, True), (Y, False)]

def file_hist(path):
    HY = np.zeros((AC + 1, 512), np.int32); HC = np.zeros((AC + 1, 512), np.int32); n = 0
    for P, n9 in planes(path):
        H, W = P.shape; n = H * W
        Z = np.zeros((H + 1, W + 1), np.int32); Z[1:, 1:] = P
        a = Z[1:, :-1].copy(); b = Z[:-1, 1:].copy(); c = Z[:-1, :-1].copy()
        a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
        aa = a.copy(); aa[:, 2:] = P[:, :-2]
        mn = np.minimum(a, b); mx = np.maximum(a, b); e = P - np.clip(a + b - c, mn, mx)
        e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
        z = np.where(e >= 0, 2 * e, -2 * e - 1)
        act = np.minimum(np.abs(a - c) + np.abs(c - b) + np.abs(a - aa), AC - 1)
        h = np.bincount((act * 512 + z).ravel(), minlength=AC * 512).reshape(AC, 512)
        (HC if n9 else HY)[1:] += np.cumsum(h, axis=0).astype(np.int32)
    return HY, HC, n

def ent(h):
    h = h[h > 0].astype(np.float64); return float(-(h * np.log2(h / h.sum())).sum()) if h.size else 0.0

def cost(HY, HC, ty, tc):
    by = [0] + list(ty) + [AC]; bc = [0] + list(tc) + [AC]
    t = ent((HY[by[1]] - HY[0]).astype(np.int64) + (HC[bc[1]] - HC[0]))
    for j in range(1, 6): t += ent(HY[by[j + 1]] - HY[by[j]]) + ent(HC[bc[j + 1]] - HC[bc[j]])
    return t

def total(F, ty, tc): return sum(cost(HY, HC, ty, tc) for HY, HC, _ in F)

def greedy(F, ty, tc, rounds=4, log=False):
    th = list(ty) + list(tc); best = total(F, th[:5], th[5:])
    for r in range(rounds):
        moved = False
        for i in range(10):
            lo = (th[i - 1] + 1) if i % 5 else 1; hi = (th[i + 1] - 1) if i % 5 != 4 else AC - 1
            if hi < lo: continue
            for v in sorted(set(int(x) for x in np.geomspace(lo, hi, 24).round()) | set(range(max(lo, th[i] - 2), min(hi, th[i] + 2) + 1))):
                if v == th[i]: continue
                t2 = th.copy(); t2[i] = v; c = total(F, t2[:5], t2[5:])
                if c < best: best, th, moved = c, t2, True
        if log: print(f'  round {r + 1}: Y {th[:5]}  UV {th[5:]}', flush=True)
        if not moved: break
    return th[:5], th[5:], best

def main():
    args = sys.argv[1:]; mx = 0; perfile = '-perfile' in args
    if '-max' in args: i = args.index('-max'); mx = int(args[i + 1]); del args[i:i + 2]
    args = [a for a in args if a != '-perfile']
    if not args: print(__doc__); return
    files = []
    for a in args:
        if os.path.isdir(a): files += sorted(f for f in glob.glob(os.path.join(a, '*')) if f.lower().endswith(('.ppm', '.pnm', '.png')))
        else: files.append(a)
    if mx: files = files[:mx]
    F = []; px = 0
    for k, f in enumerate(files):
        HY, HC, n = file_hist(f); F.append((HY, HC, n)); px += n
        if (k + 1) % 50 == 0: print(f'  read {k + 1}/{len(files)}', flush=True)
    b0 = total(F, PCTX_Y, PCTX_C)
    print(f'{len(files)} files, {px / 1e6:.1f} Mpx')
    print(f'pctx   Y {PCTX_Y}  UV {PCTX_C}: {b0 / px:.4f} bpp')
    ty, tc, b1 = greedy(F, PCTX_Y, PCTX_C, log=True)
    print(f'common Y {ty}  UV {tc}: {b1 / px:.4f} bpp  ({100 * b1 / b0 - 100:+.2f}% vs pctx)')
    if perfile:
        b2 = 0.0
        for HY, HC, n in F: b2 += greedy([(HY, HC, n)], ty, tc, rounds=2)[2]
        print(f'per-file thresholds: {b2 / px:.4f} bpp  ({100 * b2 / b1 - 100:+.2f}% vs common)')

if __name__ == '__main__':
    main()
