# Per-tile context thresholds for HLC5: streams / tables stay per file, thresholds per T x T tile.
# Alternating fit: file models from current assignment -> each tile picks thresholds minimising its cross-entropy
# under those models (greedy) -> refit. Final cost = true per-file stream entropy + 80 bits per tile (10 bytes).
import numpy as np, sys, glob, os
from PIL import Image
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ctxest.py')).read().split('def run')[0])
AC = 256; T = int(sys.argv[2]) if len(sys.argv) > 2 else 128
SY, SC = [1, 4, 9, 19, 47], [1, 3, 7, 12, 25]                 # best common static (thsearch.py)
TY0, TC0 = [1, 3, 10, 18, 35], [1, 4, 8, 16, 30]              # pctx

def tile_cums(f):
    pl = planes(f); H, W = pl[0][0].shape; data = []
    for ci, (P, n9) in enumerate(pl):
        e, a, b, c, d, aa = resid(P, n9); z = np.where(e >= 0, 2 * e, -2 * e - 1)
        act = np.minimum(np.abs(a - c) + np.abs(c - b) + np.abs(a - aa), AC - 1); data.append((act, z, n9))
    tiles = []
    for y0 in range(0, H, T):
        for x0 in range(0, W, T):
            CY = np.zeros((AC + 1, 512)); CC = np.zeros((AC + 1, 512))
            for act, z, n9 in data:
                h = np.bincount((act[y0:y0 + T, x0:x0 + T] * 512 + z[y0:y0 + T, x0:x0 + T]).ravel(), minlength=AC * 512).reshape(AC, 512)
                (CC if n9 else CY)[1:] += np.cumsum(h, axis=0)
            tiles.append((CY, CC))
    return tiles, W * H

def class_hists(cum, ty, tc):
    CY, CC = cum; by = [0] + list(ty) + [AC]; bc = [0] + list(tc) + [AC]
    hs = [(CY[by[1]] - CY[0]) + (CC[bc[1]] - CC[0])]
    hs += [CY[by[j + 1]] - CY[by[j]] for j in range(1, 6)] + [CC[bc[j + 1]] - CC[bc[j]] for j in range(1, 6)]
    return hs

def ent(h): n = h.sum(); return float(-(h[h > 0] * np.log2(h[h > 0] / n)).sum()) if n else 0.0

def fit_file(tiles, init, iters=3, greedy_rounds=2):
    th = [list(init) for _ in tiles]
    for _ in range(iters):
        S = [np.zeros(512) for _ in range(11)]
        for cum, t in zip(tiles, th):
            for s, h in zip(S, class_hists(cum, t[:5], t[5:])): s += h
        L = [-np.log2((s + 0.5) / (s.sum() + 256)) for s in S]           # smoothed code lengths
        for i, cum in enumerate(tiles):
            def ce(t): return sum(float(h @ l) for h, l in zip(class_hists(cum, t[:5], t[5:]), L))
            t = th[i]; best = ce(t)
            for _ in range(greedy_rounds):
                moved = False
                for k in range(10):
                    lo = (t[k - 1] + 1) if k % 5 else 1; hi = (t[k + 1] - 1) if k % 5 != 4 else AC - 1
                    if hi < lo: continue
                    for v in sorted(set(np.geomspace(lo, hi, 12).round().astype(int))):
                        if v == t[k]: continue
                        t2 = t.copy(); t2[k] = int(v); c = ce(t2)
                        if c < best: best, t, moved = c, t2, True
                if not moved: break
            th[i] = t
    S = [np.zeros(512) for _ in range(11)]
    for cum, t in zip(tiles, th):
        for s, h in zip(S, class_hists(cum, t[:5], t[5:])): s += h
    return sum(ent(s) for s in S), th

def fixed(tiles, ty, tc):
    S = [np.zeros(512) for _ in range(11)]
    for cum in tiles:
        for s, h in zip(S, class_hists(cum, ty, tc)): s += h
    return sum(ent(s) for s in S)

files = sorted(glob.glob(sys.argv[1])); tot = np.zeros(4); P = 0; NT = 0
print(f'tile {T}x{T}; bits per pixel: pctx / common static / per-tile (+ 80 bit per tile header)')
for f in files:
    tiles, n = tile_cums(f)
    b0 = fixed(tiles, TY0, TC0); b1 = fixed(tiles, SY, SC); b2, th = fit_file(tiles, SY + SC); hb = 80 * len(tiles)
    tot += [b0, b1, b2, b2 + hb]; P += n; NT += len(tiles)
    ys = np.array([t[4] for t in th])
    print(f'{os.path.basename(f)[:6]:6s} {len(tiles):3d} tiles  {b0 / n:7.4f} {b1 / n:7.4f} {b2 / n:7.4f} (+hdr {(b2 + hb) / n:7.4f})  {100 * (b2 + hb) / b1 - 100:+6.2f}% vs common   top Y threshold per tile: min {ys.min()} med {int(np.median(ys))} max {ys.max()}', flush=True)
print(f'TOTAL {NT} tiles  pctx {tot[0] / P:.4f}  common {tot[1] / P:.4f}  per-tile {tot[2] / P:.4f}  with header {tot[3] / P:.4f}')
print(f'per-tile vs common: {100 * tot[2] / tot[1] - 100:+.2f}% (no header)  {100 * tot[3] / tot[1] - 100:+.2f}% (with header);  vs pctx {100 * tot[3] / tot[0] - 100:+.2f}%')
