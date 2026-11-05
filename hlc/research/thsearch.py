# Threshold search for HLC5 contexts: static (pctx) vs best common static vs per-file greedy.
# Cost = zero-order entropy per stream with per-file tables (header cost ignored, ~same for all variants).
import numpy as np, sys, glob, os
from PIL import Image
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ctxest.py')).read().split('def run')[0])
AC = 512                                                   # act capped (thresholds never go that high)

def file_hist(f):
    """cumulative histograms over act: CY[a][z] = #(act < a, zz = z) for Y, CC for V+U; z alphabet 512"""
    HY = np.zeros((AC + 1, 512), np.int64); HC = np.zeros((AC + 1, 512), np.int64)
    for ci, (P, n9) in enumerate(planes(f)):
        e, a, b, c, d, aa = resid(P, n9); z = np.where(e >= 0, 2 * e, -2 * e - 1)
        act = np.minimum(np.abs(a - c) + np.abs(c - b) + np.abs(a - aa), AC - 1)
        h = np.zeros(AC * 512, np.int64); np.add.at(h, act.ravel() * 512 + z.ravel(), 1)
        (HC if n9 else HY)[1:] += np.cumsum(h.reshape(AC, 512), axis=0)
    return HY, HC

def ent(h):
    n = h.sum(-1, keepdims=True); hp = np.where(h > 0, h, 1)
    return float(-(h * np.log2(hp / np.maximum(n, 1))).sum())

def cost(H, ty, tc):
    """H = list of (HY, HC); ty, tc = 5 thresholds; summed per file (per-file tables)"""
    tot = 0.0
    for HY, HC in H:
        by = [0] + list(ty) + [AC]; bc = [0] + list(tc) + [AC]
        s0 = (HY[by[1]] - HY[0]) + (HC[bc[1]] - HC[0])
        tot += ent(s0)
        for j in range(1, 6): tot += ent(HY[by[j + 1]] - HY[by[j]]) + ent(HC[bc[j + 1]] - HC[bc[j]])
    return tot

def greedy(H, ty, tc, rounds=4):
    th = list(ty) + list(tc); best = cost(H, th[:5], th[5:])
    for _ in range(rounds):
        moved = False
        for i in range(10):
            lo = (th[i - 1] + 1) if i % 5 else 1; hi = (th[i + 1] - 1) if i % 5 != 4 else AC - 1
            cands = sorted(set(int(v) for v in np.unique(np.geomspace(lo, max(lo, hi), 24).round())) | {th[i]})
            for v in cands:
                if v == th[i] or v < lo or v > hi: continue
                t2 = th.copy(); t2[i] = v; c = cost(H, t2[:5], t2[5:])
                if c < best: best, th, moved = c, t2, True
        if not moved: break
    return th[:5], th[5:], best

files = sorted(glob.glob(sys.argv[1])); px = {}; H = {}
for f in files: H[f] = file_hist(f); w, h = Image.open(f).size; px[f] = w * h
TY, TC = [1, 3, 10, 18, 35], [1, 4, 8, 16, 30]
sy, sc, _ = greedy(list(H.values()), TY, TC)
print(f'best common static: Y {sy}  UV {sc}')
tot = np.zeros(3); P = 0
print(f'{"photo":6s} {"pctx":>7s} {"common":>7s} {"per-file":>8s}   per-file thresholds (Y | UV)')
for f in files:
    b0 = cost([H[f]], TY, TC); b1 = cost([H[f]], sy, sc); fy, fc, b2 = greedy([H[f]], sy, sc)
    n = px[f]; tot += [b0, b1, b2]; P += n
    print(f'{os.path.basename(f)[:6]:6s} {b0 / n:7.4f} {b1 / n:7.4f} {b2 / n:8.4f}   {fy} | {fc}   ({100 * b1 / b0 - 100:+.2f}% {100 * b2 / b0 - 100:+.2f}%)')
print(f'TOTAL  {tot[0] / P:7.4f} {tot[1] / P:7.4f} {tot[2] / P:8.4f}   common {100 * tot[1] / tot[0] - 100:+.2f}%  per-file {100 * tot[2] / tot[0] - 100:+.2f}% vs pctx;  per-file vs common {100 * tot[2] / tot[1] - 100:+.2f}%')
