# Per photo x 11 HLC5 buckets: FSE-quantised bits/sample when the table is built from 1/1, 1/2, 1/4 of the bucket
# (prefix in decoder order = top of the image), +4 pseudo-count for every symbol 0..max of the bucket in the
# sampled variants; and "top codes bottom": table of the first half (+4) on the second half vs its own table.
import numpy as np, sys, glob, os, csv
from PIL import Image
THY, THC = [1, 3, 10, 18, 35], [1, 4, 8, 16, 30]
NAMES = ['0'] + [f'Y{j}' for j in range(1, 6)] + [f'UV{j}' for j in range(1, 6)]

def streams(f):
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2)
    H, W = Y.shape; L = W + 7
    yy, xx = np.mgrid[0:H, 0:W]; band, lane = yy >> 3, yy & 7; t = xx + lane
    S = [[] for _ in range(11)]; K = [[] for _ in range(11)]
    for ci, (P, n9) in enumerate([(V, True), (U, True), (Y, False)]):
        Z = np.zeros((H + 1, W + 1), np.int32); Z[1:, 1:] = P
        a_ = Z[1:, :-1].copy(); b = Z[:-1, 1:].copy(); c = Z[:-1, :-1].copy(); a_[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
        aa = a_.copy(); aa[:, 2:] = P[:, :-2]
        e = P - np.clip(a_ + b - c, np.minimum(a_, b), np.maximum(a_, b))
        e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
        z = np.where(e >= 0, 2 * e, -2 * e - 1)
        act = np.abs(a_ - c) + np.abs(c - b) + np.abs(a_ - aa)
        k = sum((act >= th).astype(np.int32) for th in (THC if n9 else THY))
        key = ((band.astype(np.int64) * L + t) * 3 + ci) * 8 + lane                   # decoder order
        for j in range(6):
            sid = 0 if j == 0 else (j if ci == 2 else 5 + j); m = k == j
            S[sid].append(z[m]); K[sid].append(key[m])
    out = []
    for s, kk in zip(S, K):
        s = np.concatenate(s); kk = np.concatenate(kk); out.append(s[np.argsort(kk, kind='stable')])
    return out, H * W

def bw(v): return int(v).bit_length()
def tablelog(n, mx):
    """hlc tcap + FSE_optimalTableLog(tcap, n, mx)"""
    tl = min(12, max(9, bw(n) - 1 - 4)); maxsrc = bw(n - 1) - 1 - 2
    tl = min(tl, maxsrc); minb = min(bw(n) - 1 + 1, bw(mx) - 1 + 2); tl = max(tl, minb); return min(12, max(5, tl))
def norm(cnt, tl):
    """FSE-like normalisation to 2^tl: every present symbol >= 1"""
    tot = 1 << tl; n = cnt.sum(); p = np.where(cnt > 0, np.maximum(1, np.round(cnt * tot / n)), 0).astype(np.int64)
    while p.sum() > tot: i = int(np.argmax(p)); p[i] -= min(p.sum() - tot, p[i] - 1)
    p[np.argmax(p)] += tot - p.sum(); return p
def cost(sym, cnt, tl):
    """bits to code sym with table normalised from cnt (cnt must cover every symbol of sym)"""
    p = norm(cnt, tl); assert (p[sym] > 0).all(); return float(-np.log2(p[sym] / (1 << tl)).sum())
def hist(s, A, extra=0): return np.bincount(s, minlength=A).astype(np.float64) + extra

files = sorted(sum((glob.glob(a) if '*' in a else [a] for a in sys.argv[1:]), []))
rows = []; agg = {nm: np.zeros(6) for nm in NAMES}          # n, own, half, quarter, bot_own, bot_by_top
for f in files:
    St, px = streams(f); ph = os.path.basename(f).split('.')[0]; tot = np.zeros(4); ntot = sum(len(s) for s in St)
    for i, s in enumerate(St):
        n = len(s)
        if n == 0: continue
        A = 512; mx = int(s.max()); tl = tablelog(n, mx)
        pad = np.zeros(A); pad[:mx + 1] = 4                     # +4 reserve for every symbol 0..max of the bucket
        own = cost(s, hist(s, A), tl)
        hlf = cost(s, hist(s[:max(1, n // 2)], A) + pad, tl)
        qtr = cost(s, hist(s[:max(1, n // 4)], A) + pad, tl)
        top, bot = s[:n // 2], s[n // 2:]
        bo = cost(bot, hist(bot, A), tablelog(len(bot), mx)) if len(bot) else 0
        bt = cost(bot, hist(top, A) + pad, tablelog(len(bot), mx)) if len(bot) else 0
        rows.append([ph, NAMES[i], n, 100 * n / ntot, own / n, hlf / n, qtr / n, bo / max(1, len(bot)), bt / max(1, len(bot)), 100 * (bt / bo - 1) if bo else 0])
        agg[NAMES[i]] += [n, own, hlf, qtr, bo, bt]; tot += [own, hlf, qtr, 0]
    print(f'{ph:5s}  bpp  1/1 {tot[0] / px:7.4f}   1/2 {tot[1] / px:7.4f} ({100 * tot[1] / tot[0] - 100:+.2f}%)   1/4 {tot[2] / px:7.4f} ({100 * tot[2] / tot[0] - 100:+.2f}%)', flush=True)
with open(sys.argv[0].replace('.py', '.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['photo', 'bucket', 'n', 'share%', 'bits_own_1/1', 'bits_tab_1/2+4', 'bits_tab_1/4+4', 'bottom_own', 'bottom_by_top+4', 'top->bottom overhead %'])
    for r in rows: w.writerow([r[0], r[1], r[2]] + [f'{x:.4f}' for x in r[3:]])
print('\nALL PHOTOS per bucket: share | bits/sample own 1/1 | table 1/2+4 | table 1/4+4 | bottom: own -> by top table (overhead)')
N = sum(v[0] for v in agg.values())
for nm, v in agg.items():
    n = v[0]; hn = n / 2
    print(f'{nm:4s} {100 * n / N:5.1f}%   {v[1] / n:5.2f}  {v[2] / n:5.2f} ({100 * v[2] / v[1] - 100:+5.1f}%)  {v[3] / n:5.2f} ({100 * v[3] / v[1] - 100:+5.1f}%)   {v[4] / hn:5.2f} -> {v[5] / hn:5.2f} ({100 * v[5] / v[4] - 100:+5.1f}%)')
T = np.sum([v for v in agg.values()], axis=0)
print(f'ALL         {T[1] / T[0]:5.3f}  {T[2] / T[0]:5.3f} ({100 * T[2] / T[1] - 100:+5.2f}%)  {T[3] / T[0]:5.3f} ({100 * T[3] / T[1] - 100:+5.2f}%)   {T[4] / T[0] * 2:5.3f} -> {T[5] / T[0] * 2:5.3f} ({100 * T[5] / T[4] - 100:+5.2f}%)')
