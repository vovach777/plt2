# exp_triple.py - HALIC-like colour plane triples inside PLR, costed with the PLR cost model (exp_pred.costs: L planes,
# QH letters with the format tables, tails, zero-block runs, zU split as its expected truncated-binary length).
# Basis (an assumption; HALIC's own definitions are not known): Co = R-G, Cg = B-G, D = R-B, Y = G + ((Co+Cg+2)>>2);
# with it all 17 triples of the HALIC mode table are invertible. The first plane goes to PLR's 8-bit Y path, the other
# two to the int16 pair (S = zU + zV). Predictor: MED everywhere (as HALIC FAST). Dumpers: palic2r -DPLR_DUMP
# -DPLR_NO_QH -DPLR_TRIPLE=m (m = 0..16) in TRI_DIR/d<m>, and the plain dump build (our RCT) as OURS.
# Reports per image: our RCT, fixed mode 12 (= HALIC FAST's), the best fixed mode, and the per-tile oracle choice
# among the 17 (+5 bits per 288x288 tile), the per-tile oracle among HALIC's 4 predictors (+2 bits), and both (+7 bits;
# dumpers d<m>p<2..4> built with -DPLR_PRED=2..4), next to HALIC normal / FAST (bpp from HALIC_TABLE, a markdown table).
# usage: exp_triple.py OURS_DUMPER TRI_DIR HALIC_TABLE img...
import sys, os, re, numpy as np
sys.argv, _a = sys.argv[:1] + ['x', 'y'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_pred.py')).read().split("dm, dw = sys.argv[1], sys.argv[2]")[0])
ours, tri, htab, imgs = _a[1], _a[2], _a[3], _a[4:]
H = {}
for line in open(htab):
    m = re.match(r'\|\s*(\S+)\s*\|\s*\S+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', line)
    if m: H[m.group(1)] = (float(m.group(3)), float(m.group(4)))           # HALIC normal, FAST
# index: 0 = ours; 1 + p * 17 + m = mode m with predictor p (0 MED, 1 (2(a+b)-c)/3, 2 (a+b+1)/2, 3 none)
dumpers = [ours] + [os.path.join(tri, 'd%d' % m + ('' if p == 0 else 'p%d' % (p + 1))) for p in range(4) for m in range(17)]
print('%-30s %7s %7s %7s %9s %9s %9s | %7s %7s | tiles won by predictors (joint)' % ('file', 'ours', 'mode12', 'bestfix', 'orcl-mode', 'orcl-pred', 'orcl-both', 'HALIC', 'FAST'))
tot = np.zeros(6); hal = np.zeros(2); npx = 0
for img in imgs:
    per_tile = []; d0 = None
    for dmp in dumpers:
        d, c, _ = costs(dmp, img); nt = d['tile'].max() + 1
        per_tile.append(np.bincount(d['tile'], c, nt)); d0 = d0 or d
    P = np.stack(per_tile); n = d['npx']; nt = P.shape[1]; M = P[1:].reshape(4, 17, nt)
    o, m12 = P[0].sum(), M[0, 12].sum(); fx = M[0].sum(1); bf = int(np.argmin(fx))
    om = M[0].min(0).sum() + 5 * nt                     # modes, MED
    op = M[:, 12].min(0).sum() + 2 * nt                 # predictors, mode 12 (= our RCT on the totals)
    ob = M.reshape(68, nt).min(0).sum() + 7 * nt        # both
    pw = np.bincount(np.argmin(M.reshape(68, nt), 0) // 17, minlength=4)
    hn, hf = H.get(os.path.basename(img), (np.nan, np.nan))
    print('%-30s %7.4f %7.4f %7.4f %9.4f %9.4f %9.4f | %7.4f %7.4f | MED:%d jls2:%d avg:%d none:%d' % (os.path.basename(img)[:30], o / n, m12 / n, fx[bf] / n,
          om / n, op / n, ob / n, hn, hf, *pw), flush=True)
    tot += np.array([o, m12, fx[bf], om, op, ob]); npx += n; hal += np.array([hn, hf]) * n
print('%-30s %7.4f %7.4f %7.4f %9.4f %9.4f %9.4f | %7.4f %7.4f' % ('TOTAL', *(tot / npx), *(hal / npx)))
print('vs ours: modes %+.1f%%, predictors %+.1f%%, both %+.1f%%; gap ours -> HALIC %+.1f%%, FAST -> HALIC %+.1f%%' % (
    *(100 * (tot[i] / tot[0] - 1) for i in (3, 4, 5)), 100 * (hal[0] / tot[0] - 1), 100 * (hal[0] / hal[1] - 1)))
