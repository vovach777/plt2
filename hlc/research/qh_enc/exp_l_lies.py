# exp_l_lies.py - does the L taken from the block above "lie"? Per coded block, the cost (letters + tails + planes, format
# tables, table t from sa as now) with the causal L of the format vs with the best L for that block (oracle, 0..7, the
# L itself not counted). Shares of blocks where the causal L is too high / too low, and the bits lost. usage: DUMPER img...
import sys, os, numpy as np
sys.argv, _a = sys.argv[:1] + ['x', 'y'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_pred.py')).read().split("dm, dw = sys.argv[1], sys.argv[2]")[0])
dumper, imgs = _a[1], _a[2:]
def blkcost(z, t, L):
    h = z >> L[:, None]; lt = qhsim.letters(h); lb = T['lens'][t[:, None], lt].sum(1)
    k = T['rk'][t][:, None]; x = np.maximum(h - 3, 0); q = x >> k
    tb = np.where(h >= 3, np.where(q >= T['QESC'], T['QESC'] + 10, q + 1 + k), 0).sum(1)
    return lb + tb + 16 * L
S = dict(n=0, cur=0.0, orc=0.0, hi=0, lo=0, nb=0, lost_hi=0.0, lost_lo=0.0, tot=0.0)
for img in imgs:
    d, c, _ = costs(dumper, img); k = ~d['skip']; z = d['z'].astype(np.int64)[k]; t = T['cm'][d['ptype'] * T['NQ'] + qhsim.Qsa(d['sa'])][k]
    Lc = d['Lo'][k]; C = np.stack([blkcost(z, t, np.full(len(z), L)) for L in range(8)], 1)
    cur = C[np.arange(len(z)), Lc]; Lb = C.argmin(1); best = C.min(1)
    S['n'] += d['npx']; S['cur'] += cur.sum(); S['orc'] += best.sum(); S['nb'] += len(z); S['tot'] += c.sum()
    S['hi'] += (Lc > Lb).sum(); S['lo'] += (Lc < Lb).sum()
    S['lost_hi'] += (cur - best)[Lc > Lb].sum(); S['lost_lo'] += (cur - best)[Lc < Lb].sum()
n = S['n']
print(f"blocks {S['nb']}: causal L too high {100 * S['hi'] / S['nb']:.1f}% (lost {S['lost_hi'] / n:.4f} bpp), too low {100 * S['lo'] / S['nb']:.1f}% "
      f"(lost {S['lost_lo'] / n:.4f} bpp); blocks cost {S['cur'] / n:.4f} -> oracle L {S['orc'] / n:.4f} bpp = {100 * (S['orc'] - S['cur']) / S['tot']:+.1f}% of the file (model {S['tot'] / n:.4f} bpp)")
