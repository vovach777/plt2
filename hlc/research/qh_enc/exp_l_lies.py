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
S = dict(own=0.0, own_sig=0.0, orc_sig=0.0, own_sig0=0.0, n=0, cur=0.0, orc=0.0, hi=0, lo=0, nb=0, lost_hi=0.0, lost_lo=0.0, tot=0.0)
for img in imgs:
    d, c, _ = costs(dumper, img); k = ~d['skip']; z = d['z'].astype(np.int64)[k]; t = T['cm'][d['ptype'] * T['NQ'] + qhsim.Qsa(d['sa'])][k]
    Lc = d['Lo'][k]; C = np.stack([blkcost(z, t, np.full(len(z), L)) for L in range(8)], 1)
    cur = C[np.arange(len(z)), Lc]; Lb = C.argmin(1); best = C.min(1)
    # PLN1 -m 1: L from the block's own mean (off 2), sent; signal cost = empirical conditional entropy given the causal L
    Lown = np.clip(np.floor(np.log2(np.maximum(z.sum(1) >> 4, 1))).astype(np.int64) + (z.sum(1) >> 4 > 0) - 2, 0, 7)
    def hcond(x, y):                                     # sum of -log2 p(x | y), per image (an adaptive / two-pass coder's cost)
        j = np.bincount(y * 8 + x, minlength=64).reshape(8, 8).astype(np.float64); r = j.sum(1, keepdims=True)
        p = np.where(j > 0, j / np.maximum(r, 1), 1); return float(-(j * np.log2(p)).sum())
    def h0(x):
        c = np.bincount(x, minlength=8).astype(np.float64); p = np.where(c > 0, c / c.sum(), 1); return float(-(c * np.log2(p)).sum())
    S['own'] += C[np.arange(len(z)), Lown].sum(); S['own_sig'] += hcond(Lown, Lc); S['own_sig0'] += h0(Lown); S['orc_sig'] += hcond(Lb, Lc)
    S['n'] += d['npx']; S['cur'] += cur.sum(); S['orc'] += best.sum(); S['nb'] += len(z); S['tot'] += c.sum()
    S['hi'] += (Lc > Lb).sum(); S['lo'] += (Lc < Lb).sum()
    S['lost_hi'] += (cur - best)[Lc > Lb].sum(); S['lost_lo'] += (cur - best)[Lc < Lb].sum()
n = S['n']
print(f"PLN1 -m 1 (own L, sent): blocks {S['own'] / n:.4f} + L {S['own_sig'] / n:.4f} (|causal; order-0 {S['own_sig0'] / n:.4f}) = {(S['own'] + S['own_sig']) / n:.4f} bpp "
      f"-> {100 * (S['own'] + S['own_sig'] - S['cur']) / S['tot']:+.1f}% of the file;  oracle L sent |causal: {(S['orc'] + S['orc_sig']) / n:.4f} -> {100 * (S['orc'] + S['orc_sig'] - S['cur']) / S['tot']:+.1f}%")
print(f"blocks {S['nb']}: causal L too high {100 * S['hi'] / S['nb']:.1f}% (lost {S['lost_hi'] / n:.4f} bpp), too low {100 * S['lo'] / S['nb']:.1f}% "
      f"(lost {S['lost_lo'] / n:.4f} bpp); blocks cost {S['cur'] / n:.4f} -> oracle L {S['orc'] / n:.4f} bpp = {100 * (S['orc'] - S['cur']) / S['tot']:+.1f}% of the file (model {S['tot'] / n:.4f} bpp)")
