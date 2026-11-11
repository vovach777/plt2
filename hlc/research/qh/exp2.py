# exp2.py - block-level causal contexts for static tables (ideal code lengths): which context closes the gap to RLGR?
import numpy as np
from lib import *
def qf(x, step=4): return np.floor(step * np.log2(np.asarray(x, np.float64) + 1)).astype(np.int64)
def ctx_L(d): return d['ptype'] * 8 + d['Lo'], 16
def ctx_fine(d, est, step=4, lo=6, hi=9):
    q = qf(est, step); base = qf(16 * 2.0 ** (d['Lo'] + 1), step)      # base: sa at the low edge of bin L (off 1)
    r = np.clip(q - base, -lo * step // 4, hi * step // 4 - 1) + lo * step // 4
    r = np.where(d['Lo'] == 0, np.minimum(q, (lo + hi) * step // 4 - 1), r)
    W = (lo + hi) * step // 4 + 6 * step
    r = np.where(d['Lo'] == 0, np.minimum(q, W - 1), r)
    return (d['ptype'] * 8 + d['Lo']) * W + r, 16 * W
EST = {
 'sa': lambda d: d['sa'],
 'sa+sl': lambda d: (d['sa'] + d['sl']) / 2,
 '2sa+sl+sal/2+sar/2': lambda d: (2 * d['sa'] + d['sl'] + (d['sal'] + d['sar']) / 2) / 4,
 '2sa+2sl+sal+sar': lambda d: (2 * d['sa'] + 2 * d['sl'] + d['sal'] + d['sar']) / 6,
 'sa+2sl+sar': lambda d: (d['sa'] + 2 * d['sl'] + d['sar']) / 4,
}
def run(test, train, off=1):
    d = prep(test, off); t = prep(train, off); n = d['npx']
    s = np.minimum(d['h'], A - 1); ts = np.minimum(t['h'], A - 1); esc = esc_bits(d['h']).sum() / n
    print(f'== test {test} / train {train}, off {off}: RLGR {d["bits"].sum() / n:.4f}; raw {16 * d["Lo"].sum() / n:.4f}')
    def rep(nm, cd, ct, nc):
        c = counts(cd, s, nc, A); tr = counts(ct, ts, nc, A)
        print('  %-34s ctx used %4d  self %.4f  cross %.4f' % (nm, (c.sum(1) > 0).sum(), ent(c) / n + esc, (c * ideal_len(tr)).sum() / n + esc))
    cd, nc = ctx_L(d); ct, _ = ctx_L(t); rep('L only', cd, ct, nc)
    for nm, f in EST.items():
        for step in (2, 4):
            cd, nc = ctx_fine(d, f(d), step); ct, _ = ctx_fine(t, f(t), step); rep(f'L x q({nm}) step 1/{step} oct', cd, ct, nc)
    # two-dimensional: q(sa) x q(sl)
    for step in (2,):
        c1, n1 = ctx_fine(d, d['sa'], step); c2 = np.clip(qf(d['sl'], step) - qf(d['sa'], step), -4, 4) + 4
        t1, _ = ctx_fine(t, t['sa'], step); t2 = np.clip(qf(t['sl'], step) - qf(t['sa'], step), -4, 4) + 4
        rep(f'L x q(sa) x q(sl/sa) 1/{step} oct', c1 * 9 + c2, t1 * 9 + t2, n1 * 9)
if __name__ == "__main__":
  for te, tr in [('kodak768', 'col19'), ('col19', 'kodak768'), ('gray', 'kodak768')]: run(te, tr)
