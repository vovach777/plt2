# grad_look.py - who pays in the coded blocks of a set: by sa class, zero / non-zero, equal to the 16 z above
import sys, numpy as np, qhsim, lib
T = qhsim.tables()
for s in sys.argv[1:]:
    d, c = qhsim.baseline(s, T); lb, tb, pb = qhsim.block_bits(d, T); n = d['npx']; k = ~d['skip']
    z = d['z'].astype(np.int64); zero = (z == 0).all(1); nz = (z != 0).sum(1); sa = d['sa']
    print('==', s, 'blocks', len(sa), 'zero %.1f%%' % (100 * zero.mean()), 'coded %.1f%%' % (100 * k.mean()))
    print('  sa class: share of coded blocks | zero among them | mean letter bits zero / non-zero | bpp of the class (letters+tails)')
    for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 31), (32, 1 << 30)]:
        m = k & (sa >= lo) & (sa <= hi)
        if not m.any(): continue
        mz = m & zero; mn = m & ~zero
        print('  sa %4d..%-6s %5.1f%%  zero %5.1f%%  bits %5.2f / %5.2f  nz-symbols in non-zero %4.2f  bpp %.4f' % (lo, hi if hi < 1 << 30 else '', 100 * m.sum() / k.sum(), 100 * mz.sum() / m.sum(),
              lb[mz].mean() if mz.any() else 0, (lb[mn] + tb[mn]).mean() if mn.any() else 0, nz[mn].mean() if mn.any() else 0, (lb[m] + tb[m]).sum() / n))
