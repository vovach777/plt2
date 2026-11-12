# variants.py - candidate format changes against the low-context weakness, costed on dumps (bits -> bpp of the set).
# Tables for the letter variants are retrained on TRAIN (same 32-table context map as the format, Huffman <= 11 bits)
# and the baseline is retrained the same way, so the difference is the idea alone. Tails / planes / runs as the format.
import sys, os, numpy as np, qhsim, lib
from lib import huff_len_row
T = qhsim.tables(); K = T['K']
def prep(s):
    d, c = qhsim.baseline(s, T); lb, tb, pb = qhsim.block_bits(d, T)
    d['t'] = T['cm'][d['ptype'] * T['NQ'] + qhsim.Qsa(d['sa'])]; d['lt'] = qhsim.letters(d['h']); d['lb'] = lb; d['tb'] = tb; d['pb'] = pb
    d['zero'] = (d['z'] == 0).all(1); d['k'] = ~d['skip']; d['base'] = sum(c.values())
    # the 16 z above (tile widths in these sets are multiples of 16: block index - tw / 16), same tile and plane
    step = d['tw'] // 16; idx = np.arange(len(step)) - step; ok = (d['bi'] >= step) & (d['tw'] % 16 == 0)
    d['eqab'] = ok & (d['z'] == d['z'][np.maximum(idx, 0)]).all(1)
    return d
def counts_letters(d, eob):
    C = np.zeros((K, 257)); k = d['k']; lt = d['lt'][k]; t = d['t'][k]
    if not eob: np.add.at(C, (np.repeat(t, 4), lt.ravel()), 1); return C
    last = np.where(lt != 0, np.arange(1, 5)[None, :], 0).max(1)                 # letters coded before the EOB (4 = no EOB)
    for p in range(4): m = last > p; np.add.at(C, (t[m], lt[m, p]), 1)
    np.add.at(C, (t[last < 4], 256), 1); return C
def lens_of(C, n): return np.stack([np.r_[huff_len_row(r[:n], 11), np.zeros(257 - n, np.int64)] for r in C])
def cost_letters(d, Ln, eob):
    k = d['k']; lt = d['lt'][k]; t = d['t'][k]
    if not eob: return Ln[t[:, None], lt].sum()
    last = np.where(lt != 0, np.arange(1, 5)[None, :], 0).max(1); b = 0
    for p in range(4): m = last > p; b += Ln[t[m], lt[m, p]].sum()
    return b + Ln[t[last < 4], 256].sum()
train, tests = sys.argv[1], sys.argv[2:]
dt = prep(train); Lb = lens_of(counts_letters(dt, False), 256); Le = lens_of(counts_letters(dt, True), 257)
print('train', train, ': EOB code length per table (bits):', Le[:, 256].tolist())
print('%-14s %8s %8s | %8s %8s | %s' % ('set', 'format', 'retrain', 'EOB', 'gain', 'flag variants: bpp change, eq-above flag sa 1..T / zero-block flag sa 1..T, T = 2, 8, 31'))
for s in tests:
    d = prep(s); n = d['npx']; k = d['k']; fixed = d['runbits'].sum() + d['tb'][k].sum() + d['pb'][k].sum()
    fmt = d['base']; b0 = fixed + cost_letters(d, Lb, False); b1 = fixed + cost_letters(d, Le, True)
    out = []
    for Tm in (2, 8, 31):
        m = k & (d['sa'] >= 1) & (d['sa'] <= Tm); blk = d['lb'] + d['tb'] + d['pb']
        ea = m.sum() - blk[m & d['eqab']].sum(); zf = m.sum() - blk[m & d['zero']].sum()   # 1 flag bit per block of the class; a hit costs nothing more
        out.append('%+.4f / %+.4f' % (ea / n, zf / n))
    print('%-14s %8.4f %8.4f | %8.4f %+7.2f%% | %s' % (s, fmt / n, b0 / n, b1 / n, 100 * (b1 - b0) / b0, '   '.join(out)))
    kk = k & ~d['zero']; print('     coded blocks: zero %.1f%%, equal to the 16 z above %.1f%% (of the non-zero coded ones %.1f%%)' % (100 * (k & d['zero']).sum() / k.sum(), 100 * (k & d['eqab']).sum() / k.sum(), 100 * (kk & d['eqab']).sum() / max(kk.sum(), 1)))
