# tails_gap.py - how far are the Rice(k_t) tails (h - 3) from a static per-table code? Train on one set, cost on another.
# ideal: -log2 p of the training counts (x capped at 31, above: +Rice escape); huff: Huffman <= 12 bits on x = 0..15 + ESC(rice of x - 16)
import sys, numpy as np, qhsim, lib
from lib import huff_len_row
T = qhsim.tables(); K = T['K']; XC = 16
def get(s):
    d, c = qhsim.baseline(s, T); k = ~d['skip']; t = T['cm'][d['ptype'] * T['NQ'] + qhsim.Qsa(d['sa'])][k]; h = d['h'][k]; m = h >= 3
    return d, c, np.repeat(t, 16).reshape(-1, 16)[m], (h[m] - 3)
def rice(x, k): q = x >> k; return np.where(q >= 24, 34, q + 1 + k)
dt, ct, tt, xt = get(sys.argv[1]); C = np.zeros((K, XC + 1)); np.add.at(C, (tt, np.minimum(xt, XC)), 1)
Hl = np.stack([huff_len_row(r, 12) for r in C]); Il = -np.log2((C + 0.5) / (C + 0.5).sum(1, keepdims=True))
for s in sys.argv[2:]:
    d, c, t, x = get(s); n = d['npx']; k = T['rk'][t]; base = rice(x, k).sum(); assert base == c['tail']
    esc = np.where(x >= XC, rice(np.maximum(x - XC, 0), k + 1), 0)
    hb = (Hl[t, np.minimum(x, XC)] + esc).sum(); ib = (Il[t, np.minimum(x, XC)] + esc).sum(); tot = sum(c.values())
    print('%-10s tails: Rice %.4f bpp | Huffman(17) %.4f (%+.2f%% of the file) | ideal static %.4f (%+.2f%% of the file) | tails per coded block %.2f' % (s, base / n, hb / n, 100 * (hb - base) / tot, ib / n, 100 * (ib - base) / tot, len(x) / (~d['skip']).sum()))
