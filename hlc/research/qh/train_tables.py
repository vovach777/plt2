# train_tables.py TRAINSET K MAXLEN OUT [TESTSET...] - static quad-Huffman tables ("QH") for the C++ prototype.
#  context of a block = (plane type, Q(sa)), sa = sum of the 16 z one row above (the number L is made from):
#    Q(sa) = sa for sa < 32, else 32 + (bw(sa) - 6) * 4 + two bits under the top bit   (68 levels)
#  letter = 4 neighbours of the block, each min(h, 3), 2 bits each (first symbol in the low bits) -> 256 letters, Huffman, len <= MAXLEN
#  every symbol with min(h,3) == 3 then sends x = h - 3 as Rice(k of the table), escape: 24 ones + 10 raw bits
import numpy as np, sys, struct
from lib import *
NQ = 68; QMAXQ = 24
def Q(sa):
    sa = np.asarray(sa, np.int64); e = bwv(sa); m = (sa >> np.maximum(e - 3, 0)) & 3
    return np.where(sa < 32, sa, np.minimum(32 + (e - 6) * 4 + m, NQ - 1))
def ctx_of(d): return d['ptype'] * NQ + Q(d['sa'])
def letters(h):
    c = np.minimum(h, 3).reshape(len(h), 4, 4); return c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)
def rice_esc_bits(x, k): q = x >> k; return np.where(q >= QMAXQ, QMAXQ + 10, q + 1 + k)
def cluster(tr, K):
    rows = [r.copy() for r in tr if r.sum() > 0]; ids = [[i] for i, r in enumerate(tr) if r.sum() > 0]
    def H(c): n = c.sum(); p = c[c > 0] / n; return -(c[c > 0] * np.log2(p)).sum()
    h = [H(r) for r in rows]; n = len(rows); D = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n): D[i, j] = H(rows[i] + rows[j]) - h[i] - h[j]
    alive = list(range(n))
    while len(alive) > K:
        sub = D[np.ix_(alive, alive)]; a, b = np.unravel_index(np.argmin(sub), sub.shape); i, j = alive[a], alive[b]
        rows[i] = rows[i] + rows[j]; ids[i] += ids[j]; h[i] = H(rows[i]); alive.remove(j)
        for x in alive:
            if x != i: lo, hi = min(x, i), max(x, i); D[lo, hi] = H(rows[i] + rows[x]) - h[i] - h[x]
    mp = -np.ones(len(tr), np.int64)
    for k, a in enumerate(alive): mp[ids[a]] = k
    return mp, np.stack([rows[a] for a in alive])
def train(trainset, K, maxlen, off=1):
    t = prep(trainset, off); ct = ctx_of(t); lt = letters(t['h'])
    trc = counts(ct, lt, 2 * NQ, 256); mp, tabs = cluster(trc, K)
    for c in np.where(mp < 0)[0]:              # unseen contexts: nearest seen one of the same plane type
        g = c // NQ; cand = [x for x in range(g * NQ, (g + 1) * NQ) if mp[x] >= 0] or [x for x in range(2 * NQ) if mp[x] >= 0]
        mp[c] = mp[min(cand, key=lambda x: abs(x - c))]
    K = len(tabs); lens = huff_len(tabs, maxlen).astype(np.int64); tt = mp[ct]
    ks = np.zeros(K, np.int64); x = np.maximum(t['h'] - 3, 0); m = t['h'] >= 3
    for k in range(K):
        sel = (tt == k)[:, None] & m; ks[k] = int(np.argmin([rice_esc_bits(x[sel], kk).sum() for kk in range(5)]))
    return dict(map=mp, lens=lens, ks=ks, K=K, maxlen=maxlen)
def cost(T, testset, off=1):
    d = prep(testset, off); tb = T['map'][ctx_of(d)]; l = letters(d['h'])
    lb = T['lens'][tb[:, None], l].sum(); x = np.maximum(d['h'] - 3, 0); m = d['h'] >= 3
    xb = np.where(m, rice_esc_bits(x, T['ks'][tb][:, None]), 0).sum()
    return lb, xb, d
if __name__ == '__main__':
    trainset, K, maxlen, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    T = train(trainset, K, maxlen)
    with open(out, 'wb') as f:
        f.write(struct.pack('<4i', T['K'], maxlen, NQ, QMAXQ)); f.write(T['map'].astype(np.uint8).tobytes())
        f.write(T['lens'].astype(np.uint8).tobytes()); f.write(T['ks'].astype(np.uint8).tobytes())
    print(f'tables from {trainset}: K={T["K"]} maxlen={maxlen} rice k per table {T["ks"].tolist()}  -> {out}')
    for ts in sys.argv[5:]:
        lb, xb, d = cost(T, ts); n = d['npx']; r = d['bits'].sum()
        print(f'  test {ts}: letters {lb / n:.4f} + excess {xb / n:.4f} = {(lb + xb) / n:.4f} bpp  vs RLGR {r / n:.4f}  ({100 * ((lb + xb) / r - 1):+.2f}% of the RLGR part, {100 * (lb + xb - r) / d["fbits"]:+.2f}% of the file -> {(d["fbits"] + lb + xb - r) / n:.4f} bpp)')
