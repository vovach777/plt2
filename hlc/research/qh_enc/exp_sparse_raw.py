# exp_sparse_raw.py - Argon's "Sparse-Raw" (zero-pinned shift) checked with the PLR cost model on whole images.
# Format block: h = z >> L, 16*L raw bits. Sparse-Raw(Ls): h = 0 if z == 0 else ((z-1) >> Ls) + 1, Ls raw bits only for z > 0
# (the decoder knows which h are 0 from the letters + tails). Letters/tails with the format tables (same context t),
# escape: format (24 ones + 10 bits) or (8,2) = 8 ones + EG2 (Fable).
# Variants: B2 = causal trigger (any of above / left / above-left blocks bimodal: max z >= 64 and no z in [1,31]) -> Sparse(5),
# B3 = 1 bit per tile, oracle: whole tile format or Sparse(5); B3* = per tile best of format / Sparse(Ls), Ls 0..7 (+4 bits).
# usage: exp_sparse_raw.py DUMPER NAME img...
import sys, os, numpy as np
sys.argv, _a = sys.argv[:1] + ['x', 'y'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_pred.py')).read().split("dm, dw = sys.argv[1], sys.argv[2]")[0])
from lib import bwv
dumper, name, imgs = _a[1], _a[2], _a[3:]
def tail(h, k, rule):
    x = np.maximum(h - 3, 0); q = x >> k
    if rule == 'fmt': b = np.where(q >= 24, 34, q + 1 + k)
    else: v = np.maximum(x - (8 << k), 0); b = np.where(q >= 8, 8 + 2 * bwv((v >> 2) + 1) - 1 + 2, q + 1 + k)
    return np.where(h >= 3, b, 0).sum(1)
def cost(h, raw, t, rule): return T['lens'][t[:, None], qhsim.letters(h)].sum(1) + tail(h, T['rk'][t][:, None], rule) + raw
S = {}
for img in imgs:
    d, c, parts = costs(dumper, img); k = ~d['skip']; z = d['z'].astype(np.int64); n = len(z); idx = np.arange(n)
    other = parts['runs'] + parts['split']; t = T['cm'][d['ptype'] * T['NQ'] + qhsim.Qsa(d['sa'])]; tile = d['tile']; nt = tile.max() + 1
    pk = tile.astype(np.int64) * 4 + d['pl']; step = np.maximum(d['tw'] // 16, 1); bi = d['bi']
    bim = (z.max(1) >= 64) & ~((z >= 1) & (z < 32)).any(1)
    def nb(off, cond):
        j = np.clip(idx - off, 0, n - 1); return cond & (pk[j] == pk) & (bi[j] == bi - off) & bim[j]
    trig = nb(step, bi >= step) | nb(1, bi % step > 0) | nb(step + 1, (bi >= step) & (bi % step > 0))
    L = d['Lo']
    for rule in ('fmt', '82'):
        f = np.where(k, cost(z >> L[:, None], 16 * L, t, rule), 0)
        sp = {}
        for Ls in range(8):
            h = np.where(z > 0, ((z - 1) >> Ls) + 1, 0); sp[Ls] = np.where(k, cost(h, Ls * (z > 0).sum(1), t, rule), 0)
        ft = np.bincount(tile, f, nt); b2 = np.where(trig, sp[5], f).sum()
        b3 = np.minimum(ft, np.bincount(tile, sp[5], nt)).sum() + nt
        b3s = np.minimum(ft, np.min([np.bincount(tile, sp[l], nt) for l in range(8)], 0)).sum() + 4 * nt
        for key, v in (('F', f.sum()), ('B2', b2), ('B3', b3), ('B3*', b3s)):
            S[(rule, key)] = S.get((rule, key), 0) + v + other
    S['npx'] = S.get('npx', 0) + d['npx']
n = S['npx']; base = S[('fmt', 'F')]
print(name, '  '.join('%s/%s %.4f (%+.2f%%)' % (r, kk, S[(r, kk)] / n, 100 * (S[(r, kk)] / base - 1)) for r in ('fmt', '82') for kk in ('F', 'B2', 'B3', 'B3*')))
