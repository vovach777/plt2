# exp4.py - real code costs for static tables chosen by a causal block context, after clustering the contexts to K tables.
#  codes: ideal, tANS (counts normalised to 2^tl, per-table escape cap), Huffman on single symbols,
#         Huffman on pairs (cap 8 -> 64 letters + excess code), Huffman on quads (cap 4 -> 256 letters + excess code)
import numpy as np, sys
from lib import *
from exp2 import qf, ctx_fine, EST
def cluster(tr, K):                            # agglomerative: merge the pair of tables with the least entropy increase
    rows = [r.copy() for r in tr if r.sum() > 0]; ids = [[i] for i, r in enumerate(tr) if r.sum() > 0]
    def H(c): n = c.sum(); p = c[c > 0] / n; return -(c[c > 0] * np.log2(p)).sum()
    h = [H(r) for r in rows]
    while len(rows) > K:
        best = None
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                dlt = H(rows[i] + rows[j]) - h[i] - h[j]
                if best is None or dlt < best[0]: best = (dlt, i, j)
        _, i, j = best; rows[i] += rows[j]; ids[i] += ids[j]; h[i] = H(rows[i]); del rows[j], ids[j], h[j]
    mp = -np.ones(len(tr), np.int64)
    for k, g in enumerate(ids): mp[g] = k
    return mp, np.stack(rows)
def fill_map(mp, tr_all, tabs, W):             # contexts unseen in training: the table of the nearest seen context of the same (plane, L)
    mp = mp.copy()
    for c in np.where(mp < 0)[0]:
        g = c // W; cand = [x for x in range(g * W, (g + 1) * W) if mp[x] >= 0]
        if not cand: cand = [x for x in range(len(mp)) if mp[x] >= 0]
        mp[c] = mp[min(cand, key=lambda x: abs(x - c))]
    return mp
def tans_best(ctr_row, test_row, h_exc_bits_fn, tl):
    return None
def capped(ctr, E):                            # merge symbols >= E into one escape letter E
    out = ctr[:, :E + 1].copy(); out[:, E] = ctr[:, E:].sum(1); return out
def eg_bits(x, ke): y = (x >> ke) + 1; return 2 * (bwv(y) - 1) + 1 + ke
def evaluate(test, train, est='sa', step=2, K=16, off=1, verbose=True, tls=(9, 10, 11), ret=False):
    d = prep(test, off); t = prep(train, off); n = d['npx']
    cd, nc = ctx_fine(d, EST[est](d), step); ct, _ = ctx_fine(t, EST[est](t), step); W = nc // 16
    AA = 64; hd = d['h']; ht = t['h']
    trc = counts(ct, np.minimum(ht, AA - 1), nc, AA)
    mp, tabs = cluster(trc, K); mp = fill_map(mp, trc, tabs, W); K = len(tabs)
    td = mp[cd]; tt = mp[ct]
    res = {}
    cte = counts(td, np.minimum(hd, AA - 1), K, AA); ctr = counts(tt, np.minimum(ht, AA - 1), K, AA)
    big_d = np.bincount(td, eg_bits(np.maximum(hd - (AA - 1), 0), 2).sum(1) * 0 + np.where(hd >= AA - 1, eg_bits(np.maximum(hd - (AA - 1), 0), 2), 0).sum(1), K)   # symbols >= 63: EG2 after the last letter
    res['ideal (unclustered ctx)'] = (counts(cd, np.minimum(hd, AA - 1), nc, AA) * ideal_len(trc)).sum() + big_d.sum()
    res['ideal'] = (cte * ideal_len(ctr)).sum() + big_d.sum()
    # tANS with a per-table cap E: letters 0..E-1, E = escape followed by EG(ke) of h - E
    def esc_cost(cnt_row, E, ke): x = np.arange(E, AA); return (cnt_row[E:] * eg_bits(x - E, ke)).sum()
    for tl in tls:
        tot = 0
        for k in range(K):
            best = None
            for E in (3, 4, 6, 8, 12, 16, 24, 32, 47):
                for ke in (0, 1, 2):
                    ln = tans_len(capped(ctr[k:k + 1], E), tl)[0]
                    trc_ = (capped(ctr[k:k + 1], E)[0] * ln).sum() + esc_cost(ctr[k], E, ke)
                    if best is None or trc_ < best[0]: best = (trc_, E, ke, ln)
            _, E, ke, ln = best; tot += (capped(cte[k:k + 1], E)[0] * ln).sum() + esc_cost(cte[k], E, ke)
        res[f'tANS tl={tl}'] = tot + big_d.sum()
    # Huffman, single symbols
    def huff_single(maxlen):
        tot = 0
        for k in range(K):
            best = None
            for E in (8, 16, 24, 32, 47):
                for ke in (0, 2):
                    ln = huff_len(capped(ctr[k:k + 1], E), maxlen)[0]; c_ = (capped(ctr[k:k + 1], E)[0] * ln).sum() + esc_cost(ctr[k], E, ke)
                    if best is None or c_ < best[0]: best = (c_, E, ke, ln)
            _, E, ke, ln = best; tot += (capped(cte[k:k + 1], E)[0] * ln).sum() + esc_cost(cte[k], E, ke)
        return tot + big_d.sum()
    res['Huffman single, len<=12'] = huff_single(12)
    # letters: groups of g symbols capped at P; a symbol >= P-1 is followed by the excess code (own static Huffman per table)
    def letters(h, g, P):
        c = np.minimum(h, P - 1).reshape(len(h), 16 // g, g); l = np.zeros(c.shape[:2], np.int64)
        for j in range(g): l = l * P + c[:, :, j]
        return l
    def group_cost(g, P, maxlen):
        ld = letters(hd, g, P); lt = letters(ht, g, P); nl = P ** g
        cl_d = counts(td, ld, K, nl); cl_t = counts(tt, lt, K, nl)
        XA = 40
        def exc_counts(h, tb):
            m = h >= P - 1; x = np.minimum(h - (P - 1), XA - 1)
            return np.bincount((tb[:, None] * XA + x)[m], minlength=K * XA).reshape(K, XA).astype(np.float64)
        xd = exc_counts(hd, td); xt = exc_counts(ht, tt)
        hb = lambda h, tb: np.bincount(tb, np.where(h - (P - 1) >= XA - 1, eg_bits(np.maximum(h - (P - 1) - (XA - 1), 0), 2), 0).sum(1), K)
        per = (cl_d * huff_len(cl_t, maxlen)).sum(1) + (xd * huff_len(xt, 12)).sum(1) + hb(hd, td)
        per_tr = (cl_t * huff_len(cl_t, maxlen)).sum(1) + (xt * huff_len(xt, 12)).sum(1) + hb(ht, tt)
        per_id = (cl_d * ideal_len(cl_t)).sum(1) + (xd * ideal_len(xt)).sum(1) + hb(hd, td)
        return per, per_tr, per_id
    modes = {}
    for nm, g, P, ml in [('pair cap8', 2, 8, 12), ('pair cap16', 2, 16, 12), ('quad cap4', 4, 4, 12), ('quad cap4 len<=10', 4, 4, 10), ('quad cap4 len<=11', 4, 4, 11), ('quad cap3', 4, 3, 12)]:
        per, per_tr, per_id = group_cost(g, P, ml); modes[nm] = (per, per_tr); res[f'Huffman {nm}'] = per.sum(); res[f'  (ideal {nm})'] = per_id.sum()
    # best structure per table, chosen on the training set: single / pair / quad
    sing = []
    for k in range(K):
        ln = huff_len(capped(ctr[k:k + 1], 32), 12)[0]
        sing.append(((capped(cte[k:k + 1], 32)[0] * ln).sum() + esc_cost(cte[k], 32, 2), (capped(ctr[k:k + 1], 32)[0] * ln).sum() + esc_cost(ctr[k], 32, 2)))
    sing = np.array(sing)
    for q in ('quad cap4', 'quad cap4 len<=10'):
        alt = np.stack([sing[:, 1], modes['pair cap8'][1], modes[q][1]]); pick = alt.argmin(0)
        te_ = np.stack([sing[:, 0], modes['pair cap8'][0], modes[q][0]])
        res[f'Huffman best of single/pair/{q}'] = te_[pick, np.arange(K)].sum() + big_d.sum() * 0
        res[f'Huffman best of single/pair/{q}'] += 0
        if verbose: print('   structure per table (0 single, 1 pair, 2 quad):', pick.tolist(), ' P(0) per table:', np.round(ctr[:, 0] / ctr.sum(1), 2).tolist())
    for a_, b_ in (('quad cap4', 'pair cap16'), ('quad cap4 len<=11', 'pair cap16')):
        pick = np.stack([modes[a_][1], modes[b_][1]]).argmin(0); res[f'Huffman best of {a_} / {b_} per table'] = np.stack([modes[a_][0], modes[b_][0]])[pick, np.arange(K)].sum()
        if verbose: print('   quad(0)/pair16(1) per table:', pick.tolist())
    res['RAW planes (16 L per block)'] = 16.0 * d['Lo'].sum()
    if verbose:
        print(f'== test {test} / train {train}  ctx L x q({est}) 1/{step} oct, K={K}, off {off}: RLGR {d["bits"].sum() / n:.4f} bpp')
        for k, v in res.items(): print('   %-46s %.4f  (%+.2f%% of the RLGR part, %+.2f%% of the file)' % (k, v / n, 100 * (v / d['bits'].sum() - 1), 100 * (v - d['bits'].sum()) / d['fbits']))
    if ret: return res, d
if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 16; est = sys.argv[2] if len(sys.argv) > 2 else 'sa'; off = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for te, tr in [('kodak768', 'col19'), ('col19', 'kodak768'), ('gray', 'kodak768')]: evaluate(te, tr, est=est, K=K, off=off, tls=(10,))
