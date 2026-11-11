# exp5.py - zero-heavy tables: "oct" letters (zero mask of 8 neighbours, 256 letters, then h-1 of every non-zero symbol)
# versus the quad letters of QH, chosen per table on the training set. Costs with Huffman lengths (len <= 11) + Rice / Huffman excess.
import numpy as np, sys
from lib import *
from train_tables import *
def run(test, train, K=32, maxlen=11):
    T = train_tables_cached(train, K, maxlen); t = prep(train); d = prep(test); n = d['npx']
    out = {}
    for nm, dd in (('tr', t), ('te', d)):
        tb = T['map'][ctx_of(dd)]; h = dd['h']
        lq = letters(h); qb = np.bincount(tb, T['lens'][tb[:, None], lq].sum(1), K) + np.bincount(tb, np.where(h >= 3, rice_esc_bits(np.maximum(h - 3, 0), T['ks'][tb][:, None]), 0).sum(1), K)
        nz = (h > 0).reshape(len(h), 2, 8); lo = (nz * (1 << np.arange(8))).sum(2)
        out[nm] = dict(tb=tb, qb=qb, lo=lo, co=counts(tb, lo, K, 256), h=h)
    lens_o = huff_len(out['tr']['co'], maxlen)
    def oct_bits(o, ks, xlens):
        tb, h = o['tb'], o['h']; x = np.maximum(h - 1, 0); m = h > 0
        a = np.bincount(tb, lens_o[tb[:, None], o['lo']].sum(1), K)
        r = np.bincount(tb, np.where(m, rice_esc_bits(x, ks[tb][:, None]), 0).sum(1), K)
        xc = np.minimum(x, 15); hx = np.bincount(tb, np.where(m, xlens[tb[:, None], xc] + np.where(x >= 15, 10, 0), 0).sum(1), K)
        return a + r, a + hx
    # excess of the oct mode: Rice k per table (best on train) or a 16-letter Huffman (len <= 8, 15 = escape + 10 raw bits)
    tr = out['tr']; x = np.maximum(tr['h'] - 1, 0); m = tr['h'] > 0
    ks = np.array([int(np.argmin([rice_esc_bits(x[(tr['tb'] == k)[:, None] & m], kk).sum() for kk in range(4)])) for k in range(K)])
    xcnt = np.bincount((tr['tb'][:, None] * 16 + np.minimum(x, 15))[m], minlength=K * 16).reshape(K, 16).astype(float); xlens = huff_len(xcnt, 8)
    otr_r, otr_h = oct_bits(out['tr'], ks, xlens); ote_r, ote_h = oct_bits(out['te'], ks, xlens)
    r = d['bits'].sum(); q = out['te']['qb'].sum()
    pr = otr_r < out['tr']['qb']; ph = otr_h < out['tr']['qb']
    br = np.where(pr, ote_r, out['te']['qb']).sum(); bh = np.where(ph, ote_h, out['te']['qb']).sum()
    f = lambda v: '%.4f bpp (%+.2f%% of the RLGR part, file %.4f)' % (v / n, 100 * (v / r - 1), (d['fbits'] + v - r) / n)
    print(f'== test {test} / train {train}: RLGR {r / n:.4f}')
    print('   QH quads everywhere                 ', f(q))
    print('   quad / oct+Rice per table  (%2d oct) ' % pr.sum(), f(br))
    print('   quad / oct+Huff16 per table (%2d oct)' % ph.sum(), f(bh))
_tc = {}
def train_tables_cached(train_, K, maxlen):
    if (train_, K, maxlen) not in _tc: _tc[(train_, K, maxlen)] = train(train_, K, maxlen)
    return _tc[(train_, K, maxlen)]
for te, tr_ in [('kodak768', 'col19'), ('col19', 'kodak768'), ('gray', 'kodak768'), ('uv_kodak768', 'uv_col19'), ('uv_col19', 'uv_kodak768')]: run(te, tr_)
