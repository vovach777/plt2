# exp1.py - (a) real RLGR, (b) order-0 per tile / image / set, (c) static single-symbol tables (cross-set)
import numpy as np, sys
from lib import *
def sym_of(d): return np.minimum(d['h'], A - 1)
def run(test, train, off=1):
    d = prep(test, off); t = prep(train, off); n = d['npx']
    ctx = d['ptype'] * 8 + d['Lo']; tctx = t['ptype'] * 8 + t['Lo']; s = sym_of(d)
    esc = esc_bits(d['h']).sum(1)
    res = {}
    res['rlgr'] = np.bincount(ctx, d['bits'], 16)
    # per tile order 0
    tk = (d['tile'].astype(np.int64) * 16 + ctx); u, inv = np.unique(tk, return_inverse=True)
    c = counts(inv, s, len(u), A); n_ = c.sum(1, keepdims=True); p = np.where(c > 0, c / n_, 1); e = -(c * np.log2(p)).sum(1)
    res['o0 tile'] = np.bincount(u % 16, e, 16) + np.bincount(ctx, esc, 16)
    c = counts(ctx, s, 16, A); e = -(c * np.log2(np.where(c > 0, c / np.maximum(c.sum(1, keepdims=True), 1), 1))).sum(1)
    res['o0 set(self)'] = e + np.bincount(ctx, esc, 16)
    ct = counts(tctx, sym_of(t), 16, A)
    for nm, ln in [('static ideal', ideal_len(ct)), ('static tANS10', tans_len(ct, 10)), ('static Huff', huff_len(ct))]:
        res[nm] = (c * ln).sum(1) + np.bincount(ctx, esc, 16)
    print(f'== test {test}, tables from {train}, off {off}: bpp of the image; raw planes {16 * d["Lo"].sum() / n:.4f}, escapes {esc.sum() / n:.5f}')
    print('%-14s' % '' + ''.join('  %s%d ' % ('YS'[i // 8], i % 8) for i in range(16) if res['rlgr'][i] > 0 or True) + '  total')
    for k, v in res.items(): print('%-14s' % k + ''.join('%6.3f' % (x / n) for x in v) + '  %.4f' % (v.sum() / n))
    nsym = np.bincount(ctx, minlength=16) * 16.0
    print('%-14s' % 'rlgr bits/sym' + ''.join('%6.2f' % (a / max(b, 1)) for a, b in zip(res['rlgr'], nsym)))
    print('%-14s' % 'P(h=0)' + ''.join('%6.2f' % (c[i, 0] / max(c[i].sum(), 1)) for i in range(16)))
for te, tr in [('kodak768', 'col19'), ('col19', 'kodak768'), ('gray', 'kodak768')]: run(te, tr)
