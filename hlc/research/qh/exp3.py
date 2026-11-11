# exp3.py - (e) "frozen adaptation": the code parameter is fixed inside a 16-block and chosen at the block start.
#  e1: Rice(k) per block, k = best static k per causal context (L x q(sa), 1/2 octave), trained on the other set
#  e2: Rice(k) per block, k from an adaptive state per bin (JPEG-LS style A/N over the previous blocks of the bin in the tile)
#  e3: ideal static tables chosen by the adaptive state (EMA of the block sums of h in the bin), alone and with q(sa)
import numpy as np
from lib import *
from exp2 import qf, ctx_fine
def ema_state(d):
    key = (d['tile'].astype(np.int64) * 2 + d['ptype']) * 8 + d['Lo']; hs = d['h'].sum(1)
    st = {}; out = np.zeros(len(key)); kj = np.zeros(len(key), np.int64); AN = {}
    for i in range(len(key)):
        k = int(key[i]); s = st.get(k, 16.0); out[i] = s; st[k] = (3 * s + hs[i]) / 4
        a, nn = AN.get(k, (16, 16)); kk = 0
        while (nn << kk) < a: kk += 1
        kj[i] = kk; a += int(hs[i]); nn += 16
        if nn >= 256: a >>= 1; nn >>= 1
        AN[k] = (a, nn)
    return out, kj
def rice_bits(h, k): return ((h >> k) + 1 + k)     # no escape limit (ideal Rice)
def run(test, train, off=1):
    d = prep(test, off); t = prep(train, off); n = d['npx']
    print(f'== test {test} / train {train}: RLGR {d["bits"].sum() / n:.4f}')
    cd, nc = ctx_fine(d, d['sa'], 2); ct, _ = ctx_fine(t, t['sa'], 2)
    # e1
    tb = np.stack([np.bincount(ct, rice_bits(t['h'], k).sum(1), nc) for k in range(6)]); kbest = tb.argmin(0)
    db = np.stack([np.bincount(cd, rice_bits(d['h'], k).sum(1), nc) for k in range(6)])
    print('  e1 Rice k by L x q(sa) (static map)        %.4f   (oracle best k per block: %.4f)' % (db[kbest, np.arange(nc)].sum() / n, np.stack([rice_bits(d['h'], k).sum(1) for k in range(6)]).min(0).sum() / n))
    for nm, dd in (('d', d), ('t', t)):
        if 'ema' not in dd: dd['ema'], dd['kj'] = ema_state(dd)
    kb = np.stack([rice_bits(d['h'], k).sum(1) for k in range(8)]); 
    print('  e2 Rice k by adaptive A/N state per bin    %.4f' % (kb[np.minimum(d['kj'], 7), np.arange(len(d['kj']))].sum() / n))
    s = np.minimum(d['h'], A - 1); ts = np.minimum(t['h'], A - 1); esc = esc_bits(d['h']).sum() / n
    def rep(nm, c1, c2, nc):
        c = counts(c1, s, nc, A); tr = counts(c2, ts, nc, A)
        print('  %-42s ctx %4d  self %.4f  cross %.4f' % (nm, (c.sum(1) > 0).sum(), ent(c) / n + esc, (c * ideal_len(tr)).sum() / n + esc))
    qe = lambda dd: np.clip(qf(dd['ema'], 2), 0, 15)
    rep('e3 tables by L x q(ema of bin)', (d['ptype'] * 8 + d['Lo']) * 16 + qe(d), (t['ptype'] * 8 + t['Lo']) * 16 + qe(t), 256)
    rep('   tables by L x q(sa) [exp2]', cd, ct, nc)
    rep('e3 tables by L x q(sa) x q(ema)', cd * 16 + qe(d), ct * 16 + qe(t), nc * 16)
if __name__ == "__main__":
  for te, tr in [('kodak768', 'col19'), ('col19', 'kodak768'), ('gray', 'kodak768')]: run(te, tr)
