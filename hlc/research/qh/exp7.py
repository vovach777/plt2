# exp7.py - zero-block run mode for QH (removes the floor of 4 bits per 16-block on flat content), simulated on the dumps.
# Rule: at a block whose causal sum sa == 0, outside a run, send n = number of all-zero blocks starting here (consecutive, same
# tile and plane, all with sa == 0) as Exp-Golomb(0); n == 0: 1 bit, then the block as usual; n > 0: n blocks cost nothing more.
import numpy as np
from lib import *
from train_tables import *
def run(test, tr):
    T = train(tr, 32, 11); d = prep(test); n = d['npx']; tb = T['map'][ctx_of(d)]; h = d['h']
    qb = T['lens'][tb[:, None], letters(h)].sum(1) + np.where(h >= 3, rice_esc_bits(np.maximum(h - 3, 0), T['ks'][tb][:, None]), 0).sum(1)
    z0 = (d['sa'] == 0) & ~d['first']; allz = h.sum(1) == 0; key = d['tile'].astype(np.int64) * 4 + d['pl']
    idx = np.where(z0)[0]; delta = 0; rem = 0; last = -2; i = 0; nrun = 0; nflag = 0
    zz = z0 & allz
    while i < len(idx):
        b = idx[i]
        # run length of zero blocks from b
        j = b
        while j < len(zz) and zz[j] and key[j] == key[b]: j += 1
        r = j - b
        delta += 2 * (int(r + 1).bit_length() - 1) + 1; nflag += 1
        if r > 0: delta -= int(qb[b:j].sum()); nrun += r; i = np.searchsorted(idx, j)
        else: i += 1
    print(f'{test:12s}: blocks with sa==0: {100 * z0.mean():.2f}%, all-zero among them {100 * zz.sum() / max(z0.sum(), 1):.1f}%; run mode changes the size by {delta / n:+.5f} bpp ({100 * delta / d["fbits"]:+.3f}% of the file)')
for te, tr in [('kodak768', 'col19'), ('col19', 'kodak768'), ('gray', 'kodak768'), ('uv_col19', 'uv_kodak768')]: run(te, tr)
