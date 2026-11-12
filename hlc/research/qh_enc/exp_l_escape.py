# exp_l_escape.py - Qwen's "escape" for L: per block a flag "the causal L misses" (context: bw(sa), previous block empty,
# above empty) and, on a miss, the cheapest L (argmin of the exact block cost) coded conditionally on the causal L.
# Costs with the format tables (letters + tails + 16 L) + runs + zU split (exp_pred.costs). Side info = ideal adaptive
# (per-image conditional entropy) -> optimistic by the table cost. Also the free oracle (no side info) as the ceiling.
# usage: exp_l_escape.py DUMPER NAME img...   (optional env TILE_CSV=out.csv TILE=k: write the blocks of tile k, plane Y)
import sys, os, numpy as np
sys.argv, _a = sys.argv[:1] + ['x', 'y'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_pred.py')).read().split("dm, dw = sys.argv[1], sys.argv[2]")[0])
from lib import bwv
dumper, name, imgs = _a[1], _a[2], _a[3:]
def H(x, ctx):          # ideal adaptive cost of x given ctx, bits
    if len(x) == 0: return 0.0
    _, ci = np.unique(ctx, return_inverse=True); _, xi = np.unique(x, return_inverse=True)
    j = np.zeros((ci.max() + 1, xi.max() + 1)); np.add.at(j, (ci, xi), 1); r = j.sum(1, keepdims=True)
    p = np.where(j > 0, j / r, 1); return float(-(j * np.log2(p)).sum())
A = dict(file=0.0, caus=0.0, esc=0.0, orc=0.0, side=0.0, npx=0, miss=0, nb=0)
for img in imgs:
    d, c, parts = costs(dumper, img); k = ~d['skip']; z = d['z'].astype(np.int64)
    t = T['cm'][d['ptype'] * T['NQ'] + qhsim.Qsa(d['sa'])]
    CL = np.zeros((len(z), 8))
    for L in range(8):
        h = z >> L; x = np.maximum(h - 3, 0); q = x >> T['rk'][t][:, None]
        CL[:, L] = T['lens'][t[:, None], qhsim.letters(h)].sum(1) + np.where(h >= 3, np.where(q >= T['QESC'], T['QESC'] + 10, q + 1 + T['rk'][t][:, None]), 0).sum(1) + 16 * L
    Lc = d['Lo'].astype(np.int64); n = len(z); i = np.arange(n); cc = CL[i, Lc]; Lb = CL.argmin(1); cb = CL[i, Lb]
    miss = (cb < cc) & k
    pk = d['tile'].astype(np.int64) * 4 + d['pl']; prev_empty = np.r_[True, (z[:-1] == 0).all(1)] | np.r_[True, pk[1:] != pk[:-1]]
    ctx = (np.minimum(bwv(d['sa'].astype(np.int64)), 15) * 2 + prev_empty) * 2 + (d['sa'] < 16) + 64 * d['ptype']
    side = H(miss[k].astype(np.int64), ctx[k]) + H(Lb[miss], Lc[miss] + 8 * d['ptype'][miss])
    other = c.sum() - cc[k].sum()            # runs + split + (equal) everything not in the block cost
    A['caus'] += c.sum(); A['esc'] += other + np.where(miss, cb, cc)[k].sum() + side; A['orc'] += other + cb[k].sum()
    A['side'] += side; A['npx'] += d['npx']; A['miss'] += int(miss.sum()); A['nb'] += int(k.sum())
    if os.environ.get('TILE_CSV'):
        sel = (d['tile'] == int(os.environ['TILE'])) & (d['pl'] == 0); step = d['tw'] // 16
        with open(os.environ['TILE_CSV'], 'w') as f:
            f.write('row,blk,sa_above,L_causal,L_best,bits_causal,bits_best,zero_run_skipped,' + ','.join('bits_L%d' % l for l in range(8)) + ',' + ','.join('z%d' % j for j in range(16)) + '\n')
            for jj in np.where(sel)[0]:
                bi = int(d['bi'][jj]); step = int(d['tw'][jj]) // 16; f.write(f"{bi // step},{bi % step},{int(d['sa'][jj])},{Lc[jj]},{Lb[jj]},{int(cc[jj])},{int(cb[jj])},{int(d['skip'][jj])}," + ','.join(str(int(v)) for v in CL[jj]) + ',' + ','.join(str(int(v)) for v in z[jj]) + '\n')
n = A['npx']; b = A['caus']
print(f"{name}: causal {b / n:.4f} bpp | escape {A['esc'] / n:.4f} ({100 * (A['esc'] / b - 1):+.2f}%), side {A['side'] / n:.4f} bpp, misses {A['miss'] / A['nb']:.1%} of blocks | free oracle {A['orc'] / n:.4f} ({100 * (A['orc'] / b - 1):+.2f}%)", flush=True)
