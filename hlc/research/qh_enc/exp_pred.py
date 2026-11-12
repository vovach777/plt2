# exp_pred.py - MED vs W (left) predictor for screen content, costed through the real PLR cost model (L planes, QH
# letters with the format tables, tails, zero-block runs; colour: + the zU split as its expected truncated-binary
# length), per image, from two dumps (palic2r -DPLR_DUMP -DPLR_NO_QH, and the same with -DPLR_PRED=1 = W).
# Variants: MED (the format), W everywhere, per-tile oracle choice (+1 bit per tile), per-block oracle choice with a
# 1-bit flag (cost of the flag = its order-0 entropy), causal per-block choice "the predictor that was cheaper on
# the block above" (no side information). Caveat for the per-block mixes: each block keeps the context sa of its own
# predictor's dump, i.e. the effect of the mix on the neighbours' contexts is ignored.
# usage: exp_pred.py DUMP_MED DUMP_W img...
import sys, os, subprocess, tempfile, numpy as np
sys.path.insert(0, '/home/claude/plt2_repo/hlc/research/qh'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp(); os.environ['QH_DUMP'] = tmp
import load as L_; L_.D = tmp
import lib, qhsim
T = qhsim.tables()
def costs(dumper, img):
    sd = os.path.join(tmp, 'one'); os.makedirs(sd, exist_ok=True)
    for x in os.listdir(sd): os.remove(os.path.join(sd, x))
    subprocess.run([dumper, 'c', img, os.path.join(sd, 'a.plr')], env=dict(os.environ, PLR_DUMP_FILE=os.path.join(sd, 'a.dmp')), check=True, stdout=subprocess.DEVNULL)
    lib._cache.clear(); d = lib.prep('one', 2); lb, tb, pb = qhsim.block_bits(d, T); k = ~d['skip']
    S = d['z'].astype(np.float64); kk = np.floor(np.log2(S + 1)); u = 2 ** (kk + 1) - (S + 1)
    split = np.where(S > 0, kk + (S + 1 - u) / (S + 1), 0).sum(1) * (d['pl'] == 1)          # S plane only
    parts = dict(letters=lb * k, tails=tb * k, planes=pb * k, runs=d['runbits'].astype(np.float64), split=split)
    tot = sum(parts.values())
    return d, tot, {a: float(b.sum()) for a, b in parts.items()}
dm, dw = sys.argv[1], sys.argv[2]
print('%-32s %8s %8s %8s %8s %8s %8s | where MED -> W goes (bpp): letters tails planes runs split' % ('file', 'MED', 'W', 'tile', 'blk+flag', 'causal', 'W-share'))
A = dict(n=0, m=0, w=0, t=0, b=0, c=0)
for img in sys.argv[3:]:
    d, cm, pm = costs(dm, img); d2, cw, pw = costs(dw, img); n = d['npx']
    assert len(cm) == len(cw) and (d['tile'] == d2['tile']).all() and (d['bi'] == d2['bi']).all()
    tile = d['tile']; nt = tile.max() + 1
    st_m = np.bincount(tile, cm, nt); st_w = np.bincount(tile, cw, nt)
    tile_cost = np.minimum(st_m, st_w).sum() + nt
    pick = cw < cm; p = pick.mean(); hflag = 0 if p in (0, 1) else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    blk_cost = np.minimum(cm, cw).sum() + len(cm) * hflag
    step = d['tw'] // 16; above = np.arange(len(cm)) - step; ok = (d['bi'] >= step) & (d['tw'] % 16 == 0)
    use_w = ok & (cw[np.maximum(above, 0)] < cm[np.maximum(above, 0)])
    caus = np.where(use_w, cw, cm).sum()
    m, w = cm.sum(), cw.sum()
    print('%-32s %8.4f %8.4f %8.4f %8.4f %8.4f %8.2f | %s' % (img.split('/')[-1][:32], m / n, w / n, tile_cost / n, blk_cost / n, caus / n,
          (st_w < st_m).mean(), '  '.join('%+.4f' % ((pw[a] - pm[a]) / n) for a in ('letters', 'tails', 'planes', 'runs', 'split'))), flush=True)
    for key, v in (('n', n), ('m', m), ('w', w), ('t', tile_cost), ('b', blk_cost), ('c', caus)): A[key] += v
n = A['n']; print('%-32s %8.4f %8.4f %8.4f %8.4f %8.4f   (vs MED: W %+.1f%%, tile %+.1f%%, block+flag %+.1f%%, causal %+.1f%%)' % (
    'TOTAL', A['m'] / n, A['w'] / n, A['t'] / n, A['b'] / n, A['c'] / n, *(100 * (A[x] / A['m'] - 1) for x in 'wtbc')))
