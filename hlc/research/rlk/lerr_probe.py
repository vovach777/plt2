#!/usr/bin/env python3
"""Does the error of PLR's causal L separate RLK planes from PLR planes better than the non-zero block share?
Per 16-pixel block: Lc = chooseL(sum16 of the 16 pixels above, or of the previous block in the first row) as PLR;
Lb = chooseL(sum16 of the block itself) (what L would have been, knowing the block). Plane feature: mean |Lc - Lb|.
Prints, for thresholds on each feature, the bits lost vs the exact per-plane choice (PLR bits from the dump,
RLK bits from the trained model) when the feature alone decides."""
import os, sys
os.environ.setdefault('RLK_LONG', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rlk_model as m

def chooseL(s):
    q = s >> 4
    b = np.where(q > 0, np.floor(np.log2(np.maximum(q, 1))).astype(np.int64) + 1, 0)
    return np.clip(b - 2, 0, 7)

def feats(z, tw):
    n = len(z); nb = (n + 15) // 16
    zp = np.zeros(nb * 16 + 16, np.int64); zp[:n] = z
    blk = zp[:nb * 16].reshape(nb, 16)
    sb = blk.sum(1) & 0xffff
    Lb = chooseL(sb)
    # causal sum: 16 pixels one row above (px - tw), else the previous block
    px = np.arange(nb) * 16
    above = np.array([zp[p - tw:p - tw + 16].sum() if (tw >= 16 and p >= tw) else -1 for p in px])
    prev = np.concatenate(([64], sb[:-1]))
    sc = np.where(above >= 0, above & 0xffff, prev)
    Lc = chooseL(sc)
    return np.abs(Lc - Lb).mean(), blk.any(1).mean()

train = sys.argv[1].split(','); dumps = sys.argv[2:]
cnt = np.zeros((2, 225, m.NSYM))
for p in train:
    for tw, th, yb, sb, zy, s, zu in m.tiles(p):
        for pl, z in ((0, zy), (1, s)):
            S, C, _ = m.symbols(z, tw, 6); np.add.at(cnt[pl], (C, S), 1)
L = np.array([[m.huff_lengths(cnt[pl, c] + 0.5) for c in range(225)] for pl in range(2)])
for p in dumps:
    rows = []
    for tw, th, yb, sb, zy, s, zu in m.tiles(p):
        for pl, z, plr, extra in ((0, zy, yb, 0), (1, s, sb, m.tb_bits(s, zu))):
            S, C, raw = m.symbols(z, tw, 6)
            rlk = int(L[pl, C, S].sum()) + raw + extra
            le, fb = feats(z, tw)
            rows.append((le, fb, plr, rlk))
    r = np.array(rows, float); best = np.minimum(r[:, 2], r[:, 3]).sum()
    print(f'== {os.path.basename(p)} ({len(r)} planes)')
    for name, col, ths in (('L error', 0, (0.1, 0.2, 0.3, 0.5, 0.8)), ('non-zero blocks', 1, (0.3, 0.5, 0.7, 0.9, 0.98))):
        for t in ths:
            # L error: large -> RLK; non-zero blocks: small -> RLK
            rl = r[:, col] > t if col == 0 else r[:, col] < t
            bits = np.where(rl, r[:, 3], r[:, 2]).sum()
            print(f'  {name:16s} thr {t:4}: RLK share {rl.mean():.2f}  +{100 * (bits - best) / best:.3f}% vs exact')
