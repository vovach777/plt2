#!/usr/bin/env python3
"""Cheap probe for the RLK/PLR plane decision (as the PLS detector): sample every 8th 16-pixel block,
count non-zero blocks and non-zero pixels; compare with the exact winner (PLR bits from the dump, RLK bits from
the trained Huffman model). Prints, per probe bucket, how many planes RLK wins and the bits at stake."""
import os, sys, collections
os.environ.setdefault('RLK_LONG', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rlk_model as m

train = sys.argv[1].split(','); dumps = sys.argv[2:]
cnt = np.zeros((2, 225, m.NSYM))
for p in train:
    for tw, th, yb, sb, zy, s, zu in m.tiles(p):
        for pl, z in ((0, zy), (1, s)):
            S, C, _ = m.symbols(z, tw, 6); np.add.at(cnt[pl], (C, S), 1)
L = np.array([[m.huff_lengths(cnt[pl, c] + 0.5) for c in range(225)] for pl in range(2)])

def probe(z):
    n = len(z); nb = (n + 15) // 16
    zp = np.zeros(nb * 16, np.int64); zp[:n] = z
    blk = zp.reshape(nb, 16)[::8]                     # every 8th block
    return (blk.any(1).mean(), (blk != 0).mean())     # non-zero blocks, non-zero pixels

for p in dumps:
    rows = []
    for tw, th, yb, sb, zy, s, zu in m.tiles(p):
        for pl, z, plr, extra in ((0, zy, yb, 0), (1, s, sb, m.tb_bits(s, zu))):
            S, C, raw = m.symbols(z, tw, 6)
            rlk = int(L[pl, C, S].sum()) + raw + extra
            fb, fp = probe(z)
            rows.append((fb, fp, plr, rlk, len(z)))
    r = np.array(rows, np.float64)
    print(f'== {os.path.basename(p)}: {len(r)} planes')
    np.save('/tmp/rlk/probe_' + os.path.basename(p) + '.npy', r)
    hi_blk = r[r[:, 0] >= .8]
    for lo, hi in ((0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)):
        sel = (hi_blk[:, 1] >= lo) & (hi_blk[:, 1] < hi)
        if sel.any():
            rl = hi_blk[sel]; win = rl[:, 3] < rl[:, 2]
            print(f'  [blocks>=0.8] nonzero px {lo:.2f}..{hi:.2f}: {sel.sum():5d} planes, RLK wins {win.sum():5d}, '
                  f'PLR loses {np.maximum(rl[:, 2] - rl[:, 3], 0).sum() / 8192:7.1f} KiB, RLK loses {np.maximum(rl[:, 3] - rl[:, 2], 0).sum() / 8192:7.1f} KiB')
    for lo, hi in ((0, .3), (.3, .5), (.5, .8)):
        sel = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if not sel.any(): continue
        rl = r[sel]; win = rl[:, 3] < rl[:, 2]
        loss_plr = np.maximum(rl[:, 2] - rl[:, 3], 0).sum()   # bits lost if we always took PLR here
        loss_rlk = np.maximum(rl[:, 3] - rl[:, 2], 0).sum()   # bits lost if we always took RLK here
        print(f'  nonzero blocks {lo:.2f}..{hi:.2f}: {sel.sum():5d} planes, RLK wins {win.sum():5d}, '
              f'always-PLR loses {loss_plr / 8 / 1024:8.1f} KiB, always-RLK loses {loss_rlk / 8 / 1024:8.1f} KiB')
