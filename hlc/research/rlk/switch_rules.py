#!/usr/bin/env python3
"""Plane switch RLK / PLR, with the fixed Huffman tables: per plane collect cheap features and exact costs, save,
then score rules against the exact choice. Features (every 8th 16-pixel block sampled, as the PLS probe):
fb = non-zero blocks, q0 = zero quads (PLR letters with all 4 pixels zero), fp = non-zero pixels; and
rbpp = RLK bits per pixel (exact price, the RLK-first variant)."""
import os, sys
os.environ.setdefault('RLK_LONG', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rlk_model as m

def feats(z):
    n = len(z); nb = (n + 15) // 16
    zp = np.zeros(nb * 16, np.int64); zp[:n] = z
    b = zp.reshape(nb, 16)[::8]
    q = b.reshape(-1, 4, 4)
    return b.any(1).mean(), (~q.any(2)).mean(), (b != 0).mean()

if __name__ == '__main__':
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
                fb, q0, fp = feats(z)
                rows.append((fb, q0, fp, plr, rlk, len(z)))
        np.save('/tmp/rlk/sw_' + os.path.basename(p) + '.npy', np.array(rows, float))
        print('saved', p, len(rows))
