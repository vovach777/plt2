#!/usr/bin/env python3
"""Decide RLK vs PLR from what the PLR pass already knows (no RLK pricing): PLR bits per pixel, the share of
16-pixel blocks coded as zero runs, the mean causal L and the mean |L error| (causal L vs the block's own).
Compares simple rules with the exact per-plane choice (PLR bits from the dump, RLK bits from the trained model)."""
import os, sys
os.environ.setdefault('RLK_LONG', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rlk_model as m
from lerr_probe import chooseL

def plr_stats(z, tw):
    n = len(z); nb = (n + 15) // 16
    zp = np.zeros(nb * 16 + 16, np.int64); zp[:n] = z
    blk = zp[:nb * 16].reshape(nb, 16); sb = blk.sum(1) & 0xffff
    px = np.arange(nb) * 16
    cs = np.cumsum(np.concatenate(([0], zp)))
    up = np.where((tw >= 16) & (px >= tw), cs[np.maximum(px - tw + 16, 0)] - cs[np.maximum(px - tw, 0)], -1)
    sc = np.where(up >= 0, up & 0xffff, np.concatenate(([64], sb[:-1])))
    Lc = chooseL(sc); Lb = chooseL(sb)
    zero_run = ((sc == 0) & (sb == 0)).mean()        # blocks PLR codes inside zero runs (gsa == 0 and empty)
    return zero_run, Lc.mean(), np.abs(Lc - Lb).mean()

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
            zr, lm, le = plr_stats(z, tw)
            rows.append((plr / len(z), zr, lm, le, plr, rlk))
    r = np.array(rows); best = np.minimum(r[:, 4], r[:, 5]).sum()
    win = r[:, 5] < r[:, 4]
    print(f'== {os.path.basename(p)}: {len(r)} planes, RLK wins {win.sum()}')
    for name, rl in (('PLR bpp > 0.3', r[:, 0] > 0.3), ('PLR bpp < 3', r[:, 0] < 3), ('mean L < 1', r[:, 2] < 1),
                     ('mean L < 0.5', r[:, 2] < 0.5), ('zero-run share > 0.2', r[:, 1] > 0.2),
                     ('mean L < 1 or zero-run > 0.2', (r[:, 2] < 1) | (r[:, 1] > 0.2)),
                     ('mean L < 1.5', r[:, 2] < 1.5), ('mean L < 2', r[:, 2] < 2)):
        bits = np.where(rl, r[:, 5], r[:, 4]).sum()
        print(f'  RLK if {name:30s}: share {rl.mean():.2f}  +{100 * (bits - best) / best:.3f}% vs exact')
