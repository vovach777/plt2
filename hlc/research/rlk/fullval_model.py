#!/usr/bin/env python3
"""Huffman on whole residual values (no k / raw-mantissa split) vs PLR, per tile plane, on RLK dumps.

Symbol: z < 64 -> z itself; z >= 64 -> 64 + bitwidth(z) (escape), then bitwidth-1 raw bits. On S, + zU split.
Context: activity a = z(left) + z(up) + z(up-left) + z(up-right) (known pixels; missing ones -> the up / left one),
class = bitwidth(a) * 2 + next bit below the top one -> up to ~26 classes. Static Huffman per (plane, class),
trained on the train dumps. Optional: RUNS=1 adds (run of zeros 1..15) symbols like RLK... not here.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rlk_model as m

NS = 64 + 12

def syms(z, tw):
    n = len(z); th = n // tw; Z = z.reshape(th, tw)
    L = np.zeros_like(Z); U = np.zeros_like(Z); UL = np.zeros_like(Z); UR = np.zeros_like(Z)
    L[:, 1:] = Z[:, :-1]; U[1:, :] = Z[:-1, :]; UL[1:, 1:] = Z[:-1, :-1]; UR[1:, :-1] = Z[:-1, 1:]
    L[:, 0] = U[:, 0]; UL[:, 0] = U[:, 0]; UR[:, -1] = U[:, -1]
    U[0, :] = L[0, :]; UL[0, :] = L[0, :]; UR[0, :] = L[0, :]
    a = (L + U + UL + UR).ravel()
    b = m.bw(a)
    sub = np.where(b >= 2, (a >> np.maximum(b - 2, 0)) & 1, 0)
    ctx = np.minimum(b * 2 + sub, 31)
    LPRED = int(os.environ.get('LPRED', '0'))
    if LPRED == 2 or os.environ.get('BLKCTX'):
        # PLR's L: per 16-pixel block (tile raster order), sum of the 16 pixels one row above, else of the
        # previous block (64 for the first one), chooseL with offset LOFF (2 in PLR)
        nb = (n + 15) // 16
        zp = np.zeros(nb * 16 + 16, np.int64); zp[:n] = z
        sb = zp[:nb * 16].reshape(nb, 16).sum(1) & 0xffff
        px = np.arange(nb) * 16
        cs = np.cumsum(np.concatenate(([0], zp)))
        up = np.where((tw >= 16) & (px >= tw), cs[np.maximum(px - tw + 16, 0)] - cs[np.maximum(px - tw, 0)], -1)
        sc = np.where(up >= 0, up & 0xffff, np.concatenate(([64], sb[:-1])))
        qb = m.bw(sc >> 4)
        Lblk = np.clip(qb - int(os.environ.get('LOFF', '2')), 0, 7)
        Lpx = np.repeat(Lblk, 16)[:n]
        if os.environ.get('BLKCTX'):            # table chosen by the block sum class too (as PLR's Qsa, coarse)
            sbits = m.bw(sc); sub = np.where(sbits >= 2, (sc >> np.maximum(sbits - 2, 0)) & 1, 0)
            ctx = np.minimum(np.repeat(sbits * 2 + sub, 16)[:n], 31)
    Lp = Lpx if LPRED == 2 else (np.clip(m.bw(a >> 2) - int(os.environ.get('LOFF', '2')), 0, 7) if LPRED else np.zeros_like(a))   # as PLR chooseL (offset LOFF), from the 4-neighbour mean
    h = z >> Lp
    k = m.bw(h)
    s = np.where(h < 64, h, 64 + k)
    raw = int(np.where(h >= 64, k - 1, 0).sum()) + int(Lp.sum())
    return s, ctx, raw

def main():
    train = sys.argv[1].split(','); tests = sys.argv[2:]
    ideal = bool(os.environ.get('IDEAL'))
    cnt = np.zeros((2, 32, NS))
    for p in train:
        for tw, th, yb, sb, zy, s, zu in m.tiles(p):
            for pl, z in ((0, zy), (1, s)):
                S, C, _ = syms(z, tw); np.add.at(cnt[pl], (C, S), 1)
    Lf = np.zeros_like(cnt)
    for pl in range(2):
        for c in range(32):
            f = cnt[pl, c] + 0.5
            Lf[pl, c] = -np.log2(f / f.sum()) if ideal else m.huff_lengths(f)
    for t in tests:
        tot = {'plr': 0, 'fv': 0, 'best': 0}; px = 0
        for tw, th, yb, sb, zy, s, zu in m.tiles(t):
            px += tw * th
            Sy, Cy, ry = syms(zy, tw); fy = float(Lf[0, Cy, Sy].sum()) + ry
            Ss, Cs, rs = syms(s, tw); fs = float(Lf[1, Cs, Ss].sum()) + rs + m.tb_bits(s, zu)
            tot['plr'] += yb + sb; tot['fv'] += fy + fs; tot['best'] += min(yb, fy) + min(sb, fs)
        print(f"{'IDEAL ' if ideal else ''}{os.path.basename(t):14s} PLR {tot['plr'] / px:.4f}  whole-value Huffman {tot['fv'] / px:.4f}  per-plane min {tot['best'] / px:.4f}")

if __name__ == '__main__':
    main()
