#!/usr/bin/env python3
"""RLK plane-mode model on real PLR tiles: JPEG-style (run, k) Huffman vs PLR, per tile plane.

Input: dumps written by the as2cpp translation of plr.ts with RLK_DUMP (per tile: tw, th, PLR bits of the Y plane,
PLR bits of the S plane incl. the zU split, then zY, S, zU as u16 x tw*th).

Symbols per plane (tile raster order, runs continue across rows): (r, k) for r = 0..15 zeros then a value of bit
width k = 1..10; ZRL = 16 zeros; EOB = the rest of the plane is zero. Then k-1 raw mantissa bits; on S also the
truncated binary code of zU over 0..S. Static Huffman tables, trained on the train dumps, one per (plane, context);
the context is taken at the first pixel of the symbol from already known neighbours.
"""
import sys, heapq, collections
import numpy as np

import os
LONG = int(os.environ.get('RLK_LONG', '0'))    # 1: runs >= 16 as LRUN(b) + (b-1) raw bits instead of ZRL chains
MANT = int(os.environ.get('RLK_MANT', '0'))    # 1: the top mantissa bit of k >= 2 goes into the symbol
NV = 19 if MANT else 10                        # value classes: k=1, then (k, top bit) for k = 2..10
NSYM = 16 * NV + 2 + 14                        # (r, value class), ZRL, EOB, LRUN(b) for b = 5..18
ZRL, EOB, LRUN0 = 16 * NV, 16 * NV + 1, 16 * NV + 2
MAXLEN = 14


def tiles(path):
    d = np.fromfile(path, dtype=np.uint8)
    i = 0
    while i < len(d):
        tw, th, yb, sb = np.frombuffer(d[i:i + 16].tobytes(), np.int32)
        n = int(tw) * int(th); i += 16
        z = np.frombuffer(d[i:i + 6 * n].tobytes(), np.uint16).reshape(3, n).astype(np.int64); i += 6 * n
        yield int(tw), int(th), int(yb), int(sb), z[0], z[1], z[2]


def bw(v):
    out = np.zeros_like(v)
    nz = v > 0
    out[nz] = np.floor(np.log2(v[nz])).astype(np.int64) + 1
    return out


def q3(k): return np.minimum(k, 3)


def vclass(z, k):
    if not MANT: return k - 1
    top = (z >> np.maximum(k - 2, 0)) & 1
    return np.where(k == 1, 0, 1 + 2 * (k - 2) + top)


def symbols(z, tw, ctxmode):
    """-> (symbol ids, context ids, raw bits) for one plane."""
    n = len(z); k = bw(z)
    nzp = np.flatnonzero(z)
    prev = np.concatenate(([-1], nzp[:-1]))
    gap = nzp - prev - 1                       # zeros before each nonzero
    kk = k[nzp]; vc = vclass(z[nzp], kk)
    raw = int(np.maximum(kk - 1 - MANT * (kk >= 2), 0).sum())
    syms, starts = [], []
    if LONG:
        lg = gap >= 16
        b = bw(np.where(lg, gap, 1))
        syms.append(LRUN0 + (b[lg] - 5)); starts.append(prev[lg] + 1)
        raw += int((b[lg] - 1).sum())
        rem = np.where(lg, 0, gap)
        syms.append(rem * NV + vc); starts.append(nzp - rem)
    else:
        nzrl = gap // 16; rem = gap % 16
        syms.append(rem * NV + vc); starts.append(nzp - rem)
        zr_cnt = int(nzrl.sum())
        if zr_cnt:
            owner = np.repeat(np.arange(len(nzp)), nzrl)
            j = np.arange(zr_cnt) - np.repeat(np.cumsum(nzrl) - nzrl, nzrl)
            syms.append(np.full(zr_cnt, ZRL)); starts.append(prev[owner] + 1 + 16 * j)
    tail = n - 1 - (nzp[-1] if len(nzp) else -1)
    if tail > 0:
        syms.append(np.array([EOB])); starts.append(np.array([n - tail]))
    S = np.concatenate(syms); P = np.concatenate(starts)
    return S, context(z, k, P, tw, ctxmode), raw


def context(z, k, P, tw, mode):
    if mode == 0:
        return np.zeros(len(P), np.int64)
    x = P % tw; y = P // tw
    up = np.where(y > 0, k[np.maximum(P - tw, 0)], 4)              # 4 = no row above
    ur = np.where((y > 0) & (x < tw - 1), k[np.maximum(P - tw + 1, 0)], up)
    lf = np.where(x > 0, k[np.maximum(P - 1, 0)], up)
    if mode == 1:                                                   # up only: 0..3 + "no row"
        return np.minimum(up, 4)
    if mode == 2:                                                   # up x up-right
        return np.minimum(up, 4) * 5 + np.minimum(ur, 4)
    if mode == 3:                                                   # left x up x up-right (as Qwen's QSM)
        return (np.minimum(lf, 4) * 5 + np.minimum(up, 4)) * 5 + np.minimum(ur, 4)
    if mode in (5, 6, 7):
        # length of the zero run in the row above, starting right above P (to the end of that row)
        zr = up_zero_run(z, tw)
        ur_len = np.where(y > 0, zr[np.maximum(P - tw, 0)], -1)
        cls = np.where(ur_len < 0, 8, np.minimum(bw(np.maximum(ur_len, 0)), 7))    # 0..7, 8 = no row above
        if mode == 5: return cls * 5 + np.minimum(lf, 4)
        if mode == 6: return (cls * 5 + np.minimum(lf, 4)) * 5 + np.minimum(ur, 4)
        if mode == 7: return (cls * 5 + np.minimum(lf, 4)) * 5 + np.minimum(up, 4)
    if mode == 4:                                                   # activity sum of the three, quantised
        a = np.minimum(lf, 4) + np.minimum(up, 4) + np.minimum(ur, 4)
        return np.minimum(a, 9)
    raise ValueError(mode)


def up_zero_run(z, tw):
    """r[p] = number of zeros from p to the right in p's row (0 if z[p] != 0)."""
    n = len(z); r = np.zeros(n, np.int64)
    zz = (z == 0).reshape(-1, tw) if n % tw == 0 else None
    rows = zz[:, ::-1].astype(np.int64)
    # reverse cumulative count reset at nonzeros
    c = np.cumsum(rows, axis=1)
    reset = np.where(rows == 0, c, 0)
    reset = np.maximum.accumulate(reset, axis=1)
    return (c - reset)[:, ::-1].ravel()


def tb_bits(S, zu):
    m = S + 1; k = np.floor(np.log2(m)).astype(np.int64); u = (2 << k) - m
    return int(np.sum(k + (zu >= u)))


def _huff(f):
    h = [(c, i, (i,)) for i, c in enumerate(f)]
    heapq.heapify(h); L = np.zeros(len(f), np.int64); cnt = len(h)
    while len(h) > 1:
        a = heapq.heappop(h); b = heapq.heappop(h)
        for s in a[2] + b[2]: L[s] += 1
        heapq.heappush(h, (a[0] + b[0], cnt, a[2] + b[2])); cnt += 1
    return L


def huff_lengths(freq):
    """Huffman code lengths for counts (all > 0), limited to MAXLEN: only the rare tail is lifted (each count is
    raised to a floor that grows until the code fits), the frequent symbols keep their lengths."""
    f = np.asarray(freq, np.float64)
    L = _huff(f)
    floor = f.sum() / 2.0 ** MAXLEN
    while L.max() > MAXLEN:
        L = _huff(np.maximum(f, floor)); floor *= 1.5
    return L


def main():
    ctxmode = int(sys.argv[1]); train = sys.argv[2].split(','); tests = sys.argv[3:]
    nctx = {0: 1, 1: 5, 2: 25, 3: 125, 4: 10, 5: 45, 6: 225, 7: 225}[ctxmode]
    cnt = np.zeros((2, nctx, NSYM))
    for p in train:
        for tw, th, yb, sb, zy, s, zu in tiles(p):
            for pl, z in ((0, zy), (1, s)):
                S, C, _ = symbols(z, tw, ctxmode)
                np.add.at(cnt[pl], (C, S), 1)
    K = int(os.environ.get('RLK_K', '0'))                            # 0: one table per context; else cluster to K
    if K:
        cnt = cluster(cnt, K)
    L = np.zeros_like(cnt, dtype=np.int64); Lf = np.zeros_like(cnt)
    for pl in range(2):
        for c in range(nctx):
            if os.environ.get('RLK_IDEAL'):                           # fractional lengths -log2 p (rANS-like bound)
                f = cnt[pl, c] + 0.5; L[pl, c] = 0; Lf[pl, c] = -np.log2(f / f.sum())
            else:
                L[pl, c] = huff_lengths(cnt[pl, c] + 0.5)               # every symbol keeps a code
                Lf[pl, c] = L[pl, c]
    for t in tests:
        tot = collections.Counter(); px = 0
        for tw, th, yb, sb, zy, s, zu in tiles(t):
            px += tw * th
            ry, (Sy, Cy, rawy) = 0, symbols(zy, tw, ctxmode)
            ry = float(Lf[0, Cy, Sy].sum()) + rawy
            Ss, Cs, raws = symbols(s, tw, ctxmode)
            rs = float(Lf[1, Cs, Ss].sum()) + raws + tb_bits(s, zu)
            tot['plr'] += yb + sb
            tot['rlk'] += ry + rs + 2
            tot['best'] += min(yb, ry) + min(sb, rs) + 2
            tot['rlk_planes'] += (ry < yb) + (rs < sb); tot['planes'] += 2
        print(f"ctx{ctxmode} {t.split('/')[-1]:14s} PLR {tot['plr'] / px:.4f}  RLK {tot['rlk'] / px:.4f}  "
              f"per-plane min {tot['best'] / px:.4f}  (RLK wins {tot['rlk_planes']}/{tot['planes']} planes)  "
              f"lane bits only, no varint/padding")


if __name__ == '__main__':
    main()
