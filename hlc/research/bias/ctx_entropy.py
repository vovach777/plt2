#!/usr/bin/env python3
"""What does a context buy over none? Conditional order-0 entropy H(e | ctx) of the MED residual (2026-11-15).

palic-3 RCT (U=R-G, V=B-G, Y=G+((U+V+2)>>2), V-=U>>2), MED residual e per plane, one histogram per
(image, plane, context); tables are NOT charged (upper bound for a semi-static / adaptive coder).
Contexts (all causal):
  none     : one histogram per plane
  Lup16    : PLR/PLN L: bit_width(sum |e| of the 16 px above the 16x1 block >> 4) - 2, clamp 0..7   (8)
  up3      : bit_width(|e_b| + |e_c| + |e_d|), clamp 0..10   (sliding error above, 3 px)       (11)
  enb      : llcomp Enb = min(8, bit_width(max(|e_a|, |e_b|)))                                 (9)
  enb_up3  : bit_width(|e_a| + |e_b| + |e_c| + |e_d|)                                          (11)
  qd       : fNBLI: lut(dvdh + 2|err_a|), 12 levels (one plane, no cross-channel error)         (12)
  tiny     : llcomp -x 0: q5(a-c, c-b, d-b) mirrored by sign (63) x Enb (9), residual sign flipped (567)
Usage: ctx_entropy.py a.ppm ...
"""
import sys
import numpy as np
from bias_entropy import read_ppm, shifted, med_res

QD_EDGES = np.array([1, 2, 4, 6, 9, 15, 25, 39, 63, 101, 151])          # fNBLI lut_qd steps (approx.)


def bw(x):
    x = np.asarray(x)
    out = np.zeros(x.shape, np.int32)
    v = x.copy()
    while np.any(v > 0):
        out += (v > 0)
        v = v >> 1
    return out


def cond_ent(e, c):
    """sum over contexts of n_c * H(e | c), in bits"""
    e = e.ravel().astype(np.int64); c = c.ravel().astype(np.int64)
    key = c * 4096 + (e + 2048)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    ctx_of = np.unique(key) // 4096
    tot = np.bincount(ctx_of, weights=cnt)
    p = cnt / tot[ctx_of]
    return float(-(cnt * np.log2(p)).sum())


def q5(g, t1):
    return np.where(g == 0, 0, np.where(np.abs(g) <= t1, np.sign(g), 2 * np.sign(g)))


def contexts(P, e, luma):
    a, b, c, d = shifted(P, 0, 1), shifted(P, 1, 0), shifted(P, 1, 1), shifted(P, 1, -1)
    ee, f, g = shifted(P, 0, 2), shifted(P, 2, 0), shifted(P, 2, -1)
    ea, eb, ec, ed = (np.abs(shifted(e, dy, dx)) for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)))
    h, w = P.shape
    out = {}
    out['none'] = np.zeros_like(P)
    # PLR L: blocks of 16 x 1, sum of |e| of the 16 px above
    ab = np.abs(shifted(e, 1, 0))
    wp = (w + 15) // 16 * 16
    abp = np.zeros((h, wp), np.int64); abp[:, :w] = ab
    s16 = abp.reshape(h, wp // 16, 16).sum(axis=2) >> 4
    L = np.clip(bw(s16) - 2, 0, 7)
    out['Lup16'] = np.repeat(L, 16, axis=1)[:, :w]
    out['up3'] = np.minimum(bw(eb + ec + ed), 10)
    out['enb'] = np.minimum(bw(np.maximum(ea, eb)), 8)
    out['enb_up3'] = np.minimum(bw(ea + eb + ec + ed), 10)
    dh = np.abs(a - ee) + np.abs(b - c) + np.abs(b - d)
    dv = np.abs(a - c) + np.abs(b - f) + np.abs(d - g)
    out['qd'] = np.searchsorted(QD_EDGES, dh + dv + 2 * ea, side='right')
    t1 = 3 if luma else 4
    g1, g2, g3 = q5(a - c, t1), q5(c - b, t1), q5(d - b, t1)
    sgn = np.where(g1 != 0, np.sign(g1), np.where(g2 != 0, np.sign(g2), np.where(g3 != 0, np.sign(g3), 1)))
    cell = ((g1 * sgn + 2) * 25 + (g2 * sgn + 2) * 5 + (g3 * sgn + 2))          # mirrored: half the cells used
    out['tiny'] = cell * 9 + out['enb']
    return out, sgn


def main():
    modes = ['none', 'Lup16', 'up3', 'enb', 'enb_up3', 'qd', 'tiny']
    tot = {m: 0.0 for m in modes}
    npx = 0
    for fn in sys.argv[1:]:
        I = read_ppm(fn)
        R, G, B = I[..., 0], I[..., 1], I[..., 2]
        U, V = R - G, B - G
        Y = G + ((U + V + 2) >> 2)
        V = V - (U >> 2)
        for P, luma in ((U, False), (V, False), (Y, True)):
            e, _ = med_res(P)
            cx, sgn = contexts(P, e, luma)
            for m in modes:
                ev = e * sgn if m == 'tiny' else e
                tot[m] += cond_ent(ev, cx[m])
        npx += Y.size
    base = tot['none'] / npx
    print(f"{len(sys.argv) - 1} files, bpp = H(e | ctx) over U, V, Y (tables not charged):")
    for m in modes:
        v = tot[m] / npx
        print(f"  {m:8s} {v:.4f}  {100 * (v / base - 1):+6.2f}%")


if __name__ == '__main__':
    main()
