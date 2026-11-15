#!/usr/bin/env python3
"""How much does bias correction of MED help, and at what granularity? (PALIC-1 question, 2026-11-15)

For each image: palic-3 RCT (U=R-G, V=B-G, Y=G+((U+V+2)>>2), V-=U>>2), MED residual e per plane.
Metric: empirical order-0 entropy of e per plane (bits/sample, one histogram per plane and image),
summed over the 3 planes = bpp. Correction is subtracted as round(mean) of e in the group:
  none      : e
  plane     : one mean per plane
  tile64    : one mean per 64x64 tile
  *_med     : the same with the median instead of the mean
  ctx_fnbli : oracle static mean per fNBLI context (12 energy x 8 texture bits = 3072), two passes
  ctx_tex   : the same, texture bits only (256 contexts)
  ctx_qd    : energy only (12 contexts)
Side information (the means) is NOT counted: this is an upper bound for a static correction.
Usage: bias_entropy.py a.ppm ...
"""
import sys
import numpy as np


def read_ppm(fn):
    with open(fn, 'rb') as f:
        data = f.read()
    tok, pos = [], 0
    while len(tok) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b'#':
            while data[pos:pos + 1] not in (b'\n', b''):
                pos += 1
            continue
        s = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tok.append(data[s:pos])
    pos += 1
    w, h = int(tok[1]), int(tok[2])
    return np.frombuffer(data, np.uint8, w * h * 3, pos).reshape(h, w, 3).astype(np.int32)


def ent(e):
    _, c = np.unique(e, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def shifted(P, dy, dx):
    """P shifted so that out[y, x] = P[y - dy, x - dx], edges replicated."""
    h, w = P.shape
    Q = np.pad(P, ((2, 2), (2, 2)), mode='edge')
    return Q[2 - dy:2 - dy + h, 2 - dx:2 - dx + w]


def med_res(P):
    a, b, c = shifted(P, 0, 1), shifted(P, 1, 0), shifted(P, 1, 1)
    p = np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b))
    return P - p, p


def group_mean_sub(e, g):
    """subtract round(mean of e) inside each group label g"""
    g = g.ravel()
    s = np.bincount(g, weights=e.ravel())
    n = np.bincount(g)
    m = np.round(s / np.maximum(n, 1)).astype(np.int32)
    return e - m[g].reshape(e.shape)


def group_median_sub(e, g):
    """subtract the median of e inside each group label g"""
    g = g.ravel(); v = e.ravel()
    o = np.lexsort((v, g)); gs, vs = g[o], v[o]
    n = np.bincount(g); start = np.concatenate(([0], np.cumsum(n)[:-1]))
    med = np.zeros(len(n), np.int32); nz = n > 0
    med[nz] = vs[start[nz] + (n[nz] - 1) // 2]
    return e - med[g].reshape(e.shape)


def ctx(P, p, e):
    a, b, c, d = shifted(P, 0, 1), shifted(P, 1, 0), shifted(P, 1, 1), shifted(P, 1, -1)
    ee, f, g = shifted(P, 0, 2), shifted(P, 2, 0), shifted(P, 2, -1)
    dh = np.abs(a - ee) + np.abs(b - c) + np.abs(b - d)
    dv = np.abs(a - c) + np.abs(b - f) + np.abs(d - g)
    err_a = np.abs(shifted(e, 0, 1))                       # causal: error of the left neighbour
    act = dh + dv + 2 * err_a
    qd = np.searchsorted(np.array([2, 4, 7, 11, 17, 26, 40, 60, 90, 135, 200]), act, side='right')  # 12 levels
    tex = np.zeros_like(P)
    for k, v in enumerate((a, b, c, d, ee, f, 2 * a - ee, 2 * b - f)):
        tex |= (p > v).astype(np.int32) << k
    return qd, tex


def main():
    files = sys.argv[1:]
    modes = ['none', 'plane', 'tile64', 'ctx_qd', 'ctx_tex', 'ctx_fnbli', 'tile64_med', 'ctx_tex_med', 'ctx_fnbli_med']
    tot = {m: 0.0 for m in modes}
    npx = 0
    for fn in files:
        I = read_ppm(fn)
        R, G, B = I[..., 0], I[..., 1], I[..., 2]
        U, V = R - G, B - G
        Y = G + ((U + V + 2) >> 2)
        V = V - (U >> 2)
        h, w = Y.shape
        ty, tx = np.indices((h, w))
        tile = (ty // 64) * ((w + 63) // 64) + tx // 64
        row = {m: 0.0 for m in modes}
        for P in (U, V, Y):
            e, p = med_res(P)
            qd, tex = ctx(P, p, e)
            row['none'] += ent(e)
            row['plane'] += ent(e - int(round(e.mean())))
            row['tile64'] += ent(group_mean_sub(e, tile))
            row['ctx_qd'] += ent(group_mean_sub(e, qd))
            row['ctx_tex'] += ent(group_mean_sub(e, tex))
            row['ctx_fnbli'] += ent(group_mean_sub(e, qd * 256 + tex))
            row['tile64_med'] += ent(group_median_sub(e, tile))
            row['ctx_tex_med'] += ent(group_median_sub(e, tex))
            row['ctx_fnbli_med'] += ent(group_median_sub(e, qd * 256 + tex))
        n = h * w
        npx += n
        for m in modes:
            tot[m] += row[m] * n
        print(f"{fn.split('/')[-1][:28]:28s} " + " ".join(f"{m} {row[m]:.4f}" for m in modes), flush=True)
    base = tot['none'] / npx
    print("TOTAL bpp (order-0 entropy, side info not counted):")
    for m in modes:
        v = tot[m] / npx
        print(f"  {m:10s} {v:.4f}  {100 * (v / base - 1):+.2f}%")


if __name__ == '__main__':
    main()
