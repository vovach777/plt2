#!/usr/bin/env python3
"""JPEG-style (run, k) Huffman on MED residuals: how much would it give vs per-value coding?

Planes as in PALIC/PLR: RCT (U=R-G, V=B-G, Y=G+((U+V+2)>>2), V-=U>>2), MED, zigzag (9-bit wrap U/V, 8-bit Y);
coded planes: Y and S = zU + zV (zU then costs a truncated binary code over 0..S).
Symbol streams (raster order per plane, one static Huffman per plane per image, table cost ignored):
  value : symbol = k = bitwidth(z) (0..10), + (k-1) raw bits                  (PALIC-1 style)
  rlk   : symbol = (run of zeros 0..15, k of the next nonzero), ZRL = 16 zeros, + (k-1) raw bits; EOB at row end
Prints bits per pixel of RGB for each scheme: Huffman lengths and ideal entropy.
"""
import sys, heapq, math, collections
import numpy as np

def read_ppm(p):
    d = open(p, 'rb').read(); tok = []; i = 0
    while len(tok) < 4:
        while d[i:i+1].isspace(): i += 1
        if d[i:i+1] == b'#':
            while d[i:i+1] != b'\n': i += 1
            continue
        j = i
        while not d[j:j+1].isspace(): j += 1
        tok.append(d[i:j]); i = j
    W, H = int(tok[1]), int(tok[2])
    return np.frombuffer(d[i+1:i+1+W*H*3], np.uint8).reshape(H, W, 3).astype(np.int32)

def med_res(x):
    a = np.zeros_like(x); b = np.zeros_like(x); c = np.zeros_like(x)
    a[:, 1:] = x[:, :-1]; b[1:, :] = x[:-1, :]; c[1:, 1:] = x[:-1, :-1]
    p = np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b))
    p[0, 1:] = x[0, :-1]; p[0, 0] = 0; p[1:, 0] = x[:-1, 0]
    return x - p

def zz(e, bits):
    m = 1 << bits; e = ((e + m // 2) % m) - m // 2
    return np.where(e >= 0, 2 * e, -2 * e - 1)

def huff_bits(cnt):
    if len(cnt) == 1: return sum(cnt.values())
    h = [(n, i, None) for i, n in enumerate(cnt.values())]; heapq.heapify(h); ln = [0] * len(h); k = len(h)
    par = {}
    while len(h) > 1:
        a = heapq.heappop(h); b = heapq.heappop(h); par[a[1]] = k; par[b[1]] = k
        heapq.heappush(h, (a[0] + b[0], k, None)); k += 1
    vals = list(cnt.values()); tot = 0
    for i, n in enumerate(vals):
        d, j = 0, i
        while j in par: j = par[j]; d += 1
        tot += n * d
    return tot

def ent_bits(cnt):
    n = sum(cnt.values()); return -sum(c * math.log2(c / n) for c in cnt.values())

def bw(v): return np.where(v > 0, np.floor(np.log2(np.maximum(v, 1))).astype(np.int64) + 1, 0)

def costs(z):
    k = bw(z); raw = int(np.maximum(k - 1, 0).sum())
    cv = collections.Counter(k.ravel().tolist())
    cr = collections.Counter()
    for row, krow in zip(z, k):
        run = 0
        for kk in krow.tolist():
            if kk == 0:
                run += 1
                if run == 16: cr['ZRL'] += 1; run = 0
            else:
                cr[(run, kk)] += 1; run = 0
        if run: cr['EOB'] += 1
    return dict(value_h=huff_bits(cv) + raw, value_e=ent_bits(cv) + raw, rlk_h=huff_bits(cr) + raw, rlk_e=ent_bits(cr) + raw)

def tb_bits(S):  # truncated binary over 0..S, average length
    m = S + 1; k = np.floor(np.log2(m)).astype(np.int64); u = (2 << k) - m
    return float(np.sum(k + (1 - u / m)))  # expected length for uniform zU

tot = collections.Counter(); px = 0
for p in sys.argv[1:]:
    a = read_ppm(p); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    U = R - G; V = B - G; Y = G + ((U + V + 2) >> 2); V = V - (U >> 2)
    zu, zv, zy = zz(med_res(U), 9), zz(med_res(V), 9), zz(med_res(Y), 8)
    S = zu + zv
    split = tb_bits(S)
    cy, cs = costs(zy), costs(S)
    for key in cy: tot[key] += cy[key] + cs[key] + split
    px += a.shape[0] * a.shape[1]
for key in ('value_h', 'value_e', 'rlk_h', 'rlk_e'):
    print(f'{key:8s} {tot[key] / px:.4f} bpp')
