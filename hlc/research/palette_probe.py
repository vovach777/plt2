#!/usr/bin/env python3
"""palette_probe.py - a P6 image with few colours -> palette-index PGMs (index order by luminance and by frequency)
for coding with the gray PLR / HALIC, plus index entropies: order 0, of MED residuals, and conditional on (W, N, NW)
indices.  usage: palette_probe.py in.ppm out_dir"""
import sys, os, numpy as np
fn, od = sys.argv[1], sys.argv[2]; os.makedirs(od, exist_ok=True)
f = open(fn, 'rb'); f.readline(); w, h = map(int, f.readline().split()); f.readline()
p = np.frombuffer(f.read(), np.uint8).reshape(h, w, 3)
code = (p[..., 0].astype(np.int64) << 16) | (p[..., 1].astype(np.int64) << 8) | p[..., 2]
cols, inv, cnt = np.unique(code.ravel(), return_inverse=True, return_counts=True)
rgb = np.stack([(cols >> 16) & 255, (cols >> 8) & 255, cols & 255], 1); lum = rgb @ np.array([299, 587, 114])
def H(x): _, c = np.unique(x, return_counts=True); q = c / c.sum(); return float(-(c * np.log2(q)).sum())
print(f'{len(cols)} colours')
for name, order in (('lum', np.argsort(lum)), ('freq', np.argsort(-cnt))):
    rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order)); idx = rank[inv].reshape(h, w)
    open(f'{od}/idx_{name}.pgm', 'wb').write(b'P5\n%d %d\n255\n' % (w, h) + idx.astype(np.uint8).tobytes())
    q = idx.astype(np.int64); W = np.zeros_like(q); W[:, 1:] = q[:, :-1]; N = np.zeros_like(q); N[1:] = q[:-1]
    NW = np.zeros_like(q); NW[1:, 1:] = q[:-1, :-1]; med = np.clip(W + N - NW, np.minimum(W, N), np.maximum(W, N))
    ctx = (W * 256 + N) * 256 + NW
    print(f'{name}: H(index) {H(q) / q.size:.3f}  H(MED resid) {H((q - med) & 255) / q.size:.3f}  H(index | W,N,NW) {(H(ctx * 256 + q) - H(ctx)) / q.size:.3f} bits/px')
