#!/usr/bin/env python3
"""sc_probe.py - why HALIC wins on screen content: per image, colours and repeats vs what predictors can do.
Columns: unique colours (whole image / max per 288x288 tile), share of pixels equal to the left / upper / either
neighbour (exact RGB), and order-0 entropy (bits per pixel, 3 channels, per-image histograms) of the residuals of
MED, and of the per-16-pixel-block best of {W, N, NW, NE, MED, (W+N)/2, GAP-like gradient} (oracle choice, the choice
itself not counted: an optimistic bound for "a better predictor"). usage: sc_probe.py img.ppm ..."""
import sys, numpy as np
def rd(fn):
    f = open(fn, 'rb'); assert f.readline().strip() == b'P6'; w, h = map(int, f.readline().split()); f.readline()
    return np.frombuffer(f.read(), np.uint8).reshape(h, w, 3).astype(np.int32)
def H(x):
    _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -(c * np.log2(p)).sum()
for fn in sys.argv[1:]:
    p = rd(fn); h, w, _ = p.shape
    if h * w > 12_000_000: print(f"{fn.split('/')[-1]:34s} skipped: {w}x{h} too big for this in-memory probe"); continue
    W = np.zeros_like(p); W[:, 1:] = p[:, :-1]; N = np.zeros_like(p); N[1:] = p[:-1]
    NW = np.zeros_like(p); NW[1:, 1:] = p[:-1, :-1]; NE = np.zeros_like(p); NE[1:, :-1] = p[:-1, 1:]
    mn, mx = np.minimum(W, N), np.maximum(W, N); med = np.clip(W + N - NW, mn, mx)
    preds = {'W': W, 'N': N, 'NW': NW, 'NE': NE, 'MED': med, 'AVG': (W + N) >> 1, 'GRAD': np.clip(W + NE - N, 0, 255)}
    code = (p[..., 0] << 16) | (p[..., 1] << 8) | p[..., 2]
    ncol = len(np.unique(code)); tmax = max(len(np.unique(code[y:y + 288, x:x + 288])) for y in range(0, h, 288) for x in range(0, w, 288))
    eqW = (p == W).all(2); eqN = (p == N).all(2)
    res = {k: ((p - v + 256) & 255) for k, v in preds.items()}
    hm = sum(H(res['MED'][..., c]) for c in range(3)) / (h * w)
    # oracle best predictor per 16-pixel row block: choose the predictor with the smallest sum |r| (wrapped) in the block
    w16 = w // 16 * 16; costs = []; keys = list(res)
    for k in keys:
        r = res[k][:, :w16]; a = np.minimum(r, 256 - r).sum(2).reshape(h, w16 // 16, 16).sum(2); costs.append(a)
    best = np.argmin(np.stack(costs), 0)
    sel = np.zeros((h, w16, 3), np.int32)
    for i, k in enumerate(keys): sel += np.repeat(best == i, 16, axis=1)[..., None] * res[k][:, :w16]
    hb = sum(H(sel[..., c]) for c in range(3)) / (h * w16)
    share = np.bincount(best.ravel(), minlength=len(keys)) / best.size
    print(f"{fn.split('/')[-1]:34s} colours {ncol:7d} (tile max {tmax:6d})  =W {eqW.mean():.2f} =N {eqN.mean():.2f} =W|N {(eqW | eqN).mean():.2f}"
          f"  H(MED) {hm:.3f}  H(best-of-7 oracle) {hb:.3f}  picks " + ' '.join(f'{k}:{s:.2f}' for k, s in zip(keys, share)))
