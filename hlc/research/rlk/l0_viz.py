#!/usr/bin/env python3
"""L = 0 on screens, visually: per pixel what PLN2 with L forced to 0 (one adaptive table per plane of a tile, ideal
order-0 cost) pays. Panels: image | zY (white = 0, darker = bigger) | S = zU + zV | bits per pixel (Y + S + zU split;
white = < 0.2 bit, yellow -> red -> black up to 12 bits).  usage: l0_viz.py dump img.ppm out.png [x0 y0 w h] [scale]"""
import sys, numpy as np
from PIL import Image
sys.path.insert(0, __import__('os').path.dirname(__file__))
import rlk_model as m
dmp, img, out = sys.argv[1:4]
crop = list(map(int, sys.argv[4:8])) if len(sys.argv) > 7 else None; sc = int(sys.argv[8]) if len(sys.argv) > 8 else 1
raw = open(img, 'rb').read(); hd = raw.split(maxsplit=4); W, H = int(hd[1]), int(hd[2])
rgb = np.frombuffer(raw[len(raw) - W * H * 3:], np.uint8).reshape(H, W, 3)
nx = (W + 287) // 288
ZY = np.zeros((H, W)); ZS = np.zeros((H, W)); BITS = np.zeros((H, W)); tot = 0.0
for ti, (tw, th, yb, sb, zy, s, zu) in enumerate(m.tiles(dmp)):
    ox, oy = (ti % nx) * 288, (ti // nx) * 288
    b = np.zeros(tw * th)
    for z in (zy, s):
        c = np.bincount(z); p = c[z] / len(z); b += -np.log2(p)
    S1 = s + 1; k = np.floor(np.log2(S1)).astype(np.int64); u = (2 << k) - S1
    b += np.where(s > 0, k + (zu >= u), 0)
    tot += b.sum()
    ZY[oy:oy + th, ox:ox + tw] = zy.reshape(th, tw); ZS[oy:oy + th, ox:ox + tw] = s.reshape(th, tw)
    BITS[oy:oy + th, ox:ox + tw] = b.reshape(th, tw)
print(f"{img.split('/')[-1]} {W}x{H}: L=0 ideal per tile plane {tot / (W * H):.4f} bpp; pixels under 0.2 bit: "
      f"{100 * (BITS < 0.2).mean():.1f}%, share of bits in pixels with zY = S = 0: {100 * BITS[(ZY == 0) & (ZS == 0)].sum() / tot:.1f}%")
def zimg(Z): v = 255 - np.minimum(np.log2(Z + 1) * 40, 255); return np.stack([v] * 3, 2).astype(np.uint8)
def heat(B):
    t = np.clip(B / 12, 0, 1); r = np.where(t < 0.5, 255, 255 * (1 - (t - 0.5) * 2)); g = np.where(t < 0.5, 255 * (1 - t * 2), 0)
    bl = np.where(B < 0.2, 255, 0); return np.stack([r, np.maximum(g, bl), bl], 2).astype(np.uint8)
P = [rgb, zimg(ZY), zimg(ZS), heat(BITS)]
if crop: x0, y0, w, h = crop; P = [p[y0:y0 + h, x0:x0 + w] for p in P]
sep = np.full((P[0].shape[0], 4, 3), 128, np.uint8)
row = np.concatenate([P[0], sep, P[1], sep, P[2], sep, P[3]], 1)
Image.fromarray(row).resize((row.shape[1] * sc, row.shape[0] * sc), Image.NEAREST).save(out)
