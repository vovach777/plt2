#!/usr/bin/env python3
"""Asymmetric quantisation after the central mirror. Flip s = sign(c-a), else sign(c-b), else +1 (tiny sign rule);
ga' = s(c-a) >= 0, gb' = s(c-b): gb' > 0 = extremum side (c above/below both), gb' < 0 = ramp side (MED planar branch).
Context = q(ga', E7) x q(gb', E+ on the extremum side / E- on the ramp side), residual flipped by s. 25 contexts.
usage: python asym_q.py <ppm/png files or globs...>"""
import numpy as np, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uv_axes import H, qa, shift, E7
from PIL import Image
VAR = [([1, 4, 16], [1, 4, 16]), ([1, 4, 16], [1, 3, 12]), ([1, 4, 16], [1, 3, 10]), ([1, 4, 16], [1, 2, 8]),
       ([1, 4, 16], [1, 5, 20]), ([1, 5, 20], [1, 4, 16]), ([1, 5, 20], [1, 3, 12]), ([1, 3, 12], [1, 4, 16]), ([1, 4, 24], [1, 3, 12])]
def main(files):
    tot = np.zeros(len(VAR)); px = 0
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        G = a0[..., 1]; Vv = a0[..., 0] - G; U = a0[..., 2] - G; Y = G + ((U + Vv) >> 2); px += Y.size
        for P, n9 in [(Vv, True), (U, True), (Y, False)]:
            a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); ga, gb = c - a, c - b
            s = np.where(ga != 0, np.sign(ga), np.where(gb != 0, np.sign(gb), 1)); ga, gb = s * ga, s * gb
            ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
            for i, (ep, em) in enumerate(VAR):
                cls = qa(ga, E7) * 16 + np.where(gb >= 0, qa(gb, ep), -qa(gb, em)) + 8
                tot[i] += sum(H(z[cls == v]) for v in np.unique(cls))
    for (ep, em), t in zip(VAR, tot):
        print(f'extremum {str(ep):12s} ramp {str(em):12s} {t / px:.4f} bpp  {100 * t / tot[0] - 100:+.2f}%')
if __name__ == '__main__':
    main(sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[1:]), [])))
