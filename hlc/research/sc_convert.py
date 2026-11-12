#!/usr/bin/env python3
"""PNG screen-content sets of github imazen/codec-corpus (v1.1.0: gb82-sc, qoi-benchmark/screenshot_web) -> PPM / PGM
for the PLR benches; alpha and palettes are dropped to RGB. Prints the PNG bpp of the set (alpha included).
usage: sc_convert.py SRC_DIR DST_DIR"""
import sys, glob, os
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True); tot = px = 0
for f in sorted(glob.glob(src + '/*.png')):
    im = Image.open(f)
    if im.mode not in ('L', 'RGB'): im = im.convert('RGB')
    n = os.path.splitext(os.path.basename(f))[0]; im.save(f'{dst}/{n}.' + ('pgm' if im.mode == 'L' else 'ppm'))
    tot += os.path.getsize(f); px += im.size[0] * im.size[1]
print(f'{src}: PNG {8 * tot / px:.4f} bpp')
