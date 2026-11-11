#!/usr/bin/env python3
"""llrice coder variants vs palic2r (PLR / PLF) on P6 sets: size (bpp) and process wall clock (best of R).
usage: llrice_vs_plr.py R dir... with env LLRB=dir of llrice builds <variant>-c/-d, PLR=palic2r binary."""
import os, sys, subprocess, time, tempfile, filecmp
R = int(sys.argv[1]); dirs = sys.argv[2:]
LLRB = os.environ['LLRB']; PLR = os.environ['PLR']
codecs = {v: ([f'{LLRB}/{v}-c'], [f'{LLRB}/{v}-d']) for v in ('grb1', 'grb3', 'sumraw', 'sumtb')}
codecs['PLR'] = ([PLR, 'c'], [PLR, 'd']); codecs['PLF'] = ([PLR, 'c'], [PLR, 'd'])
def run(cmd):
    t = time.perf_counter(); subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return time.perf_counter() - t
tmp = tempfile.mkdtemp()
for d in dirs:
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.ppm'))
    print(f'== {d} ({len(files)} files)')
    for name, (c, dc) in codecs.items():
        tot = px = te = td = 0; ok = True
        for f in files:
            a, b = os.path.join(tmp, 'x.bin'), os.path.join(tmp, 'x.ppm')
            extra = ['-f'] if name == 'PLF' else []
            be = min(run(c + [f, a] + extra) for _ in range(R)); bd = min(run(dc + [a, b]) for _ in range(R))
            ok &= filecmp.cmp(f, b, shallow=False); tot += os.path.getsize(a)
            with open(f, 'rb') as h:
                h.readline(); w, hh = map(int, h.readline().split())
            px += w * hh; te += be; td += bd
        print(f'{name:7s} {8 * tot / px:8.4f} bpp  enc {3 * px / te / 1e6:6.1f} dec {3 * px / td / 1e6:6.1f} MB/s  {"ok" if ok else "FAIL"}')
