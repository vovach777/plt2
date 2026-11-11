#!/usr/bin/env python3
"""Size (bpp) and process wall clock (best of R, incl. file I/O) of PLR (QH), PLR-RLGR, PLF, PLN2, HALIC on P5/P6 sets,
with a round-trip check of every file.  usage: plr_rank.py R dir...
env: PLR (palic2r, QH format), PLR_RLGR (palic2r built before QH, optional), PLN2 (palic2), HALIC_DIR (optional)."""
import os, sys, subprocess, time, tempfile, filecmp
R = int(sys.argv[1]); dirs = sys.argv[2:]; E = os.environ
codecs = {}
codecs['PLR (QH)'] = ([E['PLR'], 'c'], [E['PLR'], 'd'], [])
if 'PLR_RLGR' in E: codecs['PLR (RLGR)'] = ([E['PLR_RLGR'], 'c'], [E['PLR_RLGR'], 'd'], [])
codecs['PLF fast'] = ([E['PLR'], 'c'], [E['PLR'], 'd'], ['-f'])
if 'PLN2' in E: codecs['PLN2 (FSE)'] = ([E['PLN2'], 'c'], [E['PLN2'], 'd'], [])
if 'HALIC_DIR' in E:
    H = lambda n: os.path.join(E['HALIC_DIR'], n)
    codecs['HALIC 0.7.2'] = ([H('HALIC_ENCODE_0.7.2_ST_x64_avx2_Linux_Static')], [H('HALIC_DECODE_0.7.2_ST_x64_avx2_Linux_Static')], [])
    codecs['HALIC FAST'] = ([H('HALIC_ENCODE_FAST_0.7.2_ST_x64_avx2_Linux_Static')], [H('HALIC_DECODE_FAST_0.7.2_ST_x64_avx2_Linux_Static')], [])
def run(cmd):
    t = time.perf_counter(); subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return time.perf_counter() - t
tmp = tempfile.mkdtemp()
for d in dirs:
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.ppm', '.pgm')))
    print(f'== {os.path.basename(os.path.normpath(d))} ({len(files)} files)')
    for name, (c, dc, extra) in codecs.items():
        tot = px = te = td = nb = 0; bad = 0
        for f in files:
            a, b = os.path.join(tmp, 'x.bin'), os.path.join(tmp, 'x' + f[-4:])
            try:
                be = min(run(c + [f, a] + extra) for _ in range(R)); bd = min(run(dc + [a, b]) for _ in range(R))
            except subprocess.CalledProcessError:
                bad += 1; continue
            if not filecmp.cmp(f, b, shallow=False): bad += 1
            with open(f, 'rb') as h:
                t = h.readline(); w, hh = map(int, h.readline().split())
            ch = 3 if t.startswith(b'P6') else 1
            tot += os.path.getsize(a); px += w * hh; nb += w * hh * ch; te += be; td += bd
        print(f'{name:12s} {8 * tot / px:8.4f} bpp  enc {nb / te / 1e6:6.1f} dec {nb / td / 1e6:6.1f} MB/s  ' + ('ok' if not bad else f'{bad} FAIL'))
