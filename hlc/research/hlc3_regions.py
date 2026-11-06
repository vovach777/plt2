# Compress seed-grown regions (pair-count feature C on raw G, ratio rule) with real HLC3: each rectangle cropped and
# coded as one HLC3 tile (own tables + predictor), file header subtracted, +4 B geometry per rect.
# Compared with HLC3 default (256x128 tiles) and fixed 48x48 tiles. Needs score_raw.txt from tile_pair_score_rawG.c.
# usage: python hlc3_regions.py   (paths below are the research sandbox ones; edit S/HLC)
import sys, os, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path[0] = os.path.dirname(os.path.abspath(__file__))
exec(open(sys.path[0] + '/grow_rect_ratio.py').read().split('COLS=')[0]); MAXD = 1; MODE = 0; SQ = 0; ASPMAX = 0
from PIL import Image
O = S + '/h3reg'; HLC = '/tmp/rb6/hlc'; os.makedirs(O, exist_ok=True)
SC = {}
for l in open(S + '/clu/score_raw.txt'):
    f, i, j, h, s = l.split(); SC.setdefault(f.split('/')[-1][:-4], []).append(float(s) - 2303)
def vlen(v):
    n = 1
    while v >= 128: v >>= 7; n += 1
    return n
def hlc(src, dst, extra):
    if not os.path.exists(dst): subprocess.run([HLC, 'c', src, dst] + extra, check=True, capture_output=True)
    return os.path.getsize(dst)
names = [d[0] for d in D if d[0] != 'p02']; jobs = []; meta = {}
for name in names:
    img = f'{S}/photos16h/{name}.ppm' if name != 'cat_h' else f'{S}/cat_h.ppm'
    A = Image.open(img).convert('RGB'); Ww, Hh = A.size; ty, tx = (Hh + 47) // 48, (Ww + 47) // 48; meta[name] = (Ww, Hh)
    jobs.append(('base', name, None, img, f'{O}/{name}_base.hlc', []))
    jobs.append(('t48', name, None, img, f'{O}/{name}_t48.hlc', ['-tw', '48', '-th', '48']))
    for thr in (0.6, 0.75):
        rid, R = grow(np.array(SC[name]).reshape(ty, tx), thr, 0, 0)
        for k, (x0, y0, x1, y1, m) in enumerate(R):
            X0, Y0, X1, Y1 = x0 * 48, y0 * 48, min(x1 * 48, Ww), min(y1 * 48, Hh); p = f'{O}/{name}_{thr}_{k}.ppm'
            if not os.path.exists(p): A.crop((X0, Y0, X1, Y1)).save(p)
            jobs.append((f'reg{thr}', name, (X1 - X0, Y1 - Y0), p, p[:-4] + '.hlc', ['-tw', str(X1 - X0), '-th', str(Y1 - Y0)]))
with ThreadPoolExecutor(os.cpu_count()) as ex: sizes = list(ex.map(lambda j: hlc(j[3], j[4], j[5]), jobs))
tot = {}; per = {n: {} for n in names}; nreg = {}
for j, s in zip(jobs, sizes):
    kind, name, wh = j[0], j[1], j[2]
    if wh: s = s - (4 + vlen(wh[0]) + vlen(wh[1]) + 1 + vlen(wh[0]) + vlen(wh[1])) + 4
    tot[kind] = tot.get(kind, 0) + s; per[name][kind] = per[name].get(kind, 0) + s; nreg[kind] = nreg.get(kind, 0) + 1
npx = sum(w * h for w, h in meta.values()); b = tot['base']
for k in ('base', 't48', 'reg0.6', 'reg0.75'):
    print(f'  {k:8s} {tot[k]:9d} B  {8 * tot[k] / npx:.4f} bpp  {100 * (tot[k] / b - 1):+.2f}%')
for n in names: print(f'  {n:6s}  ' + '  '.join(f'{100 * (per[n][k] / per[n]["base"] - 1):+6.2f}' for k in ('t48', 'reg0.6', 'reg0.75')))
