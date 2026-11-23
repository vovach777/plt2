#!/usr/bin/env python3
"""Collect per-tile DIAG counters (n, zY, zU, zV, zS) and the 4 mode-pair byte sizes
for every image of the six screen groups + the photo group.

Outputs:
  /tmp/r8f5/diag_cache/<group>/<stem>.npy   (ntiles, 9) int64
  p2/diag/holdout_<group>.txt               "<img> DIAG ..." lines in diag_oracle.py format
"""
import os, sys, subprocess, glob, numpy as np
from concurrent.futures import ProcessPoolExecutor

SETS = '/tmp/r8f5/sets'
CACHE = '/tmp/r8f5/diag_cache'
PROBE = '/tmp/r8f5/pls_probe2'
DIAG = '/home/devin/train/bench-independent/p2/diag'
TMP = '/tmp/opencode/diagwork'

def run_one(args):
    g, path = args
    stem = os.path.basename(path)[:-4]
    out = os.path.join(CACHE, g, stem + '.npy')
    if os.path.exists(out):
        return g, stem, 0
    os.makedirs(TMP, exist_ok=True)
    tmp = os.path.join(TMP, str(os.getpid()))
    os.makedirs(tmp, exist_ok=True)
    r = subprocess.run([PROBE, 'c', path, os.path.join(tmp, 'a.pls'), '-s=probe2'],
                       env=dict(os.environ, PLS_DIAG='1'), capture_output=True, text=True)
    if r.returncode != 0:
        return g, stem, -1
    rows = [ln.split()[1:] for ln in r.stderr.splitlines() if ln.startswith('DIAG ')]
    a = np.array(rows, dtype=np.int64) if rows else np.zeros((0, 9), np.int64)
    np.save(out, a)
    return g, stem, len(a)

def main():
    groups = sys.argv[1:] or ['ios', 'desktop', 'websight', 'docs', 'scans', 'web', 'photo']
    jobs = []
    for g in groups:
        os.makedirs(os.path.join(CACHE, g), exist_ok=True)
        for p in sorted(glob.glob(os.path.join(SETS, g, 'all', '*.ppm'))):
            jobs.append((g, p))
    print('images:', len(jobs), flush=True)
    nimg = 0; nfail = 0; ntile = 0
    with ProcessPoolExecutor(max_workers=10) as ex:
        for g, stem, n in ex.map(run_one, jobs, chunksize=4):
            if n < 0: nfail += 1; print('FAIL', g, stem, flush=True)
            else: ntile += n
            nimg += 1
            if nimg % 250 == 0: print('done', nimg, 'tiles', ntile, flush=True)
    print('DIAG-DONE images=%d tiles=%d fails=%d' % (nimg, ntile, nfail), flush=True)

    # holdout logs for diag_oracle.py (prefix image name like Claude's sed does)
    os.makedirs(DIAG, exist_ok=True)
    for g in ['ios', 'desktop', 'websight', 'docs', 'scans', 'web', 'photo']:
        listfile = os.path.join(SETS, 'photo_hold.txt' if g == 'photo' else os.path.join(g, 'hold.txt'))
        stems = [x.strip() for x in open(listfile) if x.strip()]
        lines = []
        for s in stems:
            a = np.load(os.path.join(CACHE, g, s[:-4] + '.npy'))
            for row in a:
                lines.append(s + ' DIAG ' + ' '.join(str(x) for x in row))
        open(os.path.join(DIAG, 'holdout_%s.txt' % g), 'w').write('\n'.join(lines) + '\n')
        print('holdout_%s.txt tiles=%d' % (g, len(lines)), flush=True)

if __name__ == '__main__':
    main()