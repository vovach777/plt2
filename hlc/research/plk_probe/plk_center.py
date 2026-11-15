#!/usr/bin/env python3
"""PLK: let the central tile decide RLK/PLR for the whole image, luma and chroma separately (2026-11-15).

Logs as for plk_rule.py (pk_log.cpp, every plane coded both ways), two planes per 288x288 tile (Y, then S),
tiles row-major. The files are given in the order they were coded, to split the log per image.
Strategies (bits relative to the per-plane oracle):
  zq_tile   : per plane, RLK if zero-quad share >= T
  two_sum   : central + first-quarter tile coded both ways; the coder with fewer bits on the two takes all
  two_vote_zq: the two agree -> that coder for all; disagree -> per-plane zq rule
  center_Y_all: only the luma of the central tile is coded both ways; its winner takes Y and chroma of the image
  center_or : the central tile is coded both ways; its winner decides for all tiles (Y and S separately)
Usage: plk_center.py log.txt dir_or_files ... [--t 0.325]
"""
import sys, glob, os
import numpy as np

T = 0.325
args = [a for a in sys.argv[1:]]
if '--t' in args:
    i = args.index('--t'); T = float(args[i + 1]); del args[i:i + 2]
log, srcs = args[0], args[1:]
files = []
for s in srcs:
    files += sorted(glob.glob(os.path.join(s, '*.ppm'))) if os.path.isdir(s) else [s]
rows = [l.split()[1:] for l in open(log) if l.startswith('PL ')]
d = np.array(rows, dtype=float)   # chroma n pr rb plr zq b8 nz bw


def dims(fn):
    with open(fn, 'rb') as f:
        t = f.read(64).split()
    return int(t[1]), int(t[2])


tot = {k: 0.0 for k in ('oracle', 'zq_tile', 'center_or', 'center_Y_all', 'two_sum', 'two_vote_zq', 'all_rlk', 'all_plr')}
pos = 0
agree = {'Y': [0, 0], 'S': [0, 0]}
for fn in files:
    W, H = dims(fn)
    cols, rws = (W + 287) // 288, (H + 287) // 288
    nt = cols * rws
    blk = d[pos:pos + 2 * nt]; pos += 2 * nt
    assert len(blk) == 2 * nt, fn
    assert blk[0, 1] == min(288, W) * min(288, H), (fn, blk[0, 1])
    ci = (rws // 2) * cols + cols // 2
    yc = blk[0::2][ci]; y_rlk = yc[3] < yc[4]                           # luma of the central tile decides all
    for p in (0, 1):
        pl = blk[p::2]; tot['center_Y_all'] += (pl[:, 3] if y_rlk else pl[:, 4]).sum()
    for p, name in ((0, 'Y'), (1, 'S')):
        pl = blk[p::2]
        rb, plr, zq = pl[:, 3], pl[:, 4], pl[:, 5]
        tot['oracle'] += np.minimum(rb, plr).sum()
        tot['zq_tile'] += np.where(zq >= T, rb, plr).sum()
        c_rlk_or = rb[ci] < plr[ci]
        qi = (rws // 4) * cols + cols // 4                                   # second probe tile: first quarter
        if qi == ci: qi = 0
        two_rlk = rb[ci] + rb[qi] < plr[ci] + plr[qi]                        # winner of the two takes all
        tot['two_sum'] += (rb if two_rlk else plr).sum()
        a1, a2 = rb[ci] < plr[ci], rb[qi] < plr[qi]
        if a1 == a2: tot['two_vote_zq'] += (rb if a1 else plr).sum()
        else: tot['two_vote_zq'] += np.where(zq >= T, rb, plr).sum()      # disagree: per-plane zq
        tot['center_or'] += (rb if c_rlk_or else plr).sum()
        tot['all_rlk'] += rb.sum(); tot['all_plr'] += plr.sum()
        win = rb < plr
        agree[name][0] += int((win == c_rlk_or).sum()); agree[name][1] += len(win)
assert pos == len(d), (pos, len(d))
o = tot['oracle']
print(f'{log.split("/")[-1]:12s} ' + '  '.join(f'{k} {v / o:.4f}' for k, v in tot.items())
      + f'  | tiles agreeing with centre: Y {100 * agree["Y"][0] / agree["Y"][1]:.0f}%  S {100 * agree["S"][0] / agree["S"][1]:.0f}%')
