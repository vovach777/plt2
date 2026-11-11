#!/usr/bin/env python3
"""Estimate (also of (b) with the shrink of a full window whose hole ends at the head) of PLR variant (b) "one bit stream per tile + run-header holes" from the counters of a stats build of
palic2r (STATS line on stderr of `c`): size_b = file - pages + bits/8, where bits = payload + k per full window
(the hole of 1 + k when only the bit 0 is used) + the hole of an open run at the end; pages are replaced by one
bit stream per tile (rounded to 64 bits: + 4 bytes per tile on average).
STATS_BINARY: palic2r.cpp built with -DRLGR_STATS -DPLR_LAYOUT=0 (the page engine the counters live in).
usage: plr_onestream_estimate.py STATS_BINARY [-f] dir..."""
import os, sys, subprocess, tempfile
b = sys.argv[1]; args = sys.argv[2:]; fast = '-f' in args; dirs = [a for a in args if a != '-f']
tmp = os.path.join(tempfile.mkdtemp(), 'x')
for d in dirs:
    T = dict(size=0, payload=0, wfull=0, nfull=0, wflush=0, runs=0, px=0, tiles=0, head=0, nshrink=0, lost=0)
    for f in sorted(os.listdir(d)):
        if not f.endswith(('.ppm', '.pgm')): continue
        fn = os.path.join(d, f)
        r = subprocess.run([b, 'c', fn, tmp] + (['-f'] if fast else []), capture_output=True, text=True, check=True)
        st = [l for l in r.stderr.splitlines() if l.startswith('STATS')][0].split()[1:]
        size, pay, wf, nf, wfl, runs, nt = map(int, st)
        sb = [l for l in r.stderr.splitlines() if l.startswith('SIMB')][0].split()[1:]
        head, nsh, lost = map(int, sb); T['head'] += head; T['nshrink'] += nsh; T['lost'] += lost
        with open(fn, 'rb') as h:
            h.readline(); w, hh = map(int, h.readline().split())
        T['size'] += os.path.getsize(tmp); T['payload'] += pay; T['wfull'] += wf; T['nfull'] += nf
        T['wflush'] += wfl; T['runs'] += runs; T['px'] += w * hh; T['tiles'] += nt
    px = T['px']; cur = 8 * T['size'] / px
    # current file: header + tile index + per tile var(npages) + pages; payload bits + laid-unused + page tails
    est_b = (T['size'] * 8 - 0 + (T['wfull'] + T['wflush']) - (T['size'] * 8 - T['payload']) + 0) / px  # payload + holes
    over = T['size'] * 8 - T['payload']        # bits in the current file that are not payload: headers, page tails, unused
    print(f"== {d} {'PLF' if fast else 'PLR'}: current {cur:.4f} bpp; payload {T['payload']/px:.4f}; non-payload now "
          f"{over/px:.4f} bpp ({100*over/(T['size']*8):.2f}%); full windows {T['nfull']} of {T['runs']} runs, "
          f"their k bits {T['wfull']/px:.4f} bpp ({100*T['wfull']/(T['size']*8):.2f}%); flush holes {T['wflush']/px:.5f} bpp")
    hdr_idx = 0  # header and tile index are the same in both
    b_bits = T['payload'] + T['wfull'] + T['wflush'] + T['tiles'] * 32
    print(f"   variant b ~ {(b_bits)/px:.4f} bpp + header/index  vs  current {cur:.4f}: delta "
          f"{100*((b_bits) - (T['size']*8 - 0))/(T['size']*8):+.2f}% (current includes header/index too, so b is a bit better than shown)")
    hb = T['head'] + T['tiles'] * 32
    print(f"   (b) + shrink when the full window's hole still ends at the head: shrunk {T['nshrink']} of {T['nfull']} full "
          f"windows, lost {T['lost']/px:.4f} bpp; ~ {hb/px:.4f} bpp vs current {cur:.4f}: {100*(hb - T['size']*8)/(T['size']*8):+.2f}%")
