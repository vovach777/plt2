#!/usr/bin/env python3
"""Medians over 3 rounds (best-of-7 inside each round) for the probe-cost task."""
import re, statistics, os, sys
O = '/home/devin/train/bench-independent/speed'
data = {}   # (set, mode) -> list of (bpp, enc, dec)
cpu = ''
for ln in open(os.path.join(O, 'raw.txt')):
    if ln.startswith('CPU:'): cpu = ln[4:].strip(); continue
    m = re.match(r'^(\S+)\s+([ABCD])\s+r(\d)\s+\S+\s+TOTAL (\d+) files ([\d.]+) bpp\s+enc ([\d.]+) dec ([\d.]+) MB/s.*(all ok|FAIL)', ln)
    if m:
        s, mode, r, n, bpp, enc, dec, ok = m.groups()
        data.setdefault((s, mode), []).append((float(bpp), float(enc), float(dec), ok))
sets = ['ios_test', 'desktop_test', 'websight_test', 'docs_test', 'scans_test', 'full12', 'web8', 'valid41', 'kodak24']
modes = [('A', '-s'), ('B', '-s=zthr'), ('C', '-s=probe2'), ('D', '-s=autocnt')]
print('CPU:', cpu)
print()
hdr = '| набор | ' + ' | '.join('%s (%s)' % (m, f) for m, f in modes) + ' | cnt/old | probe2/old |'
print(hdr); print('|---|' + '---|' * (len(modes) + 2))
for s in sets:
    cells = []
    med = {}
    for m, f in modes:
        v = data.get((s, m), [])
        if not v: cells.append('—'); continue
        enc = statistics.median(x[1] for x in v)
        bpp = statistics.median(x[0] for x in v)
        med[m] = (bpp, enc)
        cells.append('%.4f / %.0f' % (bpp, enc))
    c = '+' if s not in med else '%+.1f%%' % (100 * (med['D'][1] / med['A'][1] - 1))
    p = '+' if s not in med else '%+.1f%%' % (100 * (med['C'][1] / med['A'][1] - 1))
    print('| %s | %s | %s | %s |' % (s, ' | '.join(cells), c, p))
print()
print('### spread of enc over 3 rounds (min..max, %)')
print('| набор | -s | zthr | probe2 | autocnt |')
print('|---|---|---|---|---|')
for s in sets:
    row = []
    for m, f in modes:
        v = [x[1] for x in data.get((s, m), [])]
        if len(v) < 2: row.append('—'); continue
        row.append('%.0f..%.0f (%.1f%%)' % (min(v), max(v), 100 * (max(v) / min(v) - 1)))
    print('| %s | %s |' % (s, ' | '.join(row)))
print()
print('### bpp per mode (median) — must match between rounds; differences show probe decisions')
print('| набор | -s | zthr | probe2 | autocnt |')
print('|---|---|---|---|---|')
for s in sets:
    row = []
    for m, f in modes:
        v = [x[0] for x in data.get((s, m), [])]
        row.append('%.4f' % statistics.median(v) if v else '—')
    print('| %s | %s |' % (s, ' | '.join(row)))
bad = [k for k, v in data.items() if any(x[3] != 'all ok' for x in v)]
print('\nfails:', bad if bad else '(none)')