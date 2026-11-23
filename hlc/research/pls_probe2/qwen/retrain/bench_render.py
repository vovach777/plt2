#!/usr/bin/env python3
"""Render the bench summary into a markdown table (bpp / enc / dec, ok flag) + deltas."""
import re, os
O = '/home/devin/train/bench-independent/p2/bench'
rows = {}
cpu = ''
for ln in open(os.path.join(O, 'summary.txt')):
    if ln.startswith('CPU:'): cpu = ln[4:].strip(); continue
    m = re.match(r'^(\S+)\s+(ctx5-auto|probe1-zthr|p2-arc-50-50|p2-new-45-50|r8f5-0448cfb)\s+.*TOTAL (\d+) files ([\d.]+) bpp\s+enc ([\d.]+) dec ([\d.]+) MB/s.*(all ok|FAIL)', ln)
    if m:
        rows[(m.group(1), m.group(2))] = (int(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)), m.group(7))
sets = ['ios_test', 'desktop_test', 'websight_test', 'docs_test', 'scans_test', 'full12', 'web8', 'valid41', 'kodak24']
cfgs = ['ctx5-auto', 'probe1-zthr', 'p2-arc-50-50', 'p2-new-45-50', 'r8f5-0448cfb']
print('CPU:', cpu)
print()
print('| набор | ' + ' | '.join(cfgs) + ' |')
print('|---|' + '---|' * len(cfgs))
for s in sets:
    cells = []
    for c in cfgs:
        v = rows.get((s, c))
        cells.append('%.4f' % v[1] if v else '—')
    print('| %s | %s |' % (s, ' | '.join(cells)))
print()
print('### enc/dec MB/s (best of 3, 1 thread)')
print('| набор | ' + ' | '.join(cfgs) + ' |')
print('|---|' + '---|' * len(cfgs))
for s in sets:
    cells = []
    for c in cfgs:
        v = rows.get((s, c))
        cells.append('%.0f / %.0f' % (v[2], v[3]) if v else '—')
    print('| %s | %s |' % (s, ' | '.join(cells)))
print()
print('### deltas, %% (negative = better), vs probe1-zthr')
for c in ['p2-arc-50-50', 'p2-new-45-50']:
    print('| набор | %s |' % c)
    print('|---|---|')
    for s in sets:
        a = rows.get((s, 'probe1-zthr')); b = rows.get((s, c))
        if a and b: print('| %s | %+.2f%% |' % (s, 100 * (b[1] - a[1]) / a[1]))
    print()
print('### deltas, %% (negative = better), vs ctx5-auto')
for c in ['probe1-zthr', 'p2-new-45-50']:
    print('| набор | %s |' % c)
    print('|---|---|')
    for s in sets:
        a = rows.get((s, 'ctx5-auto')); b = rows.get((s, c))
        if a and b: print('| %s | %+.2f%% |' % (s, 100 * (b[1] - a[1]) / a[1]))
    print()
bad = [k for k, v in rows.items() if v[4] != 'all ok']
print('fails:', bad if bad else '(none)')