# summ.py OUT.json... - totals (bpp, model) per variant and tail rule; change vs the format
import sys, json, numpy as np
RULES = ['fmt', (2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (6, 1), (8, 1), (8, 2), (12, 2)]
for fn in sys.argv[1:]:
    R = json.load(open(fn)); n = sum(r['npx'] for r in R); o = sum(r['other'] for r in R); keys = [k for k in R[0] if k[0] in 'FST' and k != 'F'] ; keys = ['F'] + sorted(keys)
    tot = {k: np.sum([r[k] for r in R], 0) + o for k in keys}; f0 = tot['F'][0]
    print('== %s: %d images, format %.4f bpp (model); rows: variant, columns: tail rule; bpp and %% vs format' % (fn, len(R), f0 / n))
    print('%-5s' % '' + ''.join('%15s' % (str(r).replace(' ', '')) for r in RULES))
    for k in keys: print('%-5s' % k + ''.join('%8.4f %+5.1f%%' % (v / n, 100 * (v / f0 - 1)) for v in tot[k]))
    for k in R[0]:
        if k.startswith('pick'): print('  share of tiles choosing S%s (rule 4,1): %.3f' % (k[4:], np.average([r[k][4] for r in R], weights=[r['nt'] for r in R])))
