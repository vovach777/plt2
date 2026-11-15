#!/usr/bin/env python3
"""PLK (RLK v9 vs PLR per plane): can one cheap tile feature decide without the double pass? (2026-11-15)

Input: logs of an instrumented palic-2r -k (pk_log.cpp) that codes EVERY plane with both RLK and PLR:
  PL <chroma> <n> <probe pr> <rb bits> <plr bits> <zero-quad share> <all-zero 8x8 share> <nonzero px share> <mean bit width>
Per set: oracle = sum min(rb, plr); current -k rule (probe 0 -> PLR, 1 -> RLK, 3 -> RLK if rb*2 <= n else min);
single-feature rules "RLK if f >= t" (zq, b8) or "RLK if f <= t" (nz, bw), t fitted on the training sets.
Usage: plk_rule.py train1.txt,train2.txt test1.txt test2.txt ...
"""
import sys
import numpy as np

COLS = ['chroma', 'n', 'pr', 'rb', 'plr', 'zq', 'b8', 'nz', 'bw']


def load(fn):
    rows = [l.split()[1:] for l in open(fn) if l.startswith('PL ')]
    a = np.array(rows, dtype=float)
    return {c: a[:, i] for i, c in enumerate(COLS)}


def bits_current(d):
    rlk = (d['pr'] == 1) | ((d['pr'] == 3) & (d['rb'] * 2 <= d['n']))
    plr = d['pr'] == 0
    out = np.where(rlk, d['rb'], np.where(plr, d['plr'], np.minimum(d['rb'], d['plr'])))
    double = (d['pr'] == 3) & (d['rb'] * 2 > d['n'])
    return out.sum(), double.sum()


def bits_rule(d, f, t, ge):
    pick_rlk = d[f] >= t if ge else d[f] <= t
    return np.where(pick_rlk, d['rb'], d['plr']).sum()


def fit(train, f, ge):
    vals = np.unique(np.concatenate([d[f] for d in train]))
    cand = np.quantile(vals, np.linspace(0, 1, 400))
    best = None
    for t in cand:
        # objective: total excess over the oracle, relative per set (each set weighs the same)
        ex = sum(bits_rule(d, f, t, ge) / np.minimum(d['rb'], d['plr']).sum() for d in train)
        if best is None or ex < best[0]:
            best = (ex, t)
    return best[1]


def main():
    train = [load(x) for x in sys.argv[1].split(',')]
    tests = [(x, load(x)) for x in sys.argv[2:]]
    rules = [('zq', True), ('b8', True), ('nz', False), ('bw', False)]
    ts = {f: fit(train, f, ge) for f, ge in rules}
    print('thresholds (fitted on ' + sys.argv[1] + '): ' + ', '.join(f'{f} {"≥" if ge else "≤"} {ts[f]:.4f}' for f, ge in rules))
    for name, d in tests:
        orc = np.minimum(d['rb'], d['plr']).sum()
        cur, dbl = bits_current(d)
        line = f'{name.split("/")[-1]:12s} planes {len(d["n"]):5d}  oracle 1.0000  -k {cur / orc:.4f} (double pass {int(dbl)})'
        for f, ge in rules:
            line += f'  {f} {bits_rule(d, f, ts[f], ge) / orc:.4f}'
        rlk_win = (d['rb'] < d['plr']).mean()
        print(line + f'  | RLK wins {100 * rlk_win:.0f}%')


if __name__ == '__main__':
    main()
