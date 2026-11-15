#!/usr/bin/env python3
"""PALIC-1 window-histogram packing: cost of the current scheme vs alternatives.

Input: dumps of per-window histograms (uint16 x NSYM per window, sum = W = 2^w), from palic built with HDUMP.
For each scheme: header bits + coding bits of the window's symbols (ideal arithmetic coding with the model the
decoder would know). Exact counts give the minimal coding bits; coarser histograms trade coding loss for header.
"""
import sys, math
import numpy as np

NSYM, W = 11, 4096

def bw(v): return int(v).bit_length()

def cur_header(f):
    """Current PALIC-1: entries 0.. while start < W: 1 flag bit, plus bw(left) bits if nonzero."""
    bits, start = 0, 0
    for i in range(NSYM - 1):
        if start >= W: break
        left = W - start
        bits += 1 + (bw(left) if f[i] else 0)
        start += f[i]
    return bits

def code_bits(f, q, P):
    """Ideal coding bits of counts f with probabilities q / 2^P (q > 0 wherever f > 0)."""
    m = f > 0
    return float(-(f[m] * (np.log2(q[m]) - P)).sum())

def quantise(f, P):
    """Counts -> frequencies summing to 2^P, every used symbol >= 1 (largest absorbs the rounding)."""
    T = 1 << P
    q = np.where(f > 0, np.maximum(1, np.floor(f * T / W + 0.5)), 0).astype(np.int64)
    q[np.argmax(f)] += T - q.sum()
    if q[np.argmax(f)] < 1: return None
    return q

def header_left(q, P):
    """The same 'flag + bits of what is left' packing, at precision P."""
    T = 1 << P; bits, start = 0, 0
    for i in range(NSYM - 1):
        if start >= T: break
        left = T - start
        bits += 1 + (bw(left) if q[i] else 0)
        start += q[i]
    return bits

def egamma(v):  # Elias gamma of v >= 1
    return 2 * bw(v) - 1

def header_delta(f, prev):
    """Each count as zigzag delta from the previous window's count, Elias gamma (last symbol implied)."""
    bits = 0
    for i in range(NSYM - 1):
        d = int(f[i]) - int(prev[i]); z = 2 * d if d >= 0 else -2 * d - 1
        bits += egamma(z + 1)
    return bits

def main():
    for path in sys.argv[1:]:
        H = np.fromfile(path, dtype=np.uint16).reshape(-1, NSYM).astype(np.int64)
        assert (H.sum(1) == W).all()
        exact = sum(code_bits(f, f, 12) for f in H)
        res = {'current (exact, flag+left)': sum(cur_header(f) for f in H) + exact}
        prev = np.zeros(NSYM, np.int64)
        d = 0
        for f in H:
            d += header_delta(f, prev); prev = f
        res['exact, delta vs previous window (gamma)'] = d + exact
        for P in (10, 9, 8, 7, 6):
            hb = cb = 0
            for f in H:
                q = quantise(f, P)
                hb += header_left(q, P); cb += code_bits(f, q, P)
            res[f'quantised to {P} bits, flag+left'] = hb + cb
        base = res['current (exact, flag+left)']
        nwin = len(H)
        print(f'{path}: {nwin} windows, symbols {nwin * W}, current {base / 8 / 1024:.1f} KiB (hist+symbols, mantissas not counted)')
        for k, v in res.items():
            print(f'  {k:42s} {v / 8 / 1024:10.1f} KiB  {100 * (v - base) / base:+.2f}%  ({(v - base) / nwin:+.1f} bits/window)')

if __name__ == '__main__':
    main()
