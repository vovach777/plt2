#!/usr/bin/env python3
"""fsm_eval.py - reference evaluation for the RLH context task (Claude, 2026-11-16).

Data: *.tok - for each image, for each of 3 streams (g, r, b of llrice CODER 3): uint32 n, then n x uint16 tokens,
sym = cr << 6 | cv:  cr = bit width of the zero run before the value (0..27), cv = half-octave class of the value
(1 -> 1, 2..3 -> 2 + bit, 4..7 -> 4 + top bit ..., i.e. cv = 2*bw - 2 + next bit for bw >= 2), cv = 0 only for the
trailing zeros of a stream. Mantissas are not in the files: they do not depend on the context.

A context function maps the token history of ONE stream (causally: only tokens before i) to an integer.
Tables: per (stream index 0..2, context), trained on screens_train + photos_train together (one table set must serve
both), add-0.5 smoothing over the full alphabet (28 x 64). Score: cross-entropy of the TEST sets in bits per token.

Run:  python3 fsm_eval.py   (prints the baselines; add your context function to CTX and run again)
"""
import numpy as np, sys, math

def load(path):
    b = open(path, 'rb').read(); off = 0; streams = []
    while off < len(b):
        n = int.from_bytes(b[off:off + 4], 'little'); off += 4
        streams.append(np.frombuffer(b, dtype='<u2', count=n, offset=off)); off += 2 * n
    return streams   # list in order img0.s0, img0.s1, img0.s2, img1.s0, ...

def vb_of(cv):       # value bit width from the half-octave class
    return 0 if cv == 0 else 1 if cv == 1 else (cv + 2) // 2

# ---------------- context functions: f(tokens of one stream) -> int array of contexts (same length) ----------------
def ctx_one(t):
    return np.zeros(len(t), dtype=np.int64)

def ctx_prev(t):     # previous symbol: value-width group (4) x run > 0 (2), + start
    out = np.zeros(len(t), dtype=np.int64); c = 8
    for i, s in enumerate(t.tolist()):
        out[i] = c
        cr, vb = s >> 6, vb_of(s & 63)
        g = 0 if vb <= 1 else 1 if vb == 2 else 2 if vb <= 4 else 3
        c = g * 2 + (cr > 0)
    return out

def ctx_ema48(t):    # two rounded EMAs as automata, 48 states each: value bw (weight 1/8), run bw (1/4), 8 units/bit;
    out = np.zeros(len(t), dtype=np.int64)   # ctx = bucket(a) (6, 1 bit) x bucket(r) (6) x prev value group (5)
    a = r = 0; pg = 4; P = 8
    for i, s in enumerate(t.tolist()):
        out[i] = (min(a // P, 5) * 6 + min(r // P, 5)) * 5 + pg
        cr, vb = s >> 6, vb_of(s & 63)
        a += (vb * P - a + 4) >> 3
        r += (cr * P - r + 2) >> 2
        pg = 0 if vb <= 1 else 1 if vb == 2 else 2 if vb <= 4 else 3
    return out

CTX = {'one table': ctx_one, 'prev symbol (9)': ctx_prev, 'EMA automata 48+48 states (181)': ctx_ema48}

# ---------------- evaluation ----------------
ALPHA = 28 * 64
def evaluate(fn, train, tests):
    counts = {}                                   # (ctx * 4 + stream) -> {sym: count}
    for k, t in train:
        key = fn(t) * 4 + k
        for kk, s in zip(key.tolist(), t.tolist()):
            d = counts.setdefault(kk, {}); d[s] = d.get(s, 0) + 1
    tot = {kk: sum(d.values()) for kk, d in counts.items()}
    res = []
    for name, ts in tests:
        bits = 0.0; n = 0
        for k, t in ts:
            key = fn(t) * 4 + k
            for kk, s in zip(key.tolist(), t.tolist()):
                d = counts.get(kk, {})
                bits -= math.log2((d.get(s, 0) + 0.5) / (tot.get(kk, 0) + 0.5 * ALPHA)); n += 1
        res.append((name, bits / n, n))
    return res, len(counts)

if __name__ == '__main__':
    def tagged(path): return [(i % 3, s) for i, s in enumerate(load(path))]
    train = tagged('screens_train.tok') + tagged('photos_train.tok')
    tests = [('screens_test', tagged('screens_test.tok')), ('photos_test', tagged('photos_test.tok'))]
    for name, fn in CTX.items():
        res, nctx = evaluate(fn, train, tests)
        print(f'{name:36s} used ctx x stream {nctx:5d} | ' + '  '.join(f'{n} {b:.4f} bits/token ({cnt} tok)' for n, b, cnt in res), flush=True)
