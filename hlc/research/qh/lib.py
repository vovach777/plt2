# lib.py - common tools: bins, contexts, counting, ideal / Huffman / tANS-normalised code costs
import numpy as np, heapq
from load import load
A = 48                                        # table alphabet: 0..A-2 direct, A-1 = escape (h >= A-1)
_cache = {}
def bwv(x):
    x = np.asarray(x, np.int64); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int64) + 1, 0)
def prep(setname, off=1):
    key = (setname, off)
    if key in _cache: return _cache[key]
    d = dict(load(setname)); sa = d['sa'].astype(np.int64); first = sa == 0xffffffff
    L = np.clip(bwv(np.where(first, 0, sa) >> 4) - off, 0, 7); L[first] = 2
    if off == 1: assert (L == d['L']).all()
    d['Lo'] = L; d['h'] = (d['z'] >> L[:, None].astype(np.uint16)).astype(np.int64); d['first'] = first
    d['sa'] = np.where(first, 64, sa)          # stand-in for the first block
    d['zs'] = d['z'].astype(np.int64).sum(1)   # own sum (context for later blocks)
    # previous block (same tile, plane): its own z sum; neighbours' "above" sums
    same_prev = np.r_[False, (d['tile'][1:] == d['tile'][:-1]) & (d['pl'][1:] == d['pl'][:-1])]
    d['sl'] = np.where(same_prev, np.r_[0, d['zs'][:-1]], d['sa'])
    d['sal'] = np.where(same_prev, np.r_[0, d['sa'][:-1]], d['sa'])
    same_next = np.r_[same_prev[1:], False]
    nxt_above = (d['bi'] + 1) * 16 >= d['tw']  # the next block's context is the row above (causal for this block too)
    d['sar'] = np.where(same_next & nxt_above & (d['tw'] >= 16), np.r_[d['sa'][1:], 0], d['sa'])
    d['ptype'] = np.minimum(d['pl'], 1)        # 0 = Y, 1 = S (chroma)
    # zero-block runs of the codec (QH): a block with sa == 0 sends EG0(n), n = zero blocks from it on (same tile, plane);
    # those n blocks are skipped. skip = block not coded; runbits = bits of the run codes sent at this block
    zero = (d['z'] == 0).all(1); pk = d['tile'].astype(np.int64) * 4 + d['pl']; sa0 = d['sa'] == 0
    skip = np.zeros(len(zero), bool); runbits = np.zeros(len(zero), np.int64)
    # zero streak length from every block to the end of its (tile, plane): reverse cumulative
    streak = np.zeros(len(zero) + 1, np.int64)
    for i in range(len(zero) - 1, -1, -1):
        streak[i] = 0 if not zero[i] else (streak[i + 1] + 1 if i + 1 < len(zero) and pk[i + 1] == pk[i] else 1)
    left = 0; prevkey = -1
    for i in range(len(zero)):
        if pk[i] != prevkey: left = 0; prevkey = pk[i]
        if left: left -= 1; skip[i] = True; continue
        if sa0[i]:
            n = int(streak[i]); runbits[i] = 2 * int(n + 1).bit_length() - 1
            if n: skip[i] = True; left = n - 1
    d['skip'] = skip; d['runbits'] = runbits
    _cache[key] = d; return d
def esc_bits(h, ke=2):                         # Exp-Golomb order ke of h - (A-1), for h >= A-1
    x = (np.maximum(h - (A - 1), 0) >> ke) + 1
    return np.where(h >= A - 1, 2 * (bwv(x) - 1) + 1 + ke, 0)
def counts(ctx, sym, nctx, nsym):
    return np.bincount((ctx[:, None] * nsym + sym).ravel() if sym.ndim == 2 else ctx * nsym + sym, minlength=nctx * nsym).reshape(nctx, nsym).astype(np.float64)
def ent(c):                                    # sum over rows of n * H
    n = c.sum(1, keepdims=True); p = np.where(c > 0, c / np.maximum(n, 1), 1.0); return float(-(c * np.log2(p)).sum())
def ideal_len(ctr, alpha=0.02):                # -log2 p of the training counts, smoothed
    c = ctr + alpha; return -np.log2(c / c.sum(1, keepdims=True))
def huff_len_row(c, maxlen=12):
    n = len(c); w = np.maximum(c, 1e-3)
    while True:
        heap = [(w[i], i, (i,)) for i in range(n)]; heapq.heapify(heap); ln = np.zeros(n, np.int64); uid = n
        while len(heap) > 1:
            a = heapq.heappop(heap); b = heapq.heappop(heap)
            for i in a[2]: ln[i] += 1
            for i in b[2]: ln[i] += 1
            heapq.heappush(heap, (a[0] + b[0], uid, a[2] + b[2])); uid += 1
        if ln.max() <= maxlen: return ln
        w = np.maximum(w, w.sum() / 2.0 ** (maxlen - 1) * 1.0) if False else w + w.sum() / 2.0 ** maxlen   # flatten and retry
def huff_len(ctr, maxlen=12): return np.stack([huff_len_row(r, maxlen) for r in ctr]).astype(np.float64)
def tans_len(ctr, tl=10):                      # counts normalised to 2^tl, every symbol >= 1 (cost model of tANS)
    out = np.zeros_like(ctr); T = 1 << tl
    for r, c in enumerate(ctr):
        c = np.maximum(c, 1e-9); q = np.maximum(1, np.floor(c / c.sum() * T + 0.5)).astype(np.int64)
        # fix the sum greedily by the cost change
        while q.sum() != T:
            if q.sum() > T:
                cand = np.where(q > 1, c * np.log2(q / np.maximum(q - 1, 1)), np.inf); i = int(np.argmin(cand)); q[i] -= 1
            else:
                cand = c * np.log2((q + 1) / q); i = int(np.argmax(cand)); q[i] += 1
        out[r] = tl - np.log2(q)
    return out
