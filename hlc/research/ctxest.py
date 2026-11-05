import numpy as np, sys, glob
from PIL import Image
THY = [1, 3, 10, 18, 35]; THC = [1, 4, 8, 16, 30]

def planes(f):
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    V = R - G; U = B - G; Y = G + ((U + V) >> 2)
    return [(V, True), (U, True), (Y, False)]

def nb(P):
    H, W = P.shape; Z = np.zeros((H + 1, W + 2), np.int32); Z[1:, 1:-1] = P
    a = Z[1:, :-2].copy(); b = Z[:-1, 1:-1].copy(); c = Z[:-1, :-2].copy(); d = Z[:-1, 2:].copy()
    aa = np.zeros_like(P); aa[:, 2:] = P[:, :-2]
    a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]                      # x = 0: from above
    d[:, -1] = b[:, -1]
    return a, b, c, d, aa

def resid(P, n9):
    a, b, c, d, aa = nb(P)
    mn = np.minimum(a, b); mx = np.maximum(a, b); pr = np.clip(a + b - c, mn, mx)
    e = P - pr
    e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
    return e, a, b, c, d, aa

def H0(z):
    if len(z) == 0: return 0.0
    _, c = np.unique(z, return_counts=True); p = c / c.sum(); return float(-(c * np.log2(p)).sum())

def cls(act, th): return sum((act >= t).astype(np.int32) for t in th)

def run(files):
    tot = {}
    px = 0
    for f in files:
        S = {}
        for ci, (P, n9) in enumerate(planes(f)):
            e, a, b, c, d, aa = resid(P, n9); z = np.where(e >= 0, 2 * e, -2 * e - 1)
            ea = np.zeros_like(e); ea[:, 1:] = np.abs(e[:, :-1]); eb = np.zeros_like(e); eb[1:] = np.abs(e[:-1])
            acts = {'none': None,
                    'abd (pctx)': np.abs(a - c) + np.abs(c - b) + np.abs(b - d),
                    'ac+cb': np.abs(a - c) + np.abs(c - b),
                    'ac+cb x1.5': (3 * (np.abs(a - c) + np.abs(c - b))) // 2,
                    'ac+cb+a-aa': np.abs(a - c) + np.abs(c - b) + np.abs(a - aa),
                    'ac+cb+|ea|': np.abs(a - c) + np.abs(c - b) + ea,
                    '|ea|+|eb|': ea + eb}
            for name, act in acts.items():
                st = S.setdefault(name, {})
                if act is None: st.setdefault(ci, []).append(z.ravel()); continue
                k = cls(act, THC if n9 else THY)
                for j in range(6):
                    key = 0 if j == 0 else (j if ci == 2 else 5 + j)
                    st.setdefault(key, []).append(z[k == j])
        px += planes(f)[0][0].size
        for name, st in S.items():
            bits = sum(H0(np.concatenate(v)) for v in st.values())
            tot[name] = tot.get(name, 0) + bits
    base = tot['none']
    for n, b in tot.items(): print(f'{n:14s} {b / px:7.4f} bpp  {100 * b / base - 100:+6.2f}%')

run(sorted(sum((glob.glob(a) if '*' in a else [a] for a in sys.argv[1:]), [])))
