import numpy as np, glob, sys
from PIL import Image
AC = 256
def ent(h):
    h = h[h > 0].astype(np.float64); return float(-(h * np.log2(h / h.sum())).sum()) if h.size else 0.0
def file_hists(f, nterms, signed):
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2)
    HY = np.zeros((AC + 1, 512), np.int64); HC = np.zeros((AC + 1, 512), np.int64)
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        ll = l.copy(); ll[:, 2:] = P[:, :-2]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        if signed:
            sa, sb = np.sign(tl - l), np.sign(tl - t); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1)); e = s * e
        e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
        z = np.where(e >= 0, 2 * e, -2 * e - 1)
        act = np.abs(l - tl) + np.abs(tl - t) + (np.abs(l - ll) if nterms == 3 else 0)
        act = np.minimum(act, AC - 1)
        h = np.bincount((act * 512 + z).ravel(), minlength=AC * 512).reshape(AC, 512)
        (HC if n9 else HY)[1:] += np.cumsum(h, axis=0)
    return HY, HC, Y.size
def cost(F, ty, tc):
    tot = 0.0
    for HY, HC, _ in F:
        by = [0] + list(ty) + [AC]; bc = [0] + list(tc) + [AC]
        tot += ent((HY[by[1]] - HY[0]) + (HC[bc[1]] - HC[0]))
        for j in range(1, 6): tot += ent(HY[by[j + 1]] - HY[by[j]]) + ent(HC[bc[j + 1]] - HC[bc[j]])
    return tot
def greedy(F, ty, tc, rounds=5):
    th = list(ty) + list(tc); best = cost(F, th[:5], th[5:])
    for _ in range(rounds):
        moved = False
        for i in range(10):
            lo = (th[i - 1] + 1) if i % 5 else 1; hi = (th[i + 1] - 1) if i % 5 != 4 else AC - 1
            if hi < lo: continue
            for v in sorted(set(int(x) for x in np.geomspace(lo, hi, 24).round()) | set(range(max(lo, th[i] - 2), min(hi, th[i] + 2) + 1))):
                if v == th[i]: continue
                t2 = th.copy(); t2[i] = v; c = cost(F, t2[:5], t2[5:])
                if c < best: best, th, moved = c, t2, True
        if not moved: break
    return th[:5], th[5:], best
files = sorted(glob.glob(sys.argv[1])); PY, PC = [1, 3, 10, 18, 35], [1, 4, 8, 16, 30]
base = None
for nterms in (3, 2):
    for signed in (False, True):
        F = [file_hists(f, nterms, signed) for f in files]; px = sum(x[2] for x in F)
        c0 = cost(F, PY, PC); ty, tc, c1 = greedy(F, PY, PC)
        if base is None: base = c0
        lab = ('|a-c|+|c-b|+|a-aa|' if nterms == 3 else '|c-a|+|c-b|      ') + (' + sign' if signed else '       ')
        print(f'{lab}  pctx thr {c0 / px:.4f} ({100 * c0 / base - 100:+.2f}%)   fitted {c1 / px:.4f} ({100 * c1 / base - 100:+.2f}%)  Y {ty} UV {tc}', flush=True)
