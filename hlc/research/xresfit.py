import numpy as np, glob, sys
from PIL import Image
exec(open('xresctx.py').read().split("V_ = [")[0])
_s = open("act2.py").read(); exec(_s.split("def file_hists")[0]); exec("def cost" + _s.split("def cost")[1].split("files = sorted")[0])
AC = 256
def hists(f, use_x):
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2)
    HY = np.zeros((AC + 1, 512), np.int64); HC = np.zeros((AC + 1, 512), np.int64); acts = []
    planes = []
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        ll = l.copy(); ll[:, 2:] = P[:, :-2]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t)); e = w9(e) if n9 else ((e + 128) & 255) - 128
        sa, sb = np.sign(tl - l), np.sign(tl - t); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
        act = np.minimum(np.abs(l - tl) + np.abs(tl - t) + np.abs(l - ll), AC - 1)
        planes.append([e, s, act, n9])
    if use_x: planes[0][0], planes[1][0], _ = xres(planes[0][0], planes[1][0], False)
    for e, s, act, n9 in planes:
        ee = s * e; ee = w9(ee) if n9 else ((ee + 128) & 255) - 128; z = zz(ee)
        h = np.bincount((act * 512 + z).ravel(), minlength=AC * 512).reshape(AC, 512)
        (HC if n9 else HY)[1:] += np.cumsum(h, axis=0)
    return HY, HC, Y.size
files = sorted(glob.glob(sys.argv[1]))
for use_x in (False, True):
    F = [hists(f, use_x) for f in files]; px = sum(x[2] for x in F)
    c0 = cost(F, TY, TC); ty, tc, c1 = greedy(F, TY, TC)
    print(f'xres {"on " if use_x else "off"}: thresholds fitted without xres {c0 / px:.4f}   refitted {c1 / px:.4f} ({100 * c1 / c0 - 100:+.2f}%)  Y {ty} UV {tc}', flush=True)
