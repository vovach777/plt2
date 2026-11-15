# exp_l_plane.py - "one L for the whole plane of a tile" vs the causal L of PLR (Vladimir, 2026-11-15).
# Per (tile, plane), with the format tables (letters + tails + 16 L planes per coded block; runs / zU split unchanged):
#   causal  : the format (L from the 16 z above, off 2)
#   const   : one L 0..7 for every block of the plane, the best one; 3 bits per plane
#   shift   : causal L + d, d = -3..+3 (clipped 0..7), the best d; 3 bits per plane
#   best2   : the cheaper of const / shift; 4 bits per plane
#   med     : L by MED over the own L of the left / above / above-left blocks, free (exp_l_med)
#   oracle  : the best L per block, not sent (ceiling)
# "Anomaly" tiles: dense luma, almost no chroma (S plane: < 5% non-zero z, Y plane: > 25% non-zero z).
# MAP=out.png: also writes a picture of the first image: luma | causal L | per-block best L | error (red = causal too
# high, blue = too low; brightness = bits lost). usage: [MAP=x.png] exp_l_plane.py DUMPER img...
import sys, os, numpy as np
sys.argv, _a = sys.argv[:1] + ['x', 'y'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_pred.py')).read().split("dm, dw = sys.argv[1], sys.argv[2]")[0])
dumper, imgs = _a[1], _a[2:]
VAR = ('causal', 'const', 'shift', 'best2', 'med', 'yown', 'oracle')
SIDE = dict(yown=0, causal=0, const=3, shift=3, best2=4, med=0, oracle=0)
def own_L(z):
    s = z.sum(1) >> 4
    return np.clip(np.where(s > 0, np.floor(np.log2(np.maximum(s, 1))).astype(np.int64) + 1, 0) - 2, 0, 7)
def run(img, mapfile=None):
    d, c, parts = costs(dumper, img); z = d['z'].astype(np.int64); n = len(z); idx = np.arange(n)
    t = T['cm'][d['ptype'] * T['NQ'] + qhsim.Qsa(d['sa'])]; k = ~d['skip']
    C = np.zeros((n, 8))
    for L in range(8):
        h = z >> L; x = np.maximum(h - 3, 0); kk = T['rk'][t][:, None]; q = x >> kk
        C[:, L] = (T['lens'][t[:, None], qhsim.letters(h)].sum(1)
                   + np.where(h >= 3, np.where(q >= T['QESC'], T['QESC'] + 10, q + 1 + kk), 0).sum(1) + 16 * L)
    C[~k] = 0                                            # blocks inside zero runs are not coded
    Lc = d['Lo'].astype(np.int64)
    # L_med over the own L of the block grid
    Lown = own_L(z); step = np.maximum(d['tw'] // 16, 1); bi = d['bi']; col = bi % step
    ok_up = (d['tw'] % 16 == 0) & (bi >= step); ok_left = col > 0
    a = np.where(ok_left, Lown[np.maximum(idx - 1, 0)], -1); b = np.where(ok_up, Lown[np.maximum(idx - step, 0)], -1)
    cc = np.where(ok_up & ok_left, Lown[np.maximum(idx - step - 1, 0)], -1)
    a2 = np.where(a < 0, np.where(b < 0, 2, b), a); b2 = np.where(b < 0, a2, b); c2 = np.where(cc < 0, np.minimum(a2, b2), cc)
    Lmed = np.clip(a2 + b2 - c2, np.minimum(a2, b2), np.maximum(a2, b2))
    # yown: on S, L = own L of the Y block at the same (tile, block) - Y is decoded first, so free; Y keeps causal
    yk = d['tile'].astype(np.int64) * 100000 + d['bi']; ym = d['pl'] == 0
    order = np.argsort(yk[ym]); ykeys = yk[ym][order]; pos = np.searchsorted(ykeys, yk)
    pos = np.minimum(pos, len(ykeys) - 1); hit = (ykeys[pos] == yk) & ~ym
    Lyo = np.where(hit, Lown[ym][order][pos], Lc)
    key = d['tile'].astype(np.int64) * 4 + d['pl']; uk, ki = np.unique(key, return_inverse=True); ng = len(uk)
    def gsum(v): return np.bincount(ki, weights=v, minlength=ng)
    cost = dict(yown=gsum(C[idx, Lyo]), causal=gsum(C[idx, Lc]), med=gsum(C[idx, Lmed]), oracle=gsum(C.min(1)))
    cst = np.stack([gsum(C[:, L]) for L in range(8)], 1); cost['const'] = cst.min(1) + 3
    sh = np.stack([gsum(C[idx, np.clip(Lc + dd, 0, 7)]) for dd in range(-3, 4)], 1); cost['shift'] = sh.min(1) + 3
    cost['best2'] = np.minimum(cst.min(1), sh.min(1)) + 4
    # anomaly tiles: per tile, nonzero share of Y and S
    nzY = np.bincount(d['tile'][d['pl'] == 0], weights=(z[d['pl'] == 0] > 0).mean(1), minlength=d['tile'].max() + 1)
    nbY = np.bincount(d['tile'][d['pl'] == 0], minlength=d['tile'].max() + 1)
    nzS = np.bincount(d['tile'][d['pl'] == 1], weights=(z[d['pl'] == 1] > 0).mean(1), minlength=d['tile'].max() + 1)
    nbS = np.bincount(d['tile'][d['pl'] == 1], minlength=d['tile'].max() + 1)
    fy = nzY / np.maximum(nbY, 1); fs = nzS / np.maximum(nbS, 1); anom_t = (fs < 0.05) & (fy > 0.25)
    anom_g = anom_t[uk // 4]
    if mapfile: draw(img, d, Lc, C, mapfile)
    return dict(cost=cost, anom=anom_g, file=c.sum(), npx=d['npx'], ntile=len(anom_t), nanom=int(anom_t.sum()),
                gshift=np.argmin(sh, 1) - 3, gconst=np.argmin(cst, 1), pl=uk % 4)
def draw(img, d, Lc, C, out):
    from PIL import Image
    raw = open(img, 'rb').read(); parts_ = raw.split(maxsplit=4); W, H = int(parts_[1]), int(parts_[2])
    rgb = np.frombuffer(raw[len(raw) - W * H * 3:], np.uint8).reshape(H, W, 3)
    lum = (rgb.astype(np.int32) @ np.array([77, 150, 29]) >> 8).astype(np.uint8)
    nx = (W + 287) // 288; y0 = d['pl'] == 0
    Lb = C.argmin(1); lost = C[np.arange(len(C)), Lc] - C.min(1)
    mc = np.zeros((H, W), np.uint8); mb = np.zeros((H, W), np.uint8); err = np.zeros((H, W, 3), np.uint8)
    for i in np.flatnonzero(y0):
        tl = int(d['tile'][i]); tw = int(d['tw'][i]); ox = (tl % nx) * 288; oy = (tl // nx) * 288
        p0 = int(d['bi'][i]) * 16
        for p in range(p0, p0 + 16):
            yy = oy + p // tw; xx = ox + p % tw
            if yy >= H or xx >= W: continue
            mc[yy, xx] = Lc[i] * 36; mb[yy, xx] = Lb[i] * 36
            v = min(255, int(lost[i] * 6))
            if Lc[i] > Lb[i]: err[yy, xx] = (v, 0, 0)
            elif Lc[i] < Lb[i]: err[yy, xx] = (0, v // 2, v)
    g = lambda m: np.stack([m] * 3, 2)
    Image.fromarray(np.concatenate([g(lum), g(mc), g(mb), err], 1)).save(out)
A = {v: 0.0 for v in VAR}; An = {v: 0.0 for v in VAR}; tot = 0.0; npx = 0
print('%-28s %7s ' % ('file', 'tiles') + ' '.join('%8s' % v for v in VAR) + '   (% of the file vs causal; anomaly tiles)')
for j, img in enumerate(imgs):
    r = run(img, os.environ.get('MAP') if j == 0 else None); base = r['cost']['causal']
    row = ' '.join('%+7.1f%%' % (100 * (r['cost'][v].sum() - base.sum()) / r['file']) for v in VAR)
    an = r['anom']; rowa = ' '.join('%+6.1f%%' % (100 * (r['cost'][v][an].sum() - base[an].sum()) / max(base[an].sum(), 1)) for v in ('const', 'shift', 'best2', 'oracle')) if an.any() else '-'
    sh = np.bincount(r['gshift'] + 3, minlength=7)
    print('%-28s %3d/%-3d %s | anomaly (of their blocks) const/shift/best2/oracle %s | best shift d=-3..3: %s' % (
        os.path.basename(img)[:28], r['nanom'], r['ntile'], row, rowa, sh.tolist()))
    for v in VAR: A[v] += r['cost'][v].sum()
    tot += r['file']; npx += r['npx']
print('%-28s %7s ' % ('ALL', '') + ' '.join('%+7.1f%%' % (100 * (A[v] - A['causal']) / tot) for v in VAR) + f'   model {tot / npx:.4f} bpp')
