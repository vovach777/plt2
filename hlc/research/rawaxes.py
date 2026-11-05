import numpy as np, glob
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
R_ = 64; files = sorted(glob.glob('photos16h/*.ppm')) + ['cat_h.ppm']
cnt = np.zeros((2, R_ + 1, 2 * R_ + 1)); se = np.zeros_like(cnt)
def bw(x): x = np.abs(x); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0)
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
norms = ['max(|x|,|y|)', '|x|+|y|', 'sqrt(x²+y²)']; tot = np.zeros(3); px = 0
for f in files:
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); px += Y.size
    S = [dict() for _ in norms]
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        ga, gb = tl - l, tl - t; sa, sb = np.sign(ga), np.sign(gb); s = np.where(sa != 0, sa, np.where(sb != 0, sb, 1))
        x = s * gb; y = s * ga                                   # mirrored: y >= 0
        ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
        m = (np.abs(x) <= R_) & (y <= R_)
        np.add.at(cnt[int(n9)], (y[m], x[m] + R_), 1); np.add.at(se[int(n9)], (y[m], x[m] + R_), np.abs(ee[m]))
        for ni, r in enumerate([np.maximum(np.abs(x), y), np.abs(x) + y, np.sqrt(x.astype(float) ** 2 + y ** 2).round().astype(np.int32)]):
            k = np.minimum(bw(r), 6); k = np.where(k == 0, 0, k + (100 if n9 else 0))
            for v in np.unique(k): S[ni].setdefault(int(v), []).append(z[k == v])
    for ni in range(3): tot[ni] += sum(H(np.concatenate(v)) for v in S[ni].values())
for ni, nm in enumerate(norms): print(f'classes bw({nm}) cap 6, two axes, all 17: {tot[ni] / px:.4f} bpp  {100 * tot[ni] / tot[0] - 100:+.2f}%')
blues = LinearSegmentedColormap.from_list('b', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
fig, axs = plt.subplots(1, 2, figsize=(15, 6.6), facecolor='#ffffff')
for ax, (t, nm) in zip(axs, ((0, 'Y'), (1, 'U/V'))):
    me = np.where(cnt[t] >= 30, se[t] / np.maximum(cnt[t], 1), np.nan)
    im = ax.imshow(me, origin='lower', extent=(-R_ - .5, R_ + .5, -.5, R_ + .5), cmap=blues, norm=LogNorm(vmin=0.1, vmax=40), aspect='auto', interpolation='nearest')
    yy, xx = np.mgrid[0:R_ + 1, -R_:R_ + 1]
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(np.nan_to_num(se[t]), 1.2) / np.maximum(gaussian_filter(cnt[t], 1.2), 1e-9); sm[gaussian_filter(cnt[t], 1.2) < 30] = np.nan
    cs = ax.contour(xx, yy, sm, levels=[1, 2, 4, 8, 16], colors='#1f1f1d', linewidths=1.0)
    ax.clabel(cs, fmt=lambda v: f'|e|={v:g}', fontsize=8, colors='#1f1f1d')
    ax.set_title(f'{nm}: средний |e| по сырым осям (все 17 картинок)', loc='left', fontsize=11, color='#1f1f1d')
    ax.set_xlabel('x = c−b после зеркала  (− : c экстремум, + : скат)', color='#3d3d3a'); ax.set_ylabel('y = |c−a|', color='#3d3d3a')
    for s_ in ax.spines.values(): s_.set_color('#d8d7d0')
    ax.tick_params(colors='#5e5d59', labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.035); cb.set_label('средний |e|', color='#3d3d3a'); cb.ax.tick_params(labelsize=8, colors='#5e5d59')
plt.tight_layout(); plt.savefig('raw_axes.png', dpi=150, facecolor='#ffffff')
