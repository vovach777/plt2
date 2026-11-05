#!/usr/bin/env python3
"""Per-photo shares of the q7 x q7 cells (mirrored, geometric sign) and mean |E1|+|E2| (left + top MED errors)
per cell, Y and U/V, as small-multiple heatmaps + CSV.

usage: python q7share_plot.py <out_prefix> <ppm/png files or globs...>
writes <out_prefix>_Y.png, <out_prefix>_UV.png, <out_prefix>.csv
"""
import numpy as np, glob, sys, os, csv
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
def qsym(x, edges): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in edges)
def shift(A, dy, dx):
    O = np.zeros_like(A); h, w = A.shape; O[dy:, dx:] = A[:h - dy, :w - dx]; return O
COLS = ['−16+', '−4…15', '−1…3', '0', '+1…3', '+4…15', '+16+']; ROWS = ['0', '1–3', '4–15', '16+']
def cells(f):
    a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R_, G, B = a0[..., 0], a0[..., 1], a0[..., 2]; V = R_ - G; U = B - G; Y = G + ((U + V) >> 2)
    out = np.zeros((2, 2, 4, 7))                       # [count | sum |E1|+|E2|][plane type]
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
        qa, qb = qsym(c - a, [1, 4, 16]), qsym(c - b, [1, 4, 16]); h = qb + 7 * qa; s = np.where(h < 0, -1, 1)
        e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
        E = np.abs(shift(e, 0, 1)) + np.abs(shift(e, 1, 0)); idx = ((qa * s) * 7 + (qb * s + 3)).ravel()
        out[0, int(n9)] += np.bincount(idx, minlength=28).reshape(4, 7)
        out[1, int(n9)] += np.bincount(idx, weights=E.ravel(), minlength=28).reshape(4, 7)
    return out
def main(prefix, files):
    data = [(os.path.basename(f).split('.')[0].replace('cat_h', 'cat'), cells(f)) for f in files]
    data.append(('ALL', sum(d for _, d in data)))
    cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
    with open(prefix + '.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['photo', 'plane', 'value', 'q(c-a)'] + COLS)
        for nm, d in data:
            for t, pn in ((0, 'Y'), (1, 'UV')):
                sh = 100 * d[0, t] / d[0, t].sum(); me = d[1, t] / np.maximum(d[0, t], 1)
                for i in range(4): w.writerow([nm, pn, 'share %', ROWS[i]] + [f'{x:.2f}' for x in sh[i]])
                for i in range(4): w.writerow([nm, pn, 'mean |E1|+|E2|', ROWS[i]] + [f'{x:.2f}' for x in me[i]])
    for t, pn in ((0, 'Y'), (1, 'UV')):
        n = len(data); nc = 6; nr = (n + nc - 1) // nc
        fig, axs = plt.subplots(nr, nc, figsize=(nc * 4.0, nr * 2.9), facecolor='#ffffff')
        for k, ax in enumerate(axs.flat):
            if k >= n: ax.axis('off'); continue
            nm, d = data[k]; sh = 100 * d[0, t] / d[0, t].sum(); me = d[1, t] / np.maximum(d[0, t], 1)
            vmax = 15 if pn == 'Y' else 40
            ax.imshow(np.where(sh > 0, sh, np.nan), origin='lower', cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
            for i in range(4):
                for j in range(7):
                    if i == 0 and j < 3: continue
                    v = sh[i, j]; txt = f'{v:.0f}' if v >= 10 else (f'{v:.1f}' if v >= 0.05 else '·'); ink = '#ffffff' if v > vmax * 0.55 else '#1f1f1d'
                    ax.text(j, i + 0.17, txt, ha='center', va='center', fontsize=7.5, color=ink, weight='bold')
                    if d[0, t, i, j] >= 20: m = me[i, j]; ax.text(j, i - 0.2, f'{m:.0f}' if m >= 10 else f'{m:.1f}', ha='center', va='center', fontsize=6.3, color=ink)
            ax.set_title(nm + (' (весь корпус)' if nm == 'ALL' else ''), fontsize=9, loc='left', color='#1f1f1d', weight='bold' if nm == 'ALL' else 'normal')
            ax.set_xticks(range(7)); ax.set_yticks(range(4))
            ax.set_xticklabels(COLS if k // nc == nr - 1 or k + nc >= n else [''] * 7, fontsize=6, color='#5e5d59')
            ax.set_yticklabels(ROWS if k % nc == 0 else [''] * 4, fontsize=6, color='#5e5d59')
            for s_ in ax.spines.values(): s_.set_color('#d8d7d0')
            ax.tick_params(length=0)
        fig.suptitle(f'{pn}: ячейки q7 × q7 — сверху доля %, снизу средний |E1|+|E2| (ошибки MED слева и сверху)   строки q(c−a), столбцы q(c−b) после зеркала; + = c экстремум, − = скат', fontsize=11, x=0.01, ha='left', color='#1f1f1d')
        plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(f'{prefix}_{pn}.png', dpi=130, facecolor='#ffffff'); plt.close(fig)
if __name__ == '__main__':
    main(sys.argv[1], sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[2:]), [])))
