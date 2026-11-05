#!/usr/bin/env python3
"""Full (unmirrored) q7(c-a) x q7(c-b) grid, corpus, Y and U/V: share % and mean signed MED residual per cell,
plus the entropy cost of folding point-symmetric pairs (q1,q2) ~ (-q1,-q2) with the residual sign flipped.

usage: python q7full.py <out.png> <ppm/png files or globs...>      (also writes <out>_folded.png: the same grid after the
mirror, h = q(c-b) + 7 q(c-a) >= 0 kept, the pair (q1,q2) ~ (-q1,-q2) summed, mean of s*e)
"""
import numpy as np, glob, sys
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
def H(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())
def qsym(x, edges): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in edges)
def shift(A, dy, dx):
    O = np.zeros_like(A); h, w = A.shape; O[dy:, dx:] = A[:h - dy, :w - dx]; return O
LAB = ['−16+', '−4…15', '−1…3', '0', '+1…3', '+4…15', '+16+']
def main(out, files):
    n = np.zeros((2, 7, 7)); se = np.zeros((2, 7, 7)); nf = np.zeros((2, 7, 7)); sf = np.zeros((2, 7, 7)); full = {0: 0.0, 1: 0.0}; fold = {0: 0.0, 1: 0.0}
    for f in files:
        a0 = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
        R_, G, B = a0[..., 0], a0[..., 1], a0[..., 2]; V = R_ - G; U = B - G; Y = G + ((U + V) >> 2)
        Sf = {0: {}, 1: {}}; Sm = {0: {}, 1: {}}
        for P, n9 in [(V, True), (U, True), (Y, False)]:
            t = int(n9)
            a = shift(P, 0, 1); b = shift(P, 1, 0); c = shift(P, 1, 1); a[:, 0] = b[:, 0]; c[:, 0] = b[:, 0]
            e = P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
            q1, q2 = qsym(c - a, [1, 4, 16]), qsym(c - b, [1, 4, 16]); idx = (q1 + 3) * 7 + (q2 + 3)
            n[t] += np.bincount(idx.ravel(), minlength=49).reshape(7, 7); se[t] += np.bincount(idx.ravel(), weights=e.ravel(), minlength=49).reshape(7, 7)
            h = q2 + 7 * q1; s = np.where(h < 0, -1, 1); fidx = ((s * q1 + 3) * 7 + (s * q2 + 3)).ravel()
            nf[t] += np.bincount(fidx, minlength=49).reshape(7, 7); sf[t] += np.bincount(fidx, weights=(s * e).ravel(), minlength=49).reshape(7, 7)
            for ee, key, D in ((e, idx, Sf[t]), (s * e, np.abs(h), Sm[t])):
                ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
                for v in np.unique(key): D.setdefault(int(v), []).append(z[key == v])
        for t in (0, 1):
            full[t] += sum(H(np.concatenate(v)) for v in Sf[t].values()); fold[t] += sum(H(np.concatenate(v)) for v in Sm[t].values())
    for N_, S_, tag in ((n, se, ''), (nf, sf, '_folded')):
        draw(out.replace('.png', tag + '.png'), N_, S_, full, fold, tag)
def draw(out, n, se, full, fold, tag):
    cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.6), facecolor='#ffffff')
    for ax, (t, nm) in zip(axs, ((0, 'Y'), (1, 'U/V'))):
        sh = 100 * n[t] / n[t].sum(); me = se[t] / np.maximum(n[t], 1); vmax = 6 if t == 0 else 20
        if tag: vmax = vmax  # colour scale doubled below
        ax.imshow(np.where(n[t] > 0, sh, np.nan) if tag else sh, origin='lower', cmap=cmap, vmin=0, vmax=vmax * (2 if tag else 1), aspect='auto')
        for i in range(7):
            for j in range(7):
                if tag and n[t, i, j] == 0: continue
                ink = '#ffffff' if sh[i, j] > vmax * (2 if tag else 1) * 0.55 else '#1f1f1d'
                ax.text(j, i + 0.15, f'{sh[i, j]:.1f}' if sh[i, j] < 10 else f'{sh[i, j]:.0f}', ha='center', va='center', fontsize=10, weight='bold', color=ink)
                ax.text(j, i - 0.22, f'e {me[i, j]:+.1f}', ha='center', va='center', fontsize=8, color=ink)
        ax.plot([-0.5, 6.5], [-0.5, 6.5], color='#d8d7d0', lw=0.8, ls=':')
        ax.set_xticks(range(7)); ax.set_xticklabels(LAB, fontsize=8, color='#5e5d59'); ax.set_yticks(range(7)); ax.set_yticklabels(LAB, fontsize=8, color='#5e5d59')
        ax.set_xlabel('q(c−b)', color='#3d3d3a'); ax.set_ylabel('q(c−a)', color='#3d3d3a')
        ax.set_title((f'{nm}: после зеркала — доля % пары и средний s·e' if tag else f'{nm}: доля % и средний e (без зеркала)') + f'   склейка зеркальных пар: {100 * fold[t] / full[t] - 100:+.2f}% энтропии', loc='left', fontsize=10.5, color='#1f1f1d')
        for s_ in ax.spines.values(): s_.set_color('#d8d7d0')
        ax.tick_params(length=0)
        if not tag: print(f'{nm}: fold cost {100 * fold[t] / full[t] - 100:+.3f}%   max share mismatch of mirror pairs ' +
              f'{max(abs(sh[i, j] - sh[6 - i, 6 - j]) for i in range(7) for j in range(7)):.2f} pp')
    plt.tight_layout(); plt.savefig(out, dpi=140, facecolor='#ffffff')
if __name__ == '__main__':
    main(sys.argv[1], sorted(sum((glob.glob(x) if any(ch in x for ch in '*?') else [x] for x in sys.argv[2:]), [])))
