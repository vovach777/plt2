import numpy as np, pickle, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
exec(open('axesfit.py').read().split('N0 = {')[0])
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
zv = np.arange(512); ev = np.where(zv & 1, (zv + 1) // 2, zv // 2)
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), facecolor='#ffffff')
for ax, (t, nm) in zip(axs, ((0, 'Y'), (1, 'U/V'))):
    _, mem = oracle(t, 6); n = Hc[t].sum(-1); N = n.sum()
    sc = [ (Hc[t][[i for i, j in g], [j for i, j in g]] * ev).sum() / max(1, n[[i for i, j in g], [j for i, j in g]].sum()) for g in mem]
    order = np.argsort(sc); lab = -np.ones((10, 19), int)
    for r, gi in enumerate(order):
        for i, j in mem[gi]: lab[i, j] = r
    img = np.ones((10, 19, 3))
    for i in range(10):
        for j in range(19):
            if lab[i, j] >= 0:
                c = np.array(to_rgb(PAL[lab[i, j]])); w = 0.35 + 0.65 * min(1, (100 * n[i, j] / N) / 2.0) ** 0.5
                img[i, j] = 1 - w * (1 - c)
    ax.imshow(img, origin='lower', extent=(-9.5, 9.5, -0.5, 9.5), aspect='auto')
    for i in range(10):
        for j in range(19):
            if lab[i, j] >= 0:
                sh = 100 * n[i, j] / N
                ax.text(j - 9, i + 0.12, str(lab[i, j] + 1), ha='center', va='center', fontsize=9, weight='bold', color='#1f1f1d')
                if sh >= 0.05: ax.text(j - 9, i - 0.25, f'{sh:.1f}' if sh < 10 else f'{sh:.0f}', ha='center', va='center', fontsize=6.5, color='#3d3d3a')
    ax.set_xticks(range(-9, 10)); ax.set_yticks(range(10))
    ax.set_xlabel('l2 = signed bw(c−b) после зеркала  (− : c экстремум, + : монотонный скат)', color='#3d3d3a')
    ax.set_ylabel('l1 = bw|c−a|', color='#3d3d3a')
    ax.set_title(f'{nm}: 6 корзин оракула (цифра = корзина по росту |e|, ниже — доля %)', fontsize=11, color='#1f1f1d', loc='left')
    for s in ax.spines.values(): s.set_color('#d8d7d0')
    ax.tick_params(colors='#5e5d59', labelsize=8)
    means = [sc[g] for g in order]
    ax.text(1.0, -0.13, 'средний |e| корзин: ' + '  '.join(f'{k + 1}: {m:.1f}' for k, m in enumerate(means)), transform=ax.transAxes, ha='right', fontsize=8, color='#5e5d59')
plt.tight_layout(); plt.savefig('axes_clusters.png', dpi=150, facecolor='#ffffff')
