import numpy as np, glob
from PIL import Image
def qsym(x, edges): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in edges)
E = [1, 4, 16]; files = sorted(glob.glob('photos16h/*.ppm')) + ['cat_h.ppm']
n = np.zeros((2, 4, 7)); se = np.zeros((2, 4, 7)); hz = np.zeros((2, 4, 7, 512))
for f in files:
    a = np.asarray(Image.open(f).convert('RGB')).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2)
    for P, n9 in [(V, True), (U, True), (Y, False)]:
        Z = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32); Z[1:, 1:] = P
        l = Z[1:, :-1].copy(); t = Z[:-1, 1:].copy(); tl = Z[:-1, :-1].copy(); l[:, 0] = t[:, 0]; tl[:, 0] = t[:, 0]
        e = P - np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t))
        qa, qb = qsym(tl - l, E), qsym(tl - t, E); h = qb + 7 * qa; s = np.where(h < 0, -1, 1); qa, qb = qa * s, qb * s
        ee = s * e; ee = ((ee + 256) & 511) - 256 if n9 else ((ee + 128) & 255) - 128; z = np.where(ee >= 0, 2 * ee, -2 * ee - 1)
        idx = (qa * 7 + (qb + 3)).ravel(); T = int(n9)
        n[T] += np.bincount(idx, minlength=28).reshape(4, 7); se[T] += np.bincount(idx, weights=np.abs(ee).ravel(), minlength=28).reshape(4, 7)
        hz[T] += np.bincount(idx * 512 + z.ravel(), minlength=28 * 512).reshape(4, 7, 512)
lv = ['0', '1–3', '4–15', '16+']; cols = ['−16+', '−4…15', '−1…3', '0', '+1…3', '+4…15', '+16+']
for T, nm in ((0, 'Y'), (1, 'U/V')):
    N = n[T].sum(); print(f'\n**{nm}** — доля % / средний |e| / бит на сэмпл; строки q(c−a), столбцы q(c−b) после зеркала')
    print('| q(c−a) \\ q(c−b) | ' + ' | '.join(cols) + ' |'); print('|---' * 8 + '|')
    for i in range(4):
        cells = []
        for j in range(7):
            if n[T, i, j] == 0: cells.append('—'); continue
            h = hz[T, i, j]; p = h[h > 0] / h.sum(); b = -(p * np.log2(p)).sum()
            cells.append(f'{100 * n[T, i, j] / N:.1f} / {se[T, i, j] / n[T, i, j]:.1f} / {b:.2f}')
        print(f'| {lv[i]} | ' + ' | '.join(cells) + ' |')
