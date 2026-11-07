# pick_tiles.py - cut 256x256 gray tiles of different character from P5 images: smooth (lowest residual entropy),
# mixed (median), edges (most "peaky": many zero residuals and many large ones), texture (highest entropy).
# usage: python pick_tiles.py outdir a.pgm b.pgm ...      writes t_smooth.pgm t_mixed.pgm t_edges.pgm t_texture.pgm
import sys, os, numpy as np
def pgm(p):
    b = open(p, 'rb').read(); t = b.split(maxsplit=4); w, h = int(t[1]), int(t[2]); return np.frombuffer(t[4][:w * h], np.uint8).reshape(h, w)
def zz(t):
    t = t.astype(np.int32); a = t[1:, :-1]; b = t[:-1, 1:]; c = t[:-1, :-1]; x = t[1:, 1:]
    p = np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)); e = ((x - p + 128) & 255) - 128; return np.where(e >= 0, 2 * e, -2 * e - 1)
out = sys.argv[1]; C = []
for f in sys.argv[2:]:
    im = pgm(f)
    for y in range(0, im.shape[0] - 255, 256):
        for x in range(0, im.shape[1] - 255, 256):
            t = im[y:y + 256, x:x + 256]; z = zz(t); h = np.bincount(z.ravel(), minlength=256) / z.size; H = -(h[h > 0] * np.log2(h[h > 0])).sum()
            C.append((H, (z == 0).mean() * (z >= 64).mean(), os.path.basename(f), y, x, t))
C.sort(key=lambda c: c[0]); pick = {'smooth': C[0], 'mixed': C[len(C) // 2], 'texture': C[-1]}
lower = [c for c in C if c[0] < C[len(C) // 2][0] and c is not C[0]]; pick['edges'] = max(lower, key=lambda c: c[1])
for k, c in pick.items():
    with open(os.path.join(out, 't_%s.pgm' % k), 'wb') as fo: fo.write(b'P5\n256 256\n255\n' + c[5].tobytes())
    print('%-8s %s y=%d x=%d  H0(z)=%.3f bit' % (k, c[2], c[3], c[4], c[0]))
