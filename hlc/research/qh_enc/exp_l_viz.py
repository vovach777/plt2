# exp_l_viz.py - L maps, one pixel per 16-pixel block (Y plane, or gray), scaled up with nearest neighbour.
# Panels (left to right): luma averaged over the block | causal L (PLR, from the 16 z above) | own L of the block
# (its own mean, off 2) | best L of the block (min of the exact cost, format tables) | error causal vs best
# (red = causal too high, blue = too low, brightness = bits lost).  L 0..7 as a colour ramp (black = 0, white = 7).
# VAVG=16: average every 16 rows of the map, so one pixel = 16 x 16 image pixels.
# usage: [VAVG=v] [CROP=bx,by,bw,bh] [SCALE=s] exp_l_viz.py DUMPER img out.png   (CROP in block-map pixels: x in blocks, y in rows)
import sys, os, numpy as np
sys.argv, _a = sys.argv[:1] + ['x', 'y'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_pred.py')).read().split("dm, dw = sys.argv[1], sys.argv[2]")[0])
from PIL import Image
dumper, img, out = _a[1], _a[2], _a[3]
d, c, parts = costs(dumper, img); z = d['z'].astype(np.int64); n = len(z); idx = np.arange(n)
t = T['cm'][d['ptype'] * T['NQ'] + qhsim.Qsa(d['sa'])]
C = np.zeros((n, 8))
for L in range(8):
    h = z >> L; x = np.maximum(h - 3, 0); kk = T['rk'][t][:, None]; q = x >> kk
    C[:, L] = (T['lens'][t[:, None], qhsim.letters(h)].sum(1)
               + np.where(h >= 3, np.where(q >= T['QESC'], T['QESC'] + 10, q + 1 + kk), 0).sum(1) + 16 * L)
Lc = d['Lo'].astype(np.int64); Lb = C.argmin(1); lost = C[idx, Lc] - C.min(1)
s = z.sum(1) >> 4; Lown = np.clip(np.where(s > 0, np.floor(np.log2(np.maximum(s, 1))).astype(np.int64) + 1, 0) - 2, 0, 7)
raw = open(img, 'rb').read(); hd = raw.split(maxsplit=4); W, H = int(hd[1]), int(hd[2]); ch = 3 if hd[0] == b'P6' else 1
pix = np.frombuffer(raw[len(raw) - W * H * ch:], np.uint8).reshape(H, W, ch).astype(np.int32)
lum = (pix @ np.array([77, 150, 29]) >> 8) if ch == 3 else pix[:, :, 0]
nx = (W + 287) // 288; MW = (W + 15) // 16
maps = [np.zeros((H, MW, 3), np.uint8) for _ in range(5)]
def ramp(L): v = min(255, int(L) * int(os.environ.get("LGAIN", "36"))); return (v, v, v)
for i in np.flatnonzero(d['pl'] == 0):
    tl = int(d['tile'][i]); tw = int(d['tw'][i]); ox = (tl % nx) * 288; oy = (tl // nx) * 288; p0 = int(d['bi'][i]) * 16
    yy = oy + p0 // tw; xx = ox + p0 % tw
    if yy >= H or xx >= W: continue
    bx = xx // 16
    maps[0][yy, bx] = [int(lum[yy, xx:xx + 16].mean())] * 3
    maps[1][yy, bx] = ramp(Lc[i]); maps[2][yy, bx] = ramp(Lown[i]); maps[3][yy, bx] = ramp(Lb[i])
    v = min(255, int(lost[i] * float(os.environ.get("EGAIN", "8"))))
    maps[4][yy, bx] = (v, 0, 0) if Lc[i] > Lb[i] else ((0, v // 2, v) if Lc[i] < Lb[i] else (0, 0, 0))
VA = int(os.environ.get('VAVG', '1'))            # average every VA rows of the block map (16 = square cells)
if VA > 1:
    hh = maps[0].shape[0] // VA * VA
    maps = [m[:hh].reshape(hh // VA, VA, m.shape[1], 3).astype(np.float64).mean(1).round().astype(np.uint8) for m in maps]
if os.environ.get('CROP'):
    bx, by, bw, bh = map(int, os.environ['CROP'].split(','))
    maps = [m[by:by + bh, bx:bx + bw] for m in maps]
S = int(os.environ.get('SCALE', '4'))
sep = np.full((maps[0].shape[0], 2, 3), 255, np.uint8)
row = np.concatenate(sum([[m, sep] for m in maps], [])[:-1], 1)
Image.fromarray(row).resize((row.shape[1] * S, row.shape[0] * S), Image.NEAREST).save(out)
y = d['pl'] == 0
print(f"{os.path.basename(img)} {W}x{H}: map {MW}x{H} blocks; causal == best {100 * (Lc == Lb)[y].mean():.1f}%, own == best "
      f"{100 * (Lown == Lb)[y].mean():.1f}%, causal too high {100 * (Lc > Lb)[y].mean():.1f}%, too low {100 * (Lc < Lb)[y].mean():.1f}%")
