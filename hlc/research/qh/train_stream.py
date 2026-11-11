# train_stream.py - QH tables from a large corpus, streaming: one image at a time is dumped (palic-2r built with
# -DPLR_DUMP -DPLR_NO_QH), counted and deleted, so the corpus never has to fit in memory or on disk.
# Counts per context (plane type Y/S x Q(sa), 136): letters (4 x min(h,3), 256) and tails x = h - 3; zero-run blocks
# skipped exactly as the codec does. Then the same clustering / Huffman / Rice k as train_tables.py.
# usage: train_stream.py DUMPER OFF K MAXLEN OUT.bin image... (P5 / P6)
import sys, os, subprocess, tempfile, struct, numpy as np
import lib
from train_tables import Q, NQ, QMAXQ, cluster, rice_esc_bits
from lib import huff_len
dumper, off, K, maxlen, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
imgs = sys.argv[6:]
C = np.zeros((2 * NQ, 256)); XC = 4096; X = np.zeros((2 * NQ, XC + 1))
tmp = tempfile.mkdtemp(); os.environ['QH_DUMP'] = tmp
import load as L_
L_.D = tmp
for i, f in enumerate(imgs):
    sd = os.path.join(tmp, 'one'); os.makedirs(sd, exist_ok=True)
    for x in os.listdir(sd): os.remove(os.path.join(sd, x))
    env = dict(os.environ, PLR_DUMP_FILE=os.path.join(sd, 'a.dmp'))
    subprocess.run([dumper, 'c', f, os.path.join(sd, 'a.plr')], env=env, check=True, stdout=subprocess.DEVNULL)
    lib._cache.clear(); d = lib.prep('one', off); k = ~d['skip']
    ctx = d['ptype'][k] * NQ + Q(d['sa'][k]); h = d['h'][k]
    c = np.minimum(h, 3).reshape(len(h), 4, 4); lt = c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)
    np.add.at(C, (np.repeat(ctx, 4), lt.ravel()), 1)
    m = h >= 3; xs = np.minimum(h[m] - 3, XC); np.add.at(X, (np.repeat(ctx, 16).reshape(-1, 16)[m], xs), 1)
    if (i + 1) % 10 == 0: print(f'{i + 1}/{len(imgs)}', flush=True)
mp, tabs = cluster(C, K)
for c in np.where(mp < 0)[0]:
    g = c // NQ; cand = [x for x in range(g * NQ, (g + 1) * NQ) if mp[x] >= 0] or [x for x in range(2 * NQ) if mp[x] >= 0]
    mp[c] = mp[min(cand, key=lambda x: abs(x - c))]
K = len(tabs); lens = huff_len(tabs, maxlen).astype(np.int64)
ks = np.zeros(K, np.int64); xv = np.arange(XC + 1)
for t in range(K):
    xt = X[mp == t].sum(0); ks[t] = int(np.argmin([(xt * rice_esc_bits(xv, kk)).sum() for kk in range(5)]))
with open(out, 'wb') as f:
    f.write(struct.pack('<4i', K, maxlen, NQ, QMAXQ)); f.write(mp.astype(np.uint8).tobytes())
    f.write(lens.astype(np.uint8).tobytes()); f.write(ks.astype(np.uint8).tobytes())
print(f'{len(imgs)} images, {int(C.sum())} letters -> {out}: K={K} rice k {ks.tolist()}')
