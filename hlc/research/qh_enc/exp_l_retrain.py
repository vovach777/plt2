# exp_l_retrain.py - "if L lies, the Huffman tables lie too": L and QH tables retrained as a pair, on screen content.
# Train on screen images, test on other (held-out) screen images. For each L rule the QH tables (32, <= 11 bits, Rice k)
# are retrained on the train images with h = z >> L of that rule, so the comparison is fair:
#   causal   : L from the 16 z above (the format), context (Y/S, Q(sa))
#   own      : L from the block's own mean (PLN1 -m 1), sent (cost: conditional entropy of L given the causal L,
#              per image), context (Y/S, Q(sa))
#   own+L    : as own, context (Y/S, L_own, Q(sa)) (the sent L is known to the decoder, so it can pick the table too)
#   format   : causal L with the format tables (trained on photos), for reference
# Costs per coded block: letters + tails + 16 L planes (+ run codes, + the zU split as in exp_pred, equal for all).
# usage: exp_l_retrain.py DUMPER TRAIN_LIST -- TEST_LIST   (lists: comma-separated image paths)
import sys, os, numpy as np
sys.argv, _a = sys.argv[:1] + ['x', 'y'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_pred.py')).read().split("dm, dw = sys.argv[1], sys.argv[2]")[0])
from train_tables import cluster, rice_esc_bits, QMAXQ
from lib import huff_len
NQ = T['NQ']; dumper = _a[1]; train_imgs = _a[2].split(','); test_imgs = _a[4].split(',')
def blocks(img):
    d, c, parts = costs(dumper, img); k = ~d['skip']; z = d['z'].astype(np.int64)[k]
    s = z.sum(1) >> 4; Lown = np.clip(np.where(s > 0, np.floor(np.log2(np.maximum(s, 1))).astype(np.int64) + 1, 0) - 2, 0, 7)
    q = qhsim.Qsa(d['sa'])[k]; pt = d['ptype'][k]
    return dict(z=z, Lc=d['Lo'][k], Lo=Lown, cA=pt * NQ + q, cB=(pt * 8 + Lown) * NQ + q, other=parts['runs'] + parts['split'], npx=d['npx'])
RULES = [('causal', 'Lc', 'cA'), ('own', 'Lo', 'cA'), ('own+L', 'Lo', 'cB')]
XC = 4096
def count(bs, Lk, ck, nctx):
    C = np.zeros((nctx, 256)); X = np.zeros((nctx, XC + 1))
    for b in bs:
        h = b['z'] >> b[Lk][:, None]; lt = qhsim.letters(h); ctx = b[ck]
        np.add.at(C, (np.repeat(ctx, 4), lt.ravel()), 1)
        m = h >= 3; np.add.at(X, (np.repeat(ctx, 16).reshape(-1, 16)[m], np.minimum(h[m] - 3, XC)), 1)
    return C, X
def build(C, X):
    mp, tabs = cluster(C, 32)
    for c in np.where(mp < 0)[0]:
        cand = [x for x in range(len(mp)) if mp[x] >= 0]; mp[c] = mp[min(cand, key=lambda x: abs(x - c))]
    lens = huff_len(tabs, 11).astype(np.int64); xv = np.arange(XC + 1); ks = np.zeros(len(tabs), np.int64)
    for t in range(len(tabs)):
        xt = X[mp == t].sum(0); ks[t] = int(np.argmin([(xt * rice_esc_bits(xv, kk)).sum() for kk in range(5)]))
    return mp, lens, ks
def hcond(x, y):
    j = np.bincount(y * 8 + x, minlength=64).reshape(8, 8).astype(np.float64); r = j.sum(1, keepdims=True)
    p = np.where(j > 0, j / np.maximum(r, 1), 1); return float(-(j * np.log2(p)).sum())
tr = [blocks(i) for i in train_imgs]; print(f'train: {len(tr)} images', flush=True)
TB = {}
for name, Lk, ck in RULES:
    TB[name] = build(*count(tr, Lk, ck, 2 * NQ * (8 if ck == 'cB' else 1)))
del tr
tot = {k: 0.0 for k in ['format'] + [r[0] for r in RULES]}; npx = 0
print('%-34s %8s %8s %8s %8s' % ('test image (bpp, model)', 'format', 'causal', 'own', 'own+L'))
for img in test_imgs:
    b = blocks(img); n = b['npx']; npx += n; row = {}
    h = b['z'] >> b['Lc'][:, None]; t = T['cm'][b['cA']]
    lb = T['lens'][t[:, None], qhsim.letters(h)].sum(); x = np.maximum(h - 3, 0)
    tb = np.where(h >= 3, rice_esc_bits(x, T['rk'][t][:, None]), 0).sum(); row['format'] = lb + tb + 16 * b['Lc'].sum() + b['other']
    for name, Lk, ck in RULES:
        mp, lens, ks = TB[name]; L = b[Lk]; h = b['z'] >> L[:, None]; t = mp[b[ck]]
        lb = lens[t[:, None], qhsim.letters(h)].sum(); x = np.maximum(h - 3, 0)
        tb = np.where(h >= 3, rice_esc_bits(x, ks[t][:, None]), 0).sum()
        row[name] = lb + tb + 16 * L.sum() + b['other'] + (hcond(L, b['Lc']) if Lk == 'Lo' else 0)
    for k in row: tot[k] += row[k]
    print('%-34s %8.4f %8.4f %8.4f %8.4f' % (os.path.basename(img)[:34], *(row[k] / n for k in ('format', 'causal', 'own', 'own+L'))), flush=True)
print('%-34s %8.4f %8.4f %8.4f %8.4f' % ('TOTAL', *(tot[k] / npx for k in ('format', 'causal', 'own', 'own+L'))))
print('vs format: causal retrained %+.1f%%, own %+.1f%%, own+L %+.1f%%' % tuple(100 * (tot[k] / tot['format'] - 1) for k in ('causal', 'own', 'own+L')))
