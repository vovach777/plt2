# exp_letter_size.py - held-out ideal cost of the QH letters vs the letter size n (symbols per letter, each min(h,3)):
# n = 1, 2, 4, 8, 16; context = (plane type, Q(sa)) without clustering; probabilities from the training set with
# add-alpha smoothing over the full alphabet (4^n letters). Tails (h-3) and raw planes are the same for every n and
# are left out. Zero-run blocks are skipped as in the codec. off 2.  env QH_DUMP; usage: TRAIN TEST [TEST...]
import sys, numpy as np
from lib import prep
from train_tables import Q, NQ
def cost(train, test, n, alpha):
    def letters(d):
        k = ~d['skip']; c = np.minimum(d['h'][k], 3).reshape(-1, 16 // n, n)
        w = (4 ** np.arange(n)).astype(np.int64); L = (c * w).sum(2)
        ctx = (d['ptype'][k] * NQ + Q(d['sa'][k]))[:, None].repeat(16 // n, 1)
        return ctx.ravel(), L.ravel()
    ct, lt = letters(train); A = 4 ** n
    key = ct * A + lt; uk, cnt = np.unique(key, return_counts=True); tab = dict(zip(uk.tolist(), cnt.tolist()))
    tot = np.bincount(ct, minlength=2 * NQ).astype(np.float64)
    cs, ls = letters(test); ks = (cs * A + ls).tolist()
    num = np.array([tab.get(x, 0) for x in ks], np.float64) + alpha
    den = tot[cs] + alpha * A
    return -np.log2(num / den).sum()
train = prep(sys.argv[1], 2)
for ts in sys.argv[2:]:
    d = prep(ts, 2); n_px = d['npx']; out = []
    for n in (1, 2, 4, 8, 16):
        best = min((cost(train, d, n, a), a) for a in (0.5, 0.1, 0.02, 0.005))
        out.append(f'n={n:2d}: {best[0] / n_px:.4f} (alpha {best[1]})')
    print(f'{ts}: letters bpp  ' + '  '.join(out))
