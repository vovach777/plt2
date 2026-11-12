# exp_l_med.py - L predicted by MED over the block grid: a = own L of the block to the left, b = above, c = above-left
# (all already decoded, so causal). Variants, cost of the coded blocks with the format tables (letters + tails + planes),
# in % of the file model (exp_pred.costs): causal L of the format (mean of the 16 z above), L_med (no side info),
# own L sent conditionally on the causal L, own L sent conditionally on L_med (signal = conditional entropy per image).
# usage: exp_l_med.py DUMPER img...
import sys, os, numpy as np
sys.argv, _a = sys.argv[:1] + ['x', 'y'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_pred.py')).read().split("dm, dw = sys.argv[1], sys.argv[2]")[0])
dumper, imgs = _a[1], _a[2:]
def hcond(x, y):
    j = np.bincount(y * 8 + x, minlength=64).reshape(8, 8).astype(np.float64); r = j.sum(1, keepdims=True)
    p = np.where(j > 0, j / np.maximum(r, 1), 1); return float(-(j * np.log2(p)).sum())
S = dict(tot=0.0, n=0, causal=0.0, med=0.0, own_c=0.0, own_m=0.0)
for img in imgs:
    d, c, _ = costs(dumper, img); z = d['z'].astype(np.int64); k = ~d['skip']
    s = z.sum(1) >> 4; Lown = np.clip(np.where(s > 0, np.floor(np.log2(np.maximum(s, 1))).astype(np.int64) + 1, 0) - 2, 0, 7)
    step = d['tw'] // 16; bi = d['bi']; col = bi % np.maximum(step, 1); n = len(z); idx = np.arange(n)
    ok_up = (d['tw'] % 16 == 0) & (bi >= step); ok_left = col > 0
    a = np.where(ok_left, Lown[np.maximum(idx - 1, 0)], -1); b = np.where(ok_up, Lown[np.maximum(idx - step, 0)], -1)
    cc = np.where(ok_up & ok_left, Lown[np.maximum(idx - step - 1, 0)], -1)
    a2 = np.where(a < 0, np.where(b < 0, 2, b), a); b2 = np.where(b < 0, a2, b); c2 = np.where(cc < 0, np.minimum(a2, b2), cc)
    Lmed = np.clip(a2 + b2 - c2, np.minimum(a2, b2), np.maximum(a2, b2))
    t = T['cm'][d['ptype'] * T['NQ'] + qhsim.Qsa(d['sa'])]
    def cost(L):
        h = z >> L[:, None]; lt = qhsim.letters(h); lb = T['lens'][t[:, None], lt].sum(1); kk = T['rk'][t][:, None]; x = np.maximum(h - 3, 0); q = x >> kk
        return (lb + np.where(h >= 3, np.where(q >= T['QESC'], T['QESC'] + 10, q + 1 + kk), 0).sum(1) + 16 * L)[k].sum()
    Lc = d['Lo']
    S['causal'] += cost(Lc); S['med'] += cost(Lmed); co = cost(Lown)
    S['own_c'] += co + hcond(Lown[k], Lc[k]); S['own_m'] += co + hcond(Lown[k], Lmed[k]); S['tot'] += c.sum(); S['n'] += d['npx']
base = S['causal']
print('  '.join(f'{name} {100 * (S[key] - base) / S["tot"]:+.1f}%' for name, key in (('L_med (no side info)', 'med'), ('own L sent | causal', 'own_c'), ('own L sent | L_med', 'own_m'))) + '   (of the file, vs the causal L of the format)')
