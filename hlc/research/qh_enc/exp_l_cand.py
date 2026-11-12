# exp_l_cand.py - Qwen's cheap argmin for L: per block a few candidates instead of all 8 (Lc = causal, Lo = own
# L = clamp(bw(sum z >> 4) - 2, 0, 7), 0), the encoder costs only them and sends one symbol "which candidate won"
# (only when the distinct candidates are > 1; the decoder knows Lc, but NOT Lo: Lo depends on the block itself, so the
# symbol has to name the L value - here coded as the index among the distinct candidates in the order the encoder saw,
# which the decoder can rebuild only if Lo is sent anyway -> we cost it two ways:
#   idx : ideal cost of the winner index (as Qwen) - optimistic, assumes the decoder knows Lo (it does not);
#   val : ideal cost of the winning L value given the context and Lc (decodable: the decoder knows Lc and the context).
# Context: bucket of sa (bw, <= 15), previous block empty, plane type. Threshold T: take a non-Lc winner only if it
# gains more than T bits. Costs with the format tables (exp_pred.costs). usage: CANDS=c,o,0 ESC_T=T exp_l_cand.py DUMPER NAME img...
import sys, os, numpy as np
sys.argv, _a = sys.argv[:1] + ['x', 'y'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_pred.py')).read().split("dm, dw = sys.argv[1], sys.argv[2]")[0])
from lib import bwv
dumper, name, imgs = _a[1], _a[2], _a[3:]
CANDS = os.environ.get('CANDS', 'c,o,0').split(','); TT = float(os.environ.get('ESC_T', '0'))
def H(x, ctx):
    if len(x) == 0: return 0.0
    _, ci = np.unique(ctx, return_inverse=True); _, xi = np.unique(x, return_inverse=True)
    j = np.zeros((ci.max() + 1, xi.max() + 1)); np.add.at(j, (ci, xi), 1); r = j.sum(1, keepdims=True)
    p = np.where(j > 0, j / r, 1); return float(-(j * np.log2(p)).sum())
A = dict(caus=0.0, idx=0.0, val=0.0, side_i=0.0, side_v=0.0, npx=0, chg=0, nb=0, evals=0)
for img in imgs:
    d, c, parts = costs(dumper, img); k = ~d['skip']; z = d['z'].astype(np.int64)
    t = T['cm'][d['ptype'] * T['NQ'] + qhsim.Qsa(d['sa'])]
    CL = np.zeros((len(z), 8))
    for L in range(8):
        h = z >> L; x = np.maximum(h - 3, 0); q = x >> T['rk'][t][:, None]
        CL[:, L] = T['lens'][t[:, None], qhsim.letters(h)].sum(1) + np.where(h >= 3, np.where(q >= T['QESC'], T['QESC'] + 10, q + 1 + T['rk'][t][:, None]), 0).sum(1) + 16 * L
    n = len(z); i = np.arange(n); Lc = d['Lo'].astype(np.int64)
    s = z.sum(1) >> 4; Lo = np.clip(bwv(s) - 2, 0, 7)
    cand = np.stack([{'c': Lc, 'o': Lo, '0': np.zeros(n, np.int64)}[q] for q in CANDS], 1)     # n x m
    cc = CL[i, Lc]; cv = CL[i[:, None], cand]; bj = cv.argmin(1); Lw = cand[i, bj]; cw = cv[i, bj]
    take = (cc - cw > TT) & k; Lf = np.where(take, Lw, Lc); cf = np.where(take, cw, cc)
    ndist = np.array([len(set(r)) for r in cand]); need = (ndist > 1) & k
    # index of the chosen L among the distinct candidates (first occurrence order)
    idx = np.array([list(dict.fromkeys(r)).index(l) for r, l in zip(cand, Lf)])
    pk = d['tile'].astype(np.int64) * 4 + d['pl']; prev_empty = np.r_[True, (z[:-1] == 0).all(1)] | np.r_[True, pk[1:] != pk[:-1]]
    ctx = (np.minimum(bwv(d['sa'].astype(np.int64)), 15) * 2 + prev_empty) + 64 * d['ptype']
    side_i = H(idx[need], ctx[need]); side_v = H(Lf[k], (ctx * 8 + Lc)[k])
    other = c.sum() - cc[k].sum()
    A['caus'] += c.sum(); A['idx'] += other + cf[k].sum() + side_i; A['val'] += other + cf[k].sum() + side_v
    A['side_i'] += side_i; A['side_v'] += side_v; A['npx'] += d['npx']; A['chg'] += int(take.sum()); A['nb'] += int(k.sum()); A['evals'] += int(ndist[k].sum())
n = A['npx']; b = A['caus']
print(f"{name} cands={','.join(CANDS)} T={TT:g}: causal {b / n:.4f} | idx-coded {A['idx'] / n:.4f} ({100 * (A['idx'] / b - 1):+.2f}%, side {A['side_i'] / n:.4f}) | value-coded {A['val'] / n:.4f} ({100 * (A['val'] / b - 1):+.2f}%, side {A['side_v'] / n:.4f}) | changed {A['chg'] / A['nb']:.1%}, evals/block {A['evals'] / A['nb']:.2f}", flush=True)
