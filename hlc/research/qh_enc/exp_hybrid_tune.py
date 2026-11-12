# exp_hybrid_tune.py - per-tile tuning of the PLS (hybrid H0) mode: which variant choice is worth its cost.
# Variants: Hd (Ls = bw(est >> 4) - 2 - d, d = 0..4), HFl (fixed Ls = l, l = 3..6), and Ls offsets d = -1, -2.
# Tables: own (each variant its own 32 tables, from exp_hybrid.py train) or shared (all variants with the H0 tables).
# Choice granularity: per tile (Y + S together) or per plane of a tile; side info log2(#choices) bits per choice.
# Format (PLR) is NOT a choice here (only PLS variants).  usage: exp_hybrid_tune.py TABLES.pkl DUMPER NAME img...
import sys, os, pickle, numpy as np
sys.argv, _a = sys.argv[:1] + ['train', 'x', 'y', 'z'], sys.argv
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_hybrid.py')).read().split("if sys.argv[1] == 'train':")[0]
exec(src)
tabs = pickle.load(open(_a[1], 'rb')); dumper, name, imgs = _a[2], _a[3], _a[4:]
VAR.update({'Hm1': ('H', -1), 'Hm2': ('H', -2), 'Hm3': ('H', -3)})
def hs(b, v):
    kind, p = VAR[v]; est = np.maximum(b['sa'], b['sl']); ctx = b['pt'] * NQ + qhsim.Qsa(est); z = b['z']
    Ls = np.full(len(z), p) if kind == 'HF' else np.clip(bwv(est >> 4) - 2 - p, 0, 7)
    if kind != 'HF': Ls[b['first']] = min(max(2 - p, 0), 7)
    nz = z > 0; return np.where(nz, ((z - 1) >> Ls[:, None]) + 1, 0), Ls * nz.sum(1), ctx
own = [v for v in tabs if v[0] == 'H']; shared = ['H0', 'H1', 'H2', 'H3', 'Hm1', 'Hm2', 'Hm3', 'HF3', 'HF4', 'HF5', 'HF6']
A = {}; npx = 0
def acc(k, v): A[k] = A.get(k, 0) + v
for img in imgs:
    b = blocks(dumper, img); o = b['other']; npx += b['npx']
    g_t = b['tile']; g_p = b['tile'] * 2 + b['pt']; nt = b['nt']
    co = {}; cs = {}
    for v in own: mp, lens, ks = tabs[v]; h, raw, ctx = hs(b, v); co[v] = cost(h, raw, mp[ctx], lens, ks)
    mp, lens, ks = tabs['H0']
    for v in shared: h, raw, ctx = hs(b, v); cs[v] = cost(h, raw, mp[ctx], lens, ks)
    acc('H0', co['H0'].sum() + o)
    for lab, C, vs in (('own', co, own), ('shared', cs, shared), ('shared H0,Hm1', cs, ['H0', 'Hm1']), ('shared H0,Hm1,Hm2,Hm3', cs, ['H0', 'Hm1', 'Hm2', 'Hm3']),
                       ('shared H0,HF4,HF5,HF6', cs, ['H0', 'HF4', 'HF5', 'HF6']), ('shared H0,Hm1,Hm2,HF5', cs, ['H0', 'Hm1', 'Hm2', 'HF5']),
                       ('shared H0,Hm1,HF4,HF5,HF6,..8', cs, ['H0', 'Hm1', 'Hm2', 'HF3', 'HF4', 'HF5', 'HF6', 'Hm3'])):
        for gl, g, ng in (('tile', g_t, nt), ('plane', g_p, 2 * nt)):
            m = np.min(np.stack([np.bincount(g, C[v], ng) for v in vs]), 0).sum()
            acc(f'{lab} / {gl}', m + o + np.ceil(np.log2(len(vs))) * (np.unique(g).size))
print(name, f'npx {npx}'); base = A['H0']
for k, v in A.items(): print(f'  {k:28s} {v / npx:.4f} ({100 * (v / base - 1):+.2f}% vs H0)')
