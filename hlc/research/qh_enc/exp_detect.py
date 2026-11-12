# STATUS: written, not run yet (the run was stopped; detector work handed to Argon). Kept for reference.
# exp_detect.py - cheap detector "this plane of a tile is text / synthetic" for the PLR / PLS switch and the PLS variant.
# Per plane of a tile: exact block costs (runs and the zU split are equal in both modes and left out) of PLR (format
# tables, format escape) and of PLS variants v = 0..7 (screen tables, qhs.hpp), + cheap features of the residuals:
# p0 = share of z == 0, pbig = share of z >= 32, nzmean = mean of z > 0, bim = share of "bimodal" blocks
# (max >= 64, no z in 1..31), mean = mean z.  Saves an npz per set.  usage: exp_detect.py TABLES.pkl DUMPER OUT.npz img...
import sys, os, pickle, numpy as np
sys.argv, _a = sys.argv[:1] + ['train', 'x', 'y', 'z'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_hybrid.py')).read().split("if sys.argv[1] == 'train':")[0])
mp, lens, ks = pickle.load(open(_a[1], 'rb'))['H0']; dumper, out, imgs = _a[2], _a[3], _a[4:]
kf = np.minimum(T['rk'], KMAX)
def tailfmt(h, k):
    x = np.maximum(h - 3, 0); q = x >> k; return np.where(h >= 3, np.where(q >= T['QESC'], T['QESC'] + 10, q + 1 + k), 0)
rows = []
for img in imgs:
    b = blocks(dumper, img); z = b['z']; est = np.maximum(b['sa'], b['sl']); g = b['tile'] * 2 + b['pt']; ng = 2 * b['nt']
    t = T['cm'][b['pt'] * NQ + qhsim.Qsa(b['sa'])]; h = z >> b['Lc'][:, None]
    plr = T['lens'][t[:, None], qhsim.letters(h)].sum(1) + tailfmt(h, T['rk'][t][:, None]).sum(1) + 16 * b['Lc']
    ts = mp[b['pt'] * NQ + qhsim.Qsa(est)]; pls = []
    for v in range(8):
        Ls = np.full(len(z), v - 1) if v >= 4 else np.clip(bwv(est >> 4) - 2 + v, 0, 7)
        if v < 4: Ls[b['first']] = 2 + v
        nz = z > 0; hh = np.where(nz, ((z - 1) >> Ls[:, None]) + 1, 0); pls.append(cost(hh, Ls * nz.sum(1), ts, lens, ks))
    nzc = (z > 0).sum(1); bim = (z.max(1) >= 64) & ~((z >= 1) & (z < 32)).any(1)
    cnt = np.bincount(g, None, ng); ok = cnt > 0
    f = lambda v: np.bincount(g, v, ng)[ok]
    R = np.column_stack([f(plr)] + [f(p) for p in pls] + [f(16 - nzc), f((z >= 32).sum(1)), f(z.sum(1)), f(nzc), f(bim), cnt[ok] * 16, (np.arange(ng) % 2)[ok]])
    rows.append(R); print(os.path.basename(img), flush=True)
np.savez(out, R=np.vstack(rows), img=np.concatenate([[os.path.basename(i)] * len(r) for i, r in zip(imgs, rows)]))
