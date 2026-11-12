# exp_hybrid.py - hybrid screen mode: Fable's screen mode (own retrained 32 tables, L from the causal estimate shifted
# down by d, per-tile flag) + Argon's Sparse-Raw zero-pinned split inside it: h = 0 if z == 0 else ((z - 1) >> Ls) + 1,
# Ls raw bits only for z > 0 (the decoder sees from the letters + tails which h are 0).
# Estimate: est = max(sa above, sum of the previous block) (Fable's MAX), Ls = clamp(bw(est >> 4) - 2 - d, 0, 7).
# Variants (each with its own table set trained on the train images, Rice k 0..KMAX, escape (8,2) = 8 ones + EG2):
#   S<d>  : Fable's screen mode, h = z >> L, 16 * L raw bits
#   H<d>  : hybrid, sparse split with Ls, Ls * nnz raw bits
#   HF<l> : hybrid with a fixed Ls = l in all blocks of the tile (no estimate)
# Per-tile switch (oracle on the tile cost): format (photo tables, format escape replaced by (8,2)) vs one variant (+1 bit),
# and format vs the best of all variants (+ log2(#variants + 1) bits).
#   train: exp_hybrid.py train OUT.pkl DUMPER img,img,...
#   test : exp_hybrid.py test TABLES.pkl DUMPER NAME img...
import sys, os, pickle, subprocess, tempfile, numpy as np
sys.path.insert(0, '/home/claude/plt2_repo/hlc/research/qh'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp(); os.environ['QH_DUMP'] = tmp
import load as L_; L_.D = tmp
import lib, qhsim
from train_tables import cluster
from lib import huff_len, bwv
T = qhsim.tables(); NQ = T['NQ']; HM = 1024; KMAX = 6
VAR = {**{'S%d' % d: ('S', d) for d in (2, 3)}, **{'H%d' % d: ('H', d) for d in (0, 1, 2, 3, 4)}, **{'HF%d' % l: ('HF', l) for l in (3, 4, 5, 6)}}
def tailbits(h, k):                               # escape (8,2); h, k broadcastable; 0 for h < 3
    x = np.maximum(h - 3, 0); q = x >> k; v = np.maximum(x - (8 << k), 0)
    return np.where(h >= 3, np.where(q >= 8, 8 + 2 * bwv((v >> 2) + 1) + 1, q + 1 + k), 0)
TBK = np.stack([tailbits(np.arange(HM), k) for k in range(KMAX + 1)])
def blocks(dumper, img):
    sd = os.path.join(tmp, 'one'); os.makedirs(sd, exist_ok=True)
    for x in os.listdir(sd): os.remove(os.path.join(sd, x))
    subprocess.run([dumper, 'c', img, os.path.join(sd, 'a.plr')], env=dict(os.environ, PLR_DUMP_FILE=os.path.join(sd, 'a.dmp')), check=True, stdout=subprocess.DEVNULL)
    lib._cache.clear(); d = lib.prep('one', 2); k = ~d['skip']
    S = d['z'].astype(np.float64); kk = np.floor(np.log2(S + 1)); u = 2 ** (kk + 1) - (S + 1)
    split = float((np.where(S > 0, kk + (S + 1 - u) / (S + 1), 0).sum(1) * (d['pl'] == 1)).sum())
    b = dict(z=d['z'][k].astype(np.int64), Lc=d['Lo'][k].astype(np.int64), sa=d['sa'][k].astype(np.int64), sl=d['sl'][k].astype(np.int64),
             pt=d['ptype'][k].astype(np.int64), first=d['first'][k], tile=d['tile'][k].astype(np.int64), nt=int(d['ntile']),
             other=float(d['runbits'].sum()) + split, npx=int(d['npx']))
    lib._cache.clear(); return b
def hsplit(b, v):                                 # -> h, raw bits per block, context
    kind, p = VAR[v]; est = np.maximum(b['sa'], b['sl']); ctx = b['pt'] * NQ + qhsim.Qsa(est); z = b['z']
    if kind == 'HF': Ls = np.full(len(z), p)
    else: Ls = np.clip(bwv(est >> 4) - 2 - p, 0, 7); Ls[b['first']] = max(2 - p, 0)
    if kind == 'S': return z >> Ls[:, None], 16 * Ls, ctx
    nz = z > 0; return np.where(nz, ((z - 1) >> Ls[:, None]) + 1, 0), Ls * nz.sum(1), ctx
def cost(h, raw, t, lens, ks):
    hc = np.minimum(h, HM - 1); return lens[t[:, None], qhsim.letters(h)].sum(1) + TBK[ks[t][:, None], hc].sum(1) + raw
if sys.argv[1] == 'train':
    out, dumper, imgs = sys.argv[2], sys.argv[3], sys.argv[4].split(',')
    C = {v: np.zeros((2 * NQ, 256)) for v in VAR}; X = {v: np.zeros((2 * NQ, HM)) for v in VAR}
    for i, img in enumerate(imgs):
        b = blocks(dumper, img)
        for v in VAR:
            h, raw, ctx = hsplit(b, v)
            np.add.at(C[v], (np.repeat(ctx, 4), qhsim.letters(h).ravel()), 1); np.add.at(X[v], (np.repeat(ctx, 16), np.minimum(h, HM - 1).ravel()), 1)
        print('train', i + 1, len(imgs), os.path.basename(img), flush=True)
    tabs = {}
    for v in VAR:
        mp, tb = cluster(C[v], 32)
        for c in np.where(mp < 0)[0]:
            cand = [x for x in range(len(mp)) if mp[x] >= 0]; mp[c] = mp[min(cand, key=lambda x: abs(x - c))]
        lens = huff_len(tb, 11).astype(np.int64); ks = np.array([int(np.argmin([(X[v][mp == t].sum(0) * TBK[kk]).sum() for kk in range(KMAX + 1)])) for t in range(len(tb))])
        tabs[v] = (mp, lens, ks)
    pickle.dump(tabs, open(out, 'wb'))
else:
    tabs = pickle.load(open(sys.argv[2], 'rb')); dumper, name, imgs = sys.argv[3], sys.argv[4], sys.argv[5:]
    kf = np.minimum(T['rk'], KMAX); A = {}; npx = 0
    for img in imgs:
        b = blocks(dumper, img); nt = b['nt']; tile = b['tile']; o = b['other']
        f = cost(b['Lc'][:, None] * 0 + (b['z'] >> b['Lc'][:, None]), 16 * b['Lc'], T['cm'][b['pt'] * NQ + qhsim.Qsa(b['sa'])], T['lens'], kf)
        ft = np.bincount(tile, f, nt); vt = {}
        for v in tabs:
            mp, lens, ks = tabs[v]; h, raw, ctx = hsplit(b, v); vt[v] = np.bincount(tile, cost(h, raw, mp[ctx], lens, ks), nt)
        row = {'F(8,2)': ft.sum()}
        for v in vt: row['T:' + v] = np.minimum(ft, vt[v]).sum() + nt
        allv = np.min(np.stack(list(vt.values())), 0); row['T:all'] = np.minimum(ft, allv).sum() + np.log2(len(vt) + 1) * nt
        hv = np.min(np.stack([vt[v] for v in vt if v[0] == 'H']), 0); row['T:H*'] = np.minimum(ft, hv).sum() + np.log2(sum(v[0] == 'H' for v in vt) + 1) * nt
        for kk, val in row.items(): A[kk] = A.get(kk, 0) + val + o
        npx += b['npx']
        if len(imgs) <= 12: print('%-28s ' % os.path.basename(img)[:28] + ' '.join('%s %.3f' % (kk, (val + o) / b['npx']) for kk, val in row.items() if kk in ('F(8,2)', 'T:S3', 'T:H2', 'T:H*', 'T:all')), flush=True)
    print(name, 'npx', npx, ' '.join('%s %.4f' % (kk, val / npx) for kk, val in A.items()), flush=True)
