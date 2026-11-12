# lspike2.py - (second round: more L shifts, estimators from above / left, 2-D context)
# lspike.py - screen content in PLR: escape rule of the tails, screen table set, L shift in screen mode, per-tile switch.
# Cost model = qhsim (letters with real Huffman lengths <= 11, tails, 16 L planes) + runs + the zU split (as exp_pred).
#   train: lspike.py train OUT.pkl DUMPER img,img,...          (screen tables for every L shift d, Rice k per rule)
#   test : lspike.py test  TABLES.pkl DUMPER OUT.json img...   (all variants per image, per-tile switch)
# Tail rule (T, g): q = x >> k; q < T: Rice (q + 1 + k bits); else T ones + Exp-Golomb order g of x - (T << k).
#   rule 'fmt' = the format: T = 24, then 10 raw bits.
# Screen mode S(d): L = max(L_causal - d, 0) (d = 7: L = 0 always), tables retrained for that L, context (Y/S, Q(sa)).
import sys, os, json, pickle, subprocess, tempfile, numpy as np
sys.path.insert(0, '/home/claude/plt2_repo/hlc/research/qh'); sys.path.insert(0, '/home/claude/plt2_repo/hlc/research/qh_enc')
tmp = tempfile.mkdtemp(); os.environ['QH_DUMP'] = tmp
import load as L_; L_.D = tmp
import lib, qhsim
from train_tables import cluster
from lib import huff_len, bwv
T = qhsim.tables(); NQ = T['NQ']; HM = 1024
RULES = ['fmt', (4, 1), (6, 1), (8, 2)]
def bwc(x): return np.minimum(bwv(x), 11)
# screen-mode variants: name -> (estimator of the causal sum, L shift d, context kind); L = clamp(bw(est >> 4) - 2 - d, 0, 7)
EST = dict(A=lambda a, l: a, MIN=lambda a, l: np.minimum(a, l), MAX=lambda a, l: np.maximum(a, l), AVG=lambda a, l: (a + l + 1) >> 1)
VAR = {}
for dd in (0, 1, 2, 3, 4, 7): VAR['A%d' % dd] = ('A', dd, 'q')
for e in ('MIN', 'MAX', 'AVG'):
    for dd in (1, 2, 3): VAR['%s%d' % (e, dd)] = (e, dd, 'q')
for dd in (1, 2, 3): VAR['A%d+2D' % dd] = ('A', dd, '2d')
NCTX = dict(q=2 * NQ, **{'2d': 2 * 144})
def lctx(b, name):
    e, dd, ck = VAR[name]; est = EST[e](b['sa'], b['sl']); L = np.clip(bwv(est >> 4) - 2 - dd, 0, 7).astype(np.int32); L[b['first']] = max(2 - dd, 0)
    ctx = b['pt'] * NQ + qhsim.Qsa(est) if ck == 'q' else (b['pt'] * 12 + bwc(b['sa'])) * 12 + bwc(b['sl'])
    return L, ctx.astype(np.int64)
def tailtab():                                   # TB[rule][k][h]: bits of the tail of h (0 for h < 3)
    TB = np.zeros((len(RULES), 5, HM), np.int64); h = np.arange(HM); x = np.maximum(h - 3, 0)
    for r, rule in enumerate(RULES):
        for k in range(5):
            q = x >> k
            if rule == 'fmt': b = np.where(q >= 24, 34, q + 1 + k)
            else:
                Tn, g = rule; v = np.maximum(x - (Tn << k), 0); b = np.where(q >= Tn, Tn + 2 * bwv((v >> g) + 1) - 1 + g, q + 1 + k)
            TB[r, k] = np.where(h >= 3, b, 0)
    return TB
TB = tailtab()
def blocks(dumper, img):
    sd = os.path.join(tmp, 'one'); os.makedirs(sd, exist_ok=True)
    for x in os.listdir(sd): os.remove(os.path.join(sd, x))
    subprocess.run([dumper, 'c', img, os.path.join(sd, 'a.plr')], env=dict(os.environ, PLR_DUMP_FILE=os.path.join(sd, 'a.dmp')), check=True, stdout=subprocess.DEVNULL)
    lib._cache.clear(); d = lib.prep('one', 2); k = ~d['skip']
    S = d['z'].astype(np.float32); kk = np.floor(np.log2(S + 1)); u = 2 ** (kk + 1) - (S + 1)
    split = float((np.where(S > 0, kk + (S + 1 - u) / (S + 1), 0).sum(1) * (d['pl'] == 1)).sum())
    b = dict(z=np.minimum(d['z'][k], HM - 1).astype(np.int32), Lc=d['Lo'][k].astype(np.int32), ctx=(d['ptype'][k] * NQ + qhsim.Qsa(d['sa'][k])).astype(np.int32),
             tile=d['tile'][k].astype(np.int64), sa=d['sa'][k].astype(np.int64), sl=d['sl'][k].astype(np.int64), pt=d['ptype'][k].astype(np.int64), first=d['first'][k], nt=int(d['ntile']), other=float(d['runbits'].sum()) + split, npx=int(d['npx']))
    lib._cache.clear(); return b
def cost(b, L, t, lens, kvec):                   # per block: letters + planes, and tails per rule
    h = b['z'] >> L[:, None]; base = lens[t[:, None], qhsim.letters(h)].sum(1) + 16 * L
    tails = np.stack([TB[r][kvec[r][t][:, None], h].sum(1) for r in range(len(RULES))])
    return base, tails
if sys.argv[1] == 'train':
    out, dumper, imgs = sys.argv[2], sys.argv[3], sys.argv[4].split(',')
    C = {v: np.zeros((NCTX[VAR[v][2]], 256)) for v in VAR}; X = {v: np.zeros((NCTX[VAR[v][2]], HM)) for v in VAR}
    for i, img in enumerate(imgs):
        b = blocks(dumper, img)
        for v in VAR:
            L, ctx = lctx(b, v); h = b['z'] >> L[:, None]
            np.add.at(C[v], (np.repeat(ctx, 4), qhsim.letters(h).ravel()), 1); np.add.at(X[v], (np.repeat(ctx, 16), h.ravel()), 1)
        print('train', i + 1, len(imgs), os.path.basename(img), flush=True)
    tabs = {}
    for v in VAR:
        mp, tb = cluster(C[v], 32)
        for c in np.where(mp < 0)[0]:
            cand = [x for x in range(len(mp)) if mp[x] >= 0]; mp[c] = mp[min(cand, key=lambda x: abs(x - c))]
        lens = huff_len(tb, 11).astype(np.int64); ks = np.zeros((len(RULES), len(tb)), np.int64)
        for t in range(len(tb)):
            xt = X[v][mp == t].sum(0)
            for r in range(len(RULES)): ks[r, t] = int(np.argmin([(xt * TB[r, kk]).sum() for kk in range(5)]))
        tabs[v] = (mp, lens, ks); print('tables', v, flush=True)
    pickle.dump(tabs, open(out, 'wb'))
else:
    tabs = pickle.load(open(sys.argv[2], 'rb')); dumper, outj, imgs = sys.argv[3], sys.argv[4], sys.argv[5:]
    kfmt = [T['rk']] * len(RULES); res = []
    for img in imgs:
        b = blocks(dumper, img); nt = b['nt']; r = dict(img=os.path.basename(img), npx=b['npx'], other=b['other'], nt=nt)
        fb, ft = cost(b, b['Lc'], T['cm'][b['ctx']], T['lens'], kfmt)
        r['F'] = (fb.sum() + ft.sum(1)).tolist()
        ftile = np.stack([np.bincount(b['tile'], fb + ft[i], nt) for i in range(len(RULES))])
        for v in tabs:
            mp, lens, ks = tabs[v]; L, ctx = lctx(b, v); sb, st = cost(b, L, mp[ctx], lens, ks)
            r['S:' + v] = (sb.sum() + st.sum(1)).tolist()
            stile = np.stack([np.bincount(b['tile'], sb + st[i], nt) for i in range(len(RULES))])
            r['T:' + v] = (np.minimum(ftile, stile).sum(1) + nt).tolist()
        res.append(r); json.dump(res, open(outj, 'w'))
        n = b['npx']; o = b['other']
        print('%-30s F %.4f | S:A2 %.4f T:A2 %.4f (rule 8,2)' % (r['img'][:30], (r['F'][0] + o) / n, (r['S:A2'][3] + o) / n, (r['T:A2'][3] + o) / n), flush=True)
