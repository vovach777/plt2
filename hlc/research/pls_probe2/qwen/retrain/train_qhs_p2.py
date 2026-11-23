#!/usr/bin/env python3
"""Retrain qhs (H0) for PLS-probe-2 with per-plane selection at chosen thresholds.
Same pipeline as WorkBuddy's train_qhs_probe2_full.py, but the per-tile modes come
from the precomputed DIAG counter cache (n, zY, S) instead of the PLS_PROBE_LOG sidecar.

usage: train_qhs_p2.py OUT.pkl TY TS GROUP:all_dir,...
"""
import sys, os, subprocess, tempfile, numpy as np, pickle, struct, time, collections
THROTTLE_S = float(os.environ.get('THROTTLE_S', '0.05'))
CACHE = '/tmp/r8f5/diag_cache'
DUMP = '/tmp/r8f5/plr_dump'
ROOT = '/tmp/r8f5/plt2probe/hlc/research'
sys.path.insert(0, ROOT + '/qh'); sys.path.insert(0, ROOT + '/qh_enc')
import load as L_, lib, qhsim
from train_tables import cluster
from lib import huff_len, bwv

HM, KMAX = 1024, 6
NQ = qhsim.tables()['NQ']

def tailbits_prec(k):
    h = np.arange(HM); x = np.maximum(h - 3, 0); q = x >> k; v = np.maximum(x - (8 << k), 0)
    return np.where(h >= 3, np.where(q >= 8, 8 + 2 * bwv((v >> 2) + 1) + 1, q + 1 + k), 0)
TBK = np.stack([tailbits_prec(k) for k in range(KMAX + 1)])

def ppms(d): return sorted(f for f in (os.path.join(d, x) for x in os.listdir(d)) if f.endswith('.ppm'))
def hdrpx(f):
    with open(f, 'rb') as fh:
        assert fh.readline().strip() == b'P6'
        while True:
            l = fh.readline().strip()
            if not l.startswith(b'#'): break
        w, h = map(int, l.split())
    return w * h

def modes_from_cache(group, f, ty, ts):
    stem = os.path.basename(f)[:-4]
    a = np.load(os.path.join(CACHE, group, stem + '.npy'))   # (ntiles, 9): n zY zU zV zS b00 b01 b10 b11
    if len(a) == 0: return np.zeros((0, 2), np.int64)
    n = a[:, 0]; zY = a[:, 1]; zS = a[:, 4]
    y = (zY * 100 >= ty * n).astype(np.int64)
    s = (zS * 100 >= ts * n).astype(np.int64)
    return np.stack([y, s], 1)

def image_blocks(img, tmp):
    sd = os.path.join(tmp, 'one'); os.makedirs(sd, exist_ok=True)
    for x in os.listdir(sd): os.remove(os.path.join(sd, x))
    subprocess.run([DUMP, 'c', img, os.path.join(sd, 'a.plr')],
                   env=dict(os.environ, PLR_DUMP_FILE=os.path.join(sd, 'a.dmp')),
                   check=True, stdout=subprocess.DEVNULL)
    L_.D = tmp; lib._cache.clear()
    return lib.prep('one', 2)

def hsplit_H0(b):
    est = np.maximum(b['sa'], b['sl'])
    ctx = b['pt'] * NQ + qhsim.Qsa(est)
    z = b['z']
    Ls = np.clip(bwv(est >> 4) - 2, 0, 7)
    Ls[b['first']] = 2
    nz = z > 0
    hh = np.where(nz, ((z - 1) >> Ls[:, None]) + 1, 0)
    raw = Ls * nz.sum(1)
    return hh, raw, ctx

def main():
    out, ty, ts, groups_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    groups = [tuple(x.split(':')) for x in groups_s.split(',')]
    train_all, used, hold = {}, {}, {}
    for name, d in groups:
        v = ppms(os.path.join(d, 'all'))
        tr, ho = [], []
        for i, f in enumerate(v):
            s = ('hold' if i == 0 else 'train') if name == 'web' else \
                ('test' if i % 5 == 4 else 'hold' if i % 10 == 3 else 'train')
            if s == 'train': tr.append(f)
            elif s == 'hold': ho.append(f)
        train_all[name], hold[name] = tr, ho
        used[name] = tr
        print('GROUP %-9s train=%d hold=%d' % (name, len(tr), len(ho)), flush=True)
    px = {g: sum(hdrpx(f) for f in train_all[g]) for g, _ in groups}
    meanpx = sum(px.values()) / len(px)
    wimg = {}
    for g, _ in groups:
        w = meanpx / px[g]
        for f in used[g]: wimg[f] = w
        print('WEIGHT %-9s px=%.0f w_group=%.6f' % (g, px[g], w), flush=True)
    C = np.zeros((2 * NQ, 256), np.float64)
    X = np.zeros((2 * NQ, HM), np.float64)
    stats = collections.Counter(); t0 = time.time(); nimg = 0
    tmp = tempfile.mkdtemp()
    for g, _ in groups:
        for f in used[g]:
            modes = modes_from_cache(g, f, ty, ts)
            d = image_blocks(f, tmp)
            keep = ~d['skip']
            tile_ok = np.zeros(len(keep), bool)
            if len(modes):
                my = modes[d['tile'], 0] == 1
                ms = modes[d['tile'], 1] == 1
                tile_ok = np.where(d['ptype'] == 0, my, ms)
                tile_ok[d['tile'] == 0] = False
            k = keep & tile_ok
            z = d['z'][k].astype(np.int64)
            if len(z) == 0:
                nimg += 1; continue
            b = dict(z=z, Lc=d['Lo'][k].astype(np.int64), sa=d['sa'][k].astype(np.int64),
                     sl=d['sl'][k].astype(np.int64), pt=d['ptype'][k].astype(np.int64),
                     first=d['first'][k], tile=d['tile'][k].astype(np.int64))
            hh, raw, ctx = hsplit_H0(b)
            w = wimg[f]
            np.add.at(C, (np.repeat(ctx, 4), qhsim.letters(hh).ravel()), w)
            np.add.at(X, (np.repeat(ctx, 16), np.minimum(hh, HM - 1).ravel()), w)
            stats[g + '_blocks'] += len(z); stats[g + '_imgs'] += 1
            stats['tiles'] += len(modes); stats['tiles_pls_y'] += int((modes[:, 0] == 1).sum()); stats['tiles_pls_s'] += int((modes[:, 1] == 1).sum())
            stats['blocks'] += len(z)
            nimg += 1
            if THROTTLE_S > 0: time.sleep(THROTTLE_S)
            if nimg % 100 == 0:
                print('PROGRESS %d img %.1fs blocks=%d' % (nimg, time.time() - t0, stats['blocks']), flush=True)
    mp, tb = cluster(C, 32)
    for c in np.where(mp < 0)[0]:
        cand = [x for x in range(len(mp)) if mp[x] >= 0]
        mp[c] = mp[min(cand, key=lambda x: abs(x - c))]
    lens = huff_len(tb, 11).astype(np.int64)
    ks = np.array([int(np.argmin([(X[mp == t].sum(0) * TBK[kk]).sum() for kk in range(KMAX + 1)]))
                   for t in range(len(tb))])
    pickle.dump({'H0': (mp, lens, ks)}, open(out, 'wb'))
    print('DONE %s imgs=%d time=%.1fs blocks=%d planes_y_pls=%d planes_s_pls=%d tiles=%d' %
          (out, nimg, time.time() - t0, stats['blocks'], stats['tiles_pls_y'], stats['tiles_pls_s'], stats['tiles']), flush=True)
    for g, _ in groups:
        print('STAT %-9s imgs=%d blocks=%d' % (g, stats[g + '_imgs'], stats[g + '_blocks']), flush=True)
    print('TABLES lens_max=%d ks=%s ctxmap_hash=%d' % (lens.max(), ks.tolist(), int(mp.sum())), flush=True)

if __name__ == '__main__':
    main()