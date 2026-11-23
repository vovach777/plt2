# train_qhs_probe1.py - retrain PLR screen tables (qhs, variant H0) on the six R8F5-screen groups.
# Plane selection: PLS-probe-1 -s=zthr (zero count of Y+zU+zV >= 65% of 3*tile pixels), read from
# the instrumented probe encoder sidecar (bit-exact with probe1_mode()). Tile 0 of every image is
# dropped: the PLR_NO_QH research dumper writes an all-zero S plane there (verified: the only
# systematic dump/probe disagreement; all other tiles match exactly).
#
# usage: train_qhs_probe1.py OUT.pkl STRIDE GROUP:IMGDIR,...
#   STRIDE=10 -> every 10th train image of each group (uniform per group)
import sys, os, subprocess, tempfile, numpy as np, pickle, struct, time, collections
ROOT = '/tmp/r8f5/plt2probe/hlc/research'
sys.path.insert(0, ROOT + '/qh'); sys.path.insert(0, ROOT + '/qh_enc')
import load as L_, lib, qhsim
from train_tables import cluster
from lib import huff_len, bwv

DUMP = '/tmp/r8f5/plr_dump'
PROBE = '/tmp/r8f5/pls_probe'
HM, KMAX = 1024, 6
NQ = qhsim.tables()['NQ']
TBK = np.stack([np.stack([np.where(np.arange(HM) >= 3,
        np.where((np.maximum(np.arange(HM) - 3, 0) >> k) >= 8,
                 8 + 2 * bwv(((np.maximum(np.arange(HM) - 3, 0) - (8 << k)) >> 2) + 1) + 1,
                 (np.maximum(np.arange(HM) - 3, 0) >> k) + 1 + k), 0)
        for k in range(KMAX + 1)]) for _ in range(1)])[0]   # (7,1024)

def tailbits_prec(k):
    h = np.arange(HM); x = np.maximum(h - 3, 0); q = x >> k; v = np.maximum(x - (8 << k), 0)
    return np.where(h >= 3, np.where(q >= 8, 8 + 2 * bwv((v >> 2) + 1) + 1, q + 1 + k), 0)
TBK = np.stack([tailbits_prec(k) for k in range(KMAX + 1)])

def ppms(d):
    return sorted(f for f in (os.path.join(d, x) for x in os.listdir(d)) if f.endswith('.ppm'))

def hdrpx(f):
    with open(f, 'rb') as fh:
        assert fh.readline().strip() == b'P6'
        while True:
            l = fh.readline().strip()
            if not l.startswith(b'#'): break
        w, h = map(int, l.split())
    return w * h

def sidecar(img, tmp):
    log = os.path.join(tmp, 'a.probe')
    subprocess.run([PROBE, 'c', img, os.path.join(tmp, 'a.pls'), '-s=zthr'],
                   env=dict(os.environ, PLS_PROBE_LOG=log), check=True, stdout=subprocess.DEVNULL)
    raw = open(log, 'rb').read(); nt = struct.unpack('<I', raw[:4])[0]
    modes = np.frombuffer(raw, dtype=np.uint8, count=nt * 2, offset=4).reshape(nt, 2)
    return modes

def image_blocks(img, tmp):
    sd = os.path.join(tmp, 'one'); os.makedirs(sd, exist_ok=True)
    for x in os.listdir(sd): os.remove(os.path.join(sd, x))
    subprocess.run([DUMP, 'c', img, os.path.join(sd, 'a.plr')],
                   env=dict(os.environ, PLR_DUMP_FILE=os.path.join(sd, 'a.dmp')),
                   check=True, stdout=subprocess.DEVNULL)
    L_.D = tmp; lib._cache.clear()
    d = lib.prep('one', 2)
    return d

def hsplit_H0(b):
    est = np.maximum(b['sa'], b['sl'])
    ctx = b['pt'] * NQ + qhsim.Qsa(est)
    z = b['z']
    Ls = np.clip(bwv(est >> 4) - 2, 0, 7)
    Ls[b['first']] = 2
    nz = z > 0
    h = np.where(nz, ((z - 1) >> Ls[:, None]) + 1, 0)
    raw = Ls * nz.sum(1)
    return h, raw, ctx

def main():
    out, stride_s, groups_s = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    groups = [tuple(x.split(':')) for x in groups_s.split(',')]
    stride = max(1, stride_s)
    # splits (identical to r8f5_screens.cpp): web group = 1 holdout, rest train
    train_all, used, hold = {}, {}, {}
    for gi, (name, d) in enumerate(groups):
        v = ppms(os.path.join(d, 'all'))
        tr, ho = [], []
        for i, f in enumerate(v):
            s = ('hold' if i == 0 else 'train') if name == 'web' else \
                ('test' if i % 5 == 4 else 'hold' if i % 10 == 3 else 'train')
            if s == 'train': tr.append(f)
            elif s == 'hold': ho.append(f)
        train_all[name], hold[name] = tr, ho
        used[name] = tr[::stride]
        print('GROUP %-9s train=%d used=%d hold=%d' % (name, len(tr), len(used[name]), len(ho)), flush=True)
    # equal group pixel weight, corrected for subsampling
    px = {g: sum(hdrpx(f) for f in train_all[g]) for g, _ in groups}
    meanpx = sum(px.values()) / len(px)
    wimg = {}
    for g, _ in groups:
        w = (meanpx / px[g]) * (len(train_all[g]) / max(1, len(used[g])))
        for f in used[g]: wimg[f] = w
        print('WEIGHT %-9s px=%.0f w_group=%.6f w_img=%.6f' %
              (g, px[g], meanpx / px[g], w), flush=True)
    C = np.zeros((2 * NQ, 256), np.float64)
    X = np.zeros((2 * NQ, HM), np.float64)
    stats = collections.Counter(); t0 = time.time(); nimg = 0
    tmp = tempfile.mkdtemp()
    for g, _ in groups:
        for f in used[g]:
            modes = sidecar(f, tmp)
            d = image_blocks(f, tmp)
            keep = ~d['skip']
            tile_ok = np.zeros(len(keep), bool)
            if len(modes):
                tile_ok = modes[d['tile'], 0] == 2          # -s=zthr: one choice per tile, both planes
                tile_ok[d['tile'] == 0] = False             # dump-build first-tile S anomaly
            k = keep & tile_ok
            z = d['z'][k].astype(np.int64)
            if len(z) == 0:
                nimg += 1; continue
            b = dict(z=z, Lc=d['Lo'][k].astype(np.int64), sa=d['sa'][k].astype(np.int64),
                     sl=d['sl'][k].astype(np.int64), pt=d['ptype'][k].astype(np.int64),
                     first=d['first'][k], tile=d['tile'][k].astype(np.int64))
            h, raw, ctx = hsplit_H0(b)
            w = wimg[f]
            np.add.at(C, (np.repeat(ctx, 4), qhsim.letters(h).ravel()), w)
            np.add.at(X, (np.repeat(ctx, 16), np.minimum(h, HM - 1).ravel()), w)
            stats[g + '_blocks'] += len(z); stats[g + '_imgs'] += 1
            stats['tiles'] += len(modes); stats['tiles_pls'] += int((modes[:, 0] == 2).sum())
            stats['blocks'] += len(z); stats['blocks_pls'] += len(z)
            nimg += 1
            if nimg % 25 == 0:
                print('PROGRESS %d img %.1fs blocks=%d' % (nimg, time.time() - t0, stats['blocks']), flush=True)
    mp, tb = cluster(C, 32)
    for c in np.where(mp < 0)[0]:
        cand = [x for x in range(len(mp)) if mp[x] >= 0]
        mp[c] = mp[min(cand, key=lambda x: abs(x - c))]
    lens = huff_len(tb, 11).astype(np.int64)
    ks = np.array([int(np.argmin([(X[mp == t].sum(0) * TBK[kk]).sum() for kk in range(KMAX + 1)]))
                   for t in range(len(tb))])
    pickle.dump({'H0': (mp, lens, ks)}, open(out, 'wb'))
    print('DONE %s imgs=%d time=%.1fs blocks=%d tiles_pls=%d/%d (%.2f%%)' %
          (out, nimg, time.time() - t0, stats['blocks'], stats['tiles_pls'], stats['tiles'],
           100.0 * stats['tiles_pls'] / max(1, stats['tiles'])), flush=True)
    for g, _ in groups:
        print('STAT %-9s imgs=%d blocks=%d' % (g, stats[g + '_imgs'], stats[g + '_blocks']), flush=True)
    print('TABLES lens_max=%d ks=%s ctxmap_hash=%d' % (lens.max(), ks.tolist(), int(mp.sum())), flush=True)

if __name__ == '__main__':
    main()
