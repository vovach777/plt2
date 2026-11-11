# load.py - loader of the PLR dumps (records: plane u8, L u8, rlgr_bits u32, sum_above u32, z[16] u16)
import numpy as np, glob, os
REC = np.dtype([('pl','u1'),('L','u1'),('bits','<u4'),('sa','<u4'),('z','<u2',(16,))])
D = '/tmp/fable_rlgr/dump'
def load(setname):
    out = dict(pl=[], L=[], bits=[], sa=[], z=[], img=[], tile=[], bi=[], tw=[])
    npx = 0; fbits = 0; lanebits = 0; ntile = 0
    files = sorted(glob.glob(f'{D}/{setname}/*.dmp'))
    for ii, f in enumerate(files):
        hdr = np.fromfile(f, dtype='<i4', count=4); W, H, ch, fast = map(int, hdr)
        r = np.fromfile(f, dtype=REC, offset=16)
        npx += W * H; fbits += 8 * os.path.getsize(f[:-4] + '.plr')
        mark = r['pl'] == 255
        tid = np.cumsum(mark) - mark          # tile index of every record
        lanebits += int(r['bits'][mark].sum())
        nx = (W + 287) // 288
        tws = np.array([min(288, W - (t % nx) * 288) for t in range(int(mark.sum()))])
        keep = ~mark; r2 = r[keep]; t2 = tid[keep]
        # block index within (tile, plane)
        key = t2.astype(np.int64) * 4 + r2['pl']
        # records of one (tile, plane) are contiguous
        chg = np.r_[True, key[1:] != key[:-1]]; start = np.maximum.accumulate(np.where(chg, np.arange(len(key)), 0))
        bi = np.arange(len(key)) - start
        out['pl'].append(r2['pl']); out['L'].append(r2['L']); out['bits'].append(r2['bits']); out['sa'].append(r2['sa'])
        out['z'].append(r2['z']); out['img'].append(np.full(len(r2), ii, np.int32)); out['tile'].append(t2 + ntile); out['bi'].append(bi)
        out['tw'].append(tws[t2]); ntile += int(mark.sum())
    o = {k: np.concatenate(v) for k, v in out.items()}
    o['npx'] = npx; o['fbits'] = fbits; o['lanebits'] = lanebits; o['nimg'] = len(files); o['ntile'] = ntile
    return o
if __name__ == '__main__':
    for s in ['kodak768', 'gray', 'col19']:
        d = load(s); n = d['npx']
        rl = d['bits'].sum(); raw = (16 * d['L'].astype(np.int64)).sum()
        print(s, 'blocks', len(d['L']), 'bpp file %.4f' % (d['fbits'] / n), 'rlgr %.4f raw %.4f other(lane) %.4f hdr %.4f' % (rl / n, raw / n, (d['lanebits'] - rl - raw) / n, (d['fbits'] - d['lanebits']) / n))
        for p in np.unique(d['pl']):
            m = d['pl'] == p
            print('  plane', p, ' '.join('L%d:%.1f%%/%.3f' % (L, 100 * (m & (d['L'] == L)).sum() / m.sum(), d['bits'][m & (d['L'] == L)].sum() / n) for L in range(8)))
        # check L against sa
        sa = d['sa'].astype(np.int64); ok = sa != 0xffffffff
        bwv = np.zeros(len(sa), np.int64); mm = sa >> 4; bwv = np.where(mm > 0, np.floor(np.log2(np.maximum(mm, 1))).astype(np.int64) + 1, 0)
        Lc = np.clip(bwv - 1, 0, 7); print('  L check', (Lc[ok] == d['L'][ok]).all(), 'zmax', d['z'].max())
