# qhsim.py - exact cost model of the PLR (QH) format on block dumps (palic2r -DPLR_DUMP -DPLR_NO_QH), and the same
# blocks under candidate format changes. Uses load.py / lib.py of hlc/research/qh (zero-block runs as the codec).
import sys, os, struct, numpy as np
QH = '/home/claude/plt2_repo/hlc/research/qh'; sys.path.insert(0, QH)
os.environ.setdefault('QH_DUMP', '/tmp/fable_enc/dump')
import lib
from lib import bwv
def tables(fn=QH + '/tables/qh_clic117_kodak_off2_K32_L11.bin'):
    b = open(fn, 'rb').read(); K, ML, NQ, QESC = struct.unpack('<4i', b[:16]); p = 16
    cm = np.frombuffer(b, np.uint8, 2 * NQ, p).astype(np.int64); p += 2 * NQ
    lens = np.frombuffer(b, np.uint8, K * 256, p).reshape(K, 256).astype(np.int64); p += K * 256
    rk = np.frombuffer(b, np.uint8, K, p).astype(np.int64)
    return dict(K=K, ML=ML, NQ=NQ, QESC=QESC, cm=cm, lens=lens, rk=rk)
def Qsa(sa):
    sa = np.asarray(sa, np.int64); e = bwv(sa); q = 32 + (e - 6) * 4 + ((sa >> np.maximum(e - 3, 0)) & 3)
    return np.where(sa < 32, sa, np.minimum(q, 67))
def letters(h):
    c = np.minimum(h, 3).reshape(len(h), 4, 4); return c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)
def block_bits(d, T):
    """per block: letter bits, tail bits, plane bits (for every block, also the skipped ones)"""
    t = T['cm'][d['ptype'] * T['NQ'] + Qsa(d['sa'])]; h = d['h']; lt = letters(h)
    lb = T['lens'][t[:, None], lt].sum(1)
    k = T['rk'][t][:, None]; x = np.maximum(h - 3, 0); q = x >> k
    tb = np.where(h >= 3, np.where(q >= T['QESC'], T['QESC'] + 10, q + 1 + k), 0).sum(1)
    return lb, tb, 16 * d['Lo']
def baseline(setname, T):
    d = lib.prep(setname, 2); lb, tb, pb = block_bits(d, T); k = ~d['skip']
    return d, dict(run=d['runbits'].sum(), let=lb[k].sum(), tail=tb[k].sum(), planes=pb[k].sum())
if __name__ == '__main__':
    T = tables()
    for s in sys.argv[1:]:
        d, c = baseline(s, T); n = d['npx']; tot = sum(c.values())
        print(s, 'bpp %.4f' % (tot / n), ' '.join('%s %.4f' % (a, b / n) for a, b in c.items()), ' blocks', len(d['sa']), 'coded %.1f%%' % (100 * (~d['skip']).mean()))
