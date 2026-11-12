# runs_var.py - zero-block runs beyond sa == 0, costed on dumps:
#  R(T): the run code EG0(n) is sent by every block with sa <= T (format: T = 0)  [1 raw bit when the block is coded]
#  ZR:   sa == 0 as the format; sa >= 1: "zero run" as one more symbol of the first letter (ideal cost from the
#        training share p_t of zero blocks per table: -log2 p_t + EG0(n - 1) for a run, -log2(1 - p_t) otherwise)
import sys, numpy as np, qhsim, lib
T = qhsim.tables()
def prep(s):
    d, c = qhsim.baseline(s, T); lb, tb, pb = qhsim.block_bits(d, T); d['blk'] = lb + tb + pb
    d['t'] = T['cm'][d['ptype'] * T['NQ'] + qhsim.Qsa(d['sa'])]; d['zero'] = (d['z'] == 0).all(1); d['base'] = sum(c.values())
    zero = d['zero']; pk = d['tile'].astype(np.int64) * 4 + d['pl']; n = len(zero); streak = np.zeros(n + 1, np.int64)
    for i in range(n - 1, -1, -1): streak[i] = 0 if not zero[i] else (streak[i + 1] + 1 if i + 1 < n and pk[i + 1] == pk[i] else 1)
    d['streak'] = streak; d['pk'] = pk; return d
eg0 = lambda v: 2 * int(v + 1).bit_length() - 1
def cost(d, Tm, pz=None):
    sa = d['sa']; streak = d['streak']; pk = d['pk']; blk = d['blk']; t = d['t']; bits = 0.0; left = 0; prev = -1
    for i in range(len(sa)):
        if pk[i] != prev: left = 0; prev = pk[i]
        if left: left -= 1; continue
        n = int(streak[i])
        if sa[i] <= Tm:
            bits += eg0(n)
            if n: left = n - 1; continue
        elif pz is not None:
            p = pz[t[i]]
            if n: bits += -np.log2(p) + eg0(n - 1); left = n - 1; continue
            bits += -np.log2(1 - p)
        bits += blk[i]
    return bits
train, tests = sys.argv[1], sys.argv[2:]
dt = prep(train); k = ~dt['skip'] & (dt['sa'] >= 1)
pz = np.array([(((dt['t'] == t) & k & dt['zero']).sum() + 0.5) / (((dt['t'] == t) & k).sum() + 1.0) for t in range(T['K'])]); pz = np.clip(pz, 1e-4, 0.5)
print('%-14s %8s | %s | %8s' % ('set', 'format', '  '.join('R(%d)' % x + ' ' * 9 for x in (2, 4, 8, 16, 31)), 'ZR ideal'))
for s in tests:
    d = prep(s); n = d['npx']; b0 = cost(d, 0); assert abs(b0 - d['base']) < 1e-6
    r = [cost(d, x) for x in (2, 4, 8, 16, 31)]; z = cost(d, 0, pz)
    print('%-14s %8.4f | %s | %.4f %+.2f%%' % (s, b0 / n, '  '.join('%.4f %+5.2f%%' % (x / n, 100 * (x - b0) / b0) for x in r), z / n, 100 * (z - b0) / b0))
