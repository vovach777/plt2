# exp_zr.py - "zero run as the 257th symbol of the first letter", costed with real Huffman code lengths (<= 11 bits),
# streaming over images (dump one image, cost it, delete), so the full held-out CLIC part can be used.
# Variants (everything else = the format: tails, planes, zU splits are not in the dump cost and are equal anyway):
#  base : the format (runs only at sa == 0, EG0(n)); the 4 letters of a coded block from table t
#  FL   : as base, but the FIRST letter from its own retrained table per t (isolates the gain of a positional table)
#  ZR   : FL + the first-letter table gets a 257th symbol "zero run": at a block with sa >= 1 that starts a streak of
#         n >= 1 all-zero blocks: ZR + EG0(n - 1), the n blocks are skipped (no letters, no planes);
#         sa == 0 stays EG0(n) as now. So ZR lifts the sa == 0 limit and keeps EG for sa == 0.
# Training: the same images as the format tables (CLIC train list + Kodak 24); tables t / Rice k / letters 2..4 = format.
# usage: exp_zr.py DUMPER TRAIN_LIST(file of image paths) -- TESTSET_NAME:img... [TESTSET_NAME:img...]
import sys, os, subprocess, tempfile, numpy as np
sys.path.insert(0, '/home/claude/plt2_repo/hlc/research/qh'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp(); os.environ['QH_DUMP'] = tmp
import load as L_; L_.D = tmp
import lib, qhsim
from lib import huff_len_row
T = qhsim.tables(); K = T['K']; NQ = T['NQ']
eg0 = lambda v: 2 * np.floor(np.log2(np.asarray(v, np.float64) + 1)).astype(np.int64) + 1
dumper = sys.argv[1]; PLR = os.environ.get('PLR_BIN')     # optional: the real codec, to put the deltas against the file size
def one(img):
    sd = os.path.join(tmp, 'one'); os.makedirs(sd, exist_ok=True)
    for x in os.listdir(sd): os.remove(os.path.join(sd, x))
    subprocess.run([dumper, 'c', img, os.path.join(sd, 'a.plr')], env=dict(os.environ, PLR_DUMP_FILE=os.path.join(sd, 'a.dmp')), check=True, stdout=subprocess.DEVNULL)
    lib._cache.clear(); d = lib.prep('one', 2)
    t = T['cm'][d['ptype'] * NQ + qhsim.Qsa(d['sa'])]; lt = qhsim.letters(d['h']); lb, tb, pb = qhsim.block_bits(d, T)
    zero = (d['z'] == 0).all(1); pk = d['tile'].astype(np.int64) * 4 + d['pl']; n = len(zero)
    same = np.r_[False, pk[1:] == pk[:-1]]; prevz = np.r_[False, zero[:-1]] & same
    start = zero & ~prevz; rid = np.cumsum(start) - 1; rlen = np.bincount(rid[zero], minlength=int(start.sum())) if start.any() else np.zeros(0, int)
    slen = np.zeros(n, np.int64); slen[start] = rlen[rid[start]]
    return dict(d=d, t=t, lt=lt, lb=lb, tb=tb, pb=pb, zero=zero, start=start, slen=slen, sa=d['sa'], k=~d['skip'], npx=d['npx'])
# ---- training
F1 = np.zeros((K, 256)); Z1 = np.zeros((K, 257))
args = sys.argv[2:]; sep = args.index('--'); train = [l.strip() for l in open(args[0]) if l.strip()]
for i, img in enumerate(train):
    o = one(img); k = o['k']; t = o['t']; l0 = o['lt'][:, 0]
    np.add.at(F1, (t[k], l0[k]), 1)
    nz = ~o['zero']; np.add.at(Z1, (t[nz], l0[nz]), 1)
    zr = o['start'] & (o['sa'] >= 1); np.add.at(Z1, (t[zr], 256), 1)
F1len = np.stack([huff_len_row(r, 11) for r in F1]); Z1len = np.stack([huff_len_row(r, 11) for r in Z1])
print(f'trained on {len(train)} images; ZR share of first letters {Z1[:, 256].sum() / Z1.sum():.4f}', flush=True)
# ---- tests
for spec in args[sep + 1:]:
    name, imgs = spec.split(':', 1); imgs = imgs.split(',')
    S = dict(file=0, npx=0, base=0.0, FL=0.0, ZR=0.0, tax=0.0, ntax=0, runs_b=0.0, runs_z=0.0)
    for img in imgs:
        o = one(img); k = o['k']; t = o['t']; lt = o['lt']; sa = o['sa']; zero = o['zero']; start = o['start']; slen = o['slen']
        rest = (T['lens'][t[:, None], lt[:, 1:]].sum(1) + o['tb'] + o['pb'])        # letters 2..4 + tails + planes
        runs_b = o['d']['runbits'].sum()
        base = runs_b + o['lb'][k].sum() + o['tb'][k].sum() + o['pb'][k].sum()
        fl = runs_b + (F1len[t, lt[:, 0]] + rest)[k].sum()
        nz = ~zero
        mz = start & (sa >= 1)
        runs_z = eg0(slen[start & (sa == 0)]).sum() + eg0(np.zeros(int((nz & (sa == 0)).sum()))).sum() + \
                 (Z1len[t[mz], 256] + eg0(slen[mz] - 1)).sum()
        zr = runs_z + (Z1len[t, lt[:, 0]] + rest)[nz].sum()
        m = nz & (sa >= 1); S['tax'] += (Z1len[t, lt[:, 0]] - F1len[t, lt[:, 0]])[m].sum(); S['ntax'] += int(m.sum())
        if PLR:
            subprocess.run([PLR, 'c', img, os.path.join(tmp, 'f.plr')], check=True, stdout=subprocess.DEVNULL); S['file'] += 8 * os.path.getsize(os.path.join(tmp, 'f.plr'))
        S['npx'] += o['npx']; S['base'] += base; S['FL'] += fl; S['ZR'] += zr; S['runs_b'] += runs_b; S['runs_z'] += runs_z
    n = S['npx']; b = S['base']
    print(f"{name:18s} base {b / n:.4f}  FL {S['FL'] / n:.4f} ({100 * (S['FL'] - b) / b:+.2f}%)  ZR {S['ZR'] / n:.4f} ({100 * (S['ZR'] - b) / b:+.2f}%)"
          f"  | ZR vs FL {100 * (S['ZR'] - S['FL']) / b:+.2f}%  tax on non-zero sa>=1 blocks {S['tax'] / max(S['ntax'], 1):+.3f} bit/block"
          f" = {S['tax'] / n:+.4f} bpp  | run bits base {S['runs_b'] / n:.4f} ZR {S['runs_z'] / n:.4f}" +
          (f"  || file {S['file'] / n:.4f} bpp, ZR vs file {100 * (S['ZR'] - b) / S['file']:+.3f}%, FL vs file {100 * (S['FL'] - b) / S['file']:+.3f}%" if S['file'] else ''), flush=True)
