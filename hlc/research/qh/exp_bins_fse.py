# exp_bins_fse.py - "compress the 32 table bins like the L bins of PLN2": the letters of every QH table t go to
# their own stream, per tile (or per image), compressed by real FSE / Huff0 (lib from third_party/fse, header and
# all); compared with the static Huffman tables. Tails and raw planes are the same and left out. off 2.
# env QH_DUMP, FSE_LIB (shared lib of third_party/fse incl. huf); usage: TRAIN TEST [TEST...]
import sys, os, ctypes, numpy as np
from lib import prep
from train_tables import train, ctx_of, letters
lib = ctypes.CDLL(os.environ['FSE_LIB'])
lib.FSE_compress.restype = ctypes.c_size_t; lib.HUF_compress.restype = ctypes.c_size_t
buf = ctypes.create_string_buffer(1 << 22)
def csize(f, b):                                  # bytes of a stream incl. its own header; 0 -> raw, 1 -> rle
    n = len(b)
    if n == 0: return 0
    r = f(buf, len(buf), b, n)
    if r > (1 << 40): r = 0                       # error code -> store raw
    if r == 0: r = n
    return r + (1 if n < 128 else 2) + 1          # + length and a mode byte
T = train(sys.argv[1], int(os.environ.get("QH_K", "32")), 11, 2)
for ts in sys.argv[2:]:
    d = prep(ts, 2); k = ~d['skip']; tb = T['map'][ctx_of(d)][k]; lt = letters(d['h'][k]).astype(np.uint8)
    img = d['img'][k]; tile = d['tile'][k]; npx = d['npx']
    static = T['lens'][tb[:, None], lt].sum()
    res = {}
    for name, grp in (('tile', tile), ('image', img)):
        for cn, f in (('FSE', lib.FSE_compress), ('Huff0', lib.HUF_compress)):
            tot = 0
            for g in np.unique(grp):
                m = grp == g
                for t in np.unique(tb[m]):
                    tot += 8 * csize(f, lt[m][tb[m] == t].ravel().tobytes())
            res[(name, cn)] = tot
    print(f'{ts}: letters bpp static Huffman {static / npx:.4f} | ' + ' | '.join(f'{n}/{c} {v / npx:.4f}' for (n, c), v in res.items()))
