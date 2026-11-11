# exp_own_stats.py - static QH tables vs "own statistics" of the image / the tile for the same 32 table contexts
# (letters = 4 x min(h,3); tails and raw planes are the same in all variants and left out). off 2, zero-run blocks
# skipped as in the codec. Costs: ideal -log2 p (what tANS / arithmetic coding approaches), and Huffman lengths.
#  static   : the format's tables (trained on TRAIN), held-out
#  per image: tables built from the image itself (two passes), + a header estimate for the 32 tables
#  per tile : the same per 288x288 tile
# Header estimate: for every used table, a list of its letter code lengths, ~ (#letters used) * 4 bits + 8 bits.
# env QH_DUMP; usage: TRAIN TEST [TEST...]
import sys, numpy as np
from lib import prep, huff_len
from train_tables import train, ctx_of, letters
def ent_rows(c):
    n = c.sum(1, keepdims=True); p = np.where(c > 0, c / np.maximum(n, 1), 1.0); return float(-(c * np.log2(p)).sum())
T = train(sys.argv[1], 32, 11, 2)
for ts in sys.argv[2:]:
    d = prep(ts, 2); k = ~d['skip']; tb = T['map'][ctx_of(d)][k]; lt = letters(d['h'][k]); img = d['img'][k]; tile = d['tile'][k]
    npx = d['npx']; K = T['K']
    static_h = T['lens'][tb[:, None], lt].sum()
    # ideal of the static tables: probabilities = training counts of the clusters (smoothed) -> recompute from lens? use Huffman lens
    def own(group):
        tot_ideal = tot_huff = hdr = 0.0
        for g in np.unique(group):
            m = group == g; c = np.zeros((K, 256))
            np.add.at(c, (np.repeat(tb[m], 4), lt[m].ravel()), 1)
            tot_ideal += ent_rows(c)
            used = c.sum(1) > 0
            if used.any():
                hl = huff_len(c[used], 11); tot_huff += float((hl * c[used]).sum())
                hdr += float(((c[used] > 0).sum(1) * 4 + 8).sum())
        return tot_ideal, tot_huff, hdr
    ii, ih, ihd = own(img); ti, th, thd = own(tile)
    print(f'{ts}: letters bpp  static Huffman {static_h / npx:.4f} | per image: ideal {ii / npx:.4f} Huffman {ih / npx:.4f} '
          f'+ header {ihd / npx:.4f} = {(ih + ihd) / npx:.4f} | per tile: ideal {ti / npx:.4f} Huffman {th / npx:.4f} + header {thd / npx:.4f} = {(th + thd) / npx:.4f}')
