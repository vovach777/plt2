# PLN1 — gray codec with the "noise of the block" context

`planes_codec.cpp`, default `-m 2 -off 2`. Authors: Vladimir (idea of bit planes per block and the codec line),
Argon (mode 2: L from the symbols above, no L stream), Claude (codec, vectorisation, wave decoder, measurements).

## Pipeline of one tile

1. Tiles 288 × 288, independent (as HALIC); the last column / row of tiles takes the remainder (500 → 288 + 212).
2. Inside a tile: PALIC-1 MED (row 0: left, 0 at x = 0; column 0: up), 8-bit wrap, zigzag `z` (0..255).
   The symbols are kept in the raster order of the tile: `p = y * tw + x`.
3. Blocks = every 16 consecutive symbols `p = 16 i .. 16 i + 15`, regardless of rows.
4. Each block has a noise level `L = clamp(bw(mean of 16 z) − off, 0, 7)`, computed causally (below).
5. The `L` low bits of the 16 symbols go raw as `L` bit planes (16-bit masks, movemask);
   the high part `z >> L` goes to the FSE4 stream number `L` (8 streams = 8 contexts "noise of the block").
6. Decoder: FSE → unpack blocks (vector) → MED as a wave over 16-row bands (skew 1 px per row, vector types).

## Mode 2: L comes from the symbols directly above (Argon)

The `L` of block `i` is not transmitted. Both sides compute it from the 16 symbols one tile row back:

```
L(i) = clamp(bw(mean z[p − tw .. p − tw + 15]) − off, 0, 7)      p = 16 i,  if tw >= 16 and p >= tw
L(i) = the same over z[p − 16 .. p − 1]  (the previous block)      first row of the tile, or tw < 16
L(0) = 2
```

The window `p − tw .. p − tw + 15` ends before `p` (as long as `tw >= 16`), so the decoder already has it.
`off` is part of the format here (written in the header).

## Blocks across rows — the example 500 × 500

```
          x: 0 ........ 287 | 288 ...... 499
y 0..287     tile A 288×288 | tile B 212×288
y 288..499   tile C 288×212 | tile D 212×212
```

Tile A: 288 = 18 × 16, every row is exactly 18 blocks, no block crosses a row.

Tile B: 212 = 13 × 16 + 4. A row holds 13 whole blocks and 4 symbols; the next block takes the rest from the
start of the next row. The shift grows by 4 per row and closes after 4 rows (4 × 212 = 848 = 53 × 16):

```
tile B, width 212                                    p = y*212 + x
row 0:  [b 0: x 0..15] ... [b 12: x 192..207] [b 13: x 208..211 | -> 4
row 1:   -> 12: x 0..11]  [b 14: x 12..27] ... [b 26: x 204..211 | -> 8
row 2:   -> 8: x 0..7]    [b 27: x 8..23]  ... [b 39: x 200..211 | -> 12
row 3:   -> 4: x 0..3]    [b 40: x 4..19]  ... [b 52: x 196..211]   exact
row 4:  [b 53: x 0..15] ...                                          aligned again
```

A block that crosses a row has a window that crosses the row above in the same way, so every symbol of the
window is the upper neighbour of the corresponding symbol of the block:

```
block 26 = p 416..431:  row 1, x 204..211  +  row 2, x 0..7
window   = p 204..219:  row 0, x 204..211  +  row 1, x 0..7      <- exactly above each symbol
```

The only imprecision: a crossing block mixes the right end of one row with the left end of the next (far apart
in the picture). It happens once per row and only in tiles whose width is not a multiple of 16.

Tail of a tile: only the very last block of a tile can be incomplete; it is padded with zeros. In 500 × 500 it
never happens: 212 × 212 = 44 944 = 2809 × 16, 212 × 288 = 61 056 = 3816 × 16.

## Why not align the rows (all measured, see README)

| what is done with the row remainder | gray 17 | Kodak 24 |
|---|---|---|
| **mode 2: blocks cross rows** | **4.5268** | **4.5741** |
| rows padded with zeros, padding not coded (`-m 3`) | +0.16% | +0.15% |
| padding = copy of the tail block (`-DPLN_PADCOPY`) | +0.35% | +0.57% |
| remainder stored raw, 8 bits (`-DPLN_RAWTAIL`) | +0.42% | +0.58% |
| padding coded as zeros | +0.59% | +1.05% |

Every alignment pays more on the tile edge than the imprecision of a crossing block costs.

## Results (1 thread, static builds, process start + file I/O)

| corpus | HALIC 0.7.2 | PLN1 `-m 2 -off 2` | enc / dec MB/s, HALIC → PLN1 |
|---|---|---|---|
| 17 gray photos (15.8 Mpx) | 4.6495 | **4.5268 (−2.64%)** | 201 / 123 → 220 / 211 |
| Kodak gray 24 × 500² | 4.7371 | **4.5741 (−3.44%)** | 102 / 74 → 123 / 121 |

In-process: enc ~450, dec ~415 MB/s (AVX2). HALIC 0.7.2 fails to decode 6 of these 41 files back
(g_p07; kodim01–04, 16).

## Compression ranking on the gray sets (bpp; all round-trip checked except HALIC, which fails on 6 files)

| codec | gray 17 | Kodak 24 |
|---|---|---|
| **llcomp rc `-x 4`, map K = 128 trained on both sets** | **4.2774** | **4.3602** |
| llcomp rc `-x 4`, map trained on the *other* set (held-out) | 4.2967 | 4.3706 |
| FFV1 standard (context 0, range_tab, 1 slice) | 4.3268 | 4.4279 |
| llcomp rc `-x 1` (4-D context) | 4.3443 | 4.4470 |
| llcomp rc `-x 0` (tiny) | 4.3584 | 4.4599 |
| FFV1 large (context 1, range_tab, 1 slice) | 4.3807 | 4.6031 |
| llcomp standard | 4.4135 | 4.5091 |
| llcomp rc `-x 3` (8 groups) | 4.4150 | 4.4625 |
| PLN1 `-m 2 -off 2` | 4.5268 | 4.5741 |
| HALIC 0.7.2 | 4.6495 | 4.7371 |

FFV1 tiles = slices; every slice resets the adaptive contexts: 4 slices +0.8% / +2.0%, more slices worse
(`../ffv1_sweep.py`). range_def is 1% worse than range_tab. The large context model loses on these small images
(more contexts to learn per pixel).

`-x 4` (llcomp rc branch): FFV1-large-like cells (q11(L-TL), q11(TL-T), q11(T-TR), q5(LL-L), q5(TT-T), 16638 mirrored)
clustered to K static contexts by `llcomp_large_cluster.py`. Held-out K sweep (train one set, test the other):
K = 64: Kodak 4.3631, gray 4.3003; 128: 4.3706 / 4.2967; 256: 4.3924 / 4.3004. x Enb (`-x 5`) is worse (dilution).
