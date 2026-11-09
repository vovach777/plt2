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
| **llcomp rc `-x 6` (preset CDFs), K = 256 trained on both sets** | **4.2615** | **4.3281** |
| llcomp rc `-x 6`, trained on the *other* set (held-out) | 4.2833 | 4.3418 |
| FFV1 two-pass (fitted initial states; needs level 3: 4 slices), best context | 4.2858 (large) | 4.3534 (standard) |
| llcomp rc `-x 4`, map K = 128 trained on both sets | 4.2774 | 4.3602 |
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

`-x 6`: the 256 clusters start from preset CDFs (corpus letter distributions) with a slower start of the adaptation
(count 64). Held-out, preset count sweep 0 / 16 / 32 / 64 / 128: Kodak 4.3700 / 4.3523 / 4.3454 / 4.3418 / 4.3413,
gray 4.2892 / 4.2849 / 4.2836 / 4.2833 / 4.2838. With presets K = 256 is no longer worse than 128.

## Kodak colour (24 x 500 x 500 P6), bpp, all round trips ok (`../color_rank.py`)

| codec | bpp |
|---|---|
| **llcomp rc `-x 6`, map/presets without Kodak (held-out)** | **9.1675** |
| llcomp rc `-x 6`, shipped map (its Y part saw Kodak gray) | 9.1589 |
| FFV1 standard, range_tab, 1 slice (nut container, ~0.016 bpp of it) | 9.2727 |
| llcomp rc `-x 1` (4-D) | 9.3428 |
| llcomp rc `-x 0` (tiny) | 9.4558 |
| FFV1 large, 1 slice | 9.5079 |
| llcomp standard, Y and U/V contexts split | 9.5413 |
| llcomp standard (one context set for all channels) | 9.6756 |
| HALIC 0.7.2 | 10.3490 |
| HALIC 0.7.2 FAST | 10.5793 |

## llcomp rc `-x 6` trained on Kodak colour only (Y and U/V, 256 / 256), tested on our sets

| set | rc `-x 6` | rc `-x 4` (no presets) | FFV1 standard | FFV1 large | rc tiny | HALIC 0.7.2 |
|---|---|---|---|---|---|---|
| gray 17 | **4.2836** | 4.3008 | 4.3293* | 4.3832* | 4.3584 | 4.6495 |
| 19 colour photos | **7.4710** | | 7.5536 | 7.5675 | | |
| big 4033 x 3001 | 3.0712 | | 3.0131 | **2.8581** | 2.9785 | |
| colour 20 (19 + big) | 5.6643 | 5.6842 | 5.6908* | **5.6354*** | 5.7257 | 6.5427 |

\* nut container (~500 bytes per file). The Kodak-trained map fits phone photos but not big: there FFV1 large
(and even rc tiny) adapts better; more clusters / a faster preset start are the candidates.

Sign rule A/B in rc large cells (Kodak-trained 256/256, same test sets): tiny sign (c-a, else c-b) instead of the
FFV1 index sign: -x 6 -> 4.2804 / 7.4767 / 3.0817 vs 4.2836 / 7.4710 / 3.0712 (gray 17 / 19 colour / big):
no gain once 16 k cells are clustered by data; the FFV1 rule stays.

## PLN2 colour (full Kodak, 24 x 768 x 512, 28 311 552 raw bytes)

Y is PLN1 mode 2 (8-bit wrap, causal L, `-off 2`, tile 288). U and V are the reversible RCT
(`U = R-G`, `V = B-G`, `Y = G+(U+V+2)>>2`, then `V -= U>>2`). A chroma sample is `-255..255`, so the
MED residual `-510..510` folds modulo 512 into `-256..255` before the zigzag; the symbol stays in `0..511`
and the FSE bound is `512>>L` (no per-stream max scan). Every file round-trips. One process per image,
wall clock on this machine (not the i7 times from the public table, and not the in-process 294 / 137 MB/s):

| | bytes | bpp | enc s | dec s | enc MB/s | dec MB/s |
|---|---:|---:|---:|---:|---:|---:|
| llcomp `-x 11` | 10 598 730 | 8.985 | 1.643 | 1.697 | 17 | 17 |
| NBLI `-g` | 10 681 546 | 9.055 | 2.661 | 2.419 | 11 | 12 |
| fNBLI | 10 822 082 | 9.174 | 0.435 | 0.281 | 65 | 101 |
| llcomp `-x 0` | 10 864 795 | 9.210 | 0.840 | 0.925 | 34 | 31 |
| FFV1 level 1, rice, context 0 | 11 166 333 | 9.466 | 1.579 | 1.585 | 18 | 18 |
| ctx4d | 11 220 144 | 9.511 | 0.325 | 0.329 | 87 | 86 |
| **PLN2 `-m 2 -off 2`** | **11 640 622** | **9.868** | **0.335** | **0.348** | **85** | **81** |
| HALIC 0.7.2 | 11 913 726 | 10.099 | 0.198 | 0.335 | 143 | 85 |

On this set the 9-bit fold does not change the size: no residual leaves `-256..255`. A synthetic frame
whose neighbours jump by ±510 does round-trip, which is the case the fold is for. Gray PLN1 is unchanged
(Kodak gray 4.5741).
