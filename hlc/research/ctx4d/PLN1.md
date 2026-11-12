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
   Open, no design yet: a bypass for a block that did not compress.
6. Decoder: FSE → unpack blocks (vector) → MED as a wave over 16-row bands (skew 1 px per row, vector types).

## Mode 2: L comes from the symbols directly above (Argon)

The `L` of block `i` is not transmitted. Both sides compute it from the 16 symbols one tile row back:

```
L(i) = clamp(bw(mean z[p − tw .. p − tw + 15]) − off, 0, 7)      p = 16 i,  if tw >= 16 and p >= tw
L(i) = the same over z[p − 16 .. p − 1]  (the previous block)      first row of the tile, or tw < 16
L(0) = 2
```

The window `p − tw .. p − tw + 15` ends before `p` (as long as `tw >= 16`), so the decoder already has it.
`off` is 2 and the tile side is 288. Neither is written. Tile geometry is `(W + 287) / 288` by `(H + 287) / 288`, row-major, last tile keeps the remainder.

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

## PLR: PLN2 with one-pass interleaved RLGR instead of FSE (`palic-2/palic2r.cpp`)

Same tiles, MED, causal L, bit planes, RCT and wave decoder as PLN2; only the entropy stage changes. Every plane
of a tile has 8 RLGR coders (one per L, llrice adaptation) plus one raw writer for the L bit planes; all of them
share one page buffer per tile and take 64-bit pages only at their reservation points, so the decoder takes the
same pages in the same order. No histograms, no second pass. `rlgr_fast.hpp` is a lean re-implementation of the
llrice coder: the file is byte-identical to the llrice classes (`-DPLR_LLRICE`), the run end and the Rice code go
out as one write (reserve(64) after the run end = reserve(65 + k) before it), the decoder reads a Rice code from
one 64-bit window. Static clang `-march=native`, in-process, best of 3, all round trips ok:

| set | PLN2 (FSE) bpp | PLR bpp | PLR enc / dec MB/s (llrice classes → rlgr_fast) |
|---|---|---|---|
| full Kodak 24 colour | 9.8683 | 10.2407 (+3.8%) | 75 / 68 → 97 / 74 |
| gray 17 | 4.5268 | 4.5963 (+1.5%) | 74 / 67 → 93 / 73 |
| 19 colour photos | | 8.2810 | 87 / 80 → 109 / 85 |

PLN2 FSE on full Kodak: enc 423, dec 417 MB/s. The coder is ~75% of the PLR time and the profile is instruction
bound (IPC ~3), so the llrice format itself (a serial adaptive state per symbol) is the ceiling, not the bit I/O.

### PLR spec: off = 1

**PLR uses `off = 1`** in `L = clamp(bw(mean of 16 z) − off, 0, 7)` (PLN1 / PLN2 with FSE keep `off = 2`).
`off` is not written into the file: it is part of the format (`-DPLR_OFF` only for experiments). Measured:
off 0 / 1 / 2 / 3 on full Kodak 10.5648 / **10.1833** / 10.2407 / 10.2586, gray 17 4.6979 / **4.5726** / 4.5963 /
4.6106, 19 colour 8.5352 / **8.2567** / 8.2810 / 8.2985. Why: off 1 sends one more bit of every symbol raw and
leaves RLGR smaller high parts, where its runs of zeros work; FSE codes those small alphabets near the entropy
anyway, so for it the extra raw bit was a loss.

 (llrice `CODER_G_RB_SUM`, Malvar RLGR3 idea) and `off`

Compile switches of `palic2r.cpp`, defaults unchanged (off 2, no pairs = byte-identical to llrice):
`-DPLR_PAIR` codes the high parts of neighbours `2j, 2j+1` of a block as RLGR(a+b) plus `a` raw in `bw(a+b)` bits
(llrice) or, with `-DPLR_TRUNC`, as truncated binary of `0..a+b`; `-DPLR_PAIR_LMIN/LMAX` limit it to some L;
`-DPLR_OFF` sets off. bpp, full Kodak / gray 17 / 19 colour; enc/dec MB/s on Kodak (one run, noisy ±5%):

| variant | Kodak | gray 17 | 19 colour | enc/dec |
|---|---|---|---|---|
| PLR off 2 (llrice exact) | 10.2407 | 4.5963 | 8.2810 | 97 / 74 |
| pairs all L, raw `bw(a+b)` (llrice) | 10.4654 | 4.7221 | 8.5801 | 131 / 108 |
| pairs all L, truncated binary | 10.1419 | 4.6074 | 8.3368 | 112 / 92 |
| pairs L = 0 only, truncated | 10.1328 | 4.6008 | 8.3249 | 110 / 83 |
| pairs L >= 3 only, truncated | 10.2429 | 4.6003 | 8.2849 | 98 / 74 |
| **off 1, no pairs** | 10.1833 | **4.5726** | **8.2567** | 95 / 73 |
| off 1, pairs L = 0, truncated | **10.1267** | 4.5736 | 8.2891 | 107 / 81 |
| off 0 / off 3, no pairs | 10.5648 / 10.2586 | 4.6979 / 4.6106 | 8.5352 / 8.2985 | |

RLGR wants off 1 (FSE wanted 2). The raw split `bw(a+b)` loses 2–4%; truncated binary removes most of it. Pairs help
in the L = 0 bins (the Rice floor of 1 bit per symbol at k = 0), mostly on Kodak, and are neutral or slightly worse
elsewhere. Not done yet: the llrice pairing itself is across channels (U and V of one pixel), not neighbours.

Pairs re-measured at off 1 (now the PLR default), best of 3, all round trips ok; enc/dec MB/s on Kodak:

| variant | Kodak | gray 17 | 19 colour | enc / dec |
|---|---|---|---|---|
| **no pairs (default)** | 10.1833 | **4.5726** | **8.2567** | 99 / 76 |
| pairs L = 0, truncated binary | **10.1267** | 4.5736 | 8.2891 | 109 / 82 |
| pairs L = 0..1, truncated | 10.1711 | 4.5816 | 8.3115 | 114 / 88 |
| pairs L = 0..3, truncated | 10.1974 | 4.5947 | 8.3292 | 117 / 92 |
| pairs L >= 3, truncated | 10.1994 | 4.5892 | 8.2738 | 103 / 77 |
| pairs all L, truncated | 10.2042 | 4.6049 | 8.3389 | 116 / 92 |
| pairs L = 0, raw `bw(a+b)` | 10.2292 | 4.5902 | 8.3860 | 110 / 86 |
| pairs all L, raw `bw(a+b)` (llrice) | 10.4309 | 4.6765 | 8.5140 | 128 / 109 |

Pairs of neighbours are a speed knob, not a size gain: only Kodak (L = 0) gets smaller; on the photos the
neighbours are correlated and the sum is coded worse than the two symbols.

### PLR fast mode: files PLF1 / PLF2 (`palic2r c in out -f`)

Header magic `PLF1` (gray) / `PLF2` (colour) instead of `PLR1` / `PLR2`; everything else as PLR (off = 1).
In every bin L the 16 high parts of a block go as 8 pairs `(a, b) = (h[2j], h[2j+1])`: RLGR codes `s = a + b` in the
coder of bin L, then, if `s > 0`, the raw writer gets `a` in truncated binary over the `s + 1` values
(`k = floor(log2(s + 1))`, `u = 2^(k+1) − (s + 1)`; `a < u`: `a` in k bits, else `a + u` in k + 1 bits;
the raw writer reserves k + 1 bits first). Then the L bit planes as in PLR. The decoder reads the magic, no switch.
Half the RLGR steps per block. Best of 3, all round trips ok, byte-identical with the llrice classes (`-DPLR_LLRICE`):

| mode | full Kodak | gray 17 | 19 colour | enc / dec MB/s, Kodak |
|---|---|---|---|---|
| PLR (off 1) | 10.1833 | 4.5726 | 8.2567 | 100 / 77 |
| PLF fast | 10.2042 (+0.2%) | 4.6049 (+0.7%) | 8.3389 (+1.0%) | 120 / 95 |

### llrice with the truncated split vs PLR (process wall clock incl. file I/O, best of 3, `../llrice_vs_plr.py`)

llrice branch `trunc` (local, from refactor 81613e5): `CODER_G_RB_SUM` writes `r` in truncated binary over `rb + 1`
values instead of `bit_width(rb)` bits. llrice reads only P6, so no gray set.

| codec | full Kodak bpp | enc / dec | 19 colour bpp | enc / dec |
|---|---|---|---|---|
| llrice G_RB (one coder for U, V) | 10.3801 | 66 / 52 | 8.3416 | 78 / 61 |
| llrice G_R_B (refactor default) | 10.3636 | 65 / 52 | 8.3485 | 75 / 56 |
| llrice G_RB_SUM, `bit_width` (old) | 10.6171 | 87 / 67 | 8.5036 | 100 / 76 |
| **llrice G_RB_SUM, truncated** | 10.3223 | 77 / 61 | 8.3187 | 96 / 72 |
| PLR (off 1) | **10.1833** | 76 / 65 | **8.2567** | 94 / 78 |
| PLF fast | 10.2042 | 89 / 78 | 8.3389 | **107 / 92** |

In llrice the pair is U and V of one pixel and the truncated sum is the best llrice coder (−0.4% vs G_R_B, faster).
PLF pairs a stream with itself (neighbours of one block, same plane, same bin). Partner distance inside the block,
PLF, Kodak / gray 17 / 19 colour: 1 (default) 10.2042 / 4.6049 / 8.3389, 4: 10.2391 / 4.6079 / 8.3543,
8: 10.2467 / 4.6108 / 8.3664: the nearest partner is best.

### PLR2 format: one chroma plane S = zU + zV (Vladimir's idea; the default of PLR2, PLF2 keeps U and V planes)

The zigzag residuals of U and V at the same pixel are added, `S = zU + zV` (0..1022), and S is one plane through the
usual coder (causal L on S with the same off 1, L bit planes, RLGR on `S >> L`). After every block the raw writer
gets `zU` of its 16 pixels in truncated binary over the `S + 1` values (nothing when S = 0); `zV = S − zU`. The
decoder rebuilds zU, zV per block and runs the two MED waves as before. Y unchanged, so gray is unchanged.
Best of 3, all round trips ok:

| | full Kodak | enc / dec | 19 colour | enc / dec |
|---|---|---|---|---|
| PLR (U, V planes) | 10.1833 | 100 / 77 | 8.2567 | 118 / 89 |
| **PLR, S plane** | **10.1541 (−0.29%)** | 104 / 81 | **8.2299 (−0.32%)** | 121 / 96 |
| PLF fast (U, V planes) | 10.2042 | 120 / 95 | 8.3389 | 141 / 110 |
| PLF fast, S plane | 10.2385 | 114 / 95 | 8.3085 | 133 / 111 |

off for the S plane: off 0 / 1 / 2: Kodak 10.2623 / 10.1541 / 10.1741, colour 8.3264 / 8.2299 / 8.2318.
Fractional off for S (L = bw(mean · c) − 1, c = 4/8 .. 16/8, i.e. off 2 .. 0): c = 5/8, 6/8, 7/8, 1, 10/8, 12/8:
Kodak 10.1695 / 10.1633 / 10.1574 / **10.1541** / 10.1634 / 10.1908, colour 8.2298 / 8.2275 / **8.2272** / 8.2299 /
8.2440 / 8.2678. Off 1 stays: the best fraction is within 0.03%.

### Does the interleaving cost speed? (`-DPLR_LAYOUT`, experiment; the format stays layout 0)

0 = all streams of a tile in one page buffer (the format); 1 = own stream per coder and for the raw writer, coded on
the fly; 2 = own streams, the symbols of each bin collected and RLGR-coded at the end of the tile in one tight loop
(symbol count per bin in the file; the decoder decodes every bin up front, the block loop reads arrays).
bpp and enc / dec MB/s, best of 5, two runs agree within ~2%, all round trips ok:

| layout | Kodak | gray 17 | 19 colour |
|---|---|---|---|
| PLR 0 | 10.1541 101 / 81 | 4.5726 99 / 75 | 8.2299 119 / 95 |
| PLR 1 | 10.1578 103 / 83 | 4.5741 101 / 77 | 8.2337 122 / 98 |
| PLR 2 | 10.1625 106 / 85 | 4.5759 108 / 85 | 8.2383 125 / 103 |
| PLF 0 | 10.2042 116 / 94 | 4.6049 120 / 95 | 8.3389 136 / 109 |
| PLF 1 | 10.2081 118 / 95 | 4.6063 121 / 95 | 8.3427 140 / 109 |
| PLF 2 | 10.2128 105 / 89 | 4.6080 107 / 89 | 8.3474 120 / 101 |

The interleaving is almost free (layout 1 is +1–2%, inside the noise); coding each bin at the end gains 5–13% in PLR
and loses in PLF. The size grows by 0.04–0.08% (stream headers, a half page per stream). The time is the RLGR state
machine itself, not the page sharing.

### PLR / PLF format: lazy reservation (default; `-DRLGR_RESERVE64` = old llrice rule, byte-identical to llrice)

Why a stream reserves at all: (1) **reader time ≠ writer time**: in a run the encoder writes the run header
(0, or 1 + rl in k bits) at the end of the run, the decoder reads it at the start, and other streams take pages in
between, so the header's bits must be laid at the start of the run (`reserve(1 + k)` on entering a run and after a
full window: the ring of reserved pages exists for this); (2) **unknown length**: llrice also laid 64 bits before
every Rice code because the decoder peeks a 64-bit window. (2) is not needed: a Rice code is written and read at the
same symbol, so now a page is taken when the first bit that does not fit is written / read. The encoder reserves
exactly `q + 1 + m` (with the run end: `1 + k + q + 1 + m`) bits when it writes the code; the decoder counts the
prefix only over the bits the stream already holds and takes the next page when they are all ones (the encoder
took that page while writing this same code), then makes exactly `q + 1 + m` bits available. Escape (32 ones):
`skip(32)`, `reserve(32)`. The raw writer reserves the exact bits of what it writes next. Result: fewer laid but
unused bits. Best of 5, two runs; all round trips ok incl. noise / spike / stripe stress images, ASan + UBSan clean,
clang = gcc files:

| | Kodak | gray 17 | 19 colour |
|---|---|---|---|
| PLR, reserve 64 (old) | 10.1541 101 / 81 | 4.5726 100 / 74 | 8.2299 119 / 95 |
| **PLR, lazy** | **10.1436** 97 / 82 | **4.5668** 95 / 78 | **8.2199** 114 / 98 |
| PLF, reserve 64 (old) | 10.2042 116 / 94 | 4.6049 117 / 95 | 8.3389 136 / 109 |
| **PLF, lazy** | **10.1902** 114 / 96 | **4.5991** 117 / 96 | **8.3258** 132 / 111 |

−0.10..−0.16% size everywhere, decode +1..5%, encode −0..5%.

### Variant (b) estimate: one bit stream per tile + run-header holes (`../plr_onestream_estimate.py`, `-DRLGR_STATS`)

Idea: every code is written at one global bit head the moment it is known; only the run header is a hole of
`1 + k` bits laid at the start of the run and filled at its end; one bit reader for the whole tile in the decoder.
Cost: a full window uses 1 bit of its hole, the other k bits are lost (now they stay in the stream's `cap` and are
used by its next codes). Counted on the real streams:

| | non-payload bits now | k bits of full windows | (b) vs now |
|---|---|---|---|
| PLR Kodak | 0.07% | 0.37% (312 677 full windows) | +0.31% |
| PLR gray 17 | 0.09% | 0.74% | +0.65% |
| PLR 19 colour | 0.08% | 1.57% | +1.49% |
| PLF Kodak / gray / colour | 0.09 / 0.10 / 0.10% | 0.09 / 0.08 / 1.10% | +0.01 / −0.01 / +1.01% |

The lazy page scheme already wastes < 0.1%; the holes of (b) waste 3–15 times more. (b) is not taken as it is.

(b) with Vladimir's repair: a full window whose hole still ends at the head (nobody wrote after it during the run)
gives its k unused bits back (`head −= k`). The decoder sees the same condition: after the flag 0 its head is set
past the hole, and if nothing was read by the time the window ends it steps back by k. Simulated on the real
streams (`-DRLGR_STATS`, SIMB line):

| | full windows shrunk | lost | (b) + repair vs now |
|---|---|---|---|
| PLR Kodak | 282 575 of 312 677 | 0.0043 bpp | −0.02% |
| PLR gray 17 | 434 428 of 466 488 | 0.0026 bpp | −0.02% |
| PLR 19 colour | 1 306 717 of 1 473 591 | 0.0189 bpp | +0.15% |
| PLF Kodak / gray / colour | 93 / 91 / 99% | ≤ 0.0008 bpp | −0.08 / −0.08 / −0.09% |

### PLR / PLF format: one bit lane per tile (`rlgr_lane.hpp`, `PLR_LAYOUT 3`, default)

The (b) scheme built for real. A tile is `var(number of 64-bit words)` + the words of one bit lane. Every code
(Rice, the L bit planes, the zU split) is written at the lane head when it is known. A run header gets a hole of
`1 + k` bits at the head when the run starts and right after a full window; the encoder fills it at the end of the
run (a full window is the flag 0: nothing to write), the decoder reads it by position when the run starts. A full
window whose hole still ends at the head gives back k bits (`head = hole + 1`), on both sides. An open hole at the
end of the tile stays (zeros). Each coder keeps one open hole; no pages, rings or reservation of worst cases; the
decoder is one sequential reader. Round trips ok (edge sizes, noise / spike / stripe stress), ASan + UBSan clean,
20 corrupted files rejected without a crash, clang = gcc files. Best of 5, two runs agree:

| | Kodak | gray 17 | 19 colour |
|---|---|---|---|
| PLR, pages (lazy) | 10.1436 97 / 83 | 4.5668 95 / 78 | 8.2199 114 / 98 |
| **PLR, lane** | **10.1422** 106 / 84 | **4.5666** 102 / 78 | 8.2334 (+0.16%) 125 / 101 |
| PLF, pages (lazy) | 10.1902 116 / 97 | 4.5991 118 / 97 | 8.3258 135 / 111 |
| **PLF, lane** | **10.1834** 122 / 103 | **4.5964** 123 / 104 | **8.3196** 143 / 119 |

Encoder +5..11%, decoder +1..7%; size −0.07..0% except PLR on the colour photos (+0.16%: full windows whose hole
was passed by the raw bits of the blocks). `-DPLR_LAYOUT=0` builds the page format; with `-DRLGR_RESERVE64` it is
byte-identical with `-DPLR_LLRICE`.

### GCC vector path (palic2.cpp, palic2r.cpp)

The wave decoder and the vector block unpack were clang-only (`ext_vector_type`); GCC fell back to the row decoder.
They now also build with GCC >= 12 `vector_size` types and `__builtin_shufflevector`; the only dialect difference
is the scalar broadcast (`VSPL(T, x)`: clang `(T)x`, GCC `T{} + x`). Same files from both compilers, all round trips
ok. Decode MB/s, best of 5, GCC row path → GCC vector (clang): PLN2 Kodak 170 → 412 (416), gray 165 → 418 (430);
PLR Kodak 65 → 84 (84), gray 60 → 77 (78), 19 colour 76 → 99 (99). `planes_codec.cpp` (PLN1, experiment) still
uses clang bool vectors and stays clang-only.

## PLR format: static QH tables instead of RLGR (default since this commit; PLF unchanged)

Idea and prototype: a Claude Fable subagent (`../qh/`), integrated, zero-block runs and off 2 here.
PLR (files PLR1 / PLR2) now codes the high parts of a block with static tables; PLF (`-f`) keeps the RLGR pairs.

- **off = 2** for PLR (L = clamp(bw(mean) − 2, 0, 7), like PLN2's FSE); PLF keeps off = 1. Not in the file.
- **Context of a block:** `t = ctxmap[plane type: 0 = Y, 1 = S][Q(sa)]`, sa = the causal sum of 16 z the L is made
  from (row above, or the previous block in the first row / narrow tiles; 64 for block 0). `Q(sa) = sa` for sa < 32,
  else `32 + (bw(sa) − 6)·4 + the two bits under the top bit` (max 67).
- **Zero-block runs:** a block with sa = 0 first sends n = the number of all-zero blocks from it on (same tile, same
  plane) as Exp-Golomb order 0 (`n + 1` in `2·bw(n+1) − 1` bits); those n blocks are not coded at all. n = 0: coded.
- **Block:** 4 letters (4 neighbours each, `min(h, 3)` in 2 bits, first symbol in the low bits, 256 letters)
  in canonical Huffman of table t (lengths ≤ 11); then for every symbol with h ≥ 3: `h − 3` as Rice(k_t),
  escape (`q ≥ 24`): 24 ones + 10 raw bits; then the L raw bit planes; colour: then the zU splits of the S block.
- **Tables:** `palic-2/qh_tables.h` (32 tables, 32 × 256 lengths + 136-byte map + 32 Rice k). First version: Kodak 24
  + 12 CLIC gray (`../qh/train_tables.py`); now (see below): 117 CLIC 2020 professional train images, colour and gray,
  + Kodak 24 (`../qh/train_stream.py`). Decoder lookup: 32 × 2048 × 2 B.

Process wall clock incl. file I/O, best of 3, every file round-trips (`../plr_rank.py`). Held-out sets: col19,
gray 17 and CLIC B (8 CLIC gray images not in training). Kodak is in the training set (held-out tables
trained on CLIC A + col19 give Kodak 9.9507):

| codec | Kodak 24 | 19 colour | gray 17 | CLIC B 8 gray |
|---|---|---|---|---|
| **PLR (QH)** | **9.9111** 102 / 120 | **8.0673** 136 / 161 | **4.5084** 112 / 149 | **3.8049** 134 / 192 |
| PLR (RLGR lane, before) | 10.1422 81 / 69 | 8.2334 99 / 85 | 4.5666 78 / 65 | 3.8692 81 / 66 |
| PLF fast (RLGR pairs) | 10.1834 86 / 77 | 8.3196 105 / 94 | 4.5964 92 / 81 | 3.9006 92 / 82 |
| PLN2 (FSE, two passes) | 9.8683 199 / 217 | 8.2631 232 / 257 | 4.5268 207 / 214 | 3.8077 235 / 256 |
| HALIC 0.7.2 | 10.0994 191 / 125 | 8.4471 236 / 141 | 4.6495 (1 file fails) | 3.8749 (1 fails) |
| HALIC FAST | 10.3259 240 / 139 | 8.7396 309 / 162 | 4.7115 (1 fails) | 3.9733 131 |

Synthetic gray 1024×768 (bpp, PLR RLGR → PLR QH): flat 0.0047 → 0.0037, rectangles 0.089 → 0.075, stripes
0.011 → 0.015, sparse spikes 0.067 → 0.080, **gradient 0.152 → 0.245** (residuals never zero: 4 letters ≥ 4 bits
per block), noise equal or better. ASan + UBSan clean, 40 corrupted files rejected without a crash, clang = gcc
files. CLIC colour could not be fetched through the Drive connector (P6 files are too big for one message), so
the S tables saw Kodak only.

### Tables retrained on CLIC 2020 (full corpus from github vovach777/clic-palic2, decoded with palic-2)

CLIC 2020 professional train, 585 images, colour P6 and gray P5 (1364 Mpx each). Split by sorted name: index % 5 == 0
= test (117), the training images = every 4th of the rest (117, colour + gray) + Kodak 24 (`../qh/train_stream.py`,
one image at a time). Same tables for every set; bpp (in-process bench):

| set | tables Kodak + 12 CLIC gray | tables CLIC 117 + Kodak | PLN2 |
|---|---|---|---|
| CLIC test 39 colour | 8.4527 | **8.4105** | 8.5125 |
| CLIC test 39 gray | 3.8222 | 3.8225 | 3.8370 |
| 19 colour photos | **8.0673** | 8.0754 | 8.2631 |
| gray 17 | 4.5084 | 4.5078 | 4.5268 |
| Kodak 24 (in training both times) | **9.9111** | 9.9588 | 9.8683 |
| synthetic gradient / spikes / noise σ2 | 0.245 / 0.080 / 3.688 | **0.169 / 0.069** / 3.716 | |

The format now uses the CLIC tables (the target corpus; Kodak is old film scans).
Full held-out CLIC test part (117 images, process wall clock incl. I/O, `../plr_rank.py`, every file round-trips
except where noted):

| codec | colour bpp | enc / dec MB/s | gray bpp | enc / dec MB/s |
|---|---|---|---|---|
| **PLR (QH)** | **8.6639** | 133 / 163 | **3.8826** | 140 / 205 |
| PLR (RLGR lane) | 8.8366 | 90 / 80 | 3.9406 | 86 / 70 |
| PLF fast | 8.8673 | 106 / 95 | 3.9758 | 99 / 88 |
| PLN2 (FSE) | 8.7555 | 231 / 265 | 3.8947 | 245 / 273 |
| HALIC 0.7.2 | 8.7519 | 240 / 152 | 3.9181 (2 files fail) | 272 / 156 |
| HALIC FAST | 9.1271 | 357 / 170 | 4.0265 (1 fails) | 310 / 145 |

PLR is 1.0% (colour) / 0.9% (gray) smaller than HALIC 0.7.2 on held-out CLIC and decodes faster; its encoder is
~1.8x slower. Full CLIC with PLN2 (the clic-palic2 repo): colour 8.6320, gray 3.9070 bpp.

### Encoder speed-up (Claude Fable review, byte-identical; `../qh_enc/README.md`)

Branchless register writer (`LaneW`), SSE letters, branchless tails while all codes of a block are <= 15 bits, AVX2
joining of the 16 codes of a block, branchless zU split (`tb_put16`), one loop per plane, vectorisable `rgb_split`.
In-process encode, AVX2 (decode unchanged): Kodak 146 -> 551, CLIC 117 colour 172 -> 542, gray 188 -> 548 MB/s;
without AVX2 (x86-64-v2) ~2x. Files identical on every set (clang, gcc, x86-64-v2; PLF too). The front end
(RCT + MED) is now ~40% of the encoder.

### Second round (Claude Fable): front end and decoder, byte-identical

Encoder: one pass over the rows of a tile (RCT -> MED of Y, U, V -> S) with vector MED, prefetch 4 rows ahead
(tile rows are short strided runs). Decoder: branchless zU split (`tb_get16`; the old one had ~8 mispredicts per S
block), one rare refill check for the tails, 4 planes per window. `c` reads the input again only when the file did
not shrink. In-process MB/s enc / dec (8b2c816 -> now, verified): Kodak 565 / 182 -> 700 / 304, CLIC 117 colour
554 / 205 -> 720 / 323, gray 536 / 288 -> 621 / 316. Process wall clock incl. I/O on CLIC 117 (Fable's run):
colour 401 / 230, gray 285 / 202 MB/s (HALIC 0.7.2: 240 / 152, 272 / 156). Files identical (474 cmp on clang /
gcc / x86-64-v2, PLF too), ASan + UBSan clean, corrupted files rejected without a crash.

### Screen content (github imazen/codec-corpus v1.1.0; `../sc_convert.py`, process wall clock, all round trips ok)

| set | PLR (QH) | PLR (RLGR) | PLF | PLN2 | HALIC 0.7.2 | HALIC FAST | PNG (as shipped) |
|---|---|---|---|---|---|---|---|
| gb82-sc, 10 screenshots | 1.3433 | 1.2955 | 1.3537 | 1.3469 | **1.1844** | 1.4457 | 0.78 |
| qoi screenshot_web, 14 pages | 2.0392 | **1.9603** | 2.0078 | 2.0787 | 2.0466 | 2.3042 | 2.83 (incl. alpha) |

PLR enc / dec 651 / 440 and 787 / 403 MB/s there. HALIC's lead on gb82-sc comes from a few files: windows95 (a
dithered palette image: PLR 2.94, HALIC 0.76), windows 2.23 / 1.65, imessage 1.53 / 1.23. "Zero run" as the 257th
symbol of the first letter (`qh_enc/exp_zr.py`): gb82-sc −1.42%, screenshot_web −1.25% of the file; it does not close
that gap — palette / repeated content needs another tool.

## Screen content: hybrid screen mode (2026-11-12, model estimates, not in the codec)
Cost model (qhsim + runs + zU split), screen tables trained on 12 screens of /tmp/sc, tested on the 12 halic-wins-screen-p6
(held-out) and CLIC test 39 colour / 39 gray. % vs the format (12 screens 1.8385, CLIC c 8.4241, g 3.8213 bpp).
- escape (8,2) = 8 ones + EG2 instead of 24 ones + 10 bits (Fable): screens -6.7%, CLIC -0.03 / -0.07%.
- Fable screen mode (own 32 tables, L = bw(max(sa, sl) >> 4) - 2 - 3, 1 bit per tile): screens -23.5%, CLIC -0.06 / -0.13%.
- Argon Sparse-Raw (h = 0 if z == 0 else ((z-1) >> Ls) + 1, Ls raw bits only for z > 0), format tables, per-tile Ls: -21.4%.
- Hybrid = both (Sparse-Raw inside the screen mode, tables retrained for it): H0 (Ls = bw(max(sa,sl)>>4) - 2, no shift)
  1.3173 bpp (-28.4%); best of H0..H4 / HF3..6 per tile 1.3041 (-29.1%); CLIC -0.06 / -0.12%. HALIC 1.554, PALIC-1 1.543.
Scripts: qh_enc/exp_sparse_raw.py, exp_hybrid.py, fable_l/ (Fable's lspike*.py, REPORT.md, rounds, hybrid results).

### PLS: the screen mode in the codec (palic-2r -s, files PLS1 / PLS2; qhs.hpp, qhs_tables.h)
The hybrid H0 for the whole file (no per-tile switch yet): tiles, MED, RCT, S = zU + zV with the zU split, runs of
zero blocks at sa == 0 as PLR; per block est = max(sa, sl) (sa as PLR's causal sum, sl = the sum of the previous block;
block 0: est = 64, Ls = 2), Ls = clamp(bw(est >> 4) - 2, 0, 7), table = ctxmap[Y/S][Q(est)] of 32 screen tables;
h = 0 if z == 0 else ((z - 1) >> Ls) + 1; 4 letters min(h, 3); tails h - 3: Rice(k <= 6), q >= 8: 8 ones + EG2 of
x - (8 << k); then Ls low bits of z - 1 for every z > 0 in pixel order. PLR files are bit-identical to before.
Real files (bench -r 1, clang, this sandbox):
| set | PLR bpp | PLS bpp | PLS enc / dec MB/s (PLR) |
|---|---|---|---|
| 12 halic-wins screens | 1.8361 | 1.3155 (-28.4%; HALIC 1.554, PALIC-1 1.543) | 910 / 472 (1438 / 509) |
| CLIC test 39 colour | 8.4105 | 8.6690 (+3.1%: photos keep PLR) | 246 / 236 (676 / 297) |
| windows95 green as gray | 2.2650 | 1.2084 | |
Model (exp_hybrid.py, per-tile switch) 1.3173 vs real 1.3155. Checks: edge sizes (1x1 .. 300x577, noise and sparse),
gcc build, ASan+UBSan, 180 corrupted PLS files: 154 rejected, the rest decode, 0 crashes.
Backlog (not done, by decision): automatic choice PLR / PLS (per file or a per-tile flag, encoder-side costing),
the variant choice per tile (H0..H4, HF; +0.7 pt in the model), escape (8,2) for PLR, screen tables on a larger corpus
(icons, screenshot_game), a faster PLS encoder / decoder (scalar now), PLS for gray screens is trained on Y only.
