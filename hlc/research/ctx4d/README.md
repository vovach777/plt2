# ctx4d — AVX2 kernels of the planned HLC5 4-D context

Encoder kernel (`ctx4d_enc`) and wavefront decoder (`ctx4d_dec`) of the 4-D context:
q7(c−a) × q7(c−b) × q7(d−b) × 3 levels of v = max of the top-row zigzag residuals → 8 groups per plane type
(free 1029-cell map `q7d_max3_groups.csv`, CLIC calibration), tiny mirror sign, MED via the gradients, wrap,
zigzag, packed `(z << 4) | stream`. Spec and scalar reference: `../ctx4d_ref.cpp`.

* `ctx4d_enc image.ppm q7d_max3_groups.csv [reps]` — RCT, context + symbol with AVX2 gather and with a scalar
  group lookup, the scalar reference; checks both bit-exact against the reference.
* `ctx4d_dec image.ppm q7d_max3_groups.csv [reps]` — encodes with the reference, puts the symbols into 16 streams
  in decoder order (band of 16 rows, step t = x + 2k, lane), decodes with the AVX2 wavefront, compares every pixel.
  Timed: the decode only (context + stream read + reconstruction, 3 planes), no FSE.

* `ctx4d_vec image.ppm q7d_max3_groups.csv [reps]` — both kernels written with GCC/Clang vector types only
  (`vector_size(32)`, plain operators, `__builtin_shufflevector`), no intrinsics, no SIMDe; same checks.

Input: binary P6 PPM, width ≤ 32700.

## Build

Part of the plt2 CMake build (option `PLT_BUILD_CTX4D`, ON): the binaries land next to `plt2` / `hlc`, the target ISA
is the root `PLT_ARCH`. With AVX2 the intrinsics are native; without AVX2 (Ivy Bridge, ARM) the same code goes through
the SIMDe amalgam bundled in the plt2 root (`avx2.h`) — `simd.h` picks it, the output line says which one ran.

    cmake -B build                                          # native
    cmake -B build_ivy -DCMAKE_CXX_COMPILER=clang++ -DPLT_ARCH=ivybridge   # SIMDe path on Ivy
    cmake --build build --config Release --target ctx4d_enc ctx4d_dec

`-DCTX4D_FORCE_SIMDE=ON` runs SIMDe on an AVX2 machine too, to see its cost.
By hand from this folder: `clang++ -O3 -march=native -std=c++17 ctx4d_dec.cpp -o ctx4d_dec`.

## Compiler notes (measured in the sandbox, Xeon with AVX2; ns per pixel, 3 planes, 1008x756 / 12 Mpx)

| build | encoder (gather) | decoder |
|---|---|---|
| g++ 13 -march=haswell (native AVX2) | 2.06 | 8.20 / 8.65 |
| clang 18 -march=haswell (native AVX2) | 1.78 | 6.71 / 7.50 |
| clang 18 -march=ivybridge -mno-avx (SIMDe on SSE4) | 3.78 | 6.29 / 6.67 |
| clang 18 -march=westmere (SIMDe on SSE4) | 3.86 | 6.21 / 6.73 |
| g++ 13 -march=ivybridge (SIMDe, AVX1 present) | 31 | 40 |
| g++ 13 -march=ivybridge -mno-avx (SIMDe on SSE4) | 28 | 34 |

* For SIMDe builds use clang: gcc 13 turns the emulated 256-bit integer ops into element loops (≈10x slower).
  On an AVX1-only CPU (Ivy Bridge) clang with `-mno-avx` was the fastest SIMDe variant here.
* The decoder is latency bound: the SSE build is faster than native AVX2, the cross-128-bit lane shift
  (`vperm2i128` + `vpalignr`) sits on the step-to-step chain.

### Vector types (`ctx4d_vec`) vs intrinsics, ns per pixel, 3 planes, 1008x756 (sandbox Xeon)

| build | encoder | decoder |
|---|---|---|
| clang 18 -march=haswell, vector types | 3.12 | 6.87 |
| clang 18 -march=haswell, intrinsics (gather / scalar group) | 1.78 / 2.39 | 6.71 |
| clang 18 -march=ivybridge -mno-avx, vector types | 3.83 | 6.12 |
| clang 18 -march=ivybridge -mno-avx, SIMDe | 3.78 | 6.29 |
| g++ 13 -march=haswell, vector types | 3.60 | 8.77 |
| g++ 13 -march=ivybridge -mno-avx, vector types | 23 | 44 |

* Element access `v[i]` on a vector variable inside the loop is a trap (clang: 5x slower encoder, 3x decoder):
  go through an aligned array (`store`, scalar loop, `load`).
* clang lowers 256-bit vector types to SSE as well as SIMDe does; gcc 13 does not (same element loops as with SIMDe).
* The encoder gap on AVX2 is the group lookup: the intrinsics version uses `vpgatherdd`.

### AVX-512 (`ctx4d_vec`, lanes = 32 when `__AVX512BW__`, `-DCTX4D_LANES=16|32` to force), sandbox Xeon

| build | encoder | decoder |
|---|---|---|
| clang -march=sapphirerapids, x32, VBMI group lookup | 2.23 | 6.98 |
| g++ -march=sapphirerapids, x32, VBMI group lookup | 2.41 | 6.82 |
| clang -march=skylake-avx512, x32, scalar group lookup | 3.72 | 7.17 |
| clang -march=sapphirerapids, x16 | 2.99 | 7.26 |

* Wider vectors alone do not help: both kernels carry a per-lane scalar part (group lookup in the encoder,
  stream reads in the decoder) that grows with the lane count.
* AVX-512 VBMI: the 1029-byte group table as 9 x 128 bytes in registers, `vpermi2b` by the low 7 bits of the cell
  and a masked select by the high bits - no memory access.
* The decoder needs the stream reads vectorised to gain from width (AVX-512: `vpconflictd` gives each lane the count
  of earlier lanes in the same stream = its offset; then one gather and a per-stream pointer update).

### AVX-512 vector stream read (`CTX4D_VREAD`: VBMI + VPOPCNTDQ), 12 Mpx, sandbox Xeon, g++

| decoder | ns/px |
|---|---|
| x32, scalar stream reads | 6.82 |
| x32, vector stream reads (rank = popcnt(group mask & earlier lanes), 2 gathers per step) | 6.17 |
| same, unskew pass skipped (timing only, `-DNO_UNSKEW`) | 4.90 |
| x16 AVX2, unskew skipped | 7.36 |

* The step is latency bound: the vector read removes the pointer chain but puts a gather (+ group lookup)
  on the step-to-step path. ~150 cycles per step and plane remain. Next lever: the three planes as three
  independent chains in one step loop. The scalar unskew (wave -> raster) is ~20% of the decoder.

### Three planes as three chains in one step loop (`decode_interleaved` vs `decode_serial`), 12 Mpx, ns/px

| build | serial | interleaved |
|---|---|---|
| g++ -march=sapphirerapids (x32, VBMI, vector read) | 6.53 | 5.55 |
| clang -march=sapphirerapids | 6.43 | 5.75 |
| clang -march=haswell (x16) | 7.52 | 6.98 |
| g++ -march=haswell | 9.11 | 7.19 |
| clang -march=ivybridge -mno-avx (SSE) | 6.82 | 6.90 |

Interleaving gives 10-20 % on AVX2/AVX-512 and nothing on SSE (the 256-bit ops are already two independent
128-bit halves there): the step is now mostly throughput bound (instruction count), not latency bound.

## ctx4d_codec — prototype codec (4-D context + FSE)

`ctx4d_codec c in.ppm out.c4d groups.csv | d in.c4d out.ppm groups.csv | bench groups.csv a.ppm|dir ... [-r N] [-max N]`

RCT -> context + symbol (vector kernel, raster) -> 16 streams in decoder order (band of 16 rows, step t = x + 2k,
plane Y U V, lane; streams 0..7 Y, 8..15 shared by U and V) -> FSE4 (hlc's 4-state coder) per unit of whole bands
>= 256 compute blocks of 128 x 16 px -> file. Decoder: FSE per unit, wavefront decode of the three planes in one t
loop, inverse RCT. Built with the plt2 root (needs fse_huf).

First numbers (sandbox Xeon, clang, AVX2, 1 thread) on the 18 photos + cat + 12 Mpx upscale:
5.9822 bpp vs HLC5 (-5 -cb 65536) 6.0093 (-0.45 %; photos -0.3 .. -1.9 %, smooth images +0.2 .. +1.6 %);
encoder 203 vs 201 MB/s, decoder 271 vs 316 MB/s. ns/px: enc RCT 0.42, context 4.0, wave scatter (scalar) 5.7,
FSE 4.3; dec FSE 3.6, wave 6.2, inverse RCT 0.9.

### C4D2: block-wise flush, no plane interleave (sandbox Xeon, clang, AVX2, 20 images)

Symbols of each compute block (128 x 16) go into staging streams (Y g, U g, V g); after a block, when one stream
(Y g, or U g + V g) holds >= CHUNK symbols, everything is flushed. The chunk boundary is a wave step (header: number
of steps), the decoder continues the wave across it, also mid-band. Inside a chunk: Y, then U, then V.
Raw (uninitialised) buffers, scatter through plain pointers, histograms in a separate 4-way pass.

encoder 240 MB/s (HLC5 201), decoder 254 MB/s (HLC5 316). ns/px: enc RCT 0.44, context 3.15, wave scatter 4.30,
FSE 4.52 (histograms 1.33); dec FSE 3.76 (tables 0.43), wave 7.27 (steps 6.50, unskew 0.77), inverse RCT 0.79.

| -cb (flush threshold) | 32K | 64K | 128K | 256K | 1M |
|---|---|---|---|---|---|
| bpp | 6.0380 | 6.0081 | 5.9885 | **5.9822** | 5.9936 |

### Experiment: independent 128 x 64 tiles, no wavefront (`ctx4d_tiles`)

Each tile coded as a separate image (same 4-D context, tile borders = image borders). Decoder: 16 full tiles at once,
lane = tile, raster inside the tile (no lane shifts, no border masks, two rows x 16 lanes in L1); partial tiles scalar.
20 images, -cb 64K:

| | bpp | full-tile decode core, ns/px (AVX2 / SSE) |
|---|---|---|
| C4D2 wavefront | 6.0081 | 7.27 (steps 6.50 + unskew 0.77) / 6.55 |
| independent tiles | 6.0790 (+1.18 %) | 7.00 / 6.25 |

The cheaper step (no shifts / masks) does not pay: the scalar stream reads per lane dominate both. Independent tiles
cost ~1.2 % bits for ~4 % decoder core; their gain would be thread / random-access parallelism, not single-core speed.

Edge blocks absorb the remainder (last column up to 255 wide, last row up to 127 high), grouped by shape (inner,
right column, bottom row, corner), all through the same SIMD decoder (lanes = blocks, unused lanes idle):
6.0688 bpp (wave 6.0081, +1.0 %), encoder 234 MB/s (wave 239), decoder 233 MB/s (wave 257, HLC5 316);
decode core over all pixels 8.13 ns/px AVX2 / 7.72 SSE (edge groups run with fewer lanes and wider rows).

### Fused block encoder (default; `-planes` = the old per-plane passes)

Per compute block 128 x 16 in one pass: RGB -> RCT -> context + symbol -> wave scatter of the complete steps,
then the flush check; the block is a parallelogram (row k: columns x0 - k .. x0 + 127 - k) so the top-right z of
the error axis is already computed. Output byte-identical to `-planes` (26 files incl. edge sizes).
Sandbox (L2 4 MB, L3 480 MB: every plane fits in cache, memory effects invisible): no gain, fused 235 vs 242 MB/s
(AVX2), 212 vs 211 (SSE); 48 Mpx the same. The real test is a small-cache machine (Ivy: L2 256 KB, L3 6 MB).

### ctx4d_blk: experimental codec C4B1, independent compute blocks fused in L1

`ctx4d_blk c in.ppm out.c4b groups.csv [-tw 128] [-th 16] [-cb N]`, `ctx4d_blk d in.c4b out.ppm groups.csv`, `bench`.
Every block TW x TH is a separate image (borders restart); 16 blocks of one shape = 16 lanes. Encoder per row:
RGB -> RCT -> context + symbol -> streams; decoder per row: streams -> Y U V -> inverse RCT into the image.
Only FSE is a separate pass. Sandbox, -cb 256K, 1 thread (wave = ctx4d_codec, same groups):

| | bpp photos (19) | bpp big 12 Mpx | enc photos / big MB/s | dec photos / big MB/s |
|---|---|---|---|---|
| wave, AVX2 | 7.8065 | 3.3636 | 245 / 218 | 270 / 250 |
| C4B1 128x16, AVX2 | 7.9985 (+2.5%) | 3.5989 (+7.0%) | 264 / 248 | 266 / 269 |
| wave, SSE | | | 218 / 194 | 275 / 266 |
| C4B1 128x16, SSE | | | 249 / 245 | 291 / 301 |
| C4B1 128x64, AVX2 | 7.8848 (+1.0%) | 3.4512 (+2.6%) | 250 / 260 | 229 / 270 |
| C4B1 256x256, AVX2 | 7.8312 (+0.3%) | 3.4000 (+1.1%) | (few blocks per image: lanes idle on the photos) | |

### planes_codec: gray codec PLN1 (context-free, PALIC-1 MED + noise L per 16-block)

Independent tiles 288 x 288 (as HALIC). Per tile: MED -> zigzag z; per 16-symbol block L = clamp(bw(mean z) - off, 0, 7):
the L low bits raw as bit planes (movemask), the high part z >> L in one FSE4 stream per L value (8 contexts "noise
of the block"), L itself through FSE4. Encoder: residual rows auto-vectorized by clang, blocks in ext_vector_type.
Decoder: FSE + unpack, then MED as a wave over 16-row bands (skew 1 px per row, vector types), skew in and out by
16 x 16 transposes. 17 gray photos (16 + cat, 15.8 Mpx), 1 thread, static builds, process start + file I/O included:

| | bpp | enc MB/s | dec MB/s |
|---|---|---|---|
| HALIC 0.7.2 | 4.6495 | 205 | 125 |
| HALIC 0.7.2 FAST | 4.7115 (+1.3%) | 228 | 123 |
| PLN1, -off 2 | **4.5614 (-1.9%)** | **228** | **227** |
| PLN1, -off 1 (default) | 4.5684 (-1.7%) | 217 | 218 |

In-process (AVX2 / SSE): enc 443 / 370, dec 452 / 403 MB/s. Profile ns/px AVX2: enc residual 0.37, blocks 0.53, FSE 1.42;
dec FSE 1.29, unpack 0.33, MED wave 0.63. HALIC 0.7.2 decodes g_p07 (1288 x 966) wrongly (57 K pixels from row 600).

#### PLN1 mode 2 (Argon): no L stream, L_b causal from the 16 symbols directly above (default, -m 2 -off 2)

Decoder repeats the rule from already decoded z (row 0 / tiles narrower than 16: the previous block; block 0: L = 2),
so off is part of the format here. Same 17 gray photos, process + file I/O, static builds:

| | bpp | enc MB/s | dec MB/s |
|---|---|---|---|
| HALIC 0.7.2 | 4.6495 | 201 | 123 |
| PLN1 -m 1 -off 2 (L stream) | 4.5614 (-1.89%) | 214 | 212 |
| **PLN1 -m 2 -off 2** | **4.5268 (-2.64%)** | **220** | **211** |
| PLN1 -m 2 -off 1 | 4.5258 (-2.66%) | 215 | 208 |

#### Kodak gray (24 x 500 x 500, `../data/kodak_gray_all.zip`), process + file I/O, 5 runs

| | bpp | enc MB/s | dec MB/s |
|---|---|---|---|
| HALIC 0.7.2 | 4.7371 | 102 | 74 |
| HALIC 0.7.2 FAST | 4.7703 (+0.70%) | 108 | 74 |
| PLN1 -m 1 -off 2 | 4.6172 (-2.53%) | 121 | 124 |
| **PLN1 -m 2 -off 2** | **4.5741 (-3.44%)** | **123** | **121** |
| PLN1 -m 2 -off 1 | 4.5799 (-3.32%) | 123 | 122 |

PLN1 is smaller on all 24 files. Small files: process start dominates the MB/s. HALIC 0.7.2 does not decode 5 of the
24 back (kodim01-04, 16; e.g. kodim01: 58868 pixels differ, rows 2..199 from x = 307), FAST 4 (kodim01, 03, 16, 23).

#### PLN1 mode 3 (experiment): rows padded to 16 with z = 0, blocks never straddle rows

| | gray 17 | Kodak 24 |
|---|---|---|
| mode 2 (default) | 4.5268 | 4.5741 |
| mode 3, padding coded as zeros in the high streams | 4.5535 (+0.59%) | 4.6223 (+1.05%) |
| mode 3, padding not coded (only real symbols of a row's last block) | 4.5339 (+0.16%) | 4.5808 (+0.15%) |
| same, L of a partial block = mean over the real symbols | 4.5427 | 4.6000 |

Speed the same within noise. Kept as -m 3; mode 2 stays the default.
