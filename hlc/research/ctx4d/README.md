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

Input: binary P6 PPM, width ≤ 32700.

## Build

With AVX2 the intrinsics are native; without AVX2 (Ivy Bridge, ARM) the same code goes through
[SIMDe](https://github.com/simd-everywhere/simde) — `simd.h` picks it, the output line says which one ran.

CMake (downloads SIMDe v0.8.2 unless `-DSIMDE_DIR=...` points to a checkout):

    cmake -B build -DCTX4D_ARCH=native        # or avx2 / ivybridge / ...
    cmake --build build --config Release

By hand (SIMDe is header-only: `git clone --depth 1 -b v0.8.2 https://github.com/simd-everywhere/simde`):

    g++ -O3 -march=native -std=c++17 -I simde ctx4d_enc.cpp -o ctx4d_enc
    g++ -O3 -march=native -std=c++17 -I simde ctx4d_dec.cpp -o ctx4d_dec

`-DFORCE_SIMDE` (CMake: `-DCTX4D_FORCE_SIMDE=ON`) runs SIMDe on an AVX2 machine too, to see its cost.
MSVC: `/arch:AVX2` is set; without it SIMDe is used.

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
