// ctx4d_dec_avx2.cpp - AVX2 wavefront DECODER of the planned HLC5 4-D context (spec: ctx4d_ref.cpp).
// Band of 16 rows, one lane per row, skew t = x + 2k: at step t lane k holds pixel (x = t - 2k, y = y0 + k).
// Every neighbour is from an older step (no dependence inside the vector):
//      a = (x-1, y)   = P(t-1)[k]          d = (x+1, y-1) = P(t-1)[k-1]
//      b = (x,   y-1) = P(t-2)[k-1]        c = (x-1, y-1) = P(t-3)[k-1]
// lane 0 takes b, c, d from the last row of the band above (line buffer). Shifting a vector by one lane gives d;
// b(t) = d(t-1) and c(t) = b(t-1) are reused, so one lane shift per step for pixels and one for symbols (the top-row
// symbols for v = max3 follow the same pattern). Borders as the spec: x = 0: a = c = b, zl = 0; x = W-1: d = b,
// zr = 0; first band: the line buffers are zero (b = c = d = 0, all top symbols 0).
// Symbols: 16 streams, inside a stream in decoder order (band, t, lane); the group of a lane picks the stream.
// The test encodes with the scalar reference, scatters into streams in wave order, decodes, compares every pixel.
// build:  g++ -O3 -mavx2 -std=c++17 ctx4d_dec_avx2.cpp -o ctx4d_dec_avx2
// usage:  ctx4d_dec_avx2 image.ppm groups.csv [reps]
#include <cmath>
#include "simd.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>

static const int EDGE[2][3] = {{1, 3, 10}, {1, 3, 7}};
static const int VT[2][2]   = {{8, 32}, {3, 8}};
static uint8_t G[2][1029];
static const int LN = 16;                                   // lanes = rows per band

static bool load_groups(const char* path) {
    FILE* fp = fopen(path, "r"); if (!fp) return false;
    char line[512]; int n[2] = {0, 0};
    if (!fgets(line, sizeof line, fp)) { fclose(fp); return false; }
    while (fgets(line, sizeof line, fp)) {
        char plane[8]; int t1, t2, groups, sector, dcell, level, group; long samples;
        if (sscanf(line, "%7[^,],%d,%d,%d,%d,%d,%d,%d,%ld", plane, &t1, &t2, &groups, &sector, &dcell, &level, &group, &samples) != 9) continue;
        if (groups != 8) continue;
        int t = plane[0] == 'Y' ? 0 : 1; G[t][((sector - 1) * 7 + dcell) * 3 + level] = (uint8_t)group; n[t]++;
    }
    fclose(fp); return n[0] == 1029 && n[1] == 1029;
}

// ---------------- scalar reference encoder (as ctx4d_ref.cpp): packed (z << 4) | stream in raster order ----------
static int q7s(int x, int t) { int a = x < 0 ? -x : x, q = (a >= EDGE[t][0]) + (a >= EDGE[t][1]) + (a >= EDGE[t][2]); return x < 0 ? -q : q; }
static void encode_plane_ref(const int16_t* P, uint16_t* out, int W, int H, int k) {
    int t = k ? 1 : 0, nb = k ? 9 : 8; std::vector<int> Zup(W, 0), Zc(W, 0);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            auto at = [&](int xx, int yy) { return (int)P[(size_t)yy * W + xx]; };
            int b = y ? at(x, y - 1) : 0, a = x ? at(x - 1, y) : b, c = (x && y) ? at(x - 1, y - 1) : b;
            int d = y ? (x + 1 < W ? at(x + 1, y - 1) : b) : 0, ga = c - a, gb = c - b, gd = d - b;
            int sector = (3 - q7s(ga, t)) * 7 + (q7s(gb, t) + 3), dcell = q7s(gd, t) + 3, v = 0;
            if (y) { v = std::max({x ? Zup[x - 1] : 0, Zup[x], x + 1 < W ? Zup[x + 1] : 0}); }
            int level = (v >= VT[t][0]) + (v >= VT[t][1]), cell = (sector * 7 + dcell) * 3 + level;
            int s = ga ? (ga > 0 ? 1 : -1) : (gb ? (gb > 0 ? 1 : -1) : 1);
            int f = std::clamp(ga + gb, std::min(ga, gb), std::max(ga, gb));
            int m = 1 << nb, e = (s * (at(x, y) - c + f)) & (m - 1); e = e >= m / 2 ? e - m : e;
            int z = e >= 0 ? 2 * e : -2 * e - 1; Zc[x] = z; out[(size_t)y * W + x] = (uint16_t)((z << 4) | (t * 8 + G[t][cell]));
        }
        std::swap(Zup, Zc);
    }
}
// scatter the raster symbols into the streams in decoder order (band, t, lane)
static void scatter_wave(const uint16_t* packed, int W, int H, std::vector<uint16_t>* st) {
    for (int y0 = 0; y0 < H; y0 += LN)
        for (int t = 0; t < W + 2 * (LN - 1); t++)
            for (int k = 0; k < LN; k++) {
                int x = t - 2 * k, y = y0 + k;
                if (x < 0 || x >= W || y >= H) continue;
                uint16_t p = packed[(size_t)y * W + x]; st[p & 15].push_back(p >> 4);
            }
}

// ---------------- AVX2 wavefront decoder ----------------
static inline __m256i q7v(__m256i g, __m256i e0, __m256i e1, __m256i e2) {
    __m256i a = _mm256_abs_epi16(g);
    __m256i nq = _mm256_add_epi16(_mm256_add_epi16(_mm256_cmpgt_epi16(a, e0), _mm256_cmpgt_epi16(a, e1)), _mm256_cmpgt_epi16(a, e2));
    return _mm256_sign_epi16(nq, _mm256_sub_epi16(_mm256_setzero_si256(), g));
}
// lane k <- v[k-1], lane 0 <- ins
static inline __m256i shl1(__m256i v, int16_t ins) {
    __m256i s = _mm256_alignr_epi8(v, _mm256_permute2x128_si256(v, v, 0x08), 14);
    return _mm256_or_si256(s, _mm256_zextsi128_si256(_mm_cvtsi32_si128((uint16_t)ins)));
}

struct Streams { const uint16_t* p[16]; };

template <int SH>                   // SH = 16 - plane bits, compile-time (SIMDe without AVX2 loops on run-time shifts)
static void decode_plane_k(int16_t* __restrict out, int W, int H, int k, Streams& S) {
    const int t8 = k ? 1 : 0; constexpr int sh = SH;
    const __m256i e0 = _mm256_set1_epi16(EDGE[t8][0] - 1), e1 = _mm256_set1_epi16(EDGE[t8][1] - 1), e2 = _mm256_set1_epi16(EDGE[t8][2] - 1);
    const __m256i v0 = _mm256_set1_epi16(VT[t8][0] - 1), v1 = _mm256_set1_epi16(VT[t8][1] - 1), one = _mm256_set1_epi16(1);
    const __m256i lane2k = _mm256_setr_epi16(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m256i lanek = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m256i vW = _mm256_set1_epi16((int16_t)W), vWm1 = _mm256_set1_epi16((int16_t)(W - 1)), m1 = _mm256_set1_epi16(-1);
    const uint8_t* Gt = G[t8];
    const int T = W + 2 * (LN - 1), PAD = 40;
    std::vector<int16_t> topP(W + 2 * PAD, 0), topZ(W + 2 * PAD, 0);        // last row of the band above
    int16_t* tp = topP.data() + PAD; int16_t* tz = topZ.data() + PAD;
    int16_t* wave = (int16_t*)simd_alloc(32, sizeof(int16_t) * LN * T);    // pixels in wave layout [t][lane]
    int16_t* wavez = (int16_t*)simd_alloc(32, sizeof(int16_t) * LN * T);
    alignas(32) int16_t cell[16], zz[16];
    for (int y0 = 0; y0 < H; y0 += LN) {
        const __m256i vrows = _mm256_set1_epi16((int16_t)std::min(LN, H - y0));
        const __m256i lanes_ok = _mm256_cmpgt_epi16(vrows, lanek);
        __m256i P1 = _mm256_setzero_si256(), Z1 = P1;                          // pixel / symbol vectors of step t-1
        // d(t-1), b(t-1) and their symbol twins; at t = 0 only lane 0 matters: b(0) = top[0], c(0) = top[-1]
        __m256i Dp = shl1(P1, tp[0]), Bp = shl1(P1, tp[-1]), Dz = shl1(Z1, tz[0]), Bz = shl1(Z1, tz[-1]);
        for (int t = 0; t < T; t++) {
            __m256i d = shl1(P1, tp[t + 1]), b = Dp, c = Bp;                   // raw neighbours (before border fixes)
            __m256i zr = shl1(Z1, tz[t + 1]), zc = Dz, zl = Bz;
            Bp = b; Dp = d; Bz = zc; Dz = zr;                                  // for the next step: b(t+1) = d(t), c(t+1) = b(t)
            __m256i xv = _mm256_sub_epi16(_mm256_set1_epi16((int16_t)t), lane2k);
            __m256i x0 = _mm256_cmpeq_epi16(xv, _mm256_setzero_si256()), xr = _mm256_cmpeq_epi16(xv, vWm1);
            __m256i valid = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi16(xv, m1), _mm256_cmpgt_epi16(vW, xv)), lanes_ok);
            __m256i a = _mm256_blendv_epi8(P1, b, x0);
            c = _mm256_blendv_epi8(c, b, x0); d = _mm256_blendv_epi8(d, b, xr);
            zl = _mm256_andnot_si256(x0, zl); zr = _mm256_andnot_si256(xr, zr);
            __m256i ga = _mm256_sub_epi16(c, a), gb = _mm256_sub_epi16(c, b), gd = _mm256_sub_epi16(d, b);
            __m256i qa = q7v(ga, e0, e1, e2), qb = q7v(gb, e0, e1, e2), qd = q7v(gd, e0, e1, e2);
            __m256i sec = _mm256_sub_epi16(_mm256_add_epi16(qb, _mm256_set1_epi16(24)), _mm256_sub_epi16(_mm256_slli_epi16(qa, 3), qa));
            __m256i vv = _mm256_max_epi16(_mm256_max_epi16(zl, zc), zr);
            __m256i lev = _mm256_sub_epi16(_mm256_setzero_si256(), _mm256_add_epi16(_mm256_cmpgt_epi16(vv, v0), _mm256_cmpgt_epi16(vv, v1)));
            __m256i cl = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_slli_epi16(sec, 4), _mm256_slli_epi16(sec, 2)), sec),
                                          _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(qd, qd), qd), _mm256_add_epi16(lev, _mm256_set1_epi16(9))));
            // invalid lanes -> cell 0 (any valid index), they read no symbol
            _mm256_store_si256((__m256i*)cell, _mm256_and_si256(cl, valid));
            unsigned vm = (unsigned)_mm256_movemask_epi8(valid);
            for (int i = 0; i < 16; i++) zz[i] = (vm >> (2 * i)) & 1 ? (int16_t)*S.p[t8 * 8 + Gt[cell[i]]]++ : 0;
            __m256i z = _mm256_load_si256((const __m256i*)zz);
            __m256i lo = _mm256_min_epi16(ga, gb), hi = _mm256_max_epi16(ga, gb);
            __m256i f = _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(ga, gb), lo), hi);
            __m256i s = _mm256_or_si256(_mm256_add_epi16(_mm256_add_epi16(ga, ga), _mm256_sign_epi16(one, gb)), one);
            __m256i e = _mm256_xor_si256(_mm256_srli_epi16(z, 1), _mm256_sub_epi16(_mm256_setzero_si256(), _mm256_and_si256(z, one)));
            __m256i val = _mm256_add_epi16(_mm256_sub_epi16(c, f), _mm256_sign_epi16(e, s));
            val = _mm256_srai_epi16(_mm256_slli_epi16(val, sh), sh);           // wrap; Y back to 0..255 below
            if (!k) val = _mm256_and_si256(val, _mm256_set1_epi16(0xFF));
            _mm256_store_si256((__m256i*)(wave + (size_t)t * LN), val);
            _mm256_store_si256((__m256i*)(wavez + (size_t)t * LN), z);
            P1 = val; Z1 = z;
        }
        // unskew the band into the plane, keep its last row (pixels and symbols) for the next band
        int rows = std::min(LN, H - y0);
        for (int kk = 0; kk < rows; kk++) {
            int16_t* row = out + (size_t)(y0 + kk) * W; const int16_t* src = wave + 2 * kk * LN + kk;
            for (int x = 0; x < W; x++) row[x] = src[(size_t)x * LN];
        }
        const int16_t* srcp = wave + 2 * (LN - 1) * LN + (LN - 1); const int16_t* srcz = wavez + 2 * (LN - 1) * LN + (LN - 1);
        if (rows == LN) for (int x = 0; x < W; x++) { tp[x] = srcp[(size_t)x * LN]; tz[x] = srcz[(size_t)x * LN]; }
    }
    simd_free(wave); simd_free(wavez);
}

static void decode_plane_avx2(int16_t* out, int W, int H, int k, Streams& S) {
    if (k) decode_plane_k<7>(out, W, H, k, S); else decode_plane_k<8>(out, W, H, k, S);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s image.ppm groups.csv [reps]\n", argv[0]); return 1; }
    if (!load_groups(argv[2])) { fprintf(stderr, "bad groups CSV\n"); return 1; }
    int reps = argc > 3 ? atoi(argv[3]) : 10;
    FILE* fp = fopen(argv[1], "rb"); int W, H, mx;
    if (!fp || fscanf(fp, "P6 %d %d %d", &W, &H, &mx) != 3) { fprintf(stderr, "need a binary P6 PPM\n"); return 1; }
    fgetc(fp); size_t N = (size_t)W * H; std::vector<uint8_t> rgb(N * 3);
    if (fread(rgb.data(), 1, rgb.size(), fp) != rgb.size()) return 1;
    fclose(fp);
    if (W + 2 * LN > 32767) { fprintf(stderr, "width too large for 16-bit x lanes\n"); return 1; }
    std::vector<int16_t> P[3], D[3]; std::vector<uint16_t> packed(N), st[3][16];
    for (int k = 0; k < 3; k++) { P[k].resize(N); D[k].assign(N, -999); }
    for (size_t i = 0; i < N; i++) { int R = rgb[3 * i], Gc = rgb[3 * i + 1], B = rgb[3 * i + 2]; P[0][i] = (R + 2 * Gc + B) >> 2; P[1][i] = B - Gc; P[2][i] = R - Gc; }
    for (int k = 0; k < 3; k++) { encode_plane_ref(P[k].data(), packed.data(), W, H, k); scatter_wave(packed.data(), W, H, st[k]); }
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < 3; k++) { Streams S; for (int s = 0; s < 16; s++) S.p[s] = st[k][s].data(); decode_plane_avx2(D[k].data(), W, H, k, S); }
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    size_t bad = 0; for (int k = 0; k < 3; k++) for (size_t i = 0; i < N; i++) if (D[k][i] != P[k][i]) { if (bad < 5) printf("MISMATCH plane %d x %zu y %zu: %d vs %d\n", k, i % W, i / W, D[k][i], P[k][i]); bad++; }
    printf("[%s] %dx%d (%.2f Mpx): AVX2 wavefront decode (context + symbol + reconstruct, 3 planes) %.3f ms = %.2f ns/px; %s (%zu bad)\n",
           CTX4D_SIMD, W, H, N / 1e6, best, best * 1e6 / N, bad ? "FAIL" : "round trip OK", bad);
    return bad ? 2 : 0;
}
