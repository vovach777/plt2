// ctx4d_avx2.cpp - AVX2 encoder kernel of the planned HLC5 4-D context + symbol (16 px per step, raster, no branches).
// Spec and bit-exact reference: ctx4d_ref.cpp (same constants, borders, cell layout, CSV group map).
// Based on a second-opinion AVX2 draft, fixed: aligned buffers (the draft used aligned loads on 16-byte-aligned
// std::vector + 32 B), a row tail that never writes past the row, pre-scaled q instead of three pmullw per cell,
// cheaper tiny sign (2 ga + sign(gb) | 1). Group lookup: AVX2 gather (2 x vpgatherdd per 16 px) or, -DLUT_SCALAR,
// a scalar loop over the 16 lanes (what an SSE build would do) - both are timed.
// build:  g++ -O3 -mavx2 -std=c++17 ctx4d_avx2.cpp -o ctx4d_avx2        (SIMDe: include simde/x86/avx2.h instead)
// usage:  ctx4d_avx2 image.ppm groups.csv [reps]
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
alignas(64) static uint8_t G[2][1029 + 7];               // +7: a 4-byte gather at the last cell stays inside

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

// ---------------- scalar reference (identical to ctx4d_ref.cpp encoder) ----------------
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

// ---------------- AVX2 ----------------
static void rct_avx2(const uint8_t* __restrict rgb, int16_t* __restrict Y, int16_t* __restrict U, int16_t* __restrict V, size_t N) {
    // 8 px = 24 bytes per 128-bit half: two overlapping 16-byte loads at +0 and +12, pshufb to u16 lanes
    const __m128i rl = _mm_setr_epi8(0,-1,3,-1,6,-1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1), rh = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,0,-1,3,-1,6,-1,9,-1);
    const __m128i gl = _mm_setr_epi8(1,-1,4,-1,7,-1,10,-1,-1,-1,-1,-1,-1,-1,-1,-1), gh = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,1,-1,4,-1,7,-1,10,-1);
    const __m128i bl = _mm_setr_epi8(2,-1,5,-1,8,-1,11,-1,-1,-1,-1,-1,-1,-1,-1,-1), bh = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,2,-1,5,-1,8,-1,11,-1);
    size_t i = 0;
    for (; i + 17 <= N; i += 16) {                                     // last load reads bytes up to 3*(i+8)+12+15 < 3*N
        const uint8_t* p = rgb + 3 * i;
        __m128i a0 = _mm_loadu_si128((const __m128i*)p), a1 = _mm_loadu_si128((const __m128i*)(p + 12));
        __m128i b0 = _mm_loadu_si128((const __m128i*)(p + 24)), b1 = _mm_loadu_si128((const __m128i*)(p + 36));
        __m256i R = _mm256_set_m128i(_mm_or_si128(_mm_shuffle_epi8(b0, rl), _mm_shuffle_epi8(b1, rh)), _mm_or_si128(_mm_shuffle_epi8(a0, rl), _mm_shuffle_epi8(a1, rh)));
        __m256i Gc = _mm256_set_m128i(_mm_or_si128(_mm_shuffle_epi8(b0, gl), _mm_shuffle_epi8(b1, gh)), _mm_or_si128(_mm_shuffle_epi8(a0, gl), _mm_shuffle_epi8(a1, gh)));
        __m256i B = _mm256_set_m128i(_mm_or_si128(_mm_shuffle_epi8(b0, bl), _mm_shuffle_epi8(b1, bh)), _mm_or_si128(_mm_shuffle_epi8(a0, bl), _mm_shuffle_epi8(a1, bh)));
        _mm256_storeu_si256((__m256i*)(Y + i), _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(R, B), _mm256_add_epi16(Gc, Gc)), 2));
        _mm256_storeu_si256((__m256i*)(U + i), _mm256_sub_epi16(B, Gc));
        _mm256_storeu_si256((__m256i*)(V + i), _mm256_sub_epi16(R, Gc));
    }
    for (; i < N; i++) { int R = rgb[3 * i], Gc = rgb[3 * i + 1], B = rgb[3 * i + 2]; Y[i] = (int16_t)((R + 2 * Gc + B) >> 2); U[i] = (int16_t)(B - Gc); V[i] = (int16_t)(R - Gc); }
}

static inline __m256i q7v(__m256i g, __m256i e0, __m256i e1, __m256i e2) {   // e* = edge - 1; returns q in -3..3
    __m256i a = _mm256_abs_epi16(g);
    __m256i nq = _mm256_add_epi16(_mm256_add_epi16(_mm256_cmpgt_epi16(a, e0), _mm256_cmpgt_epi16(a, e1)), _mm256_cmpgt_epi16(a, e2));
    return _mm256_sign_epi16(nq, _mm256_sub_epi16(_mm256_setzero_si256(), g));
}

template <bool GATHER, int SH>      // SH = 16 - plane bits, a compile-time shift (SIMDe without AVX2 loops on run-time shifts)
static void encode_plane_k(const int16_t* __restrict P, uint16_t* __restrict out, int W, int H, int k) {
    const int t = k ? 1 : 0; constexpr int sh = SH;
    const __m256i e0 = _mm256_set1_epi16(EDGE[t][0] - 1), e1 = _mm256_set1_epi16(EDGE[t][1] - 1), e2 = _mm256_set1_epi16(EDGE[t][2] - 1);
    const __m256i v0 = _mm256_set1_epi16(VT[t][0] - 1), v1 = _mm256_set1_epi16(VT[t][1] - 1);
    const __m256i one = _mm256_set1_epi16(1), sb = _mm256_set1_epi16(t * 8), m8 = _mm256_set1_epi32(0xFF);
    const uint8_t* Gt = G[t];
    const int L = 32, Wp = (W + 15) & ~15, stride = Wp + 2 * L;                 // 32 elements (64 B) halo on each side
    int16_t* buf = (int16_t*)simd_alloc(64, sizeof(int16_t) * stride * 4);
    memset(buf, 0, sizeof(int16_t) * stride * 4);
    int16_t *Pu = buf + L, *Pc = buf + stride + L, *Zu = buf + 2 * stride + L, *Zc = buf + 3 * stride + L;
    alignas(32) uint16_t tmp[16];
    for (int y = 0; y < H; y++) {
        memcpy(Pc, P + (size_t)y * W, W * sizeof(int16_t));
        if (y == 0) { memset(Pu - 1, 0, (Wp + 2) * sizeof(int16_t)); memset(Zu - 1, 0, (Wp + 2) * sizeof(int16_t)); Pc[-1] = 0; }
        else { Pc[-1] = Pu[0]; Pu[-1] = Pu[0]; Pu[W] = Pu[W - 1]; Zu[-1] = 0; Zu[W] = 0; }
        uint16_t* orow = out + (size_t)y * W;
        for (int x = 0; x < W; x += 16) {
            __m256i vx = _mm256_load_si256((const __m256i*)(Pc + x)), va = _mm256_loadu_si256((const __m256i*)(Pc + x - 1));
            __m256i vb = _mm256_load_si256((const __m256i*)(Pu + x)), vc = _mm256_loadu_si256((const __m256i*)(Pu + x - 1));
            __m256i vd = _mm256_loadu_si256((const __m256i*)(Pu + x + 1));
            __m256i ga = _mm256_sub_epi16(vc, va), gb = _mm256_sub_epi16(vc, vb), gd = _mm256_sub_epi16(vd, vb);
            __m256i qa = q7v(ga, e0, e1, e2), qb = q7v(gb, e0, e1, e2), qd = q7v(gd, e0, e1, e2);
            // cell = ((3 - qa) * 7 + qb + 3) * 21 + (qd + 3) * 3 + level = 21 * (24 - 7 qa + qb) + 3 qd + 9 + level
            __m256i sec = _mm256_sub_epi16(_mm256_add_epi16(qb, _mm256_set1_epi16(24)), _mm256_sub_epi16(_mm256_slli_epi16(qa, 3), qa));
            __m256i zl = _mm256_loadu_si256((const __m256i*)(Zu + x - 1)), zc = _mm256_load_si256((const __m256i*)(Zu + x));
            __m256i zr = _mm256_loadu_si256((const __m256i*)(Zu + x + 1)), vv = _mm256_max_epi16(_mm256_max_epi16(zl, zc), zr);
            __m256i lev = _mm256_sub_epi16(_mm256_setzero_si256(), _mm256_add_epi16(_mm256_cmpgt_epi16(vv, v0), _mm256_cmpgt_epi16(vv, v1)));
            __m256i sec21 = _mm256_add_epi16(_mm256_slli_epi16(sec, 4), _mm256_add_epi16(_mm256_slli_epi16(sec, 2), sec));        // 21 sec
            __m256i qd3 = _mm256_add_epi16(_mm256_add_epi16(qd, qd), qd);
            __m256i cell = _mm256_add_epi16(_mm256_add_epi16(sec21, qd3), _mm256_add_epi16(lev, _mm256_set1_epi16(9)));
            __m256i grp;
            if (GATHER) {
                __m256i lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(cell)), hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(cell, 1));
                __m256i glo = _mm256_and_si256(_mm256_i32gather_epi32((const int*)Gt, lo, 1), m8), ghi = _mm256_and_si256(_mm256_i32gather_epi32((const int*)Gt, hi, 1), m8);
                grp = _mm256_permute4x64_epi64(_mm256_packus_epi32(glo, ghi), _MM_SHUFFLE(3, 1, 2, 0));
            } else {
                _mm256_store_si256((__m256i*)tmp, cell);
                for (int i = 0; i < 16; i++) tmp[i] = Gt[tmp[i]];
                grp = _mm256_load_si256((const __m256i*)tmp);
            }
            __m256i lo = _mm256_min_epi16(ga, gb), hi = _mm256_max_epi16(ga, gb);
            __m256i f = _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(ga, gb), lo), hi);
            __m256i s = _mm256_or_si256(_mm256_add_epi16(_mm256_add_epi16(ga, ga), _mm256_sign_epi16(one, gb)), one);   // 2ga + sign(gb) | 1
            __m256i e = _mm256_sign_epi16(_mm256_add_epi16(_mm256_sub_epi16(vx, vc), f), s);
            e = _mm256_srai_epi16(_mm256_slli_epi16(e, sh), sh);
            __m256i z = _mm256_xor_si256(_mm256_add_epi16(e, e), _mm256_srai_epi16(e, 15));
            _mm256_store_si256((__m256i*)(Zc + x), z);
            __m256i pk = _mm256_or_si256(_mm256_slli_epi16(z, 4), _mm256_add_epi16(sb, grp));
            if (x + 16 <= W) _mm256_storeu_si256((__m256i*)(orow + x), pk);
            else { _mm256_store_si256((__m256i*)tmp, pk); memcpy(orow + x, tmp, (W - x) * 2); }
        }
        std::swap(Pu, Pc); std::swap(Zu, Zc);
    }
    simd_free(buf);
}

template <bool GATHER>
static void encode_plane_avx2(const int16_t* P, uint16_t* out, int W, int H, int k) {
    if (k) encode_plane_k<GATHER, 7>(P, out, W, H, k); else encode_plane_k<GATHER, 8>(P, out, W, H, k);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s image.ppm groups.csv [reps]\n", argv[0]); return 1; }
    if (!load_groups(argv[2])) { fprintf(stderr, "bad groups CSV\n"); return 1; }
    int reps = argc > 3 ? atoi(argv[3]) : 20;
    FILE* fp = fopen(argv[1], "rb"); int W, H, mx;
    if (!fp || fscanf(fp, "P6 %d %d %d", &W, &H, &mx) != 3) { fprintf(stderr, "need a binary P6 PPM\n"); return 1; }
    fgetc(fp); size_t N = (size_t)W * H; std::vector<uint8_t> rgb(N * 3);
    if (fread(rgb.data(), 1, rgb.size(), fp) != rgb.size()) return 1;
    fclose(fp);
    std::vector<int16_t> P[3]; for (auto& p : P) p.resize(N);
    std::vector<uint16_t> ref(N * 3), avx(N * 3);
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    double best[4] = {1e30, 1e30, 1e30, 1e30};
    for (int r = 0; r < reps; r++) {
        auto t0 = now(); rct_avx2(rgb.data(), P[0].data(), P[1].data(), P[2].data(), N);
        auto t1 = now(); for (int k = 0; k < 3; k++) encode_plane_avx2<true>(P[k].data(), avx.data() + k * N, W, H, k);
        auto t2 = now(); for (int k = 0; k < 3; k++) encode_plane_avx2<false>(P[k].data(), avx.data() + k * N, W, H, k);
        auto t3 = now(); best[0] = std::min(best[0], ms(t0, t1)); best[1] = std::min(best[1], ms(t1, t2)); best[2] = std::min(best[2], ms(t2, t3));
        if (r < 2) { auto t4 = now(); for (int k = 0; k < 3; k++) encode_plane_ref(P[k].data(), ref.data() + k * N, W, H, k); best[3] = std::min(best[3], ms(t4, now())); }
    }
    for (int k = 0; k < 3; k++) encode_plane_avx2<true>(P[k].data(), avx.data() + k * N, W, H, k);
    // RCT check against scalar
    size_t bad_rct = 0;
    for (size_t i = 0; i < N; i++) { int R = rgb[3 * i], Gc = rgb[3 * i + 1], B = rgb[3 * i + 2]; bad_rct += P[0][i] != ((R + 2 * Gc + B) >> 2) || P[1][i] != B - Gc || P[2][i] != R - Gc; }
    size_t bad = 0; for (size_t i = 0; i < N * 3; i++) bad += ref[i] != avx[i];
    std::vector<uint16_t> sc(N * 3); for (int k = 0; k < 3; k++) encode_plane_avx2<false>(P[k].data(), sc.data() + k * N, W, H, k);
    size_t bad2 = 0; for (size_t i = 0; i < N * 3; i++) bad2 += ref[i] != sc[i];
    double mp = N / 1e6;
    printf("%dx%d (%.2f Mpx), best of %d, %s:\n", W, H, mp, reps, CTX4D_SIMD);
    printf("  RCT AVX2                      %7.3f ms  %6.2f ns/px\n", best[0], best[0] * 1e6 / N);
    printf("  context+symbol AVX2, gather   %7.3f ms  %6.2f ns/px (3 planes)\n", best[1], best[1] * 1e6 / N);
    printf("  context+symbol AVX2, scalar G %7.3f ms  %6.2f ns/px (3 planes)\n", best[2], best[2] * 1e6 / N);
    printf("  scalar reference              %7.3f ms  %6.2f ns/px (3 planes)\n", best[3], best[3] * 1e6 / N);
    printf("  bit-exact vs reference: gather %s (%zu), scalar-G %s (%zu); RCT %s (%zu)\n", bad ? "FAIL" : "OK", bad, bad2 ? "FAIL" : "OK", bad2, bad_rct ? "FAIL" : "OK", bad_rct);
    return bad || bad2 || bad_rct ? 2 : 0;
}
