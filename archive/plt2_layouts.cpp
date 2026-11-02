// PLT2 — PALIC-tiny v2: lossless RGB, adaptive multi-symbol range coder + raw bypass bits,
// 6 logical streams per strip interleaved in ONE buffer by deterministic slot reservation.
//
//  RCT (V=R-G, U=B-G, Y=G+((U+V)>>2)) -> MED -> context = per-component corpus cluster (64) [x Enb (9)]
//  -> alphabet A31 (PALIC-2) or A15 (0, +-7 power-of-two classes) -> letter: adaptive CDF + range coder,
//     low bits of the class (bypass): plain bit stream, 64-bit words.
//
//  Range coder, HYBRID carry: low 64 / range 32 / byte renorm at 2^24. Carries are absorbed by 4 pending bytes held
//  in low; after every shift, if all 4 pending bytes are 0xFF, range is trimmed to 2^32 - low_window (Subbotin
//  guillotine) so the window can never carry into them. => every emitted byte is final. The decoder tracks low too.
//
//  Strip buffer: per strip the streams RC_Y, RC_U, RC_V, BIT_Y, BIT_U, BIT_V share one byte buffer and one counter:
//   - RC encoder reserves 4 bytes at start and 1 byte per renorm shift (emits it 4 shifts later, FIFO of slots);
//     RC decoder reads 4 bytes at start and 1 byte per shift, buf[next++].
//   - bit writer reserves an 8-byte slot when the FIRST bit of a new word is written, stores the word when full;
//     bit reader reserves the next 8-byte slot when it needs the first bit of a new word.
//   Pixel order Y, U, V; per component: letter (RC) then bypass bits — identical on both sides.
//   => the strip buffer is the final layout, nothing is moved or glued except the strip concatenation.
//
//  File: "PLT2" | u32 W | u32 H | u8 model (0 cl64, 1 cl64 x Enb) | u8 alphabet (0 A31, 1 A15) | u16 0 | u32 len[N]
//        | strips. Strips: N = max(1, W/512), width = (W/N) rounded down to a multiple of 64, last takes the tail.
//
//  Build: clang++ -O3 -march=native -std=c++17 -pthread plt2.cpp -o plt2
//         SIMD path picked at compile time: AVX-512BW, AVX2, or scalar (-DPLT_SCALAR). All give identical streams.
//  Usage: plt2 c in.ppm out.plt [-t N] [-m 0|1] [-a 31|15]   plt2 d in.plt out.ppm [-t N]
//         plt2 bench [-t N] [-m 0|1] [-a 31|15] [-r R] [-csv f] (file.ppm | dir)...
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
#if defined(PLT_SIMDE)                  // SIMDe: AVX2 (PLT_SIMDE=1) or AVX-512 (=2) intrinsics emulated on any target
#include <simde/x86/avx512.h>
#define SIMD PLT_SIMDE
// thin shim (SIMDe native aliases clash with clang 18 headers; mask_sub_epi16 is missing in SIMDe)
#define __m128i simde__m128i
#define __m256i simde__m256i
#define __m512i simde__m512i
#define __mmask16 simde__mmask16
#define __mmask32 simde__mmask32
#define _mm_cvtsi32_si128 simde_mm_cvtsi32_si128
#define _mm256_load_si256 simde_mm256_load_si256
#define _mm256_store_si256 simde_mm256_store_si256
#define _mm256_setr_epi16 simde_mm256_setr_epi16
#define _mm256_set1_epi16 simde_mm256_set1_epi16
#define _mm256_add_epi16 simde_mm256_add_epi16
#define _mm256_sub_epi16 simde_mm256_sub_epi16
#define _mm256_srl_epi16 simde_mm256_srl_epi16
#define _mm256_cmpgt_epi16 simde_mm256_cmpgt_epi16
#define _mm256_cmpeq_epi16 simde_mm256_cmpeq_epi16
#define _mm256_max_epu16 simde_mm256_max_epu16
#define _mm256_blendv_epi8 simde_mm256_blendv_epi8
#define _mm256_movemask_epi8 simde_mm256_movemask_epi8
#define _mm256_mask_add_epi16 simde_mm256_mask_add_epi16
#define _mm256_cmple_epu16_mask simde_mm256_cmple_epu16_mask
#define _mm512_load_si512 simde_mm512_load_si512
#define _mm512_store_si512 simde_mm512_store_si512
#define _mm512_set1_epi16 simde_mm512_set1_epi16
#define _mm512_sub_epi16 simde_mm512_sub_epi16
#define _mm512_srl_epi16 simde_mm512_srl_epi16
#define _mm512_mask_add_epi16 simde_mm512_mask_add_epi16
#define _mm512_cmple_epu16_mask simde_mm512_cmple_epu16_mask
static inline simde__m512i plt_mask_sub512(simde__m512i src, simde__mmask32 k, simde__m512i a, simde__m512i b) { return simde_mm512_mask_blend_epi16(k, src, simde_mm512_sub_epi16(a, b)); }
static inline simde__m256i plt_mask_sub256(simde__m256i src, simde__mmask16 k, simde__m256i a, simde__m256i b) { return simde_mm256_mask_blend_epi16(k, src, simde_mm256_sub_epi16(a, b)); }
#define _mm512_mask_sub_epi16 plt_mask_sub512
#define _mm256_mask_sub_epi16 plt_mask_sub256
#elif !defined(PLT_SCALAR) && (defined(__AVX2__) || defined(__AVX512BW__))
#include <immintrin.h>
#endif
#if defined(PLT_SIMDE)
#elif defined(__AVX512BW__) && !defined(PLT_SCALAR)
#define SIMD 2
#elif defined(__AVX2__) && !defined(PLT_SCALAR)
#define SIMD 1
#else
#define SIMD 0
#endif
#include "plt2_tables.h"
#ifdef PLT_STRESS                       // test only: trim whenever the top pending byte is 0xFF (over-trimming is always safe)
#define TRIM_COND(low) (((low) >> 56) == 0xFF)
#else
#define TRIM_COND(low) (((low) >> 32) == 0xFFFFFFFFu)
#endif
#ifndef SPLIT
#define SPLIT 3      // 5: as 4, bypass as LZMA direct bits (no division); 4: ONE buffer per strip, 3 interleaved RC streams, bypass coded inside each RC (no bit streams); 3: ONE buffer per strip, 4 interleaved streams (RC Y,U,V + one shared bypass), slot reservation; 2: per strip 3 pure range buffers (Y,U,V) + 1 shared bypass buffer (pixel order); 1: buffer per (strip, component) with RC+bits; 0: one buffer per strip
#endif

namespace plt2 {
using namespace plt2_tab;
// start CDFs in use: copies of the compiled corpus tables, replaceable at run time (-T file, see train)
alignas(64) static uint16_t C31_1[3][64][32], C31_E[3][576][32], C15_1[3][64][16], C15_E[3][576][16];
static uint32_t tables_hash() {
    uint32_t h = 2166136261u; auto mix = [&](const void* p, size_t n) { auto* b = (const uint8_t*)p; for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 16777619u; } };
    mix(C31_1, sizeof C31_1); mix(C31_E, sizeof C31_E); mix(C15_1, sizeof C15_1); mix(C15_E, sizeof C15_E); return h;
}
static uint32_t g_default_hash = 0;
static void init_cdf_tables() {
    std::memcpy(C31_1, CDF1_31, sizeof C31_1); std::memcpy(C31_E, CDFE_31, sizeof C31_E);
    std::memcpy(C15_1, CDF1_15, sizeof C15_1); std::memcpy(C15_E, CDFE_15, sizeof C15_E);
    g_default_hash = tables_hash();
}

static int32_t G1[1024], G2[1024], G3[1024];
static uint8_t BW8[512];
static void init_static() {
    auto bw = [](int v) { return v > 0 ? 32 - __builtin_clz(unsigned(v)) : 0; };
    for (int g = -511; g <= 512; ++g) {
        int lv = bw(g < 0 ? -g : g) * ((g > 0) - (g < 0)) + 16;
        G1[g + 511] = lv * 1089; G2[g + 511] = lv * 33; G3[g + 511] = lv;
    }
    for (int v = 0; v < 512; ++v) BW8[v] = uint8_t(std::min(8, bw(v)));
}

// ------------------------------------------------------------------ alphabets
template <int A> struct Alpha;
template <> struct Alpha<31> {
    static constexpr int NL = 31, LANES = 32;
    static const uint16_t (*lo())[32] { return LO31; } static const uint8_t (*raw())[32] { return RAW31; }
    static const int8_t (*sign())[32] { return SIGN31; } static const uint16_t (*enc())[513] { return ENC31; }
    static const uint16_t* cdf(bool enb, int C) { return enb ? &C31_E[C][0][0] : &C31_1[C][0][0]; }
};
template <> struct Alpha<15> {
    static constexpr int NL = 15, LANES = 16;
    static const uint16_t (*lo())[16] { return LO15; } static const uint8_t (*raw())[16] { return RAW15; }
    static const int8_t (*sign())[16] { return SIGN15; } static const uint16_t (*enc())[513] { return ENC15; }
    static const uint16_t* cdf(bool enb, int C) { return enb ? &C15_E[C][0][0] : &C15_1[C][0][0]; }
};

// ------------------------------------------------------------------ adaptive CDF (total 2^15, cum[NL] = 32768)
constexpr int MINF = 2, R0 = 6, RMAX = 8;
template <int L> struct alignas(2 * L) Cdf { uint16_t c[L]; };   // 64 B (A31) or 32 B (A15), tightly packed like the tables
static_assert(sizeof(Cdf<16>) == 32 && sizeof(Cdf<32>) == 64, "cdf layout");
template <int A> struct Tgt { alignas(64) uint16_t le[Alpha<A>::LANES], gt[Alpha<A>::LANES]; };
static Tgt<31> T31; static Tgt<15> T15;
template <int A> static Tgt<A>& tgt() { if constexpr (A == 31) return T31; else return T15; }
static void init_targets() {
    for (int i = 0; i < 32; ++i) { T31.le[i] = uint16_t(std::min(i, 31) * MINF); T31.gt[i] = uint16_t(32768 - std::max(0, 31 - i) * MINF); }
    for (int i = 0; i < 16; ++i) { T15.le[i] = uint16_t(std::min(i, 15) * MINF); T15.gt[i] = uint16_t(32768 - std::max(0, 15 - i) * MINF); }
}
static inline int rate_of(uint16_t& cnt) {
    int r = R0 + (cnt > 15) + (cnt > 31) + (cnt > 63) + (cnt > 127) + (cnt > 255) + (cnt > 511);
    cnt += cnt < 1024; return r < RMAX ? r : RMAX;
}
template <int A>
static inline void cdf_update(Cdf<Alpha<A>::LANES>& d, uint16_t& cnt, int l) {
    constexpr int NL = Alpha<A>::NL; const Tgt<A>& T = tgt<A>();
    int rate = rate_of(cnt);
#if SIMD == 2
    if constexpr (A == 31) {
        __m512i c = _mm512_load_si512(d.c); __mmask32 gt = ~((2u << l) - 1); __m128i sh = _mm_cvtsi32_si128(rate);
        __m512i up = _mm512_srl_epi16(_mm512_sub_epi16(_mm512_load_si512(T.gt), c), sh);
        __m512i dn = _mm512_srl_epi16(_mm512_sub_epi16(c, _mm512_load_si512(T.le)), sh);
        c = _mm512_mask_add_epi16(c, gt, c, up); c = _mm512_mask_sub_epi16(c, ~gt, c, dn);
        _mm512_store_si512(d.c, c); return;
    }
#endif
#if SIMD == 2
    if constexpr (A == 15) {
        __m256i c = _mm256_load_si256((const __m256i*)d.c); __mmask16 gt = __mmask16(~((2u << l) - 1)); __m128i sh = _mm_cvtsi32_si128(rate);
        __m256i up = _mm256_srl_epi16(_mm256_sub_epi16(_mm256_load_si256((const __m256i*)T.gt), c), sh);
        __m256i dn = _mm256_srl_epi16(_mm256_sub_epi16(c, _mm256_load_si256((const __m256i*)T.le)), sh);
        c = _mm256_mask_add_epi16(c, gt, c, up); c = _mm256_mask_sub_epi16(c, __mmask16(~gt), c, dn);
        _mm256_store_si256((__m256i*)d.c, c); return;
    }
#endif
#if SIMD >= 1
    {
        const __m256i idx = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m128i sh = _mm_cvtsi32_si128(rate);
        for (int h = 0; h < Alpha<A>::LANES; h += 16) {
            __m256i c = _mm256_load_si256((const __m256i*)(d.c + h));
            __m256i gt = _mm256_cmpgt_epi16(_mm256_add_epi16(idx, _mm256_set1_epi16(short(h))), _mm256_set1_epi16(short(l)));
            __m256i up = _mm256_srl_epi16(_mm256_sub_epi16(_mm256_load_si256((const __m256i*)(T.gt + h)), c), sh);
            __m256i dn = _mm256_srl_epi16(_mm256_sub_epi16(c, _mm256_load_si256((const __m256i*)(T.le + h))), sh);
            c = _mm256_blendv_epi8(_mm256_sub_epi16(c, dn), _mm256_add_epi16(c, up), gt);
            _mm256_store_si256((__m256i*)(d.c + h), c);
        }
        return;
    }
#endif
    for (int i = 1; i <= l; ++i) d.c[i] = uint16_t(d.c[i] - (uint16_t(d.c[i] - T.le[i]) >> rate));
    for (int i = l + 1; i < NL; ++i) d.c[i] = uint16_t(d.c[i] + (uint16_t(T.gt[i] - d.c[i]) >> rate));
}
template <int A>
static inline int cdf_find(const Cdf<Alpha<A>::LANES>& d, uint32_t v) {
#if SIMD == 2
    if constexpr (A == 31) return __builtin_popcount(_mm512_cmple_epu16_mask(_mm512_load_si512(d.c), _mm512_set1_epi16(short(v)))) - 1;
    else return __builtin_popcount(_mm256_cmple_epu16_mask(_mm256_load_si256((const __m256i*)d.c), _mm256_set1_epi16(short(v)))) - 1;
#endif
#if SIMD >= 1
    {
        __m256i vv = _mm256_set1_epi16(short(v)); int n = 0;
        for (int h = 0; h < Alpha<A>::LANES; h += 16) {
            __m256i c = _mm256_load_si256((const __m256i*)(d.c + h));
            n += __builtin_popcount(unsigned(_mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_max_epu16(c, vv), vv))));
        }
        return n / 2 - 1;
    }
#endif
    const uint16_t* c = d.c; int l = 0;
    if constexpr (A == 31) l += (c[l + 16] <= v) ? 16 : 0;
    l += (c[l + 8] <= v) ? 8 : 0; l += (c[l + 4] <= v) ? 4 : 0; l += (c[l + 2] <= v) ? 2 : 0; l += (c[l + 1] <= v) ? 1 : 0;
    return l;
}

// ------------------------------------------------------------------ strip buffer + streams
constexpr uint32_t TOP = 1u << 24;
struct Buf {
    std::vector<uint8_t> b; uint32_t next = 0;
    inline uint32_t take(uint32_t n) { uint32_t s = next; next += n; if (__builtin_expect(next + 16 > b.size(), 0)) b.resize(b.size() * 2 + 64); return s; }
};
static inline void st64(uint8_t* p, uint64_t v) { std::memcpy(p, &v, 8); }
static inline uint64_t ld64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

struct RcEnc {
    uint64_t low = 0; uint32_t range = 0xFFFFFFFFu; int p = 0;
    uint32_t fifo[16]; uint32_t head = 0, tail = 0; uint64_t trims = 0, escapes = 0;
    void init(Buf& B) { for (int i = 0; i < 4; ++i) fifo[tail++ & 15] = B.take(1); }
    inline void put(Buf& B, uint32_t c, uint32_t f) {
        uint32_t r = range >> 15;
#ifdef PLT_DEBUG
        uint64_t o = low;
#endif
        low += uint64_t(c) * r; range = f * r;
#ifdef PLT_DEBUG
        escapes += low < o;                                                    // must stay 0
#endif
        while (range < TOP) {
            if (p == 4) B.b[fifo[head++ & 15]] = uint8_t(low >> 56); else ++p;
            fifo[tail++ & 15] = B.take(1);
            low <<= 8; range <<= 8;
            if (__builtin_expect(TRIM_COND(low), 0)) {             // guillotine
                uint64_t lim = (1ull << 32) - (low & 0xFFFFFFFFu); if (range > lim) { range = uint32_t(lim); ++trims; }
            }
        }
    }
    // n uniform bits without division: halve the range per bit (LZMA "direct bits"), renormalize once at the end
    inline void bits(Buf& B, uint32_t v, int n) {
        for (int i = n - 1; i >= 0; --i) { range >>= 1; if ((v >> i) & 1) low += range; }
        while (range < TOP) {
            if (p == 4) B.b[fifo[head++ & 15]] = uint8_t(low >> 56); else ++p;
            fifo[tail++ & 15] = B.take(1);
            low <<= 8; range <<= 8;
            if (__builtin_expect(TRIM_COND(low), 0)) { uint64_t lim = (1ull << 32) - (low & 0xFFFFFFFFu); if (range > lim) { range = uint32_t(lim); ++trims; } }
        }
    }
    void flush(Buf& B) { uint64_t l = low << (8 * (4 - p)); for (int i = 0; i < p + 4; ++i) { B.b[fifo[head++ & 15]] = uint8_t(l >> 56); l <<= 8; } }
};
struct BitW {
    uint64_t acc = 0; int fill = 0; uint32_t slot = 0; bool has = false;
    inline void put(Buf& B, uint32_t v, int n) {                              // n in 1..8
        if (!has) { slot = B.take(8); has = true; }
        acc |= uint64_t(v) << fill; fill += n;
        if (fill >= 64) {
            st64(&B.b[slot], acc); fill -= 64;
            if (fill) { slot = B.take(8); acc = uint64_t(v) >> (n - fill); } else { has = false; acc = 0; }
        }
    }
    void flush(Buf& B) { if (has) st64(&B.b[slot], acc); }
};
struct RcDec {
    uint64_t low = 0; uint32_t range = 0xFFFFFFFFu, code = 0;
    void init(const uint8_t* b, uint32_t& nx) { for (int i = 0; i < 4; ++i) code = (code << 8) | b[nx++]; }
    inline void step(uint32_t c, uint32_t f, uint32_t r, const uint8_t* b, uint32_t& nx) {
        code -= c * r; low += uint64_t(c) * r; range = f * r;
        while (range < TOP) {
            code = (code << 8) | b[nx++]; low <<= 8; range <<= 8;
            if (__builtin_expect(TRIM_COND(low), 0)) { uint64_t lim = (1ull << 32) - (low & 0xFFFFFFFFu); if (range > lim) range = uint32_t(lim); }
        }
    }
};
inline uint32_t rc_bits(RcDec& D, int n, const uint8_t* b, uint32_t& nx) {
    uint32_t v = 0;
    for (int i = 0; i < n; ++i) {                                   // branchless: bit = code >= half
        D.range >>= 1; uint32_t bit = D.code >= D.range;
        uint32_t m = 0u - bit; D.code -= D.range & m; D.low += D.range & m; v = (v << 1) | bit;
    }
    while (D.range < TOP) {
        D.code = (D.code << 8) | b[nx++]; D.low <<= 8; D.range <<= 8;
        if (__builtin_expect(TRIM_COND(D.low), 0)) { uint64_t lim = (1ull << 32) - (D.low & 0xFFFFFFFFu); if (D.range > lim) D.range = uint32_t(lim); }
    }
    return v;
}
struct BitR {
    uint64_t cur = 0; int avail = 0;
    inline uint32_t get(const uint8_t* b, uint32_t& nx, int n) {             // n in 1..8
        if (avail >= n) { uint32_t v = uint32_t(cur) & ((1u << n) - 1); cur >>= n; avail -= n; return v; }
        uint64_t w = ld64(b + nx); nx += 8;
        uint32_t v = uint32_t(cur | (w << avail)) & ((1u << n) - 1);         // avail < n <= 8: low bits from cur
        int k = n - avail; cur = w >> k; avail = 64 - k; return v;
    }
};

// ------------------------------------------------------------------ model / context
template <int A> struct Model {
    std::vector<Cdf<Alpha<A>::LANES>> cdf; std::vector<uint16_t> cnt;
    void init(int C, bool enb) { int n = enb ? 576 : 64; cdf.resize(n); cnt.assign(n, 0); std::memcpy(cdf.data(), Alpha<A>::cdf(enb, C), size_t(n) * Alpha<A>::LANES * 2); }
};
struct Ctx { int pred, ctx, s; };
template <int C, bool ENB>
static inline Ctx context(const int16_t* P, const int16_t* Q, const uint16_t* Ec, const uint16_t* Ep, int x, int y, int w) {
    constexpr int MID = C == 0 ? 128 : 0;
    int a, b, c, d, el = 0, et = 0;
    if (y == 0) { a = x ? P[x - 1] : MID; b = c = d = a; if (x) el = Ec[x - 1]; }
    else { b = Q[x]; if (x == 0) { a = c = b; } else { a = P[x - 1]; c = Q[x - 1]; el = Ec[x - 1]; } d = (x + 1 < w) ? Q[x + 1] : b; et = Ep[x]; }
    int mn = a < b ? a : b, mx = a < b ? b : a, pr = a + b - c; pr = pr < mn ? mn : (pr > mx ? mx : pr);
    int m = LUT[C][G1[a - c + 511] + G2[c - b + 511] + G3[d - b + 511]];
    return {pr, ENB ? (m & 63) * 9 + (el > et ? el : et) : (m & 63), (m & 128) ? -1 : 1};
}

// per-component working state (rows are 2-row rings, strip-local)
template <int A, int C, bool ENB>
struct Comp {
    static constexpr int G = C ? 1 : 0;
    Model<A> M; std::vector<int16_t> ring; std::vector<uint16_t> e0, e1; int w;     // u16, not u8: no char aliasing
    int16_t* P; const int16_t* Q; uint16_t* Ec; const uint16_t* Ep;
    explicit Comp(int w_) : ring(2 * size_t(w_)), e0(w_), e1(w_), w(w_) { M.init(C, ENB); }
    inline void row(int y) { P = ring.data() + (y & 1) * w; Q = y ? ring.data() + ((y - 1) & 1) * w : P; Ec = (y & 1) ? e1.data() : e0.data(); Ep = (y & 1) ? e0.data() : e1.data(); }
    __attribute__((always_inline)) inline void enc_px(Buf& B, RcEnc& E, BitW& Bw, int x, int y, bool ad) {
        constexpr int HALF = C ? 256 : 128, MASK = C ? 511 : 255;
        Ctx k = context<C, ENB>(P, Q, Ec, Ep, x, y, w);
        int e = ((P[x] - k.pred + HALF) & MASK) - HALF;
        Ec[x] = BW8[e < 0 ? -e : e];
        uint16_t q = Alpha<A>::enc()[G][k.s * e + 256]; int l = q & 31;
        auto& d = M.cdf[k.ctx];
        E.put(B, d.c[l], d.c[l + 1] - d.c[l]);           // cum[NL] = 32768 is stored
        if (ad) cdf_update<A>(d, M.cnt[k.ctx], l);
        if (int nb = Alpha<A>::raw()[G][l]) {
            if constexpr (SPLIT == 5) E.bits(B, q >> 5, nb);                                     // bypass = direct bits in RC (no division)
            else if constexpr (SPLIT == 4) E.put(B, uint32_t(q >> 5) << (15 - nb), 1u << (15 - nb));   // bypass = uniform pseudo-letter in RC
            else Bw.put(B, q >> 5, nb);
        }
    }
    __attribute__((always_inline)) inline void dec_px(const uint8_t* b, uint32_t& nx, RcDec& D, BitR& Br, const uint8_t* bb, uint32_t& nbx, int x, int y, bool ad) {
        Ctx k = context<C, ENB>(P, Q, Ec, Ep, x, y, w);
        auto& d = M.cdf[k.ctx];
        uint32_t r = D.range >> 15, v = D.code / r; v = v < 32767 ? v : 32767;
        int l = cdf_find<A>(d, v);
        D.step(d.c[l], d.c[l + 1] - d.c[l], r, b, nx);
        if (ad) cdf_update<A>(d, M.cnt[k.ctx], l);
        int a = Alpha<A>::lo()[G][l];
        if (int nb = Alpha<A>::raw()[G][l]) {
            if constexpr (SPLIT == 5) a += int(rc_bits(D, nb, b, nx));
            else if constexpr (SPLIT == 4) {
                uint32_t r2 = D.range >> 15, val = (D.code / r2) >> (15 - nb);
                D.step(val << (15 - nb), 1u << (15 - nb), r2, b, nx); a += int(val);
            } else a += int(Br.get(bb, nbx, nb));
        }
        Ec[x] = BW8[a];
        int vv = k.pred + k.s * Alpha<A>::sign()[G][l] * a;
        P[x] = int16_t(C == 0 ? (vv & 255) : (((vv + 256) & 511) - 256));
    }
};

// ------------------------------------------------------------------ strip coding
constexpr int NB = (SPLIT >= 3) ? 1 : SPLIT == 2 ? 4 : SPLIT ? 3 : 1;                                       // buffers per strip
template <int A, bool ENB>
static void enc_strip(const uint8_t* rgb, int W, int x0, int w, int H, std::vector<uint8_t>* out, uint64_t* trims, bool ad) {
    Buf B[NB]; for (auto& b : B) b.b.resize(size_t(w) * H / NB + 256);
    auto bf = [&](int C) -> Buf& { return B[(SPLIT == 1 || SPLIT == 2) ? C : 0]; };
    RcEnc E[3]; BitW Bw[3]; BitW& W0 = Bw[0]; BitW& W1 = Bw[SPLIT == 3 ? 0 : 1]; BitW& W2 = Bw[SPLIT == 3 ? 0 : 2];
    for (int C = 0; C < 3; ++C) E[C].init(bf(C));
    Comp<A, 0, ENB> cy(w); Comp<A, 1, ENB> cu(w); Comp<A, 2, ENB> cv(w);
    for (int y = 0; y < H; ++y) {
        cy.row(y); cu.row(y); cv.row(y);
        const uint8_t* p = rgb + (size_t(y) * W + x0) * 3;
        for (int x = 0; x < w; ++x) {
            int r = p[3 * x], g = p[3 * x + 1], b = p[3 * x + 2], v = r - g, u = b - g;
            cy.P[x] = int16_t(g + ((u + v) >> 2)); cu.P[x] = int16_t(u); cv.P[x] = int16_t(v);
        }
        for (int x = 0; x < w; ++x) { cy.enc_px(bf(0), E[0], W0, x, y, ad); cu.enc_px(bf(1), E[1], W1, x, y, ad); cv.enc_px(bf(2), E[2], W2, x, y, ad); }
    }
    for (int C = 0; C < 3; ++C) E[C].flush(bf(C));
    for (int C = 0; C < (SPLIT == 3 ? 1 : SPLIT >= 4 ? 0 : 3); ++C) Bw[C].flush(bf(C));
    if (trims) for (auto& e : E) { trims[0] += e.trims; trims[1] += e.escapes; }
    for (int k = 0; k < NB; ++k) { B[k].b.resize(B[k].next); out[k] = std::move(B[k].b); }
}
template <int A, bool ENB>
static void dec_strip(const uint8_t* const* bs, uint8_t* rgb, int W, int x0, int w, int H, bool ad) {
    RcDec D0, D1, D2; Comp<A, 0, ENB> cy(w); Comp<A, 1, ENB> cu(w); Comp<A, 2, ENB> cv(w);
    auto out_row = [&](int y) {
        uint8_t* p = rgb + (size_t(y) * W + x0) * 3;
        for (int x = 0; x < w; ++x) { int u = cu.P[x], v = cv.P[x], g = cy.P[x] - ((u + v) >> 2); p[3 * x] = uint8_t(v + g); p[3 * x + 1] = uint8_t(g); p[3 * x + 2] = uint8_t(u + g); }
    };
    if constexpr (SPLIT == 3 || SPLIT == 0 || SPLIT >= 4) {   // one buffer, one counter (3: shared bit reader; 4: bypass inside RC)
        const uint8_t* b = bs[0]; uint32_t n = 0; BitR R0, R1, R2; BitR& r1 = SPLIT == 3 ? R0 : R1; BitR& r2 = SPLIT == 3 ? R0 : R2;
        D0.init(b, n); D1.init(b, n); D2.init(b, n);
        for (int y = 0; y < H; ++y) {
            cy.row(y); cu.row(y); cv.row(y);
            for (int x = 0; x < w; ++x) { cy.dec_px(b, n, D0, R0, b, n, x, y, ad); cu.dec_px(b, n, D1, r1, b, n, x, y, ad); cv.dec_px(b, n, D2, r2, b, n, x, y, ad); }
            out_row(y);
        }
    } else {                                              // SPLIT 1: buffer per component; SPLIT 2: + shared bit buffer
        const uint8_t *b0 = bs[0], *b1 = bs[1], *b2 = bs[2], *bb = SPLIT == 2 ? bs[3] : nullptr;
        uint32_t n0 = 0, n1 = 0, n2 = 0, nb = 0; BitR R0, R1, R2;
        D0.init(b0, n0); D1.init(b1, n1); D2.init(b2, n2);
        for (int y = 0; y < H; ++y) {
            cy.row(y); cu.row(y); cv.row(y);
            if constexpr (SPLIT == 2) for (int x = 0; x < w; ++x) { cy.dec_px(b0, n0, D0, R0, bb, nb, x, y, ad); cu.dec_px(b1, n1, D1, R0, bb, nb, x, y, ad); cv.dec_px(b2, n2, D2, R0, bb, nb, x, y, ad); }
            else for (int x = 0; x < w; ++x) { cy.dec_px(b0, n0, D0, R0, b0, n0, x, y, ad); cu.dec_px(b1, n1, D1, R1, b1, n1, x, y, ad); cv.dec_px(b2, n2, D2, R2, b2, n2, x, y, ad); }
            out_row(y);
        }
    }
}

// SPLIT=1: a component's buffer depends only on its own symbols -> encode each component in its own pass/task
template <int A, int C, bool ENB>
static std::vector<uint8_t> enc_comp(const uint8_t* rgb, int W, int x0, int w, int H, uint64_t* trims, std::vector<uint64_t>* bl, bool ad) {
    constexpr int G = C ? 1 : 0, HALF = C ? 256 : 128, MASK = C ? 511 : 255;
    Buf B; B.b.resize(size_t(w) * H / 2 + 256);
    RcEnc E; BitW Bw; E.init(B);
    Model<A> M; M.init(C, ENB);
    std::vector<int16_t> ring(2 * size_t(w)); std::vector<uint16_t> e0(w), e1(w);
    const uint16_t* enc = Alpha<A>::enc()[G]; const uint8_t* raw = Alpha<A>::raw()[G];
    for (int y = 0; y < H; ++y) {                               // plain locals (no struct members): nothing to reload
        int16_t* P = ring.data() + (y & 1) * w; const int16_t* Q = y ? ring.data() + ((y - 1) & 1) * w : P;
        uint16_t* Ec = (y & 1) ? e1.data() : e0.data(); const uint16_t* Ep = (y & 1) ? e0.data() : e1.data();
        const uint8_t* p = rgb + (size_t(y) * W + x0) * 3;
        for (int x = 0; x < w; ++x) {
            int r = p[3 * x], g = p[3 * x + 1], b = p[3 * x + 2];
            P[x] = int16_t(C == 0 ? g + (((b - g) + (r - g)) >> 2) : C == 1 ? b - g : r - g);
        }
        for (int x = 0; x < w; ++x) {
            Ctx k = context<C, ENB>(P, Q, Ec, Ep, x, y, w);
            int e = ((P[x] - k.pred + HALF) & MASK) - HALF;
            Ec[x] = BW8[e < 0 ? -e : e];
            uint16_t q = enc[k.s * e + 256]; int l = q & 31;
            auto& d = M.cdf[k.ctx];
            E.put(B, d.c[l], d.c[l + 1] - d.c[l]);
            if (ad) cdf_update<A>(d, M.cnt[k.ctx], l);
            if (int nb = raw[l]) {
                if constexpr (SPLIT == 2) bl->push_back(uint64_t(uint32_t(y) * uint32_t(w) + uint32_t(x)) << 16 | uint64_t(nb) << 8 | (q >> 5));
                else Bw.put(B, q >> 5, nb);
            }
        }
    }
    E.flush(B); Bw.flush(B);
    if (trims) { __atomic_fetch_add(&trims[0], E.trims, __ATOMIC_RELAXED); __atomic_fetch_add(&trims[1], E.escapes, __ATOMIC_RELAXED); }
    B.b.resize(B.next); return std::move(B.b);
}
template <int A, bool ENB>
static std::vector<uint8_t> enc_comp_rt(int C, const uint8_t* rgb, int W, int x0, int w, int H, uint64_t* trims, std::vector<uint64_t>* bl, bool ad) {
    return C == 0 ? enc_comp<A, 0, ENB>(rgb, W, x0, w, H, trims, bl, ad) : C == 1 ? enc_comp<A, 1, ENB>(rgb, W, x0, w, H, trims, bl, ad) : enc_comp<A, 2, ENB>(rgb, W, x0, w, H, trims, bl, ad);
}
// SPLIT=2: merge the three per-component bypass lists in pixel order (Y, U, V within a pixel) into one bit stream
static std::vector<uint8_t> merge_bits(const std::vector<uint64_t>* L) {
    size_t tot = L[0].size() + L[1].size() + L[2].size();
    Buf B; B.b.resize(tot + 64); BitW Bw; size_t i[3] = {0, 0, 0};
    for (size_t k = 0; k < tot; ++k) {
        int c = -1; uint64_t best = ~0ull;
        for (int j = 0; j < 3; ++j) if (i[j] < L[j].size() && (L[j][i[j]] >> 16) < best) { best = L[j][i[j]] >> 16; c = j; }
        uint64_t e = L[c][i[c]++]; Bw.put(B, uint32_t(e & 0xFF), int((e >> 8) & 0xF));
    }
    Bw.flush(B); B.b.resize(B.next); return std::move(B.b);
}

// Strip geometry: N from header byte 15 (0 = default rule N = W/512; otherwise the encoder's N, clamped so every
// strip is >= 64 px). The ⌊W/64⌋ blocks of 64 px are dealt out evenly: every strip gets ⌊B/N⌋ blocks, the first
// B mod N strips one more; the last strip also takes the < 64 px tail. Strip starts stay multiples of 64 px.
struct Strips {
    int n, W, q, r;
    int x(int i) const { return n == 1 ? 0 : 64 * (i * q + std::min(i, r)); }
    int w(int i, int) const { return i + 1 < n ? 64 * (q + (i < r)) : W - x(i); }
};
static Strips strips_for(int W, int req) {
    int n = req ? std::min(req, std::max(1, W / 64)) : std::max(1, W / 512);
    int B = W / 64; return Strips{n, W, n > 1 ? B / n : 0, n > 1 ? B % n : 0};
}
template <class F> static void parallel_for(int n, int threads, F f) {
    if (threads <= 1 || n <= 1) { for (int i = 0; i < n; ++i) f(i); return; }
    std::atomic<int> next{0}; std::vector<std::thread> th;
    for (int t = 0; t < std::min(threads, n); ++t) th.emplace_back([&] { for (int i; (i = next.fetch_add(1)) < n;) f(i); });
    for (auto& t : th) t.join();
}
static void put32(std::vector<uint8_t>& o, uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back(uint8_t(v >> (8 * i))); }
static uint32_t get32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24; }

uint64_t g_trims = 0, g_esc = 0;
std::vector<uint8_t> encode(const uint8_t* rgb, int W, int H, int model, int alpha, int threads, int nstrips = 0, bool stat = false) {
    const bool ad = !stat;
    nstrips = std::max(0, std::min(nstrips, 255)); Strips S = strips_for(W, nstrips); std::vector<std::vector<uint8_t>> st(NB * S.n); std::vector<uint64_t> tr(2 * S.n, 0);
    std::vector<std::vector<uint64_t>> bl(SPLIT == 2 ? 3 * S.n : 0);
    if (SPLIT == 1 || SPLIT == 2) parallel_for(3 * S.n, threads, [&](int t) {
        int i = t / 3, C = t % 3, x0 = S.x(i), w = S.w(i, W); uint64_t* tq = &tr[2 * i];
        std::vector<uint8_t>& o = st[NB * i + C]; std::vector<uint64_t>* L = SPLIT == 2 ? &bl[t] : nullptr;
        if (alpha == 15) o = model ? enc_comp_rt<15, true>(C, rgb, W, x0, w, H, tq, L, ad) : enc_comp_rt<15, false>(C, rgb, W, x0, w, H, tq, L, ad);
        else             o = model ? enc_comp_rt<31, true>(C, rgb, W, x0, w, H, tq, L, ad) : enc_comp_rt<31, false>(C, rgb, W, x0, w, H, tq, L, ad);
    });
    if (SPLIT == 2) parallel_for(S.n, threads, [&](int i) { st[NB * i + 3] = merge_bits(&bl[3 * i]); });
    if (SPLIT == 0 || SPLIT >= 3) parallel_for(S.n, threads, [&](int i) {
        int x0 = S.x(i), w = S.w(i, W); std::vector<uint8_t>* o = &st[NB * i];
        if (alpha == 15) { if (model) enc_strip<15, true>(rgb, W, x0, w, H, o, &tr[2 * i], ad); else enc_strip<15, false>(rgb, W, x0, w, H, o, &tr[2 * i], ad); }
        else             { if (model) enc_strip<31, true>(rgb, W, x0, w, H, o, &tr[2 * i], ad); else enc_strip<31, false>(rgb, W, x0, w, H, o, &tr[2 * i], ad); }
    });
    for (int i = 0; i < S.n; ++i) { g_trims += tr[2 * i]; g_esc += tr[2 * i + 1]; }
    std::vector<uint8_t> o = {'P', 'L', 'T', '2'}; put32(o, W); put32(o, H); o.push_back(uint8_t(model)); const uint32_t th = tables_hash(); const bool custom = th != g_default_hash;
    o.push_back(uint8_t((alpha == 15 ? 1 : 0) | (stat ? 2 : 0) | (custom ? 4 : 0))); o.push_back(uint8_t(SPLIT)); o.push_back(uint8_t(nstrips));
    if (custom) put32(o, th);                                       // fingerprint of non-default tables
    size_t total = o.size() + 4 * st.size(); for (auto& s : st) total += s.size(); o.reserve(total);
    for (auto& s : st) put32(o, uint32_t(s.size()));
    for (auto& s : st) o.insert(o.end(), s.begin(), s.end());
    return o;
}
bool decode(const std::vector<uint8_t>& in, std::vector<uint8_t>& rgb, int& W, int& H, int threads) {
    if (in.size() < 16 || std::memcmp(in.data(), "PLT2", 4)) return false;
    W = int(get32(&in[4])); H = int(get32(&in[8])); int model = in[12], alpha = (in[13] & 1) ? 15 : 31; const bool ad = !(in[13] & 2);
    if (W <= 0 || H <= 0 || model > 1 || in[14] != SPLIT) return false;
    const size_t base = (in[13] & 4) ? 20 : 16;                     // bit 2: custom tables, fingerprint follows
    if (in[13] & 4) { if (in.size() < 20) return false; if (get32(&in[16]) != tables_hash()) { std::fprintf(stderr, "table fingerprint mismatch: decode with the same -T tables\n"); return false; } }
    else if (tables_hash() != g_default_hash) { std::fprintf(stderr, "file uses the built-in tables, but -T tables are loaded\n"); return false; }
    Strips S = strips_for(W, in[15]); int NS = NB * S.n; size_t hdr = base + 4 * size_t(NS); if (in.size() < hdr) return false;
    std::vector<size_t> off(NS + 1, hdr);
    for (int i = 0; i < NS; ++i) off[i + 1] = off[i] + get32(&in[base + 4 * i]);
    if (off.back() != in.size()) return false;
    std::vector<uint8_t> buf(in); buf.resize(in.size() + 16, 0);
    rgb.resize(size_t(W) * H * 3);
    parallel_for(S.n, threads, [&](int i) {
        const uint8_t* b[4]; for (int k = 0; k < NB; ++k) b[k] = buf.data() + off[NB * i + k]; int x0 = S.x(i), w = S.w(i, W);
        if (alpha == 15) { if (model) dec_strip<15, true>(b, rgb.data(), W, x0, w, H, ad); else dec_strip<15, false>(b, rgb.data(), W, x0, w, H, ad); }
        else             { if (model) dec_strip<31, true>(b, rgb.data(), W, x0, w, H, ad); else dec_strip<31, false>(b, rgb.data(), W, x0, w, H, ad); }
    });
    return true;
}

// ------------------------------------------------------------------ training of start/static CDFs on a corpus
// Counts letters per (component, cluster x Enb) for both alphabets over whole images (no strips: contexts and
// letters do not depend on the model), then quantizes: freq = round(p * (2^15 - 2 NL)) >= 2, sum exactly 2^15.
// Contexts with few samples are pulled to the current table (PRIOR virtual samples).
struct TrainCounts { std::vector<uint64_t> c31, c15; TrainCounts() : c31(3 * 576 * 31, 0), c15(3 * 576 * 15, 0) {} };
template <int C>
static void train_comp(const int16_t* pl, int W, int H, TrainCounts& T) {
    constexpr int G = C ? 1 : 0, HALF = C ? 256 : 128, MASK = C ? 511 : 255;
    std::vector<uint16_t> e0(W), e1(W);
    for (int y = 0; y < H; ++y) {
        const int16_t* P = pl + size_t(y) * W; const int16_t* Q = y ? P - W : P;
        uint16_t* Ec = (y & 1) ? e1.data() : e0.data(); const uint16_t* Ep = (y & 1) ? e0.data() : e1.data();
        for (int x = 0; x < W; ++x) {
            Ctx k = context<C, true>(P, Q, Ec, Ep, x, y, W);
            int e = ((P[x] - k.pred + HALF) & MASK) - HALF;
            Ec[x] = BW8[e < 0 ? -e : e];
            int v = k.s * e + 256;
            T.c31[(size_t(C) * 576 + k.ctx) * 31 + (ENC31[G][v] & 31)]++;
            T.c15[(size_t(C) * 576 + k.ctx) * 15 + (ENC15[G][v] & 31)]++;
        }
    }
}
static void train_image(const uint8_t* rgb, int W, int H, TrainCounts& T) {
    size_t n = size_t(W) * H; std::vector<int16_t> pl[3]; for (auto& p : pl) p.resize(n);
    for (size_t i = 0; i < n; ++i) { int r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2], v = r - g, u = b - g; pl[0][i] = int16_t(g + ((u + v) >> 2)); pl[1][i] = int16_t(u); pl[2][i] = int16_t(v); }
    train_comp<0>(pl[0].data(), W, H, T); train_comp<1>(pl[1].data(), W, H, T); train_comp<2>(pl[2].data(), W, H, T);
}
static void quantize(const double* cnt, const uint16_t* prior_cum, int NL, int lanes, double prior, uint16_t* out) {
    double p[32], tot = 0;
    for (int l = 0; l < NL; ++l) { double pf = (prior_cum[l + 1] - prior_cum[l]) / 32768.0; p[l] = cnt[l] + prior * pf; tot += p[l]; }
    int f[32], sum = 0, mx = 0;
    for (int l = 0; l < NL; ++l) { f[l] = std::max(2, int(p[l] / tot * (32768 - 2 * NL) + 0.5)); sum += f[l]; if (f[l] > f[mx]) mx = l; }
    f[mx] += 32768 - sum;                                            // the largest absorbs the rounding
    int c = 0; for (int l = 0; l < NL; ++l) { out[l] = uint16_t(c); c += f[l]; }
    for (int l = NL; l < lanes; ++l) out[l] = 32768;
}
static void build_tables(const TrainCounts& T, double prior) {
    for (int C = 0; C < 3; ++C) for (int a = 0; a < 2; ++a) {
        int NL = a ? 15 : 31, lanes = a ? 16 : 32; const uint64_t* cnt = a ? T.c15.data() : T.c31.data();
        for (int cl = 0; cl < 64; ++cl) {
            double sum1[32] = {};
            for (int e = 0; e < 9; ++e) {
                int ctx = cl * 9 + e; double ce[32];
                for (int l = 0; l < NL; ++l) { ce[l] = double(cnt[(size_t(C) * 576 + ctx) * NL + l]); sum1[l] += ce[l]; }
                if (a) quantize(ce, CDFE_15[C][ctx], NL, lanes, prior, C15_E[C][ctx]); else quantize(ce, CDFE_31[C][ctx], NL, lanes, prior, C31_E[C][ctx]);
            }
            if (a) quantize(sum1, CDF1_15[C][cl], NL, lanes, prior, C15_1[C][cl]); else quantize(sum1, CDF1_31[C][cl], NL, lanes, prior, C31_1[C][cl]);
        }
    }
}
static bool save_tables(const char* path) {
    FILE* f = std::fopen(path, "wb"); if (!f) return false;
    std::fwrite("PLTT", 1, 4, f); std::fwrite(C31_1, 1, sizeof C31_1, f); std::fwrite(C31_E, 1, sizeof C31_E, f);
    std::fwrite(C15_1, 1, sizeof C15_1, f); std::fwrite(C15_E, 1, sizeof C15_E, f); std::fclose(f); return true;
}
static bool load_tables(const char* path) {
    FILE* f = std::fopen(path, "rb"); if (!f) return false; char m[4];
    bool ok = std::fread(m, 1, 4, f) == 4 && !std::memcmp(m, "PLTT", 4) && std::fread(C31_1, 1, sizeof C31_1, f) == sizeof C31_1 &&
              std::fread(C31_E, 1, sizeof C31_E, f) == sizeof C31_E && std::fread(C15_1, 1, sizeof C15_1, f) == sizeof C15_1 &&
              std::fread(C15_E, 1, sizeof C15_E, f) == sizeof C15_E;
    std::fclose(f); return ok;
}
static bool save_header(const char* path) {                         // same arrays as C++ for plt2_tables.h / wasm
    FILE* f = std::fopen(path, "w"); if (!f) return false;
    auto arr = [&](const char* name, const char* dims, const uint16_t* p, size_t n, int per) {
        std::fprintf(f, "alignas(64) static const uint16_t %s%s = {", name, dims);
        for (size_t i = 0; i < n; ++i) std::fprintf(f, "%s%u,", i % per ? "" : "\n", p[i]);
        std::fprintf(f, "};\n");
    };
    std::fprintf(f, "// trained start CDFs (plt2 train): replace the CDF1_31/CDFE_31/CDF1_15/CDFE_15 arrays in plt2_tables.h\n");
    arr("CDF1_31", "[3][64][32]", &C31_1[0][0][0], sizeof C31_1 / 2, 32); arr("CDFE_31", "[3][576][32]", &C31_E[0][0][0], sizeof C31_E / 2, 32);
    arr("CDF1_15", "[3][64][16]", &C15_1[0][0][0], sizeof C15_1 / 2, 16); arr("CDFE_15", "[3][576][16]", &C15_E[0][0][0], sizeof C15_E / 2, 16);
    std::fclose(f); return true;
}
}  // namespace plt2

// ------------------------------------------------------------------ CLI
static bool read_ppm(const std::string& path, std::vector<uint8_t>& px, int& W, int& H) {
    FILE* f = std::fopen(path.c_str(), "rb"); if (!f) return false;
    char m[3] = {}; int mx;
    if (std::fscanf(f, "%2s", m) != 1 || std::strcmp(m, "P6")) { std::fclose(f); return false; }
    auto num = [&](int& v) { int c; while ((c = std::fgetc(f)) != EOF) { if (c == '#') { while ((c = std::fgetc(f)) != '\n' && c != EOF) {} } else if (!isspace(c)) { std::ungetc(c, f); break; } } return std::fscanf(f, "%d", &v) == 1; };
    if (!num(W) || !num(H) || !num(mx) || mx != 255) { std::fclose(f); return false; }
    std::fgetc(f); px.resize(size_t(W) * H * 3);
    bool ok = std::fread(px.data(), 1, px.size(), f) == px.size(); std::fclose(f); return ok;
}
static bool write_ppm(const std::string& path, const std::vector<uint8_t>& px, int W, int H) {
    FILE* f = std::fopen(path.c_str(), "wb"); if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", W, H); bool ok = std::fwrite(px.data(), 1, px.size(), f) == px.size(); std::fclose(f); return ok;
}
static bool read_file(const std::string& p, std::vector<uint8_t>& b) {
    FILE* f = std::fopen(p.c_str(), "rb"); if (!f) return false; std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    b.resize(size_t(n)); bool ok = std::fread(b.data(), 1, b.size(), f) == b.size(); std::fclose(f); return ok;
}
int main(int argc, char** argv) {
    plt2::init_static(); plt2::init_targets(); plt2::init_cdf_tables();
    if (argc < 2) { std::fprintf(stderr, "usage: plt2 c in.ppm out.plt [-t N] [-m 0|1] [-a 31|15] [-s strips] [-x static CDF] [-T tables.pltt] | plt2 train (ppm|dir)... -o tables.pltt [-H tables.h] [-p prior] [-t N] | plt2 d in.plt out.ppm [-t N] | plt2 bench [-t N] [-m 0|1] [-a 31|15] [-r R] [-csv f] (ppm|dir)...\n"); return 1; }
    std::string mode = argv[1]; int threads = int(std::thread::hardware_concurrency()), model = 1, alpha = 31, reps = 3, nstrips = 0; bool stat = false; std::string csv, tables, outp, hdr; double prior = 32; std::vector<std::string> pos;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-t" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "-m" && i + 1 < argc) model = std::atoi(argv[++i]);
        else if (a == "-a" && i + 1 < argc) alpha = std::atoi(argv[++i]);
        else if (a == "-r" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "-csv" && i + 1 < argc) csv = argv[++i];
        else if (a == "-s" && i + 1 < argc) nstrips = std::atoi(argv[++i]);
        else if (a == "-x") stat = true;
        else if (a == "-T" && i + 1 < argc) tables = argv[++i];
        else if (a == "-o" && i + 1 < argc) outp = argv[++i];
        else if (a == "-H" && i + 1 < argc) hdr = argv[++i];
        else if (a == "-p" && i + 1 < argc) prior = std::atof(argv[++i]);
        else pos.push_back(a);
    }
    if (alpha != 31 && alpha != 15) alpha = 31;
    if (!tables.empty() && !plt2::load_tables(tables.c_str())) { std::fprintf(stderr, "bad tables %s\n", tables.c_str()); return 1; }
    auto list_files = [&](const std::vector<std::string>& in) {
        std::vector<std::string> files;
        for (auto& p : in) {
            if (std::filesystem::is_directory(p)) { for (auto& e : std::filesystem::directory_iterator(p)) if (e.path().extension() == ".ppm") files.push_back(e.path().string()); }
            else files.push_back(p);
        }
        std::sort(files.begin(), files.end()); return files;
    };
    if (mode == "train") {                                           // plt2 train (ppm|dir)... -o t.pltt [-H t.h] [-p prior] [-t N]
        auto files = list_files(pos); if (files.empty() || outp.empty()) { std::fprintf(stderr, "train needs files and -o\n"); return 1; }
        std::vector<plt2::TrainCounts> part(std::max(1, threads)); std::atomic<size_t> next{0}; std::atomic<uint64_t> npx{0};
        std::vector<std::thread> th;
        for (int t = 0; t < std::max(1, threads); ++t) th.emplace_back([&, t] {
            for (size_t i; (i = next.fetch_add(1)) < files.size();) {
                std::vector<uint8_t> px; int W, H; if (!read_ppm(files[i], px, W, H)) { std::fprintf(stderr, "skip %s\n", files[i].c_str()); continue; }
                plt2::train_image(px.data(), W, H, part[t]); npx += uint64_t(W) * H;
            }
        });
        for (auto& t : th) t.join();
        plt2::TrainCounts T; for (auto& p : part) { for (size_t i = 0; i < T.c31.size(); ++i) T.c31[i] += p.c31[i]; for (size_t i = 0; i < T.c15.size(); ++i) T.c15[i] += p.c15[i]; }
        plt2::build_tables(T, prior);
        if (!plt2::save_tables(outp.c_str())) return 1;
        if (!hdr.empty() && !plt2::save_header(hdr.c_str())) return 1;
        std::printf("trained on %zu files, %.1f Mpx -> %s%s%s\n", files.size(), npx / 1e6, outp.c_str(), hdr.empty() ? "" : ", ", hdr.c_str());
        return 0;
    }
    if (mode == "c" && pos.size() == 2) {
        std::vector<uint8_t> px; int W, H; if (!read_ppm(pos[0], px, W, H)) { std::fprintf(stderr, "bad ppm\n"); return 1; }
        auto o = plt2::encode(px.data(), W, H, model, alpha, threads, nstrips, stat);
        FILE* f = std::fopen(pos[1].c_str(), "wb"); std::fwrite(o.data(), 1, o.size(), f); std::fclose(f); return 0;
    }
    if (mode == "d" && pos.size() == 2) {
        std::vector<uint8_t> b, px; int W, H; if (!read_file(pos[0], b) || !plt2::decode(b, px, W, H, threads)) { std::fprintf(stderr, "bad stream\n"); return 1; }
        return write_ppm(pos[1], px, W, H) ? 0 : 1;
    }
    if (mode == "bench") {
        auto files = list_files(pos);
        FILE* fc = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
        if (fc) std::fprintf(fc, "file,width,height,bytes,bpp,enc_s,dec_s,ok\n");
        double npx = 0, bytes = 0, te = 0, td = 0; bool all = true;
        #ifdef PLT_SIMDE
        const char* simd = SIMD == 2 ? "simde-avx512" : "simde-avx2";
#else
        const char* simd = SIMD == 2 ? "avx512" : SIMD == 1 ? "avx2" : "scalar";
#endif
        std::printf("PLT2 model=%d alphabet=A%d %s threads=%d simd=%s\n", model, alpha, stat ? "STATIC" : "adaptive", threads, simd);
        for (auto& fn : files) {
            std::vector<uint8_t> px, o, out; int W, H, w, h; if (!read_ppm(fn, px, W, H)) { std::fprintf(stderr, "skip %s\n", fn.c_str()); continue; }
            double be = 1e30, bd = 1e30;
            for (int r = 0; r < reps; ++r) {
                auto t0 = std::chrono::steady_clock::now(); o = plt2::encode(px.data(), W, H, model, alpha, threads, nstrips, stat);
                auto t1 = std::chrono::steady_clock::now(); bool d = plt2::decode(o, out, w, h, threads); (void)d;
                auto t2 = std::chrono::steady_clock::now();
                be = std::min(be, std::chrono::duration<double>(t1 - t0).count()); bd = std::min(bd, std::chrono::duration<double>(t2 - t1).count());
            }
            bool ok = out == px; all &= ok; double p = double(W) * H; npx += p; bytes += o.size(); te += be; td += bd;
            std::string nm = std::filesystem::path(fn).filename().string();
            std::printf("%-40s %5dx%-5d %10zu B %7.4f bpp  enc %7.1f  dec %7.1f MB/s  %s\n", nm.c_str(), W, H, o.size(), 8.0 * o.size() / p, 3 * p / 1e6 / be, 3 * p / 1e6 / bd, ok ? "ok" : "FAIL");
            if (fc) std::fprintf(fc, "%s,%d,%d,%zu,%.5f,%.6f,%.6f,%d\n", nm.c_str(), W, H, o.size(), 8.0 * o.size() / p, be, bd, int(ok));
        }
        std::printf("TOTAL %zu files  %.4f bpp  enc %.1f  dec %.1f MB/s (RGB)  trims %llu  escapes %llu  %s\n", files.size(), 8 * bytes / npx, 3 * npx / 1e6 / te, 3 * npx / 1e6 / td,
                    (unsigned long long)plt2::g_trims, (unsigned long long)plt2::g_esc, all ? "all ok" : "FAIL");
        if (fc) std::fclose(fc);
        return all ? 0 : 2;
    }
    std::fprintf(stderr, "bad args\n"); return 1;
}
