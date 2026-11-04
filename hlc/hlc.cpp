// hlc — HALIC-class codec built on PALIC-1: one colour transform, MED only, independent 256x256 tiles,
// in-place residuals, one FSE table per plane-tile (or none: -n).
//
//   RGB -> RCT (V = R-G, U = B-G, Y = G + ((U+V)>>2); HALIC's mode 0 Y,Co,Cg with Cg = -U), SIMD 16 px
//   -> tiles of 256 x 256 (edge tiles smaller), independent (prediction restarts at the tile border)
//   -> MED, branchless (HALIC's cmov form); the encoder writes the residual in place over the plane, walking the tile
//      backwards (bottom-right to top-left) so the neighbours are still the original samples; SSE4.1 8 lanes
//   -> residual wrapped (Y mod 256, U/V mod 512), zigzag, in place
//   -> FSE (Collet): byte FSE when max zz < 256, else FSE-U16; -n = no entropy coding (raw u16 residuals)
//   -s: skewed 8-row bands (wavefront MED in 8 SSE lanes); only real pixels are stored, the band triangles are not.
//   decoder: FSE writes straight into the tile buffer, MED restores the plane in place (left neighbour in a
//   register), SSSE3 inverse RCT to interleaved RGB.  HALIC itself writes encoder residuals to a separate buffer.
//
//   hlc c in.ppm out.hlc [-t threads] [-n]     hlc d in.hlc out.ppm [-t threads]     hlc bench [opts] a.ppm|dir ... [-max N]
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <filesystem>
#include <vector>
#include <smmintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include "../palic-1/rgb.hpp"
extern "C" {
#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
size_t FSE_buildCTableU16(FSE_CTable* ct, const short* normalizedCounter, unsigned maxSymbolValue, unsigned tableLog);
}

namespace hlc {
static int g_tw = 256, g_th = 128;                              // tile size (-tw, -th), stored in the header
#if defined(_MSC_VER) && !defined(__clang__)
static inline uint32_t bw(uint32_t v) { unsigned long i; return _BitScanReverse(&i, v) ? uint32_t(i) + 1 : 0; }
#else
static inline uint32_t bw(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
#endif
static inline uint16_t zz(int e) { return uint16_t(e >= 0 ? 2 * e : -2 * e - 1); }
static inline int unzz(uint32_t z) { return int(z >> 1) ^ -int(z & 1); }
static inline int med(int a, int b, int c) {                     // HALIC 0x1400026e0: two cmov for min/max, two for the clamp
    int mx = a > b ? a : b, mn = a > b ? b : a, p = a + b - c;
    p = c <= mn ? mx : p; p = c >= mx ? mn : p; return p;
}
template <bool N9> static inline int wrape(int e) { return N9 ? ((e + 256) & 511) - 256 : int(int8_t(e)); }
template <bool N9> static inline int wrapv(int v) { return N9 ? ((v + 256) & 511) - 256 : (v & 255); }

template <class T> struct noinit_alloc : std::allocator<T> {                // vector::resize without zero-fill (decoder output)
    template <class U> struct rebind { using other = noinit_alloc<U>; };
    noinit_alloc() = default; template <class U> noinit_alloc(const noinit_alloc<U>&) {}
    template <class U> void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
    template <class U, class... A> void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
};
struct Img { int W = 0, H = 0; std::vector<uint8_t, noinit_alloc<uint8_t>> rgb; };
struct Out { std::vector<uint8_t> b; void u8(uint8_t v) { b.push_back(v); } void var(uint64_t v) { while (v >= 128) { b.push_back(uint8_t(v | 128)); v >>= 7; } b.push_back(uint8_t(v)); } void raw(const void* p, size_t n) { b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n); } };
struct In { const uint8_t* p; const uint8_t* e;
    uint8_t u8() { if (p >= e) throw std::runtime_error("truncated"); return *p++; }
    uint64_t var() { uint64_t v = 0; int s = 0; for (;;) { uint8_t c = u8(); v |= uint64_t(c & 127) << s; if (!(c & 128)) return v; s += 7; } }
    const uint8_t* take(size_t n) { if (size_t(e - p) < n) throw std::runtime_error("truncated"); const uint8_t* r = p; p += n; return r; } };

// ------------------------------------------------------------------ RCT (SIMD 16 px; tail block overlapped)
static void rct_row(const uint8_t* rgb, int w, int16_t* v, int16_t* u, int16_t* y) {
    if (w < 16) { for (int x = 0; x < w; ++x) { int R = rgb[3 * x], G = rgb[3 * x + 1], B = rgb[3 * x + 2]; v[x] = int16_t(R - G); u[x] = int16_t(B - G); y[x] = int16_t(G + ((R - G + B - G) >> 2)); } return; }
    for (int x = 0; x < w; x += 16) { int xs = std::min(x, w - 16); conv::convert_to_plannar_line_8to16_48(rgb + 3 * xs, v + xs, u + xs, y + xs); }
}
alignas(16) static uint8_t SHUF[3][3][16];
static void init_shuf() { for (int o = 0; o < 3; ++o) for (int c = 0; c < 3; ++c) for (int i = 0; i < 16; ++i) { int j = 16 * o + i; SHUF[o][c][i] = j % 3 == c ? uint8_t(j / 3) : 0x80; } }
static inline void irct16(const int16_t* v, const int16_t* u, const int16_t* y, uint8_t* out) {
    __m128i v0 = _mm_loadu_si128((const __m128i*)v), v1 = _mm_loadu_si128((const __m128i*)(v + 8));
    __m128i u0 = _mm_loadu_si128((const __m128i*)u), u1 = _mm_loadu_si128((const __m128i*)(u + 8));
    __m128i y0 = _mm_loadu_si128((const __m128i*)y), y1 = _mm_loadu_si128((const __m128i*)(y + 8));
    __m128i g0 = _mm_sub_epi16(y0, _mm_srai_epi16(_mm_add_epi16(u0, v0), 2)), g1 = _mm_sub_epi16(y1, _mm_srai_epi16(_mm_add_epi16(u1, v1), 2));
    const __m128i ff = _mm_set1_epi16(255);
    __m128i R8 = _mm_packus_epi16(_mm_and_si128(_mm_add_epi16(v0, g0), ff), _mm_and_si128(_mm_add_epi16(v1, g1), ff));
    __m128i G8 = _mm_packus_epi16(_mm_and_si128(g0, ff), _mm_and_si128(g1, ff));
    __m128i B8 = _mm_packus_epi16(_mm_and_si128(_mm_add_epi16(u0, g0), ff), _mm_and_si128(_mm_add_epi16(u1, g1), ff));
    for (int o = 0; o < 3; ++o)
        _mm_storeu_si128((__m128i*)(out + 16 * o), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(R8, _mm_load_si128((const __m128i*)SHUF[o][0])),
                                                                             _mm_shuffle_epi8(G8, _mm_load_si128((const __m128i*)SHUF[o][1]))),
                                                                _mm_shuffle_epi8(B8, _mm_load_si128((const __m128i*)SHUF[o][2]))));
}
static void irct_row(const int16_t* v, const int16_t* u, const int16_t* y, int w, uint8_t* rgb) {
    if (w < 16) { for (int x = 0; x < w; ++x) { int g = y[x] - ((u[x] + v[x]) >> 2); rgb[3 * x] = uint8_t(v[x] + g); rgb[3 * x + 1] = uint8_t(g); rgb[3 * x + 2] = uint8_t(u[x] + g); } return; }
    for (int x = 0; x < w; x += 16) { int xs = std::min(x, w - 16); irct16(v + xs, u + xs, y + xs, rgb + 3 * xs); }
}

// ------------------------------------------------------------------ MED over a tile plane (stride = w), in place
// borders: row 0 -> left neighbour (first sample: 0); column 0 -> top neighbour.
// encoder: residual + zigzag in place, backwards so that every neighbour read is still an original sample
template <bool N9> static inline __m128i wrapv8(__m128i e) { return N9 ? _mm_srai_epi16(_mm_slli_epi16(e, 7), 7) : _mm_srai_epi16(_mm_slli_epi16(e, 8), 8); }
template <bool N9> static void residuals_inplace(int16_t* p, int w, int h) {
    for (int y = h - 1; y >= 0; --y) {
        int16_t* r = p + size_t(y) * w; const int16_t* u = r - w;
        if (y == 0) { for (int x = w - 1; x >= 1; --x) r[x] = int16_t(zz(wrape<N9>(r[x] - r[x - 1]))); r[0] = int16_t(zz(wrape<N9>(r[0]))); continue; }
        int x = w - 8;
        for (; x >= 1; x -= 8) {                                   // 8 lanes: a = r[x-1..], b = u[x..], c = u[x-1..]
            __m128i cur = _mm_loadu_si128((const __m128i*)(r + x)), a = _mm_loadu_si128((const __m128i*)(r + x - 1));
            __m128i b = _mm_loadu_si128((const __m128i*)(u + x)), c = _mm_loadu_si128((const __m128i*)(u + x - 1));
            __m128i mx = _mm_max_epi16(a, b), mn = _mm_min_epi16(a, b), pr = _mm_sub_epi16(_mm_add_epi16(a, b), c);
            pr = _mm_min_epi16(_mm_max_epi16(pr, mn), mx);           // = MED (clamp form)
            __m128i e = wrapv8<N9>(_mm_sub_epi16(cur, pr));
            _mm_storeu_si128((__m128i*)(r + x), _mm_xor_si128(_mm_add_epi16(e, e), _mm_srai_epi16(e, 15)));
        }
        for (x += 7; x >= 1; --x) r[x] = int16_t(zz(wrape<N9>(r[x] - med(r[x - 1], u[x], u[x - 1]))));
        r[0] = int16_t(zz(wrape<N9>(r[0] - u[0])));
    }
}
// ---- per-plane predictor choice (HALIC's four): 0 MED, 1 trunc((2(a+b)-c)/3), 2 (a+b+1)>>1, 3 none.
// Mode applies to interior samples; row 0 / column 0 of the tile always predict from the left / top neighbour.
static int g_pmask = 15;                                            // -P: allowed predictors (bit m = mode m)
template <int PM> static inline int pred_s(int a, int b, int c) {
    if (PM == 0) return med(a, b, c);
    if (PM == 1) return (2 * (a + b) - c) / 3;
    if (PM == 2) return (a + b + 1) >> 1;
    return 0;
}
template <int PM> static inline __m128i pred_v(__m128i a, __m128i b, __m128i c) {
    if (PM == 0) { __m128i mx = _mm_max_epi16(a, b), mn = _mm_min_epi16(a, b); return _mm_min_epi16(_mm_max_epi16(_mm_sub_epi16(_mm_add_epi16(a, b), c), mn), mx); }
    if (PM == 1) { __m128i s = _mm_add_epi16(a, b); __m128i x = _mm_sub_epi16(_mm_add_epi16(s, s), c);   // |x| <= 1275; trunc(x / 3)
                   return _mm_sub_epi16(_mm_mulhi_epi16(x, _mm_set1_epi16(21846)), _mm_srai_epi16(x, 15)); }
    if (PM == 2) return _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(a, b), _mm_set1_epi16(1)), 1);
    return _mm_setzero_si128();
}
// one row: o[x] = zigzag(wrap(X - P)) of row y of plane p (stride w); plane unchanged
template <bool N9, int PM> static inline void residual_row(const int16_t* p, int w, int y, int16_t* o) {
    const int16_t* r = p + size_t(y) * w; const int16_t* u = r - w;
    // tile borders (row 0, column 0): left / top neighbour for every mode (= MED with zero padding)
    if (y == 0) { o[0] = int16_t(zz(wrape<N9>(r[0]))); for (int x = 1; x < w; ++x) o[x] = int16_t(zz(wrape<N9>(r[x] - r[x - 1]))); return; }
    o[0] = int16_t(zz(wrape<N9>(r[0] - u[0])));
    int x = 1;
    for (; x + 8 <= w; x += 8) {
        __m128i cur = _mm_loadu_si128((const __m128i*)(r + x)), a = _mm_loadu_si128((const __m128i*)(r + x - 1));
        __m128i b = _mm_loadu_si128((const __m128i*)(u + x)), c = _mm_loadu_si128((const __m128i*)(u + x - 1));
        __m128i e = wrapv8<N9>(_mm_sub_epi16(cur, pred_v<PM>(a, b, c)));
        _mm_storeu_si128((__m128i*)(o + x), _mm_xor_si128(_mm_add_epi16(e, e), _mm_srai_epi16(e, 15)));
    }
    for (; x < w; ++x) o[x] = int16_t(zz(wrape<N9>(r[x] - pred_s<PM>(r[x - 1], u[x], u[x - 1]))));
}
// out = zigzag(wrap(X - P)), plane p (stride w) unchanged
template <bool N9, int PM> static void residuals_pm(const int16_t* p, int w, int h, int16_t* out) {
    for (int y = 0; y < h; ++y) residual_row<N9, PM>(p, w, y, out + size_t(y) * w);
}
template <bool N9> static void residuals_mode(int m, const int16_t* p, int w, int h, int16_t* out) {
    switch (m) { case 0: residuals_pm<N9, 0>(p, w, h, out); break; case 1: residuals_pm<N9, 1>(p, w, h, out); break;
                 case 2: residuals_pm<N9, 2>(p, w, h, out); break; default: residuals_pm<N9, 3>(p, w, h, out); }
}
static double entropy_bits(const int16_t* z, size_t n) {           // zero-order entropy of the zigzag residuals, in bits
    unsigned cnt[512] = {0}; for (size_t i = 0; i < n; ++i) cnt[uint16_t(z[i]) & 511]++;
    double t = double(n) * std::log2(double(n)); for (unsigned v = 0; v < 512; ++v) if (cnt[v]) t -= double(cnt[v]) * std::log2(double(cnt[v]));
    return t;
}
// zero-order entropy estimate of mode PM on every step-th interior row (no residual buffer, histogram only)
static int g_pxstep = 1;                                            // -Px: column step of the predictor probe (HALIC: 9 x 9)
template <bool N9, int PM> static double entropy_rows(const int16_t* p, int w, int h, int step) {
    unsigned cnt[512] = {0}; size_t n = 0; alignas(16) uint16_t z[8];
    if (g_pxstep > 1) {                                              // sparse grid, scalar
        for (int y = 1; y < h; y += step) { const int16_t* r = p + size_t(y) * w; const int16_t* u = r - w;
            for (int x = 1; x < w; x += g_pxstep) { cnt[zz(wrape<N9>(r[x] - pred_s<PM>(r[x - 1], u[x], u[x - 1]))) & 511]++; ++n; } }
        if (!n) return 0;
        double t = double(n) * std::log2(double(n)); for (unsigned v = 0; v < 512; ++v) if (cnt[v]) t -= double(cnt[v]) * std::log2(double(cnt[v]));
        return t;
    }
    for (int y = 1; y < h; y += step) {
        const int16_t* r = p + size_t(y) * w; const int16_t* u = r - w; int x = 1;
        for (; x + 8 <= w; x += 8) {
            __m128i e = wrapv8<N9>(_mm_sub_epi16(_mm_loadu_si128((const __m128i*)(r + x)), pred_v<PM>(_mm_loadu_si128((const __m128i*)(r + x - 1)), _mm_loadu_si128((const __m128i*)(u + x)), _mm_loadu_si128((const __m128i*)(u + x - 1)))));
            _mm_store_si128((__m128i*)z, _mm_xor_si128(_mm_add_epi16(e, e), _mm_srai_epi16(e, 15)));
            for (int k = 0; k < 8; ++k) cnt[z[k] & 511]++;
            n += 8;
        }
        for (; x < w; ++x) { cnt[zz(wrape<N9>(r[x] - pred_s<PM>(r[x - 1], u[x], u[x - 1]))) & 511]++; ++n; }
    }
    if (!n) return 0;
    double t = double(n) * std::log2(double(n)); for (unsigned v = 0; v < 512; ++v) if (cnt[v]) t -= double(cnt[v]) * std::log2(double(cnt[v]));
    return t;
}
static std::atomic<unsigned> g_pstat[3][4];                         // chosen predictor per plane (V, U, Y) x mode, for bench
static int g_pstep = 8;                                             // -Ps: rows sampled for the predictor choice (1 = all)
// best predictor by entropy (HALIC: argmin, no hysteresis), mode only
template <bool N9> static int choose_pred(const int16_t* p, int w, int h) {
    if ((g_pmask & (g_pmask - 1)) == 0) { int m = 0; while (!(g_pmask >> m & 1)) ++m; return m; }   // single allowed mode: no probe
    int best = -1; double bh = 0;
    for (int m = 0; m < 4; ++m) {
        if (!(g_pmask >> m & 1)) continue;
        double hm;
        switch (m) { case 0: hm = entropy_rows<N9, 0>(p, w, h, g_pstep); break; case 1: hm = entropy_rows<N9, 1>(p, w, h, g_pstep); break;
                     case 2: hm = entropy_rows<N9, 2>(p, w, h, g_pstep); break; default: hm = entropy_rows<N9, 3>(p, w, h, g_pstep); }
        if (best < 0 || hm < bh) { best = m; bh = hm; }
    }
    return best;
}
// decoder: on entry p[] holds zigzag residuals, on exit the plane
template <bool N9> static void restore(int16_t* p, int w, int h) {
    for (int y = 0; y < h; ++y) {
        int16_t* r = p + size_t(y) * w; const int16_t* u = r - w;
        int a;
        if (y == 0) { a = wrapv<N9>(unzz(uint16_t(r[0]))); r[0] = int16_t(a); for (int x = 1; x < w; ++x) { a = wrapv<N9>(a + unzz(uint16_t(r[x]))); r[x] = int16_t(a); } continue; }
        a = wrapv<N9>(u[0] + unzz(uint16_t(r[0]))); r[0] = int16_t(a);
        for (int x = 1; x < w; ++x) { a = wrapv<N9>(med(a, u[x], u[x - 1]) + unzz(uint16_t(r[x]))); r[x] = int16_t(a); }
    }
}

// ------------------------------------------------------------------ FSE4: tANS over u16 symbols, 4 interleaved states, one bitstream
// Library parts: histogram normalisation, NCount header, CTable + encoder primitives. The decoder is ours:
// flat u32 table (newState | nbBits << 16 | symbol << 20), reload once per 4 symbols, output straight to int16.
// stream: mode u8 (0 raw u16, 1 rle var, 6 FSE4 var csize + data)
static int g_tl = 12, g_tadapt = 1;
static int g_joint = 0, g_K = 2;                                    // -J 0: three separate streams, -K states per plane
static bool joint_put(Out& o, const uint16_t* const* s, size_t n, int K, std::vector<uint8_t>& tmp);                                               // -T: max tableLog
static int g_S = 4;                                                 // -S: FSE states per stream (1, 2, 4, 8); 4 = mode 6, others = mode 8
static size_t fse4_enc(uint8_t* dst, size_t cap, const uint16_t* s, size_t n, unsigned mx, const unsigned* cnt) {
    // table size follows the stream: <= n/16 entries (decoder table build ~ 1/16 of the symbols), never below 9
    const unsigned tcap = g_tadapt ? std::min(unsigned(g_tl), std::max(9u, bw(uint32_t(n)) - 1 - 4)) : unsigned(g_tl);
    const unsigned tl = FSE_optimalTableLog(tcap, n, mx); short norm[1024];
    if (FSE_isError(FSE_normalizeCount(norm, tl, cnt, n, mx))) return 0;
    size_t h = FSE_writeNCount(dst, cap, norm, mx, tl); if (FSE_isError(h)) return 0;
    thread_local std::vector<FSE_CTable> ct(FSE_CTABLE_SIZE_U32(13, 1023));
    if (FSE_isError(FSE_buildCTableU16(ct.data(), norm, mx, tl))) return 0;
    BIT_CStream_t bc; if (FSE_isError(BIT_initCStream(&bc, dst + h, cap - h))) return 0;
    const size_t S = size_t(g_S), M = S - 1; FSE_CState_t st[8]; const size_t L = std::min<size_t>(n, S); size_t i = n;   // S is a power of two
    for (size_t j = 0; j < L; ++j) { --i; FSE_initCState2(&st[i & M], ct.data(), s[i]); }   // last symbol of each lane
    while (i > 0 && (i & 3)) { --i; FSE_encodeSymbol(&bc, &st[i & M], s[i]); if ((i & 3) == 0) BIT_flushBits(&bc); }   // align to 4
    if (S == 4) {                                                   // lanes fixed per unrolled step: states stay in registers
        FSE_CState_t c0 = st[0], c1 = st[1], c2 = st[2], c3 = st[3];
        for (; i >= 4; i -= 4) {
            FSE_encodeSymbol(&bc, &c3, s[i - 1]); FSE_encodeSymbol(&bc, &c2, s[i - 2]); FSE_encodeSymbol(&bc, &c1, s[i - 3]); FSE_encodeSymbol(&bc, &c0, s[i - 4]);
            BIT_flushBits(&bc);                                         // <= 4 x 12 bits between flushes
        }
        st[0] = c0; st[1] = c1; st[2] = c2; st[3] = c3;
    }
    while (i > 0) { --i; FSE_encodeSymbol(&bc, &st[i & M], s[i]); if ((i & 3) == 0) BIT_flushBits(&bc); }   // generic S / tail
    for (size_t k = L; k-- > 0;) FSE_flushCState(&bc, &st[k]);    // lane 0 last -> read first
    size_t c = BIT_closeCStream(&bc); return c ? h + c : 0;
}
static void fse4_dec(int16_t* out, size_t n, const uint8_t* src, size_t c) {
    short norm[1024]; unsigned mx = 1023, tl;
    size_t h = FSE_readNCount(norm, &mx, &tl, src, c); if (FSE_isError(h) || tl > 13 || tl < 1) throw std::runtime_error("ncount");
    thread_local std::vector<uint32_t> T(1 << 13); thread_local std::vector<uint16_t> nxt(1024);
    const uint32_t ts = 1u << tl, mask = ts - 1, step = (ts >> 1) + (ts >> 3) + 3; uint32_t hi = ts - 1;
    for (unsigned s = 0; s <= mx; ++s) { if (norm[s] == -1) { T[hi--] = s; nxt[s] = 1; } else nxt[s] = uint16_t(norm[s]); }
    uint32_t pos = 0;
    for (unsigned s = 0; s <= mx; ++s) for (int k = 0; k < norm[s]; ++k) { T[pos] = s; pos = (pos + step) & mask; while (pos > hi) pos = (pos + step) & mask; }
    if (pos) throw std::runtime_error("ncount spread");
    for (uint32_t u = 0; u < ts; ++u) { uint32_t s = T[u], ns = nxt[s]++, nb = tl - bw(ns) + 1; T[u] = (((ns << nb) - ts) & 0xffff) | nb << 16 | s << 20; }
    BIT_DStream_t bd; if (FSE_isError(BIT_initDStream(&bd, src + h, c - h))) throw std::runtime_error("bitstream");
    const size_t L = std::min<size_t>(n, 4); uint32_t st[4] = {0, 0, 0, 0};
    for (size_t k = 0; k < L; ++k) { st[k] = uint32_t(BIT_readBits(&bd, tl)); BIT_reloadDStream(&bd); }
    const uint32_t* t = T.data(); size_t i = 0;
#define HLC_DEC1(k) { uint32_t e = t[st[k]]; out[i + k] = int16_t(e >> 20); st[k] = (e & 0xffff) + uint32_t(BIT_readBits(&bd, (e >> 16) & 15)); }
    for (; i + 4 <= n; i += 4) { BIT_reloadDStream(&bd); HLC_DEC1(0) HLC_DEC1(1) HLC_DEC1(2) HLC_DEC1(3) }
    BIT_reloadDStream(&bd); for (size_t k = 0; i < n; ++i, ++k) { uint32_t e = t[st[k]]; out[i] = int16_t(e >> 20); st[k] = (e & 0xffff) + uint32_t(BIT_readBits(&bd, (e >> 16) & 15)); }
#undef HLC_DEC1
}
// A = alphabet bound: 256 for Y (residual mod 256), 512 for U/V (mod 512)
static void put_stream(Out& o, const uint16_t* s, size_t n, std::vector<uint8_t>& tmp, unsigned A = 512) {
    thread_local unsigned cnt[512], c4[4][512]; std::memset(c4, 0, sizeof c4); unsigned mx = 0;
    size_t i = 0;                                                   // 4 interleaved histograms: repeated symbols do not serialise on one counter
    for (; i + 4 <= n; i += 4) { c4[0][s[i]]++; c4[1][s[i + 1]]++; c4[2][s[i + 2]]++; c4[3][s[i + 3]]++; }
    for (; i < n; ++i) c4[0][s[i]]++;
    for (unsigned v = 0; v < A; ++v) cnt[v] = c4[0][v] + c4[1][v] + c4[2][v] + c4[3][v];
    for (unsigned v = 0; v < A; ++v) if (cnt[v]) mx = v;
    if (cnt[s[0]] == n) { o.u8(1); o.var(s[0]); return; }
    tmp.resize(n * 2 + 4096); size_t c = fse4_enc(tmp.data(), tmp.size(), s, n, mx, cnt);
    if (!c || c >= n * 2) { o.u8(0); o.raw(s, n * 2); return; }
    if (g_S == 4) o.u8(6); else { o.u8(8); o.u8(uint8_t(g_S)); } o.var(c); o.raw(tmp.data(), c);
}
// decode straight into the tile buffer (int16 holding zigzag values)
static void get_stream(In& in, int16_t* dst, size_t n) {
    uint8_t mode = in.u8();
    if (mode == 1) { int16_t v = int16_t(in.var()); std::fill(dst, dst + n, v); return; }
    if (mode == 0) { std::memcpy(dst, in.take(n * 2), n * 2); return; }
    if (mode != 6) throw std::runtime_error("stream mode");
    size_t c = in.var(); fse4_dec(dst, n, in.take(c), c);
}

template <class F> static void parallel_for(int n, int threads, F f) {
    if (threads <= 1 || n <= 1) { for (int i = 0; i < n; ++i) f(i); return; }
    std::atomic<int> next{0}; std::vector<std::thread> th;
    for (int t = 0; t < std::min(threads, n); ++t) th.emplace_back([&] { for (int i; (i = next.fetch_add(1)) < n;) f(i); });
    for (auto& t : th) t.join();
}

// ------------------------------------------------------------------ skewed 8-row bands (wavefront MED, 8 lanes)
// Band of 8 rows; pixel (x, k) lives at (x + k) * 8 + k, so vector t holds the antidiagonal (t - k, k), k = 0..7.
// MED neighbours of vector t: a = V(t-1), b = V(t-1) shifted up one lane (lane 0 = row above the band, x = t),
// c = b of step t-1. Borders are virtual zeros, which gives exactly the raster rules (row 0: P = a, column 0: P = b).
static inline void tr8x8(__m128i& r0, __m128i& r1, __m128i& r2, __m128i& r3, __m128i& r4, __m128i& r5, __m128i& r6, __m128i& r7) {
    __m128i a0 = _mm_unpacklo_epi16(r0, r1), a1 = _mm_unpackhi_epi16(r0, r1), a2 = _mm_unpacklo_epi16(r2, r3), a3 = _mm_unpackhi_epi16(r2, r3);
    __m128i a4 = _mm_unpacklo_epi16(r4, r5), a5 = _mm_unpackhi_epi16(r4, r5), a6 = _mm_unpacklo_epi16(r6, r7), a7 = _mm_unpackhi_epi16(r6, r7);
    __m128i b0 = _mm_unpacklo_epi32(a0, a2), b1 = _mm_unpackhi_epi32(a0, a2), b2 = _mm_unpacklo_epi32(a1, a3), b3 = _mm_unpackhi_epi32(a1, a3);
    __m128i b4 = _mm_unpacklo_epi32(a4, a6), b5 = _mm_unpackhi_epi32(a4, a6), b6 = _mm_unpacklo_epi32(a5, a7), b7 = _mm_unpackhi_epi32(a5, a7);
    r0 = _mm_unpacklo_epi64(b0, b4); r1 = _mm_unpackhi_epi64(b0, b4); r2 = _mm_unpacklo_epi64(b1, b5); r3 = _mm_unpackhi_epi64(b1, b5);
    r4 = _mm_unpacklo_epi64(b2, b6); r5 = _mm_unpackhi_epi64(b2, b6); r6 = _mm_unpacklo_epi64(b3, b7); r7 = _mm_unpackhi_epi64(b3, b7);
}
static inline size_t skew_len(int w, int h) { return size_t(w) * h; }      // only real pixels are stored, no padding
// raster (stride w) -> skewed bands; rows of the band are first copied into zero-padded rows (8 left, 16 right)
static void skew_pack(const int16_t* R, int w, int h, int16_t* S) {
    const int L = w + 7, PW = w + 32; thread_local std::vector<int16_t> pad; pad.assign(size_t(8) * PW, 0);
    thread_local std::vector<int16_t> band; band.resize(size_t(((L + 7) & ~7) + 8) * 8);
    for (int b0 = 0; b0 < h; b0 += 8) {
        for (int k = 0; k < 8; ++k) { int16_t* row = pad.data() + size_t(k) * PW; std::fill(row, row + PW, int16_t(0)); if (b0 + k < h) std::memcpy(row + 8, R + size_t(b0 + k) * w, size_t(w) * 2); }
        for (int t = 0; t < L; t += 8) {                       // row k, x = t - k .. t - k + 7 -> transpose -> vectors t .. t+7
            __m128i v[8]; for (int k = 0; k < 8; ++k) v[k] = _mm_loadu_si128((const __m128i*)(pad.data() + size_t(k) * PW + 8 + t - k));
            tr8x8(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            for (int j = 0; j < 8; ++j) _mm_storeu_si128((__m128i*)(band.data() + size_t(t + j) * 8), v[j]);
        }
        const int hb = std::min(8, h - b0);                    // store only the lanes holding real pixels
        for (int t = 0; t < L; ++t) {
            int klo = std::max(0, t - w + 1), khi = std::min(hb - 1, t);
            if (klo > khi) continue;
            std::memcpy(S, band.data() + size_t(t) * 8 + klo, size_t(khi - klo + 1) * 2); S += khi - klo + 1;
        }
    }
}
// fused encoder pass: residual rows of a band go straight into the zero-padded band rows, then transpose; full
// antidiagonals leave with one 16-byte store, only the band triangles are copied piecewise
template <bool N9, int PM> static void residuals_pack(const int16_t* P, int w, int h, int16_t* S) {
    const int L = w + 7, PW = w + 32; thread_local std::vector<int16_t> pad; pad.assign(size_t(8) * PW, 0);
    thread_local std::vector<int16_t> band; band.resize(size_t(((L + 7) & ~7) + 8) * 8);
    for (int b0 = 0; b0 < h; b0 += 8) {
        const int hb = std::min(8, h - b0);
        for (int k = 0; k < 8; ++k) { int16_t* row = pad.data() + size_t(k) * PW + 8;
            if (k < hb) residual_row<N9, PM>(P, w, b0 + k, row); else std::fill(row, row + w, int16_t(0)); }
        for (int t = 0; t < L; t += 8) {
            __m128i v[8]; for (int k = 0; k < 8; ++k) v[k] = _mm_loadu_si128((const __m128i*)(pad.data() + size_t(k) * PW + 8 + t - k));
            tr8x8(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            for (int j = 0; j < 8; ++j) _mm_store_si128((__m128i*)(band.data() + size_t(t + j) * 8), v[j]);
        }
        for (int t = 0; t < L; ++t) {
            int klo = std::max(0, t - w + 1), khi = std::min(hb - 1, t);
            if (klo == 0 && khi == 7) { _mm_storeu_si128((__m128i*)S, _mm_load_si128((const __m128i*)(band.data() + size_t(t) * 8))); S += 8; continue; }
            if (klo > khi) continue;
            std::memcpy(S, band.data() + size_t(t) * 8 + klo, size_t(khi - klo + 1) * 2); S += khi - klo + 1;
        }
    }
}
template <bool N9> static void residuals_pack_m(int m, const int16_t* P, int w, int h, int16_t* S) {
    switch (m) { case 0: residuals_pack<N9, 0>(P, w, h, S); break; case 1: residuals_pack<N9, 1>(P, w, h, S); break;
                 case 2: residuals_pack<N9, 2>(P, w, h, S); break; default: residuals_pack<N9, 3>(P, w, h, S); }
}
// in place over the skewed residuals: zigzag residuals in, plane values out; then back to raster rows of P (stride w)
template <bool N9> static void skew_restore(const int16_t* S, int w, int h, int16_t* P) {
    const int L = w + 7, PW = w + 32; const __m128i one = _mm_set1_epi16(1), zero = _mm_setzero_si128(), m255 = _mm_set1_epi16(255);
    thread_local std::vector<int16_t> band, pad; band.resize(size_t(((L + 7) & ~7) + 8) * 8); pad.resize(size_t(8) * PW);
    alignas(16) int16_t lane[8];
    for (int b0 = 0; b0 < h; b0 += 8) {
        const int16_t* top = b0 ? P + size_t(b0 - 1) * w : nullptr;
        const int hb = std::min(8, h - b0);
        __m128i V = zero, Bp = zero;
        for (int t = 0; t < L; ++t) {
            int klo = std::max(0, t - w + 1), khi = std::min(hb - 1, t);
            __m128i Z;
            if (klo == 0 && khi == 7) { Z = _mm_loadu_si128((const __m128i*)S); S += 8; }       // full antidiagonal
            else {                                                                              // band triangles / short band
                std::memset(lane, 0, sizeof lane);
                if (klo <= khi) { std::memcpy(lane + klo, S, size_t(khi - klo + 1) * 2); S += khi - klo + 1; }
                Z = _mm_load_si128((const __m128i*)lane);
            }
            __m128i e = _mm_xor_si128(_mm_srli_epi16(Z, 1), _mm_sub_epi16(zero, _mm_and_si128(Z, one)));
            __m128i b = _mm_insert_epi16(_mm_slli_si128(V, 2), (top && t < w) ? top[t] : 0, 0), c = Bp, a = V;
            __m128i mx = _mm_max_epi16(a, b), mn = _mm_min_epi16(a, b);
            __m128i pr = _mm_min_epi16(_mm_max_epi16(_mm_sub_epi16(_mm_add_epi16(a, b), c), mn), mx);
            __m128i v = _mm_add_epi16(pr, e);
            V = N9 ? _mm_srai_epi16(_mm_slli_epi16(v, 7), 7) : _mm_and_si128(v, m255);
            _mm_storeu_si128((__m128i*)(band.data() + size_t(t) * 8), V); Bp = b;
        }
        std::memset(band.data() + size_t(L) * 8, 0, size_t(((L + 7) & ~7) + 8 - L) * 16);
        for (int t = 0; t < L; t += 8) {                       // vectors t .. t+7 -> transpose -> row k, x = t - k ..
            __m128i v[8]; for (int j = 0; j < 8; ++j) v[j] = _mm_loadu_si128((const __m128i*)(band.data() + size_t(t + j) * 8));
            tr8x8(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            for (int k = 0; k < 8; ++k) _mm_storeu_si128((__m128i*)(pad.data() + size_t(k) * PW + 8 + t - k), v[k]);
        }
        for (int k = 0; k < 8 && b0 + k < h; ++k) std::memcpy(P + size_t(b0 + k) * w, pad.data() + size_t(k) * PW + 8, size_t(w) * 2);
    }
}


// ---- fused decoder for -s: wavefront MED of a band for all three planes, then transpose + inverse RCT straight to RGB
// one wavefront step; Z = residual vector (zigzag for N9)
template <bool N9, int PM = 0, bool BL = false> static inline __m128i wave_step(__m128i Z, __m128i& V, __m128i& Bp, int16_t topv, __m128i bm = _mm_setzero_si128()) {
    const __m128i one = _mm_set1_epi16(1), zero = _mm_setzero_si128(), m255 = _mm_set1_epi16(255);
    Z = _mm_and_si128(Z, _mm_set1_epi16(0x0fff));                    // FSE writes sym | nb << 12
    __m128i e = _mm_xor_si128(_mm_srli_epi16(Z, 1), _mm_sub_epi16(zero, _mm_and_si128(Z, one)));
    __m128i b = _mm_insert_epi16(_mm_slli_si128(V, 2), topv, 0), c = Bp, a = V;
    __m128i pr = pred_v<PM>(a, b, c);
    if (BL) pr = _mm_blendv_epi8(pr, pred_v<0>(a, b, c), bm);          // border lanes (x = 0 / y = 0) predict as MED
    __m128i v = _mm_add_epi16(pr, e);
    V = N9 ? _mm_srai_epi16(_mm_slli_epi16(v, 7), 7) : _mm_and_si128(v, m255); Bp = b; return V;
}
// g_wsplit: 0 = per-step klo/khi (old), 1 = three phases (triangle, full middle, triangle)
static int g_wsplit = 1;
static inline __m128i ld8(const int16_t* p) { return _mm_loadu_si128((const __m128i*)p); }
template <bool N9, int PM = 0, class RT = int16_t> static const RT* wave_band(const RT* S, int w, int hb, const int16_t* top, int16_t* band, bool first = false) {
    const int L = w + 7; const __m128i zero = _mm_setzero_si128();
    const __m128i idx = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7); __m128i V = zero, Bp = zero;
    // MED-on-border lanes for PM != 0: lane k at step k is column 0; lane 0 of the tile's first band is row 0
    const __m128i lane0 = first ? _mm_cmpeq_epi16(idx, zero) : zero;
    auto tri = [&](int t) {   // band triangle: load 8 so that lane klo = S[0], keep lanes klo..khi (the stream is padded, earlier data exists)
        int klo = std::max(0, t - w + 1), khi = std::min(hb - 1, t); __m128i Z;
        if (klo == 0 && khi == 7) { Z = ld8(S); S += 8; }
        else if (khi < klo) Z = zero;
        else { __m128i m = _mm_and_si128(_mm_cmpgt_epi16(idx, _mm_set1_epi16(short(klo - 1))), _mm_cmpgt_epi16(_mm_set1_epi16(short(khi + 1)), idx));
               Z = _mm_and_si128(ld8(S - klo), m); S += khi - klo + 1; }
        __m128i r;
        if (PM == 0) r = wave_step<N9, 0>(Z, V, Bp, top[t]);
        else r = wave_step<N9, PM, true>(Z, V, Bp, top[t], _mm_or_si128(lane0, _mm_cmpeq_epi16(idx, _mm_set1_epi16(short(t)))));
        _mm_store_si128((__m128i*)(band + size_t(t) * 8), r);
    };
    if (!g_wsplit || hb < 8 || w < 9 || (PM != 0 && first)) { for (int t = 0; t < L; ++t) tri(t); return S; }
    const int t0 = PM == 0 ? 7 : 8;                                // PM != 0: step 7 still has a column-0 lane
    for (int t = 0; t < t0; ++t) tri(t);
    for (int t = t0; t < w; ++t) { _mm_store_si128((__m128i*)(band + size_t(t) * 8), wave_step<N9, PM>(ld8(S), V, Bp, top[t])); S += 8; }
    for (int t = w; t < L; ++t) tri(t);
    return S;
}
static inline void irct8(__m128i v, __m128i u, __m128i y, uint8_t* out24) {        // 8 px -> 24 bytes RGB
    const __m128i ff = _mm_set1_epi16(255), z = _mm_setzero_si128();
    __m128i g = _mm_sub_epi16(y, _mm_srai_epi16(_mm_add_epi16(u, v), 2));
    __m128i R8 = _mm_packus_epi16(_mm_and_si128(_mm_add_epi16(v, g), ff), z), G8 = _mm_packus_epi16(_mm_and_si128(g, ff), z);
    __m128i B8 = _mm_packus_epi16(_mm_and_si128(_mm_add_epi16(u, g), ff), z);
    __m128i o0 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(R8, _mm_load_si128((const __m128i*)SHUF[0][0])), _mm_shuffle_epi8(G8, _mm_load_si128((const __m128i*)SHUF[0][1]))), _mm_shuffle_epi8(B8, _mm_load_si128((const __m128i*)SHUF[0][2])));
    __m128i o1 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(R8, _mm_load_si128((const __m128i*)SHUF[1][0])), _mm_shuffle_epi8(G8, _mm_load_si128((const __m128i*)SHUF[1][1]))), _mm_shuffle_epi8(B8, _mm_load_si128((const __m128i*)SHUF[1][2])));
    _mm_storeu_si128((__m128i*)out24, o0); _mm_storel_epi64((__m128i*)(out24 + 16), o1);
}
template <bool N9> static const int16_t* wave_band_m(int m, const int16_t* S, int w, int hb, const int16_t* top, int16_t* band, bool first) {
    switch (m) { case 0: return wave_band<N9, 0>(S, w, hb, top, band); case 1: return wave_band<N9, 1>(S, w, hb, top, band, first);
                 case 2: return wave_band<N9, 2>(S, w, hb, top, band, first); default: return wave_band<N9, 3>(S, w, hb, top, band, first); }
}
template <class YT> static void decode_tile_skew(const int16_t* S0, const int16_t* S1, const YT* S2, int w, int h, uint8_t* rgb, size_t rstride, const uint8_t* pm = nullptr) {
    static const uint8_t pm0[3] = {0, 0, 0}; if (!pm) pm = pm0;
    const int L = w + 7, LB = ((L + 7) & ~7) + 8;
    thread_local std::vector<int16_t> band[3], top[3];
    for (int c = 0; c < 3; ++c) { band[c].assign(size_t(LB) * 8, 0); top[c].assign(size_t(w) + 48, 0); }
    const int16_t* S[2] = {S0, S1}; const YT* SY = S2;
    alignas(16) uint8_t tmp[32];
    for (int b0 = 0; b0 < h; b0 += 8) {
        const int hb = std::min(8, h - b0);
        S[0] = wave_band_m<true>(pm[0], S[0], w, hb, top[0].data() + 8, band[0].data(), b0 == 0);
        S[1] = wave_band_m<true>(pm[1], S[1], w, hb, top[1].data() + 8, band[1].data(), b0 == 0);
        SY = wave_band_m<false>(pm[2], SY, w, hb, top[2].data() + 8, band[2].data(), b0 == 0);
        for (int t = 0; t < L; t += 8) {
            __m128i P[3][8];
            for (int c = 0; c < 3; ++c) { for (int j = 0; j < 8; ++j) P[c][j] = _mm_load_si128((const __m128i*)(band[c].data() + size_t(t + j) * 8));
                tr8x8(P[c][0], P[c][1], P[c][2], P[c][3], P[c][4], P[c][5], P[c][6], P[c][7]); }
            for (int k = 0; k < hb; ++k) {                         // row k, pixels x = t - k .. t - k + 7
                int x0 = t - k; uint8_t* row = rgb + size_t(b0 + k) * rstride;
                if (x0 >= 0 && x0 + 8 <= w) irct8(P[0][k], P[1][k], P[2][k], row + size_t(x0) * 3);
                else { irct8(P[0][k], P[1][k], P[2][k], tmp); int xa = std::max(0, x0), xb = std::min(w, x0 + 8); if (xa < xb) std::memcpy(row + size_t(xa) * 3, tmp + (xa - x0) * 3, size_t(xb - xa) * 3); }
                if (k == hb - 1) for (int c = 0; c < 3; ++c) _mm_storeu_si128((__m128i*)(top[c].data() + 8 + x0), P[c][k]);   // row above the next band
            }
        }
    }
}

// ---- cross-chroma residual prediction (per tile). V = R-G and U = B-G share -G, so their MED residuals correlate:
// target' = wrap9(e_t - ((k * e_s + 4) >> 3)), k in [-8, 8], dir 1: U from V, dir 2: V from U. Works on the zigzag
// residual streams (skew order: same index = same pixel); the decoder adds the term back before the wave.
// byte: dir | (k + 8) << 2, 0 = off
static int g_xres = 0, g_xstep = 11;                                 // -X 1 on (default off), -Xs probe stride (samples)
static std::atomic<unsigned> g_xstat[3];
static inline int unzz_i(int z) { return (z >> 1) ^ -(z & 1); }
static inline int wrap9(int e) { return ((e + 256) & 511) - 256; }
static double hist_bits(const unsigned* c, unsigned n) { double t = n ? double(n) * std::log2(double(n)) : 0; for (int v = 0; v < 512; ++v) if (c[v]) t -= double(c[v]) * std::log2(double(c[v])); return t; }
static int g_xfull = 0, g_xclip = 8;                                             // -Xf 1: exhaustive k search (slow, for reference)
static uint8_t choose_xres(const int16_t* zV, const int16_t* zU, size_t n) {
    if (!g_xres || n < 64) return 0;
    const size_t st = size_t(std::max(1, g_xstep)); thread_local std::vector<int16_t> eV, eU; eV.resize(n / st + 1); eU.resize(n / st + 1);
    unsigned m = 0; double sVV = 0, sUU = 0, sUV = 0;
    for (size_t i = 0; i < n; i += st, ++m) { const int v = unzz_i(zV[i] & 0x0fff), u = unzz_i(zU[i] & 0x0fff); eV[m] = int16_t(v); eU[m] = int16_t(u);
        if (std::abs(v) <= g_xclip && std::abs(u) <= g_xclip) { sVV += v * v; sUU += u * u; sUV += u * v; } }   // robust: small residuals only
    unsigned c[512];
    auto H = [&](const std::vector<int16_t>& t, const std::vector<int16_t>& src, int k) {
        std::memset(c, 0, sizeof c);
        for (unsigned i = 0; i < m; ++i) { int e = wrap9(t[i] - ((k * src[i] + 4) >> 3)); c[uint16_t(zz(e)) & 511]++; }
        return hist_bits(c, m); };
    const double hU = H(eU, eV, 0), hV = H(eV, eU, 0);
    double best = 0; uint8_t bx = 0;
    auto tryk = [&](int dir, int k) {
        if (k == 0 || k < -8 || k > 8) return;
        const double g = dir == 1 ? hU - H(eU, eV, k) : hV - H(eV, eU, k);
        if (g > best) { best = g; bx = uint8_t(dir | (k + 8) << 2); } };
    if (g_xfull) { for (int k = -8; k <= 8; ++k) { tryk(1, k); tryk(2, k); } }
    else {                                                            // robust slope, then descend on the probe entropy
        auto cost = [&](int dir, int k) { return k == 0 ? 0.0 : (dir == 1 ? hU - H(eU, eV, k) : hV - H(eV, eU, k)); };
        for (int dir = 1; dir <= 2; ++dir) {
            const double sxx = dir == 1 ? sVV : sUU; int k = sxx > 0 ? int(std::lround(8.0 * sUV / sxx)) : 0; k = std::max(-8, std::min(8, k));
            double g = cost(dir, k);
            for (int stp : {1, -1}) for (;;) { const int k2 = k + stp; if (k2 < -8 || k2 > 8) break; const double g2 = cost(dir, k2); if (g2 > g) { g = g2; k = k2; } else break; }
            if (k && g > best) { best = g; bx = uint8_t(dir | (k + 8) << 2); }
        }
    }
    return best * double(st) > 16.0 ? bx : 0;                        // must beat the header byte with margin
}
static void xres_apply(uint8_t xr, int16_t* zV, int16_t* zU, size_t n, bool dec) {
    const int dir = xr & 3, k = (xr >> 2) - 8; if (!dir) return;
    int16_t* t = dir == 1 ? zU : zV; const int16_t* src = dir == 1 ? zV : zU; size_t i = 0;
    if (dec) {                                                        // SIMD: t = zz(wrap9(unzz(t) + ((k * unzz(s) + 4) >> 3)))
        const __m128i m12 = _mm_set1_epi16(0x0fff), one = _mm_set1_epi16(1), z = _mm_setzero_si128(), kv = _mm_set1_epi16(short(k)), r4 = _mm_set1_epi16(4);
        for (; i + 8 <= n; i += 8) {
            __m128i zt = _mm_and_si128(_mm_loadu_si128((const __m128i*)(t + i)), m12), zs = _mm_and_si128(_mm_loadu_si128((const __m128i*)(src + i)), m12);
            __m128i et = _mm_xor_si128(_mm_srli_epi16(zt, 1), _mm_sub_epi16(z, _mm_and_si128(zt, one)));
            __m128i es = _mm_xor_si128(_mm_srli_epi16(zs, 1), _mm_sub_epi16(z, _mm_and_si128(zs, one)));
            __m128i e = _mm_add_epi16(et, _mm_srai_epi16(_mm_add_epi16(_mm_mullo_epi16(es, kv), r4), 3));
            e = _mm_srai_epi16(_mm_slli_epi16(e, 7), 7);
            _mm_storeu_si128((__m128i*)(t + i), _mm_xor_si128(_mm_add_epi16(e, e), _mm_srai_epi16(e, 15)));
        }
    }
    for (; i < n; ++i) { const int es = unzz_i(src[i] & 0x0fff), et = unzz_i(t[i] & 0x0fff); t[i] = int16_t(zz(wrap9(dec ? et + ((k * es + 4) >> 3) : et - ((k * es + 4) >> 3)))); }
}

static bool g_raw = false, g_skew = true;                          // -n: no entropy coding, -s: skewed 8-row bands
// file: "HLC3" var W var H u8 flags var TW var TH, then per tile (raster order): var size, payload = [u8 predictors: 2 bits per plane
// V,U,Y if flags bit 2] 3 x stream
std::vector<uint8_t> compress(const Img& im, int threads) {
    const int W = im.W, H = im.H, TW = g_tw, TH = g_th, tx = (W + TW - 1) / TW, ty = (H + TH - 1) / TH, nt = tx * ty;
    std::vector<std::vector<uint8_t>> tiles(nt);
    parallel_for(nt, threads, [&](int t) {
        int x0 = (t % tx) * TW, y0 = (t / tx) * TH, w = std::min(TW, W - x0), h = std::min(TH, H - y0); size_t n = size_t(w) * h;
        thread_local std::vector<int16_t> P[3], SK; thread_local std::vector<uint8_t> tmp;
        for (auto& p : P) p.resize(n); const size_t ns = g_skew ? skew_len(w, h) : n; SK.resize(ns);
        for (int y = 0; y < h; ++y) rct_row(im.rgb.data() + (size_t(y0 + y) * W + x0) * 3, w, P[0].data() + size_t(y) * w, P[1].data() + size_t(y) * w, P[2].data() + size_t(y) * w);
        Out o;
        thread_local std::vector<int16_t> SKp[3];
        const uint16_t* src[3]; uint8_t pm = 0;
        for (int c = 0; c < 3; ++c) {
            if (g_skew) {                                                // predictor per plane (entropy probe), residuals straight to skew order
                const int m = c < 2 ? choose_pred<true>(P[c].data(), w, h) : choose_pred<false>(P[c].data(), w, h);
                pm |= uint8_t(m << (2 * c)); g_pstat[c][m]++;
                SKp[c].resize(ns + 8);
                if (c < 2) residuals_pack_m<true>(m, P[c].data(), w, h, SKp[c].data()); else residuals_pack_m<false>(m, P[c].data(), w, h, SKp[c].data());
                src[c] = (const uint16_t*)SKp[c].data();
                continue;
            }
            if (c < 2) residuals_inplace<true>(P[c].data(), w, h); else residuals_inplace<false>(P[c].data(), w, h);
            src[c] = (const uint16_t*)P[c].data();
        }
        uint8_t xr = 0;
        if (g_skew) { xr = choose_xres(SKp[0].data(), SKp[1].data(), ns); g_xstat[xr & 3]++; if (xr) xres_apply(xr, SKp[0].data(), SKp[1].data(), ns, false); o.u8(pm); o.u8(xr); }
        if (g_raw) { for (int c = 0; c < 3; ++c) o.raw(src[c], ns * 2); }
        else if (!(g_joint && g_skew && joint_put(o, src, ns, g_K, tmp))) for (int c = 0; c < 3; ++c) put_stream(o, src[c], ns, tmp, c == 2 ? 256 : 512);
        tiles[t] = std::move(o.b);
    });
    Out f; f.raw("HLC3", 4); f.var(W); f.var(H); f.u8(uint8_t(g_raw | g_skew << 1 | (g_skew ? 4 | 8 : 0))); f.var(TW); f.var(TH);   // bit 2: per-tile predictor byte, bit 3: cross-chroma byte
    for (int t = 0; t < nt; ++t) { f.var(tiles[t].size()); f.raw(tiles[t].data(), tiles[t].size()); }
    return std::move(f.b);
}

// size estimate only: MED + skew over the whole image width (one wave per 8-row band), FSE statistics per
// "parallelogram" block = wave steps t in [jB, (j+1)B) of R/8 consecutive bands; returns payload bytes
size_t estimate_sb(const Img& im, int B, int R) {
    const int W = im.W, H = im.H; size_t n = size_t(W) * H; std::vector<int16_t> P[3], SK(n); std::vector<uint8_t> tmp; size_t total = 0;
    for (auto& p : P) p.resize(n);
    for (int y = 0; y < H; ++y) rct_row(im.rgb.data() + size_t(y) * W * 3, W, P[0].data() + size_t(y) * W, P[1].data() + size_t(y) * W, P[2].data() + size_t(y) * W);
    const int nb = std::max(1, (W + B / 2) / B), bands_per = std::max(1, R / 8), nbands = (H + 7) / 8, nrows = (nbands + bands_per - 1) / bands_per;
    for (int c = 0; c < 3; ++c) {
        if (c < 2) residuals_inplace<true>(P[c].data(), W, H); else residuals_inplace<false>(P[c].data(), W, H);
        skew_pack(P[c].data(), W, H, SK.data());
        std::vector<std::vector<uint16_t>> G(size_t(nb) * nrows); const int16_t* S = SK.data();
        for (int b = 0; b < nbands; ++b) { const int hb = std::min(8, H - 8 * b), L = W + 7;
            for (int t = 0; t < L; ++t) { int klo = std::max(0, t - W + 1), khi = std::min(hb - 1, t); if (klo > khi) continue;
                auto& g = G[size_t(b / bands_per) * nb + std::min(t / B, nb - 1)]; for (int k = klo; k <= khi; ++k) g.push_back(uint16_t(*S++)); } }
        for (auto& g : G) if (!g.empty()) { Out o; put_stream(o, g.data(), g.size(), tmp); total += o.b.size() + 2; }
    }
    return total;
}

// ================================================================== HLC4: lanes + decoupled FSE blocks
// lane  = R rows x full width; MED wave runs across the whole width (8-row bands), prediction resets at the lane top,
//         lanes are independent (threads).
// block = FSE statistics region: wave steps t in [jB, (j+1)B) of all bands of the lane (a parallelogram, skew 7 px);
//         the last block of a band takes the tail t >= (nb-1)B. One stream per plane per block.
// lane payload: for plane c: for block j: stream.
static int g_B = 128, g_R = 128, g_dord = 1, g_legacy = 1;           // -B, -R, -dord 0 per band / 1 per block, -L: HLC3
static inline int lane_nb(int W, int B) { return std::max(1, (W + B / 2) / B); }
// symbols of band (height hb) in steps [t0, t1)
static inline size_t seg_count(int W, int hb, int t0, int t1) {
    size_t n = 0; for (int t = t0; t < t1; ++t) { int klo = std::max(0, t - W + 1), khi = std::min(hb - 1, t); if (klo <= khi) n += size_t(khi - klo + 1); } return n;
}
// NCount header -> flat decode table T: low u16 newState, high u16 sx = symbol | nbBits << 12 (sx goes to the output
// as is, the wave masks it with 0x0fff); returns header size
static size_t read_dtable(const uint8_t* src, size_t c, std::vector<uint32_t>& T, unsigned& tl) {
    short norm[1024]; unsigned mx = 1023;
    size_t h = FSE_readNCount(norm, &mx, &tl, src, c); if (FSE_isError(h) || tl > 13 || tl < 1) throw std::runtime_error("ncount");
    uint16_t nxt[1024]; const uint32_t ts = 1u << tl, mask = ts - 1, step = (ts >> 1) + (ts >> 3) + 3; uint32_t hi = ts - 1; T.resize(ts);
    for (unsigned s = 0; s <= mx; ++s) { if (norm[s] == -1) { T[hi--] = s; nxt[s] = 1; } else nxt[s] = uint16_t(norm[s]); }
    uint32_t pos = 0;
    for (unsigned s = 0; s <= mx; ++s) for (int k = 0; k < norm[s]; ++k) { T[pos] = s; pos = (pos + step) & mask; while (pos > hi) pos = (pos + step) & mask; }
    if (pos) throw std::runtime_error("ncount spread");
    for (uint32_t u = 0; u < ts; ++u) { uint32_t s = T[u], ns = nxt[s]++, nb = tl - bw(ns) + 1; T[u] = (((ns << nb) - ts) & 0xffff) | (s | nb << 12) << 16; }   // {u16 newState, u16 sym | nb << 12}
    return h;
}
// resumable reader: raw / rle / FSE4 with 4 interleaved states (symbol g uses state g & 3, reload when g % 4 == 0)
struct RStream {
    uint8_t mode = 0; const uint8_t* raw = nullptr; int16_t rle = 0;
    BIT_DStream_t bd; uint32_t st[8] = {}, g = 0, S = 4; std::vector<uint32_t> T;
    void open(In& in, size_t n) {
        mode = in.u8();
        if (mode == 1) { rle = int16_t(in.var()); return; }
        if (mode == 0) { raw = in.take(n * 2); return; }
        S = 4;
        if (mode == 8) { S = in.u8(); if (S != 1 && S != 2 && S != 4 && S != 8) throw std::runtime_error("states"); }
        else if (mode != 6) throw std::runtime_error("stream mode");
        size_t c = in.var(); const uint8_t* src = in.take(c);
        unsigned tl; size_t h = read_dtable(src, c, T, tl);
        if (FSE_isError(BIT_initDStream(&bd, src + h, c - h))) throw std::runtime_error("bitstream");
        const size_t L = std::min<size_t>(n, S);
        for (size_t k = 0; k < L; ++k) { st[k] = uint32_t(BIT_readBits(&bd, tl)); BIT_reloadDStream(&bd); }
        g = 0;
    }
    void get(int16_t* out, size_t cnt) {
        if (mode == 1) { std::fill(out, out + cnt, rle); return; }
        if (mode == 0) { std::memcpy(out, raw, cnt * 2); raw += cnt * 2; return; }
        const uint32_t* t = T.data(); size_t i = 0; BIT_DStream_t b = bd; uint32_t s0 = st[0], s1 = st[1], s2 = st[2], s3 = st[3]; uint32_t* sp[4] = {&s0, &s1, &s2, &s3};
#define HLC_D(sv) { uint32_t e = t[sv], sx = e >> 16; out[i++] = int16_t(sx); sv = (e & 0xffff) + uint32_t(BIT_readBits(&b, sx >> 12)); }
        while (i < cnt && (g & 3)) { uint32_t& sv = *sp[g & 3]; HLC_D(sv) ++g; }
        for (; i + 4 <= cnt; g += 4) { BIT_reloadDStream(&b); HLC_D(s0) HLC_D(s1) HLC_D(s2) HLC_D(s3) }
        if (i < cnt) { BIT_reloadDStream(&b); while (i < cnt) { uint32_t& sv = *sp[g & 3]; HLC_D(sv) ++g; } }
#undef HLC_D
        bd = b; st[0] = s0; st[1] = s1; st[2] = s2; st[3] = s3;
    }
    template <int NS, class OT = int16_t> void all_t(OT* out, size_t n) {   // whole stream, NS states, reload every 4 symbols
        const uint32_t* t = T.data(); BIT_DStream_t b = bd; uint32_t s[NS]; for (int k = 0; k < NS; ++k) s[k] = st[k];
        constexpr int U = NS < 4 ? 4 : NS; size_t i = 0;
#define HLC_DS(sv) { uint32_t e = t[sv], sx = e >> 16; out[i + j] = OT(sx); sv = (e & 0xffff) + uint32_t(BIT_readBits(&b, sx >> 12)); }
        for (; i + U <= n; i += U) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 8
#endif
            for (int j = 0; j < U; ++j) { if ((j & 3) == 0) BIT_reloadDStream(&b); HLC_DS(s[j % NS]) }
        }
        for (int j = 0; i + size_t(j) < n; ++j) { if ((j & 3) == 0) BIT_reloadDStream(&b); HLC_DS(s[j % NS]) }
#undef HLC_DS
    }
    void get_all(int16_t* out, size_t n) {                          // fresh stream decoded at once (states stay in registers)
        if (mode != 6 && mode != 8) { get(out, n); return; }
        switch (S) { case 1: all_t<1>(out, n); break; case 2: all_t<2>(out, n); break; case 4: all_t<4>(out, n); break; default: all_t<8>(out, n); }
    }
};
// three FSE4 streams of equal length decoded in one loop: 3 bit readers x 4 states = 12 independent chains
static int g_dec3 = 0, g_al = 1;   // -al 0: old (unaligned) residual buffer start
                                             // -d3 0: decode the planes one after another
static void dec3(RStream* r, int16_t* o0, int16_t* o1, int16_t* o2, size_t n) {
    const uint32_t *t0 = r[0].T.data(), *t1 = r[1].T.data(), *t2 = r[2].T.data();
    BIT_DStream_t b0 = r[0].bd, b1 = r[1].bd, b2 = r[2].bd;
    uint32_t a0 = r[0].st[0], a1 = r[0].st[1], a2 = r[0].st[2], a3 = r[0].st[3];
    uint32_t c0 = r[1].st[0], c1 = r[1].st[1], c2 = r[1].st[2], c3 = r[1].st[3];
    uint32_t d0 = r[2].st[0], d1 = r[2].st[1], d2 = r[2].st[2], d3 = r[2].st[3];
    size_t i = 0;
#define HLC_D3(t, b, sv, o, k) { uint32_t e = t[sv], sx = e >> 16; o[i + k] = int16_t(sx); sv = (e & 0xffff) + uint32_t(BIT_readBits(&b, sx >> 12)); }
    for (; i + 4 <= n; i += 4) {
        BIT_reloadDStream(&b0); BIT_reloadDStream(&b1); BIT_reloadDStream(&b2);
        HLC_D3(t0, b0, a0, o0, 0) HLC_D3(t1, b1, c0, o1, 0) HLC_D3(t2, b2, d0, o2, 0)
        HLC_D3(t0, b0, a1, o0, 1) HLC_D3(t1, b1, c1, o1, 1) HLC_D3(t2, b2, d1, o2, 1)
        HLC_D3(t0, b0, a2, o0, 2) HLC_D3(t1, b1, c2, o1, 2) HLC_D3(t2, b2, d2, o2, 2)
        HLC_D3(t0, b0, a3, o0, 3) HLC_D3(t1, b1, c3, o1, 3) HLC_D3(t2, b2, d3, o2, 3)
    }
#undef HLC_D3
    r[0].bd = b0; r[1].bd = b1; r[2].bd = b2;
    r[0].st[0] = a0; r[0].st[1] = a1; r[0].st[2] = a2; r[0].st[3] = a3;
    r[1].st[0] = c0; r[1].st[1] = c1; r[1].st[2] = c2; r[1].st[3] = c3;
    r[2].st[0] = d0; r[2].st[1] = d1; r[2].st[2] = d2; r[2].st[3] = d3;
    for (int p = 0; p < 3; ++p) r[p].g = uint32_t(i);
    if (i < n) { r[0].get(o0 + i, n - i); r[1].get(o1 + i, n - i); r[2].get(o2 + i, n - i); }
}
// ---- mode 7: the three planes of a tile in ONE bitstream: symbol order Y_i U_i V_i Y_i+1 ...,
// K states per plane (state of (plane p, sample i) = p*K + i mod K), each plane its own table (tableLog <= 12),
// decoder reloads once per 4 symbols. Layout: u8 7, u8 K, 3 x NCount, var csize, bitstream.
static bool joint_put(Out& o, const uint16_t* const* s, size_t n, int K, std::vector<uint8_t>& tmp) {
    thread_local unsigned cnt[3][512]; thread_local std::vector<FSE_CTable> ct[3]; unsigned mx[3], tl[3]; short norm[1024];
    tmp.resize(n * 6 + 8192); uint8_t* op = tmp.data(); uint8_t* const oe = tmp.data() + tmp.size();
    for (int p = 0; p < 3; ++p) {
        std::memset(cnt[p], 0, sizeof cnt[p]); mx[p] = 0;
        for (size_t i = 0; i < n; ++i) cnt[p][s[p][i]]++;
        for (unsigned v = 0; v < 512; ++v) if (cnt[p][v]) mx[p] = v;
        if (cnt[p][s[p][0]] == n) return false;                      // constant plane: separate streams (rle)
        const unsigned tcap = g_tadapt ? std::min(12u, std::min(unsigned(g_tl), std::max(9u, bw(uint32_t(n)) - 1 - 4))) : std::min(12u, unsigned(g_tl));
        tl[p] = FSE_optimalTableLog(tcap, n, mx[p]); if (tl[p] > 12) return false;
        if (FSE_isError(FSE_normalizeCount(norm, tl[p], cnt[p], n, mx[p]))) return false;
        size_t h = FSE_writeNCount(op, size_t(oe - op), norm, mx[p], tl[p]); if (FSE_isError(h)) return false; op += h;
        ct[p].resize(FSE_CTABLE_SIZE_U32(13, 1023)); if (FSE_isError(FSE_buildCTableU16(ct[p].data(), norm, mx[p], tl[p]))) return false;
    }
    const size_t hdr = size_t(op - tmp.data());
    BIT_CStream_t bc; if (FSE_isError(BIT_initCStream(&bc, op, size_t(oe - op)))) return false;
    FSE_CState_t st[3][4]; const size_t L = std::min<size_t>(n, size_t(K));
    size_t i = n; int k = int((n - 1) % size_t(K));
    for (size_t j = 0; j < L; ++j, k = k ? k - 1 : K - 1) { --i; for (int p = 3; p-- > 0;) FSE_initCState2(&st[p][k], ct[p].data(), s[p][i]); }   // last sample of each lane
    while (i > 0) {
        --i; FSE_encodeSymbol(&bc, &st[2][k], s[2][i]); FSE_encodeSymbol(&bc, &st[1][k], s[1][i]); FSE_encodeSymbol(&bc, &st[0][k], s[0][i]);
        BIT_flushBits(&bc); k = k ? k - 1 : K - 1;                     // <= 3 x 12 bits per sample
    }
    for (int p = 3; p-- > 0;) for (size_t k = L; k-- > 0;) FSE_flushCState(&bc, &st[p][k]);   // decoder reads p = 0.., k = 0.. first
    size_t c = BIT_closeCStream(&bc); if (!c || hdr + c >= n * 6) return false;
    o.u8(7); o.u8(uint8_t(K)); o.raw(tmp.data(), hdr); o.var(c); o.raw(op, c);
    return true;
}
template <int K> static void joint_dec_t(const uint32_t* const* T, BIT_DStream_t& bd, uint32_t (*st)[4], int16_t* const* o, size_t n) {
    const uint32_t *t0 = T[0], *t1 = T[1], *t2 = T[2]; int16_t *o0 = o[0], *o1 = o[1], *o2 = o[2];
    uint32_t s0[K], s1[K], s2[K]; for (int k = 0; k < K; ++k) { s0[k] = st[0][k]; s1[k] = st[1][k]; s2[k] = st[2][k]; }
    BIT_DStream_t b = bd; constexpr int U = K == 3 ? 12 : 4;           // samples per unrolled round: lane index and reload slots are constants
    size_t i = 0;
#define HLC_J(t, sv, out) { uint32_t e = t[sv], sx = e >> 16; out[i + j] = int16_t(sx); sv = (e & 0xffff) + uint32_t(BIT_readBits(&b, sx >> 12)); }
    for (; i + U <= n; i += U) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (int j = 0; j < U; ++j) { const int k = j % K;
            if (((3 * j) & 3) == 0) BIT_reloadDStream(&b); HLC_J(t0, s0[k], o0)
            if (((3 * j + 1) & 3) == 0) BIT_reloadDStream(&b); HLC_J(t1, s1[k], o1)
            if (((3 * j + 2) & 3) == 0) BIT_reloadDStream(&b); HLC_J(t2, s2[k], o2) }
    }
    for (int j = 0; i + j < n; ++j) { const int k = j % K; BIT_reloadDStream(&b); HLC_J(t0, s0[k], o0) HLC_J(t1, s1[k], o1) HLC_J(t2, s2[k], o2) }   // tail: U | i, lanes continue
#undef HLC_J
}
static void joint_get(In& in, int16_t* const* o, size_t n) {
    if (in.u8() != 7) throw std::runtime_error("joint mode"); const int K = in.u8(); if (K < 1 || K > 4) throw std::runtime_error("joint K");
    thread_local std::vector<uint32_t> T[3]; unsigned tl[3];
    for (int p = 0; p < 3; ++p) { const uint8_t* q = in.p; size_t h = read_dtable(q, size_t(in.e - q), T[p], tl[p]); if (tl[p] > 12) throw std::runtime_error("joint tl"); in.take(h); }
    size_t c = in.var(); const uint8_t* src = in.take(c);
    BIT_DStream_t bd; if (FSE_isError(BIT_initDStream(&bd, src, c))) throw std::runtime_error("bitstream");
    uint32_t st[3][4] = {}; const size_t L = std::min<size_t>(n, size_t(K));
    for (int p = 0; p < 3; ++p) for (size_t k = 0; k < L; ++k) { st[p][k] = uint32_t(BIT_readBits(&bd, tl[p])); BIT_reloadDStream(&bd); }
    const uint32_t* tp[3] = {T[0].data(), T[1].data(), T[2].data()};
    switch (K) { case 1: joint_dec_t<1>(tp, bd, st, o, n); break; case 2: joint_dec_t<2>(tp, bd, st, o, n); break;
                 case 3: joint_dec_t<3>(tp, bd, st, o, n); break; default: joint_dec_t<4>(tp, bd, st, o, n); }
}
// fused lane decoder: fill(c, band) returns the band's skew residuals of plane c (8 valid slack values before, 16 after)
template <class F> static void decode_lane(int w, int h, uint8_t* rgb, size_t rstride, F fill) {
    const int L = w + 7, LB = ((L + 7) & ~7) + 8;
    thread_local std::vector<int16_t> band[3], top[3];
    for (int c = 0; c < 3; ++c) { band[c].resize(size_t(LB) * 8); top[c].assign(size_t(w) + 48, 0); }
    alignas(16) uint8_t tmp[32];
    for (int b0 = 0, bi = 0; b0 < h; b0 += 8, ++bi) {
        const int hb = std::min(8, h - b0);
        wave_band<true>(fill(0, bi), w, hb, top[0].data() + 8, band[0].data());
        wave_band<true>(fill(1, bi), w, hb, top[1].data() + 8, band[1].data());
        wave_band<false>(fill(2, bi), w, hb, top[2].data() + 8, band[2].data());
        for (int c = 0; c < 3; ++c) std::memset(band[c].data() + size_t(L) * 8, 0, size_t(LB - L) * 16);
        for (int t = 0; t < L; t += 8) {
            __m128i P[3][8];
            for (int c = 0; c < 3; ++c) { for (int j = 0; j < 8; ++j) P[c][j] = _mm_load_si128((const __m128i*)(band[c].data() + size_t(t + j) * 8));
                tr8x8(P[c][0], P[c][1], P[c][2], P[c][3], P[c][4], P[c][5], P[c][6], P[c][7]); }
            for (int k = 0; k < hb; ++k) {
                int x0 = t - k; uint8_t* row = rgb + size_t(b0 + k) * rstride;
                if (x0 >= 0 && x0 + 8 <= w) irct8(P[0][k], P[1][k], P[2][k], row + size_t(x0) * 3);
                else { irct8(P[0][k], P[1][k], P[2][k], tmp); int xa = std::max(0, x0), xb = std::min(w, x0 + 8); if (xa < xb) std::memcpy(row + size_t(xa) * 3, tmp + (xa - x0) * 3, size_t(xb - xa) * 3); }
                if (k == hb - 1) for (int c = 0; c < 3; ++c) _mm_storeu_si128((__m128i*)(top[c].data() + 8 + x0), P[c][k]);
            }
        }
    }
}
std::vector<uint8_t> compress4(const Img& im, int threads) {
    const int W = im.W, H = im.H, B = g_B, R = g_R, nl = (H + R - 1) / R, nb = lane_nb(W, B);
    std::vector<std::vector<uint8_t>> lanes(nl);
    parallel_for(nl, threads, [&](int l) {
        const int y0 = l * R, h = std::min(R, H - y0), L = W + 7; const size_t n = size_t(W) * h;
        thread_local std::vector<int16_t> P[3], SK; thread_local std::vector<uint8_t> tmp; thread_local std::vector<std::vector<uint16_t>> G;
        for (auto& p : P) p.resize(n); SK.resize(n); G.resize(nb);
        for (int y = 0; y < h; ++y) rct_row(im.rgb.data() + size_t(y0 + y) * W * 3, W, P[0].data() + size_t(y) * W, P[1].data() + size_t(y) * W, P[2].data() + size_t(y) * W);
        Out o;
        for (int c = 0; c < 3; ++c) {
            if (c < 2) residuals_inplace<true>(P[c].data(), W, h); else residuals_inplace<false>(P[c].data(), W, h);
            skew_pack(P[c].data(), W, h, SK.data());
            for (auto& g : G) g.clear();
            const uint16_t* S = (const uint16_t*)SK.data();
            for (int b0 = 0; b0 < h; b0 += 8) { const int hb = std::min(8, h - b0);
                for (int j = 0; j < nb; ++j) { int t0 = j * B, t1 = j == nb - 1 ? L : std::min(L, (j + 1) * B); size_t k = seg_count(W, hb, t0, t1); G[j].insert(G[j].end(), S, S + k); S += k; } }
            for (int j = 0; j < nb; ++j) {
                if (g_raw) { o.u8(0); o.raw(G[j].data(), G[j].size() * 2); }
                else if (G[j].empty()) { o.u8(1); o.var(0); }
                else put_stream(o, G[j].data(), G[j].size(), tmp, c == 2 ? 256 : 512);
            }
        }
        lanes[l] = std::move(o.b);
    });
    Out f; f.raw("HLC4", 4); f.var(W); f.var(H); f.var(B); f.var(R);
    for (int l = 0; l < nl; ++l) { f.var(lanes[l].size()); f.raw(lanes[l].data(), lanes[l].size()); }
    return std::move(f.b);
}
Img decompress4(In in, int threads) {
    Img im; im.W = int(in.var()); im.H = int(in.var()); const int W = im.W, H = im.H, B = int(in.var()), R = int(in.var());
    if (W < 1 || H < 1 || B < 8 || R < 8 || (R & 7)) throw std::runtime_error("header");
    const int nl = (H + R - 1) / R, nb = lane_nb(W, B);
    im.rgb.resize(size_t(W) * H * 3 + 64);
    std::vector<const uint8_t*> lp(nl); std::vector<size_t> ln(nl);
    for (int l = 0; l < nl; ++l) { ln[l] = in.var(); lp[l] = in.take(ln[l]); }
    parallel_for(nl, threads, [&](int l) {
        const int y0 = l * R, h = std::min(R, H - y0), L = W + 7, nbands = (h + 7) / 8;
        In li{lp[l], lp[l] + ln[l]};
        thread_local std::vector<RStream> rs; thread_local std::vector<size_t> segn, segoff; rs.resize(size_t(3) * nb);
        segn.assign(size_t(nbands) * nb, 0); segoff.assign(size_t(nbands) * nb + 1, 0); std::vector<size_t> tot(nb, 0);
        for (int bi = 0; bi < nbands; ++bi) { const int hb = std::min(8, h - 8 * bi);
            for (int j = 0; j < nb; ++j) { int t0 = j * B, t1 = j == nb - 1 ? L : std::min(L, (j + 1) * B); size_t k = seg_count(W, hb, t0, t1); segn[size_t(bi) * nb + j] = k; tot[j] += k; } }
        for (int c = 0; c < 3; ++c) for (int j = 0; j < nb; ++j) rs[size_t(c) * nb + j].open(li, tot[j]);
        uint8_t* out = im.rgb.data() + size_t(y0) * W * 3;
        if (g_dord == 0) {                                              // per band: decode B-segments of every block into a band buffer (L1)
            thread_local std::vector<int16_t> bb[3]; for (auto& v : bb) v.resize(size_t(L) * 8 + 32);
            decode_lane(W, h, out, size_t(W) * 3, [&](int c, int bi) {
                int16_t* d = bb[c].data() + 8; std::memset(bb[c].data(), 0, 16);
                for (int j = 0; j < nb; ++j) { size_t k = segn[size_t(bi) * nb + j]; rs[size_t(c) * nb + j].get(d, k); d += k; }
                return (const int16_t*)(bb[c].data() + 8); });
        } else {                                                        // per block: decode each stream whole (hot table) into the lane buffer
            thread_local std::vector<int16_t> lb[3]; thread_local std::vector<size_t> boff; const size_t n = size_t(W) * h;
            boff.assign(size_t(nbands) + 1, 0);
            for (int bi = 0; bi < nbands; ++bi) { size_t k = 0; for (int j = 0; j < nb; ++j) k += segn[size_t(bi) * nb + j]; boff[bi + 1] = boff[bi] + k; }
            for (int c = 0; c < 3; ++c) { lb[c].resize(n + 32); std::memset(lb[c].data(), 0, 16);
                for (int j = 0; j < nb; ++j) { size_t pre = 0; for (int jj = 0; jj < j; ++jj) pre += segn[jj];
                    for (int bi = 0; bi < nbands; ++bi) { size_t off = boff[bi]; for (int jj = 0; jj < j; ++jj) off += segn[size_t(bi) * nb + jj];
                        rs[size_t(c) * nb + j].get(lb[c].data() + 8 + off, segn[size_t(bi) * nb + j]); } } }
            decode_lane(W, h, out, size_t(W) * 3, [&](int c, int bi) { return (const int16_t*)(lb[c].data() + 8 + boff[bi]); });
        }
    });
    im.rgb.resize(size_t(W) * H * 3);
    return im;
}

// ================================================================== HLC5 (experiment): whole-image skew + activity contexts
// MED over the whole image (8-row skew bands across the full width, no tiles). Every sample gets a class from its
// causal neighbourhood: act = |a-c| + |c-b| + |a-aa| (aa = two to the left; the up-right d of pctx is not decoded
// yet in a wave step), class = #thresholds <= act (Y: 1 3 10 18 35, U/V: 1 4 8 16 30) -> 0..5.
// 11 streams: class 0 shared by all planes, classes 1..5 of Y -> 1..5, classes 1..5 of U/V -> 6..10.
// Symbol order inside a stream = decoder order: band, wave step t, plane V U Y, lane 0..7.
// Borders: x = 0 -> a = c = b; x <= 1 -> aa = a; y = 0 -> b = c = 0 (as the wave's zero top row).
// Each stream is cut into FSE chunks of -cb symbols (own table each; 0 = one table per stream).
// file: "HLC5" var W var H u8 flags (bit 0 raw) var CB, 11 x var n_i, then the chunks of stream 0, 1, ...
static const int TH5Y[5] = {1, 3, 10, 18, 35}, TH5C[5] = {1, 4, 8, 16, 30};
static int g_v5 = 0, g_cb = 0, g_g5 = 1;                           // -5, -cb, -g5 0: serial gather (old)
constexpr int NS5 = 11;
static unsigned long long g_c5stat[NS5], g_f5stat[2];                // bench: symbols per stream; full steps uniform / mixed class
template <bool N9> static inline int cls5(int act) { const int* th = N9 ? TH5C : TH5Y; int k = 0; for (int i = 0; i < 5; ++i) k += act >= th[i]; return k; }
static inline int sid5(int c, int k) { return k == 0 ? 0 : (c == 2 ? k : 5 + k); }
alignas(16) static const uint8_t ID5[2][8] = {{0, 6, 7, 8, 9, 10, 0, 0}, {0, 1, 2, 3, 4, 5, 0, 0}};   // [Y?][class]

std::vector<uint8_t> compress5(const Img& im) {
    const int W = im.W, H = im.H; const size_t n = size_t(W) * H;
    std::vector<int16_t> P[3], Z[3]; std::vector<uint8_t> K[3];
    for (int c = 0; c < 3; ++c) { P[c].resize(n); Z[c].resize(n + 8); K[c].resize(n); }
    for (int y = 0; y < H; ++y) rct_row(im.rgb.data() + size_t(y) * W * 3, W, P[0].data() + size_t(y) * W, P[1].data() + size_t(y) * W, P[2].data() + size_t(y) * W);
    for (int c = 0; c < 3; ++c) {
        const bool n9 = c < 2; const int16_t* p = P[c].data();
        for (int y = 0; y < H; ++y) {
            if (n9) residual_row<true, 0>(p, W, y, Z[c].data() + size_t(y) * W); else residual_row<false, 0>(p, W, y, Z[c].data() + size_t(y) * W);
            const int16_t* r = p + size_t(y) * W; const int16_t* u = r - W; uint8_t* kr = K[c].data() + size_t(y) * W;
            for (int x = 0; x < W; ++x) {
                int b = y ? u[x] : 0, cc = (y && x) ? u[x - 1] : 0, a = x ? r[x - 1] : b; if (!x) cc = b;
                int aa = x >= 2 ? r[x - 2] : a, act = std::abs(a - cc) + std::abs(cc - b) + std::abs(a - aa);
                kr[x] = uint8_t(n9 ? cls5<true>(act) : cls5<false>(act));
            }
        }
    }
    std::vector<uint16_t> S[NS5]; for (auto& s : S) s.reserve(n / 4);
    unsigned long long st[NS5] = {}, fu = 0, fm = 0;
    for (int b0 = 0; b0 < H; b0 += 8) { const int hb = std::min(8, H - b0), L = W + 7;
        for (int t = 0; t < L; ++t) { const int klo = std::max(0, t - W + 1), khi = std::min(hb - 1, t);
            for (int c = 0; c < 3; ++c) {
                if (hb == 8 && t >= 9 && t < W) { bool same = true; for (int k = 1; k < 8; ++k) same &= K[c][size_t(b0 + k) * W + (t - k)] == K[c][size_t(b0) * W + t]; (same ? fu : fm)++; }
                for (int k = klo; k <= khi; ++k) { size_t i = size_t(b0 + k) * W + (t - k); int id = sid5(c, K[c][i]); S[id].push_back(uint16_t(Z[c][i])); st[id]++; }
            }
        }
    }
    for (int i = 0; i < NS5; ++i) g_c5stat[i] += st[i]; g_f5stat[0] += fu; g_f5stat[1] += fm;
    Out f; f.raw("HLC5", 4); f.var(W); f.var(H); f.u8(uint8_t(g_raw)); f.var(size_t(g_cb));
    for (auto& s : S) f.var(s.size());
    std::vector<uint8_t> tmp;
    for (int i = 0; i < NS5; ++i) { const size_t m = S[i].size(), cb = g_cb > 0 ? size_t(g_cb) : std::max<size_t>(m, 1);
        for (size_t o = 0; o < m; o += cb) { const size_t k = std::min(cb, m - o);
            if (g_raw) f.raw(S[i].data() + o, k * 2); else put_stream(f, S[i].data() + o, k, tmp, (i >= 1 && i <= 5) ? 256 : 512); } }
    return std::move(f.b);
}
// one wave step of one plane with the context gather; FULL = all 8 lanes real and no border lane (x >= 2)
template <bool N9, bool FULL> static inline void step5(int t, int klo, int khi, __m128i& V, __m128i& Bp, __m128i& V2, int16_t topv,
                                                       const uint8_t* idm, const int16_t* sb, uint32_t* pos, int16_t* out) {
    const __m128i zero = _mm_setzero_si128(), one = _mm_set1_epi16(1);
    __m128i a = V, b = _mm_insert_epi16(_mm_slli_si128(V, 2), topv, 0), cc = Bp, aa = V2, a_ = a, c_ = cc, aa_ = aa;
    if (!FULL) { const __m128i idx = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7), m0 = _mm_cmpeq_epi16(idx, _mm_set1_epi16(short(t))), m1 = _mm_cmpeq_epi16(idx, _mm_set1_epi16(short(t - 1)));
                 a_ = _mm_blendv_epi8(a, b, m0); c_ = _mm_blendv_epi8(cc, b, m0); aa_ = _mm_blendv_epi8(aa, a_, _mm_or_si128(m0, m1)); }
    __m128i act = _mm_add_epi16(_mm_add_epi16(_mm_abs_epi16(_mm_sub_epi16(a_, c_)), _mm_abs_epi16(_mm_sub_epi16(c_, b))), _mm_abs_epi16(_mm_sub_epi16(a_, aa_)));
    const int* th = N9 ? TH5C : TH5Y; __m128i k = zero;
    for (int i = 0; i < 5; ++i) k = _mm_sub_epi16(k, _mm_cmpgt_epi16(act, _mm_set1_epi16(short(th[i] - 1))));
    __m128i Z; alignas(16) int16_t kl[8], zl[8];
    if (FULL) {
        const __m128i k0 = _mm_shuffle_epi8(k, _mm_set1_epi16(0x0100));
        if (g_g5 == 2 || _mm_movemask_epi8(_mm_cmpeq_epi16(k, k0)) == 0xffff) { const int id = idm[_mm_cvtsi128_si32(k) & 7]; Z = _mm_loadu_si128((const __m128i*)(sb + pos[id])); pos[id] += 8; }
        else if (g_g5 == 0) { _mm_store_si128((__m128i*)kl, k); for (int j = 0; j < 8; ++j) { const int id = idm[kl[j]]; zl[j] = sb[pos[id]++]; } Z = _mm_load_si128((const __m128i*)zl); }
        else {   // rank of each lane among the earlier lanes of its class -> 8 independent loads; classes + 1 so shifted-in zeros never match
            const __m128i k1 = _mm_add_epi16(k, one); __m128i r = zero;
            r = _mm_sub_epi16(r, _mm_cmpeq_epi16(k1, _mm_slli_si128(k1, 2)));  r = _mm_sub_epi16(r, _mm_cmpeq_epi16(k1, _mm_slli_si128(k1, 4)));
            r = _mm_sub_epi16(r, _mm_cmpeq_epi16(k1, _mm_slli_si128(k1, 6)));  r = _mm_sub_epi16(r, _mm_cmpeq_epi16(k1, _mm_slli_si128(k1, 8)));
            r = _mm_sub_epi16(r, _mm_cmpeq_epi16(k1, _mm_slli_si128(k1, 10))); r = _mm_sub_epi16(r, _mm_cmpeq_epi16(k1, _mm_slli_si128(k1, 12)));
            r = _mm_sub_epi16(r, _mm_cmpeq_epi16(k1, _mm_slli_si128(k1, 14)));
            alignas(16) int16_t rl[8]; _mm_store_si128((__m128i*)kl, k); _mm_store_si128((__m128i*)rl, r);
            uint32_t bs[8]; int ids[8];
            for (int j = 0; j < 8; ++j) { ids[j] = idm[kl[j]]; bs[j] = pos[ids[j]] + uint32_t(rl[j]); }
            Z = _mm_cvtsi32_si128(uint16_t(sb[bs[0]]));
            Z = _mm_insert_epi16(Z, sb[bs[1]], 1); Z = _mm_insert_epi16(Z, sb[bs[2]], 2); Z = _mm_insert_epi16(Z, sb[bs[3]], 3);
            Z = _mm_insert_epi16(Z, sb[bs[4]], 4); Z = _mm_insert_epi16(Z, sb[bs[5]], 5); Z = _mm_insert_epi16(Z, sb[bs[6]], 6); Z = _mm_insert_epi16(Z, sb[bs[7]], 7);
            for (int j = 0; j < 8; ++j) pos[ids[j]] = bs[j] + 1;   // in lane order: the last lane of a class leaves old + count
        }
    } else {
        _mm_store_si128((__m128i*)kl, k); _mm_store_si128((__m128i*)zl, zero);
        for (int j = klo; j <= khi; ++j) { const int id = idm[kl[j]]; zl[j] = sb[pos[id]++]; }
        Z = _mm_load_si128((const __m128i*)zl);
    }
    Z = _mm_and_si128(Z, _mm_set1_epi16(0x0fff));
    __m128i e = _mm_xor_si128(_mm_srli_epi16(Z, 1), _mm_sub_epi16(zero, _mm_and_si128(Z, one)));
    __m128i v = _mm_add_epi16(pred_v<0>(a, b, cc), e);
    V2 = V; V = N9 ? _mm_srai_epi16(_mm_slli_epi16(v, 7), 7) : _mm_and_si128(v, _mm_set1_epi16(255)); Bp = b;
    _mm_store_si128((__m128i*)out, V);
}
Img decompress5(In in) {
    Img im; im.W = int(in.var()); im.H = int(in.var()); const bool raw = in.u8() & 1; const size_t cb0 = size_t(in.var()); const int W = im.W, H = im.H;
    if (W < 1 || H < 1 || W > 65536 || H > 65536) throw std::runtime_error("header");
    size_t ns[NS5], off[NS5 + 1], tot = 0; off[0] = 8;
    for (int i = 0; i < NS5; ++i) { ns[i] = size_t(in.var()); tot += ns[i]; off[i + 1] = off[i] + ((ns[i] + 15) & ~size_t(7)); }
    if (tot != size_t(W) * H * 3) throw std::runtime_error("stream sizes");
    thread_local std::vector<int16_t> SB; SB.resize(off[NS5] + 32);
    thread_local RStream rs;
    for (int i = 0; i < NS5; ++i) { const size_t cb = cb0 ? cb0 : std::max<size_t>(ns[i], 1);
        for (size_t o = 0; o < ns[i]; o += cb) { const size_t k = std::min(cb, ns[i] - o); int16_t* d = SB.data() + off[i] + o;
            if (raw) std::memcpy(d, in.take(k * 2), k * 2); else { rs.open(in, k); rs.get_all(d, k); } } }
    im.rgb.resize(size_t(W) * H * 3 + 64);
    const int L = W + 7, LB = ((L + 7) & ~7) + 8;
    thread_local std::vector<int16_t> band[3], top[3];
    for (int c = 0; c < 3; ++c) { band[c].assign(size_t(LB) * 8, 0); top[c].assign(size_t(W) + 48, 0); }
    uint32_t pos[NS5]; for (int i = 0; i < NS5; ++i) pos[i] = uint32_t(off[i]);
    const int16_t* sb = SB.data(); uint8_t* rgb = im.rgb.data(); const size_t rstride = size_t(W) * 3;
    alignas(16) uint8_t tmp[32];
    for (int b0 = 0; b0 < H; b0 += 8) {
        const int hb = std::min(8, H - b0);
        __m128i V[3], Bp[3], V2[3]; for (int c = 0; c < 3; ++c) V[c] = Bp[c] = V2[c] = _mm_setzero_si128();
        const int16_t *t0 = top[0].data() + 8, *t1 = top[1].data() + 8, *t2 = top[2].data() + 8;
        int16_t *o0 = band[0].data(), *o1 = band[1].data(), *o2 = band[2].data();
        auto edge = [&](int t) { const int klo = std::max(0, t - W + 1), khi = std::min(hb - 1, t);
            step5<true, false>(t, klo, khi, V[0], Bp[0], V2[0], t0[t], ID5[0], sb, pos, o0 + size_t(t) * 8);
            step5<true, false>(t, klo, khi, V[1], Bp[1], V2[1], t1[t], ID5[0], sb, pos, o1 + size_t(t) * 8);
            step5<false, false>(t, klo, khi, V[2], Bp[2], V2[2], t2[t], ID5[1], sb, pos, o2 + size_t(t) * 8); };
        if (hb < 8 || W < 10) { for (int t = 0; t < L; ++t) edge(t); }
        else {
            for (int t = 0; t < 9; ++t) edge(t);
            for (int t = 9; t < W; ++t) {
                step5<true, true>(t, 0, 7, V[0], Bp[0], V2[0], t0[t], ID5[0], sb, pos, o0 + size_t(t) * 8);
                step5<true, true>(t, 0, 7, V[1], Bp[1], V2[1], t1[t], ID5[0], sb, pos, o1 + size_t(t) * 8);
                step5<false, true>(t, 0, 7, V[2], Bp[2], V2[2], t2[t], ID5[1], sb, pos, o2 + size_t(t) * 8);
            }
            for (int t = W; t < L; ++t) edge(t);
        }
        for (int t = 0; t < L; t += 8) {                                // transpose + inverse RCT (as decode_tile_skew)
            __m128i Pv[3][8];
            for (int c = 0; c < 3; ++c) { for (int j = 0; j < 8; ++j) Pv[c][j] = _mm_load_si128((const __m128i*)(band[c].data() + size_t(t + j) * 8));
                tr8x8(Pv[c][0], Pv[c][1], Pv[c][2], Pv[c][3], Pv[c][4], Pv[c][5], Pv[c][6], Pv[c][7]); }
            for (int k = 0; k < hb; ++k) {
                int x0 = t - k; uint8_t* row = rgb + size_t(b0 + k) * rstride;
                if (x0 >= 0 && x0 + 8 <= W) irct8(Pv[0][k], Pv[1][k], Pv[2][k], row + size_t(x0) * 3);
                else { irct8(Pv[0][k], Pv[1][k], Pv[2][k], tmp); int xa = std::max(0, x0), xb = std::min(W, x0 + 8); if (xa < xb) std::memcpy(row + size_t(xa) * 3, tmp + (xa - x0) * 3, size_t(xb - xa) * 3); }
                if (k == hb - 1) for (int c = 0; c < 3; ++c) _mm_storeu_si128((__m128i*)(top[c].data() + 8 + x0), Pv[c][k]);
            }
        }
    }
    if (g_g5 != 2) for (int i = 0; i < NS5; ++i) if (pos[i] != off[i] + ns[i]) throw std::runtime_error("context streams out of sync");
    im.rgb.resize(size_t(W) * H * 3);
    return im;
}

Img decompress(const std::vector<uint8_t>& data, int threads) {
    In in{data.data(), data.data() + data.size()};
    { const uint8_t* m = in.take(4); if (!std::memcmp(m, "HLC5", 4)) return decompress5(in); if (!std::memcmp(m, "HLC4", 4)) return decompress4(in, threads); if (std::memcmp(m, "HLC3", 4)) throw std::runtime_error("not HLC3/HLC4/HLC5"); }
    Img im; im.W = int(in.var()); im.H = int(in.var()); const uint8_t fl = in.u8(); const bool raw = fl & 1, skew = fl >> 1 & 1, haspm = fl >> 2 & 1, hasxr = fl >> 3 & 1; const int W = im.W, H = im.H, TW = int(in.var()), TH = int(in.var());
    if (TW < 1 || TH < 1 || TW > 65536 || TH > 65536) throw std::runtime_error("tile size"); const int tx = (W + TW - 1) / TW, ty = (H + TH - 1) / TH, nt = tx * ty;
    im.rgb.resize(size_t(W) * H * 3 + 64);
    std::vector<const uint8_t*> tp(nt); std::vector<size_t> tn(nt);
    for (int t = 0; t < nt; ++t) { tn[t] = in.var(); tp[t] = in.take(tn[t]); }
    parallel_for(nt, threads, [&](int t) {
        int x0 = (t % tx) * TW, y0 = (t / tx) * TH, w = std::min(TW, W - x0), h = std::min(TH, H - y0); size_t n = size_t(w) * h;
        thread_local std::vector<int16_t> P[3];
        In ti{tp[t], tp[t] + tn[t]}; const size_t ns = skew ? skew_len(w, h) : n;
        uint8_t pmb[3] = {0, 0, 0}; if (haspm) { uint8_t v = ti.u8(); for (int c = 0; c < 3; ++c) pmb[c] = uint8_t(v >> (2 * c) & 3); }
        const uint8_t xrb = hasxr ? ti.u8() : 0; if ((xrb & 3) == 3 || (xrb >> 2) > 16) throw std::runtime_error("xres");
        if (skew) {                                                     // fused: streams -> wavefront MED -> RGB
            thread_local RStream rs[3];
            // residuals start so that the full middle of every band is 16-byte aligned: the top triangle holds 28 samples and a
            // band holds w*8, so start + 28 = 0 mod 8 for all bands (unaligned middles split a cache line on every 4th load);
            // >= 8 samples of zeroed slack before (masked triangle loads read S - klo)
            int16_t* Pd[3];
            for (int c = 0; c < 3; ++c) { P[c].resize(ns + 32 + 64); int16_t* a = (int16_t*)((uintptr_t(P[c].data()) + 63) & ~uintptr_t(63));
                Pd[c] = a + (g_al ? 12 : 8); std::memset(Pd[c] - 8, 0, 16); }
            if (!raw && ti.p < ti.e && *ti.p == 7) { int16_t* op[3] = {Pd[0], Pd[1], Pd[2]}; joint_get(ti, op, ns); }
            else {
            for (int c = 0; c < 3; ++c) { if (raw) std::memcpy(Pd[c], ti.take(ns * 2), ns * 2); else rs[c].open(ti, ns); }
            if (!raw) {
                if (g_dec3 && rs[0].mode == 6 && rs[1].mode == 6 && rs[2].mode == 6) dec3(rs, Pd[0], Pd[1], Pd[2], ns);
                else for (int c = 0; c < 3; ++c) rs[c].get_all(Pd[c], ns);
            }
            }
            if (xrb) xres_apply(xrb, Pd[0], Pd[1], ns, true);
            decode_tile_skew(Pd[0], Pd[1], (const int16_t*)Pd[2], w, h, im.rgb.data() + (size_t(y0) * W + x0) * 3, size_t(W) * 3, pmb);
            return;
        }
        for (int c = 0; c < 3; ++c) {
            P[c].resize(n + 16);
            if (raw) std::memcpy(P[c].data(), ti.take(n * 2), n * 2); else get_stream(ti, P[c].data(), n);
            if (c < 2) restore<true>(P[c].data(), w, h); else restore<false>(P[c].data(), w, h);
        }
        for (int y = 0; y < h; ++y) irct_row(P[0].data() + size_t(y) * w, P[1].data() + size_t(y) * w, P[2].data() + size_t(y) * w, w, im.rgb.data() + (size_t(y0 + y) * W + x0) * 3);
    });
    im.rgb.resize(size_t(W) * H * 3);
    return im;
}
}  // namespace hlc

static hlc::Img read_ppm(const char* fn) {
    FILE* f = std::fopen(fn, "rb"); if (!f) throw std::runtime_error("open");
    auto tok = [&]() { int c; std::string s; while ((c = std::fgetc(f)) != EOF) { if (c == '#') { while ((c = std::fgetc(f)) != '\n' && c != EOF) {} continue; } if (isspace(c)) { if (!s.empty()) break; } else s += char(c); } return s; };
    if (tok() != "P6") throw std::runtime_error("P6 only");
    hlc::Img im; im.W = std::stoi(tok()); im.H = std::stoi(tok()); if (std::stoi(tok()) != 255) throw std::runtime_error("maxval 255 only");
    im.rgb.resize(size_t(im.W) * im.H * 3); if (std::fread(im.rgb.data(), 1, im.rgb.size(), f) != im.rgb.size()) throw std::runtime_error("short");
    std::fclose(f); return im;
}
static std::vector<uint8_t> slurp(const char* fn) { FILE* f = std::fopen(fn, "rb"); if (!f) throw std::runtime_error("open"); std::vector<uint8_t> v; uint8_t b[1 << 16]; size_t k; while ((k = std::fread(b, 1, sizeof b, f))) v.insert(v.end(), b, b + k); std::fclose(f); return v; }

static int g_max = 0; static std::vector<std::string> g_files;
int main(int argc, char** argv) {
    using namespace hlc; init_shuf();
    if (argc < 3) { std::fprintf(stderr, "usage: hlc c in.ppm out.hlc [-t N] [-n] | hlc d in.hlc out.ppm [-t N] | hlc bench [opts] a.ppm|dir ... [-max N]\n"); return 1; }
    std::string mode = argv[1]; int threads = 1; std::vector<const char*> pos;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i]; if (a == "-t" && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i])); else if (a == "-n") g_raw = true; else if (a == "-s") g_skew = true; else if (a == "-r") g_skew = false; else if (a == "-tw" && i + 1 < argc) g_tw = std::max(8, std::atoi(argv[++i])); else if (a == "-th" && i + 1 < argc) g_th = std::max(8, std::atoi(argv[++i])); else if (a == "-B" && i + 1 < argc) g_B = std::max(8, std::atoi(argv[++i])); else if (a == "-R" && i + 1 < argc) g_R = std::max(8, std::atoi(argv[++i]) & ~7); else if (a == "-dord" && i + 1 < argc) g_dord = std::atoi(argv[++i]); else if (a == "-L") g_legacy = 1; else if (a == "-4") g_legacy = 0; else if (a == "-5") g_v5 = 1; else if (a == "-g5" && i + 1 < argc) g_g5 = std::atoi(argv[++i]); else if (a == "-cb" && i + 1 < argc) g_cb = std::max(0, std::atoi(argv[++i])); else if (a == "-X" && i + 1 < argc) g_xres = std::atoi(argv[++i]); else if (a == "-Xc" && i + 1 < argc) g_xclip = std::atoi(argv[++i]); else if (a == "-Xf" && i + 1 < argc) g_xfull = std::atoi(argv[++i]); else if (a == "-Xs" && i + 1 < argc) g_xstep = std::max(1, std::atoi(argv[++i])); else if (a == "-P" && i + 1 < argc) { g_pmask = std::atoi(argv[++i]) & 15; if (!g_pmask) g_pmask = 1; } else if (a == "-Px" && i + 1 < argc) g_pxstep = std::max(1, std::atoi(argv[++i])); else if (a == "-Ps" && i + 1 < argc) g_pstep = std::max(1, std::atoi(argv[++i])); else if (a == "-al" && i + 1 < argc) g_al = std::atoi(argv[++i]); else if (a == "-S" && i + 1 < argc) { g_S = std::atoi(argv[++i]); if (g_S != 1 && g_S != 2 && g_S != 8) g_S = 4; } else if (a == "-max" && i + 1 < argc) g_max = std::atoi(argv[++i]); else if (a == "-J" && i + 1 < argc) g_joint = std::atoi(argv[++i]); else if (a == "-K" && i + 1 < argc) g_K = std::max(1, std::min(4, std::atoi(argv[++i]))); else if (a == "-d3" && i + 1 < argc) g_dec3 = std::atoi(argv[++i]); else if (a == "-Ta" && i + 1 < argc) g_tadapt = std::atoi(argv[++i]); else if (a == "-ws" && i + 1 < argc) g_wsplit = std::atoi(argv[++i]); else if (a == "-T" && i + 1 < argc) g_tl = std::max(5, std::min(13, std::atoi(argv[++i]))); else pos.push_back(argv[i]); }
    try {
        if (mode == "c" && pos.size() == 2) { auto o = g_v5 ? compress5(read_ppm(pos[0])) : g_legacy ? compress(read_ppm(pos[0]), threads) : compress4(read_ppm(pos[0]), threads); FILE* f = std::fopen(pos[1], "wb"); std::fwrite(o.data(), 1, o.size(), f); std::fclose(f); return 0; }
        if (mode == "d" && pos.size() == 2) { Img im = decompress(slurp(pos[0]), threads); FILE* f = std::fopen(pos[1], "wb"); std::fprintf(f, "P6\n%d %d\n255\n", im.W, im.H); std::fwrite(im.rgb.data(), 1, im.rgb.size(), f); std::fclose(f); return 0; }
        if (mode == "est") { size_t px = 0, by = 0; for (auto fn : pos) { Img im = read_ppm(fn); px += size_t(im.W) * im.H; by += estimate_sb(im, g_tw, g_th); } std::printf("EST B=%d R=%d  %.4f bpp  %zu B\n", g_tw, g_th, 8.0 * by / px, by); return 0; }
        if (mode == "bench") {
            {   // directories expand to their .ppm/.pnm files (sorted); -max N keeps the first N
                std::vector<std::string> ex;
                for (auto a : pos) { std::filesystem::path pa(a); std::error_code ec;
                    if (std::filesystem::is_directory(pa, ec)) { std::vector<std::string> d;
                        for (auto& e : std::filesystem::directory_iterator(pa, ec)) { auto x = e.path().extension().string(); for (auto& ch : x) ch = char(std::tolower(ch)); if (x == ".ppm" || x == ".pnm") d.push_back(e.path().string()); }
                        std::sort(d.begin(), d.end()); ex.insert(ex.end(), d.begin(), d.end()); }
                    else ex.push_back(a); }
                if (g_max > 0 && ex.size() > size_t(g_max)) ex.resize(size_t(g_max));
                g_files = std::move(ex); pos.clear(); for (auto& f : g_files) pos.push_back(f.c_str());
            }
            double te = 0, td = 0; size_t px = 0, by = 0; bool ok = true;
            for (auto fn : pos) {
                Img im = read_ppm(fn); double be = 1e9, bd = 1e9; std::vector<uint8_t> o; Img r;
                for (int k = 0; k < 3; ++k) {
                    auto t0 = std::chrono::steady_clock::now(); o = g_v5 ? compress5(im) : g_legacy ? compress(im, threads) : compress4(im, threads); auto t1 = std::chrono::steady_clock::now(); r = decompress(o, threads); auto t2 = std::chrono::steady_clock::now();
                    be = std::min(be, std::chrono::duration<double>(t1 - t0).count()); bd = std::min(bd, std::chrono::duration<double>(t2 - t1).count());
                }
                bool g = r.rgb == im.rgb; ok &= g; size_t n = size_t(im.W) * im.H; px += n; by += o.size(); te += be; td += bd;
                std::printf("%-28s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s %s\n", fn, im.W, im.H, o.size(), 8.0 * o.size() / n, 3e-6 * n / be, 3e-6 * n / bd, g ? "ok" : "FAIL");
            }
            std::printf("TOTAL %zu files  %.4f bpp  enc %.1f  dec %.1f MB/s (RGB)  t=%d  %s  %s\n", pos.size(), 8.0 * by / px, 3e-6 * px / te, 3e-6 * px / td, threads, g_v5 ? (g_raw ? "HLC5 raw" : "HLC5 fse") : g_raw ? (g_skew ? "raw skew" : "raw") : (g_skew ? "fse skew" : "fse"), ok ? "all ok" : "FAIL");
            {   unsigned tot = 0; for (auto& r : g_pstat) for (auto& v : r) tot += v.load();
                if (tot) { static const char* nm[3] = {"V", "U", "Y"};             // counts are summed over the 3 bench rounds
                    std::printf("predictors (tiles per plane: MED / (2(a+b)-c)/3 / (a+b+1)/2 / none):");
                    for (int c = 0; c < 3; ++c) { unsigned t = 0; for (auto& v : g_pstat[c]) t += v.load();
                        std::printf("  %s %u/%u/%u/%u", nm[c], g_pstat[c][0].load() / 3, g_pstat[c][1].load() / 3, g_pstat[c][2].load() / 3, g_pstat[c][3].load() / 3); }
                    std::printf("\n");
                    std::printf("cross-chroma (tiles): off %u  U<-V %u  V<-U %u\n", g_xstat[0].load() / 3, g_xstat[1].load() / 3, g_xstat[2].load() / 3); } }
            if (g_v5) { unsigned long long t = 0; for (auto v : g_c5stat) t += v;
                if (t) { std::printf("contexts (%% of symbols): shared0 %.1f  Y1-5", 100.0 * g_c5stat[0] / t); for (int i = 1; i <= 5; ++i) std::printf(" %.1f", 100.0 * g_c5stat[i] / t);
                    std::printf("  UV1-5"); for (int i = 6; i <= 10; ++i) std::printf(" %.1f", 100.0 * g_c5stat[i] / t);
                    std::printf("   full steps one class %.1f%%\n", 100.0 * g_f5stat[0] / std::max(1ull, g_f5stat[0] + g_f5stat[1])); } }
            return ok ? 0 : 3;
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    std::fprintf(stderr, "bad arguments\n"); return 1;
}
