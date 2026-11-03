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
//   hlc c in.ppm out.hlc [-t threads] [-n]     hlc d in.hlc out.ppm [-t threads]     hlc bench [-t N] [-n] a.ppm ...
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <smmintrin.h>
#include "../palic-1/rgb.hpp"
extern "C" {
#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
size_t FSE_buildCTableU16(FSE_CTable* ct, const short* normalizedCounter, unsigned maxSymbolValue, unsigned tableLog);
}

namespace hlc {
static int g_tw = 256, g_th = 256;                              // tile size (-tw, -th), stored in the header
static inline uint32_t bw(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
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
static int g_tl = 12, g_tadapt = 1;                                               // -T: max tableLog
static size_t fse4_enc(uint8_t* dst, size_t cap, const uint16_t* s, size_t n, unsigned mx, const unsigned* cnt) {
    // table size follows the stream: <= n/16 entries (decoder table build ~ 1/16 of the symbols), never below 9
    const unsigned tcap = g_tadapt ? std::min(unsigned(g_tl), std::max(9u, bw(uint32_t(n)) - 1 - 4)) : unsigned(g_tl);
    const unsigned tl = FSE_optimalTableLog(tcap, n, mx); short norm[1024];
    if (FSE_isError(FSE_normalizeCount(norm, tl, cnt, n, mx))) return 0;
    size_t h = FSE_writeNCount(dst, cap, norm, mx, tl); if (FSE_isError(h)) return 0;
    thread_local std::vector<FSE_CTable> ct(FSE_CTABLE_SIZE_U32(13, 1023));
    if (FSE_isError(FSE_buildCTableU16(ct.data(), norm, mx, tl))) return 0;
    BIT_CStream_t bc; if (FSE_isError(BIT_initCStream(&bc, dst + h, cap - h))) return 0;
    FSE_CState_t st[4]; const size_t L = std::min<size_t>(n, 4); size_t i = n;
    for (size_t j = 0; j < L; ++j) { --i; FSE_initCState2(&st[i & 3], ct.data(), s[i]); }   // last symbol of each lane
    while (i > 0) { --i; FSE_encodeSymbol(&bc, &st[i & 3], s[i]); if ((i & 3) == 0) BIT_flushBits(&bc); }   // <= 4 x 12 bits between flushes
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
static void put_stream(Out& o, const uint16_t* s, size_t n, std::vector<uint8_t>& tmp) {
    thread_local unsigned cnt[1024]; std::memset(cnt, 0, sizeof cnt); unsigned mx = 0;
    for (size_t i = 0; i < n; ++i) cnt[s[i]]++;
    for (unsigned v = 0; v < 1024; ++v) if (cnt[v]) mx = v;
    if (cnt[s[0]] == n) { o.u8(1); o.var(s[0]); return; }
    tmp.resize(n * 2 + 4096); size_t c = fse4_enc(tmp.data(), tmp.size(), s, n, mx, cnt);
    if (!c || c >= n * 2) { o.u8(0); o.raw(s, n * 2); return; }
    o.u8(6); o.var(c); o.raw(tmp.data(), c);
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
template <bool N9> static inline __m128i wave_step(__m128i Z, __m128i& V, __m128i& Bp, int16_t topv) {
    const __m128i one = _mm_set1_epi16(1), zero = _mm_setzero_si128(), m255 = _mm_set1_epi16(255);
    __m128i e = _mm_xor_si128(_mm_srli_epi16(Z, 1), _mm_sub_epi16(zero, _mm_and_si128(Z, one)));
    __m128i b = _mm_insert_epi16(_mm_slli_si128(V, 2), topv, 0), c = Bp, a = V;
    __m128i mx = _mm_max_epi16(a, b), mn = _mm_min_epi16(a, b);
    __m128i v = _mm_add_epi16(_mm_min_epi16(_mm_max_epi16(_mm_sub_epi16(_mm_add_epi16(a, b), c), mn), mx), e);
    V = N9 ? _mm_srai_epi16(_mm_slli_epi16(v, 7), 7) : _mm_and_si128(v, m255); Bp = b; return V;
}
// g_wsplit: 0 = per-step klo/khi (old), 1 = three phases (triangle, full middle, triangle)
static int g_wsplit = 1;
template <bool N9> static const int16_t* wave_band(const int16_t* S, int w, int hb, const int16_t* top, int16_t* band) {
    const int L = w + 7; const __m128i zero = _mm_setzero_si128();
    const __m128i idx = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7); __m128i V = zero, Bp = zero;
    auto tri = [&](int t) {   // band triangle: load 8 so that lane klo = S[0], keep lanes klo..khi (the stream is padded, earlier data exists)
        int klo = std::max(0, t - w + 1), khi = std::min(hb - 1, t); __m128i Z;
        if (klo == 0 && khi == 7) { Z = _mm_loadu_si128((const __m128i*)S); S += 8; }
        else if (khi < klo) Z = zero;
        else { __m128i m = _mm_and_si128(_mm_cmpgt_epi16(idx, _mm_set1_epi16(short(klo - 1))), _mm_cmpgt_epi16(_mm_set1_epi16(short(khi + 1)), idx));
               Z = _mm_and_si128(_mm_loadu_si128((const __m128i*)(S - klo)), m); S += khi - klo + 1; }
        _mm_store_si128((__m128i*)(band + size_t(t) * 8), wave_step<N9>(Z, V, Bp, top[t]));
    };
    if (!g_wsplit || hb < 8 || w < 8) { for (int t = 0; t < L; ++t) tri(t); return S; }
    for (int t = 0; t < 7; ++t) tri(t);
    for (int t = 7; t < w; ++t) { _mm_store_si128((__m128i*)(band + size_t(t) * 8), wave_step<N9>(_mm_loadu_si128((const __m128i*)S), V, Bp, top[t])); S += 8; }
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
static void decode_tile_skew(const int16_t* S0, const int16_t* S1, const int16_t* S2, int w, int h, uint8_t* rgb, size_t rstride) {
    const int L = w + 7, LB = ((L + 7) & ~7) + 8;
    thread_local std::vector<int16_t> band[3], top[3];
    for (int c = 0; c < 3; ++c) { band[c].assign(size_t(LB) * 8, 0); top[c].assign(size_t(w) + 48, 0); }
    const int16_t* S[3] = {S0, S1, S2};
    alignas(16) uint8_t tmp[32];
    for (int b0 = 0; b0 < h; b0 += 8) {
        const int hb = std::min(8, h - b0);
        S[0] = wave_band<true>(S[0], w, hb, top[0].data() + 8, band[0].data());
        S[1] = wave_band<true>(S[1], w, hb, top[1].data() + 8, band[1].data());
        S[2] = wave_band<false>(S[2], w, hb, top[2].data() + 8, band[2].data());
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

static bool g_raw = false, g_skew = false;                          // -n: no entropy coding, -s: skewed 8-row bands
// file: "HLC3" var W var H u8 flags var TW var TH, then per tile (raster order): var size, payload = 3 x stream
std::vector<uint8_t> compress(const Img& im, int threads) {
    const int W = im.W, H = im.H, TW = g_tw, TH = g_th, tx = (W + TW - 1) / TW, ty = (H + TH - 1) / TH, nt = tx * ty;
    std::vector<std::vector<uint8_t>> tiles(nt);
    parallel_for(nt, threads, [&](int t) {
        int x0 = (t % tx) * TW, y0 = (t / tx) * TH, w = std::min(TW, W - x0), h = std::min(TH, H - y0); size_t n = size_t(w) * h;
        thread_local std::vector<int16_t> P[3], SK; thread_local std::vector<uint8_t> tmp;
        for (auto& p : P) p.resize(n); const size_t ns = g_skew ? skew_len(w, h) : n; SK.resize(ns);
        for (int y = 0; y < h; ++y) rct_row(im.rgb.data() + (size_t(y0 + y) * W + x0) * 3, w, P[0].data() + size_t(y) * w, P[1].data() + size_t(y) * w, P[2].data() + size_t(y) * w);
        Out o;
        for (int c = 0; c < 3; ++c) {
            if (c < 2) residuals_inplace<true>(P[c].data(), w, h); else residuals_inplace<false>(P[c].data(), w, h);
            const int16_t* src = P[c].data();
            if (g_skew) { skew_pack(P[c].data(), w, h, SK.data()); src = SK.data(); }
            if (g_raw) o.raw(src, ns * 2);
            else put_stream(o, (const uint16_t*)src, ns, tmp);
        }
        tiles[t] = std::move(o.b);
    });
    Out f; f.raw("HLC3", 4); f.var(W); f.var(H); f.u8(uint8_t(g_raw | g_skew << 1)); f.var(TW); f.var(TH);
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
static int g_B = 128, g_R = 128, g_dord = 1, g_legacy = 0;           // -B, -R, -dord 0 per band / 1 per block, -L: HLC3
static inline int lane_nb(int W, int B) { return std::max(1, (W + B / 2) / B); }
// symbols of band (height hb) in steps [t0, t1)
static inline size_t seg_count(int W, int hb, int t0, int t1) {
    size_t n = 0; for (int t = t0; t < t1; ++t) { int klo = std::max(0, t - W + 1), khi = std::min(hb - 1, t); if (klo <= khi) n += size_t(khi - klo + 1); } return n;
}
// resumable reader: raw / rle / FSE4 with 4 interleaved states (symbol g uses state g & 3, reload when g % 4 == 0)
struct RStream {
    uint8_t mode = 0; const uint8_t* raw = nullptr; int16_t rle = 0;
    BIT_DStream_t bd; uint32_t st[4] = {0, 0, 0, 0}, g = 0; std::vector<uint32_t> T;
    void open(In& in, size_t n) {
        mode = in.u8();
        if (mode == 1) { rle = int16_t(in.var()); return; }
        if (mode == 0) { raw = in.take(n * 2); return; }
        if (mode != 6) throw std::runtime_error("stream mode");
        size_t c = in.var(); const uint8_t* src = in.take(c);
        short norm[1024]; unsigned mx = 1023, tl;
        size_t h = FSE_readNCount(norm, &mx, &tl, src, c); if (FSE_isError(h) || tl > 13 || tl < 1) throw std::runtime_error("ncount");
        uint16_t nxt[1024]; const uint32_t ts = 1u << tl, mask = ts - 1, step = (ts >> 1) + (ts >> 3) + 3; uint32_t hi = ts - 1; T.resize(ts);
        for (unsigned s = 0; s <= mx; ++s) { if (norm[s] == -1) { T[hi--] = s; nxt[s] = 1; } else nxt[s] = uint16_t(norm[s]); }
        uint32_t pos = 0;
        for (unsigned s = 0; s <= mx; ++s) for (int k = 0; k < norm[s]; ++k) { T[pos] = s; pos = (pos + step) & mask; while (pos > hi) pos = (pos + step) & mask; }
        if (pos) throw std::runtime_error("ncount spread");
        for (uint32_t u = 0; u < ts; ++u) { uint32_t s = T[u], ns = nxt[s]++, nb = tl - bw(ns) + 1; T[u] = (((ns << nb) - ts) & 0xffff) | nb << 16 | s << 20; }
        if (FSE_isError(BIT_initDStream(&bd, src + h, c - h))) throw std::runtime_error("bitstream");
        const size_t L = std::min<size_t>(n, 4);
        for (size_t k = 0; k < L; ++k) { st[k] = uint32_t(BIT_readBits(&bd, tl)); BIT_reloadDStream(&bd); }
        g = 0;
    }
    void get(int16_t* out, size_t cnt) {
        if (mode == 1) { std::fill(out, out + cnt, rle); return; }
        if (mode == 0) { std::memcpy(out, raw, cnt * 2); raw += cnt * 2; return; }
        const uint32_t* t = T.data(); size_t i = 0; BIT_DStream_t b = bd; uint32_t s0 = st[0], s1 = st[1], s2 = st[2], s3 = st[3]; uint32_t* sp[4] = {&s0, &s1, &s2, &s3};
#define HLC_D(sv) { uint32_t e = t[sv]; out[i++] = int16_t(e >> 20); sv = (e & 0xffff) + uint32_t(BIT_readBits(&b, (e >> 16) & 15)); }
        while (i < cnt && (g & 3)) { uint32_t& sv = *sp[g & 3]; HLC_D(sv) ++g; }
        for (; i + 4 <= cnt; g += 4) { BIT_reloadDStream(&b); HLC_D(s0) HLC_D(s1) HLC_D(s2) HLC_D(s3) }
        if (i < cnt) { BIT_reloadDStream(&b); while (i < cnt) { uint32_t& sv = *sp[g & 3]; HLC_D(sv) ++g; } }
#undef HLC_D
        bd = b; st[0] = s0; st[1] = s1; st[2] = s2; st[3] = s3;
    }
};
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
                else put_stream(o, G[j].data(), G[j].size(), tmp);
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

Img decompress(const std::vector<uint8_t>& data, int threads) {
    In in{data.data(), data.data() + data.size()};
    { const uint8_t* m = in.take(4); if (!std::memcmp(m, "HLC4", 4)) return decompress4(in, threads); if (std::memcmp(m, "HLC3", 4)) throw std::runtime_error("not HLC3/HLC4"); }
    Img im; im.W = int(in.var()); im.H = int(in.var()); const uint8_t fl = in.u8(); const bool raw = fl & 1, skew = fl >> 1 & 1; const int W = im.W, H = im.H, TW = int(in.var()), TH = int(in.var());
    if (TW < 1 || TH < 1 || TW > 65536 || TH > 65536) throw std::runtime_error("tile size"); const int tx = (W + TW - 1) / TW, ty = (H + TH - 1) / TH, nt = tx * ty;
    im.rgb.resize(size_t(W) * H * 3 + 64);
    std::vector<const uint8_t*> tp(nt); std::vector<size_t> tn(nt);
    for (int t = 0; t < nt; ++t) { tn[t] = in.var(); tp[t] = in.take(tn[t]); }
    parallel_for(nt, threads, [&](int t) {
        int x0 = (t % tx) * TW, y0 = (t / tx) * TH, w = std::min(TW, W - x0), h = std::min(TH, H - y0); size_t n = size_t(w) * h;
        thread_local std::vector<int16_t> P[3];
        In ti{tp[t], tp[t] + tn[t]}; const size_t ns = skew ? skew_len(w, h) : n;
        if (skew) {                                                     // fused: streams -> wavefront MED -> RGB
            for (int c = 0; c < 3; ++c) { P[c].resize(ns + 32); int16_t* d = P[c].data() + 8; std::memset(P[c].data(), 0, 16);   // 8 slack before: masked loads read S - klo
                if (raw) std::memcpy(d, ti.take(ns * 2), ns * 2); else get_stream(ti, d, ns); }
            decode_tile_skew(P[0].data() + 8, P[1].data() + 8, P[2].data() + 8, w, h, im.rgb.data() + (size_t(y0) * W + x0) * 3, size_t(W) * 3);
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

int main(int argc, char** argv) {
    using namespace hlc; init_shuf();
    if (argc < 3) { std::fprintf(stderr, "usage: hlc c in.ppm out.hlc [-t N] [-n] | hlc d in.hlc out.ppm [-t N] | hlc bench [-t N] [-n] a.ppm ...\n"); return 1; }
    std::string mode = argv[1]; int threads = 1; std::vector<const char*> pos;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i]; if (a == "-t" && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i])); else if (a == "-n") g_raw = true; else if (a == "-s") g_skew = true; else if (a == "-tw" && i + 1 < argc) g_tw = std::max(8, std::atoi(argv[++i])); else if (a == "-th" && i + 1 < argc) g_th = std::max(8, std::atoi(argv[++i])); else if (a == "-B" && i + 1 < argc) g_B = std::max(8, std::atoi(argv[++i])); else if (a == "-R" && i + 1 < argc) g_R = std::max(8, std::atoi(argv[++i]) & ~7); else if (a == "-dord" && i + 1 < argc) g_dord = std::atoi(argv[++i]); else if (a == "-L") g_legacy = 1; else if (a == "-Ta" && i + 1 < argc) g_tadapt = std::atoi(argv[++i]); else if (a == "-ws" && i + 1 < argc) g_wsplit = std::atoi(argv[++i]); else if (a == "-T" && i + 1 < argc) g_tl = std::max(5, std::min(13, std::atoi(argv[++i]))); else pos.push_back(argv[i]); }
    try {
        if (mode == "c" && pos.size() == 2) { auto o = g_legacy ? compress(read_ppm(pos[0]), threads) : compress4(read_ppm(pos[0]), threads); FILE* f = std::fopen(pos[1], "wb"); std::fwrite(o.data(), 1, o.size(), f); std::fclose(f); return 0; }
        if (mode == "d" && pos.size() == 2) { Img im = decompress(slurp(pos[0]), threads); FILE* f = std::fopen(pos[1], "wb"); std::fprintf(f, "P6\n%d %d\n255\n", im.W, im.H); std::fwrite(im.rgb.data(), 1, im.rgb.size(), f); std::fclose(f); return 0; }
        if (mode == "est") { size_t px = 0, by = 0; for (auto fn : pos) { Img im = read_ppm(fn); px += size_t(im.W) * im.H; by += estimate_sb(im, g_tw, g_th); } std::printf("EST B=%d R=%d  %.4f bpp  %zu B\n", g_tw, g_th, 8.0 * by / px, by); return 0; }
        if (mode == "bench") {
            double te = 0, td = 0; size_t px = 0, by = 0; bool ok = true;
            for (auto fn : pos) {
                Img im = read_ppm(fn); double be = 1e9, bd = 1e9; std::vector<uint8_t> o; Img r;
                for (int k = 0; k < 3; ++k) {
                    auto t0 = std::chrono::steady_clock::now(); o = g_legacy ? compress(im, threads) : compress4(im, threads); auto t1 = std::chrono::steady_clock::now(); r = decompress(o, threads); auto t2 = std::chrono::steady_clock::now();
                    be = std::min(be, std::chrono::duration<double>(t1 - t0).count()); bd = std::min(bd, std::chrono::duration<double>(t2 - t1).count());
                }
                bool g = r.rgb == im.rgb; ok &= g; size_t n = size_t(im.W) * im.H; px += n; by += o.size(); te += be; td += bd;
                std::printf("%-28s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s %s\n", fn, im.W, im.H, o.size(), 8.0 * o.size() / n, 3e-6 * n / be, 3e-6 * n / bd, g ? "ok" : "FAIL");
            }
            std::printf("TOTAL %zu files  %.4f bpp  enc %.1f  dec %.1f MB/s (RGB)  t=%d  %s  %s\n", pos.size(), 8.0 * by / px, 3e-6 * px / te, 3e-6 * px / td, threads, g_raw ? (g_skew ? "raw skew" : "raw") : (g_skew ? "fse skew" : "fse"), ok ? "all ok" : "FAIL");
            return ok ? 0 : 3;
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    std::fprintf(stderr, "bad arguments\n"); return 1;
}
