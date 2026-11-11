// palic-2: PLN2 colour (and P5 gray) codec, mode 2 only.
// Independent 288x288 tiles, MED + zigzag, causal L from the block above (off=2),
// L low bit-planes raw, high part z>>L in one FSE4 stream per L.
// Colour: Y is the 8-bit path; U,V are signed, residual folded mod 512 into -256..255
// so the zigzag stays in 0..511 and the FSE bound is 512>>L.
// P5/P6 I/O through palic-1/netpbm.hpp (comments in the header are accepted).
#include <cstdint>
#include <utility>
#include <type_traits>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <memory>
#include <chrono>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#if defined(PLN_SSE)
#include <smmintrin.h>
#endif
extern "C" {
#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
size_t FSE_buildCTableU16(FSE_CTable* ct, const short* normalizedCounter, unsigned maxSymbolValue, unsigned tableLog);
}
#include "../palic-1/netpbm.hpp"

static inline uint32_t bw(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
struct Out { std::vector<uint8_t> b;
    void u8(uint8_t v) { b.push_back(v); }
    void var(uint64_t v) { while (v >= 128) { b.push_back(uint8_t(v | 128)); v >>= 7; } b.push_back(uint8_t(v)); }
    void raw(const void* p, size_t n) { b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n); }
};
struct In { const uint8_t* p; const uint8_t* e;
    uint8_t u8() { if (p >= e) throw std::runtime_error("truncated"); return *p++; }
    uint64_t var() { uint64_t v = 0; int s = 0; for (;;) { uint8_t c = u8(); v |= uint64_t(c & 127) << s; if (!(c & 128)) return v; s += 7; } }
    const uint8_t* take(size_t n) { if (size_t(e - p) < n) throw std::runtime_error("truncated"); const uint8_t* r = p; p += n; return r; }
};
template <class T> struct UBuf {
    std::unique_ptr<T[]> p; size_t n = 0;
    void need(size_t m) { if (m > n) { p.reset(new T[m]); n = m; } }
    T* data() { return p.get(); }
};
static size_t fse4_enc(uint8_t* dst, size_t cap, const uint16_t* s, size_t n, unsigned mx, const unsigned* cnt) {
    const unsigned tcap = std::min(12u, std::max(9u, bw(uint32_t(n)) - 1 - 4));
    const unsigned tl = FSE_optimalTableLog(tcap, n, mx); short norm[1024];
    if (FSE_isError(FSE_normalizeCount(norm, tl, cnt, n, mx))) return 0;
    size_t h = FSE_writeNCount(dst, cap, norm, mx, tl); if (FSE_isError(h)) return 0;
    static std::vector<FSE_CTable> ct(FSE_CTABLE_SIZE_U32(13, 1023));
    if (FSE_isError(FSE_buildCTableU16(ct.data(), norm, mx, tl))) return 0;
    BIT_CStream_t bc; if (FSE_isError(BIT_initCStream(&bc, dst + h, cap - h))) return 0;
    FSE_CState_t st[4]; const size_t L = std::min<size_t>(n, 4); size_t i = n;
    for (size_t j = 0; j < L; ++j) { --i; FSE_initCState2(&st[i & 3], ct.data(), s[i]); }
    while (i > 0 && (i & 3)) { --i; FSE_encodeSymbol(&bc, &st[i & 3], s[i]); if ((i & 3) == 0) BIT_flushBits(&bc); }
    if (i >= 4) {
        const FSE_symbolCompressionTransform* const tt = (const FSE_symbolCompressionTransform*)st[0].symbolTT;
        const U16* const stt = (const U16*)st[0].stateTable;
        uint32_t v0 = uint32_t(st[0].value), v1 = uint32_t(st[1].value), v2 = uint32_t(st[2].value), v3 = uint32_t(st[3].value);
        size_t box = bc.bitContainer; unsigned bp = bc.bitPos; char* bptr = bc.ptr; char* const bend = bc.endPtr; const unsigned* const msk = BIT_mask;
#define C4_ENC1(v, sym) { const FSE_symbolCompressionTransform T = tt[sym]; const uint32_t nb = (v + T.deltaNbBits) >> 16; \
                          box |= size_t(v & msk[nb]) << bp; bp += nb; v = stt[uint32_t((v >> nb) + uint32_t(T.deltaFindState))]; }
        for (; i >= 4; i -= 4) {
            C4_ENC1(v3, s[i - 1]) C4_ENC1(v2, s[i - 2]) C4_ENC1(v1, s[i - 3]) C4_ENC1(v0, s[i - 4])
            memcpy(bptr, &box, sizeof box); const unsigned nbB = bp >> 3; bptr += nbB; box >>= nbB * 8; bp &= 7;
        }
#undef C4_ENC1
        if (bptr > bend) return 0;
        bc.bitContainer = box; bc.bitPos = bp; bc.ptr = bptr;
        st[0].value = ptrdiff_t(v0); st[1].value = ptrdiff_t(v1); st[2].value = ptrdiff_t(v2); st[3].value = ptrdiff_t(v3);
    }
    while (i > 0) { --i; FSE_encodeSymbol(&bc, &st[i & 3], s[i]); if ((i & 3) == 0) BIT_flushBits(&bc); }
    for (size_t k = L; k-- > 0;) FSE_flushCState(&bc, &st[k]);
    size_t c = BIT_closeCStream(&bc); return c ? h + c : 0;
}
static void fse4_dec(uint16_t* out, size_t n, const uint8_t* src, size_t c) {
    short norm[1024]; unsigned mx = 1023, tl;
    size_t h = FSE_readNCount(norm, &mx, &tl, src, c); if (FSE_isError(h) || tl > 13 || tl < 1) throw std::runtime_error("ncount");
    static std::vector<uint32_t> T(1u << 13); static std::vector<uint16_t> nxt(1024);
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
#define C4_DEC1(k) { uint32_t e = t[st[k]]; out[i + k] = uint16_t(e >> 20); st[k] = (e & 0xffff) + uint32_t(BIT_readBits(&bd, (e >> 16) & 15)); }
    for (; i + 4 <= n; i += 4) { BIT_reloadDStream(&bd); C4_DEC1(0) C4_DEC1(1) C4_DEC1(2) C4_DEC1(3) }
    BIT_reloadDStream(&bd); for (size_t k = 0; i < n; ++i, ++k) { uint32_t e = t[st[k]]; out[i] = uint16_t(e >> 20); st[k] = (e & 0xffff) + uint32_t(BIT_readBits(&bd, (e >> 16) & 15)); }
#undef C4_DEC1
}
static void put_stream(Out& o, const uint16_t* s, size_t n, unsigned A, std::vector<uint8_t>& tmp) {
    o.var(n); if (!n) return;
    static unsigned cnt[512], c4[4][512]; for (auto& r : c4) memset(r, 0, A * sizeof(unsigned));
    unsigned mx = 0; size_t i = 0;
    for (; i + 4 <= n; i += 4) { c4[0][s[i]]++; c4[1][s[i + 1]]++; c4[2][s[i + 2]]++; c4[3][s[i + 3]]++; }
    for (; i < n; ++i) c4[0][s[i]]++;
    for (unsigned v = 0; v < A; ++v) { cnt[v] = c4[0][v] + c4[1][v] + c4[2][v] + c4[3][v]; if (cnt[v]) mx = v; }
    if (cnt[s[0]] == n) { o.u8(1); o.var(s[0]); return; }
    if (tmp.size() < n * 2 + 4096) tmp.resize(n * 2 + 4096);
    size_t c = fse4_enc(tmp.data(), n * 2 + 4096, s, n, mx, cnt);
    if (!c || c >= n * 2) { o.u8(0); o.raw(s, n * 2); return; }
    o.u8(6); o.var(c); o.raw(tmp.data(), c);
}
static size_t get_stream(In& in, UBuf<uint16_t>& dst) {
    size_t n = in.var(); dst.need(n + 2); if (!n) return 0;
    uint8_t mode = in.u8();
    if (mode == 1) { uint16_t v = uint16_t(in.var()); std::fill(dst.data(), dst.data() + n, v); }
    else if (mode == 0) memcpy(dst.data(), in.take(n * 2), n * 2);
    else if (mode == 6) { size_t c = in.var(); fse4_dec(dst.data(), n, in.take(c), c); }
    else throw std::runtime_error("stream mode");
    return n;
}

namespace palic2 {
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static inline uint32_t bwz(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
static inline int med(int a, int b, int c) { int mn = std::min(a, b), mx = std::max(a, b), p = a + b - c; return p < mn ? mn : (p > mx ? mx : p); }
struct Cfg { int off = 2; };
static constexpr int TILE = 288;

static inline uint16_t zz8(int e) { e = int8_t(e); return uint16_t(uint8_t((e << 1) ^ (e >> 7))); }
static void resid_row0(const uint8_t* __restrict r, uint16_t* __restrict z, int w) {
    z[0] = zz8(r[0]);
    for (int x = 1; x < w; ++x) z[x] = zz8(r[x] - r[x - 1]);
}
static void resid_row(const uint8_t* __restrict r, const uint8_t* __restrict up, uint16_t* __restrict z, int w) {
    z[0] = zz8(r[0] - up[0]);
    for (int x = 1; x < w; ++x) {
        const int a = r[x - 1], b = up[x], c = up[x - 1], mn = a < b ? a : b, mx = a < b ? b : a;
        int p = a + b - c; p = p < mn ? mn : p; p = p > mx ? mx : p;
        z[x] = zz8(r[x] - p);
    }
}
static inline int block_L(const uint16_t* s, int off) {
#if defined(PLN_SSE)
    const __m128i a0 = _mm_loadu_si128((const __m128i*)s), a1 = _mm_loadu_si128((const __m128i*)(s + 8));
    __m128i t = _mm_add_epi16(a0, a1); t = _mm_add_epi16(t, _mm_srli_si128(t, 8)); t = _mm_add_epi16(t, _mm_srli_si128(t, 4)); t = _mm_add_epi16(t, _mm_srli_si128(t, 2));
    return std::min(7, std::max(0, int(bwz(uint32_t(_mm_extract_epi16(t, 0)) >> 4)) - off));
#else
    uint32_t sum = 0; for (int j = 0; j < 16; ++j) sum += s[j];
    return std::min(7, std::max(0, int(bwz(sum >> 4)) - off));
#endif
}
template<int Off>
static inline int block_Lt(const uint16_t* s) { return block_L(s, Off); }
template<int Off>
static inline int causal_Lt(const uint16_t* z, size_t i, int tw) {
    const size_t px = i * 16;
    if (tw >= 16 && px >= size_t(tw)) return block_Lt<Off>(z + px - tw);
    return i ? block_Lt<Off>(z + px - 16) : 2;
}
static inline int causal_L(const uint16_t* z, size_t i, int tw, int off) {
    const size_t px = i * 16;
    if (tw >= 16 && px >= size_t(tw)) return block_L(z + px - tw, off);
    return i ? block_L(z + px - 16, off) : 2;
}
static inline void block_split(const uint16_t* s, int L, uint16_t*& pw, uint16_t* hi) {
#if defined(PLN_SSE)
    const __m128i a0 = _mm_loadu_si128((const __m128i*)s), a1 = _mm_loadu_si128((const __m128i*)(s + 8));
    for (int k = 0; k < L; ++k) { const __m128i q = _mm_cvtsi32_si128(15 - k);
        *pw++ = uint16_t(_mm_movemask_epi8(_mm_packs_epi16(_mm_sll_epi16(a0, q), _mm_sll_epi16(a1, q)))); }
    const __m128i q = _mm_cvtsi32_si128(L);
    _mm_storeu_si128((__m128i*)hi, _mm_srl_epi16(a0, q)); _mm_storeu_si128((__m128i*)(hi + 8), _mm_srl_epi16(a1, q));
#else
    for (int k = 0; k < L; ++k) { uint16_t m = 0; for (int j = 0; j < 16; ++j) m |= uint16_t(((s[j] >> k) & 1u) << j); *pw++ = m; }
    for (int j = 0; j < 16; ++j) hi[j] = uint16_t(s[j] >> L);
#endif
}
// Vector path: clang ext_vector_type, or GCC >= 12 vector_size (+ __builtin_shufflevector). The two dialects differ
// only in how a scalar is broadcast: VSPL(T, x).
#if defined(__clang__)
#define PLR_VEC 1
typedef uint16_t v16u __attribute__((ext_vector_type(16)));
typedef int16_t v16s __attribute__((ext_vector_type(16)));
#define VSPL(T, x) ((T)(x))
#elif defined(__GNUC__) && __GNUC__ >= 12
#define PLR_VEC 1
typedef uint16_t v16u __attribute__((vector_size(32)));
typedef int16_t v16s __attribute__((vector_size(32)));
template <class T, class E> static inline T vspl_(E x) {                               // broadcast
    using El = typename std::remove_cv<typename std::remove_reference<decltype(std::declval<T&>()[0])>::type>::type;
    const El e = El(x); return T{e, e, e, e, e, e, e, e, e, e, e, e, e, e, e, e}; }
#define VSPL(T, x) (vspl_<T>(x))
#endif
static inline void block_unpack(const uint16_t* h, int L, const uint8_t*& pp, uint16_t* z) {
#if defined(PLR_VEC)                                                    // vector: weights & mask != 0 -> bit k
    const v16u w = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    v16u v; memcpy(&v, h, 32); v = v << uint16_t(L);
    for (int k = 0; k < L; ++k) { uint16_t m; memcpy(&m, pp, 2); pp += 2; v |= (v16u)((VSPL(v16u, m) & w) != 0) & VSPL(v16u, uint16_t(1u << k)); }
    memcpy(z, &v, 32);
#else
    for (int j = 0; j < 16; ++j) z[j] = uint16_t(h[j] << L);
    for (int k = 0; k < L; ++k) { uint16_t m; memcpy(&m, pp, 2); pp += 2; for (int j = 0; j < 16; ++j) z[j] |= uint16_t(((m >> j) & 1u) << k); }
#endif
}
#if defined(PLR_VEC)
// ---- decoder MED as a wave over bands of 16 rows (from PLN1 planes_codec.cpp): lane k = row y0 + k, step t,
// lane k at x = t - k (skew 1 px per row: MED needs a, b, c only). a = own lane, previous step; b = lane k-1,
// previous step; c = lane k-1, two steps back; lane 0 takes b, c from the row above the band. Borders: tile row 0:
// b = c = a (pred = left, 0 at x = 0); x = 0: a = c = b (pred = up). Skew in and out by 16 x 16 transposes.
// NINE = false: Y, uint8 samples, value = (pred + e) & 255. NINE = true: U/V, int16 samples, value = fold9(pred + e).
#define SV __builtin_shufflevector
static inline void tr16s(v16s* m) {
    for (int i = 0; i < 16; i++) if (!(i & 8)) { v16s a = m[i], b = m[i + 8];
        m[i] = SV(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23); m[i + 8] = SV(a, b, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31); }
    for (int i = 0; i < 16; i++) if (!(i & 4)) { v16s a = m[i], b = m[i + 4];
        m[i] = SV(a, b, 0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27); m[i + 4] = SV(a, b, 4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31); }
    for (int i = 0; i < 16; i++) if (!(i & 2)) { v16s a = m[i], b = m[i + 2];
        m[i] = SV(a, b, 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29); m[i + 2] = SV(a, b, 2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31); }
    for (int i = 0; i < 16; i += 2) { v16s a = m[i], b = m[i + 1];
        m[i] = SV(a, b, 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30); m[i + 1] = SV(a, b, 1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31); }
}
static void unzz16w(const uint16_t* __restrict z, int16_t* __restrict e, int w) { for (int x = 0; x < w; ++x) e[x] = int16_t(int(z[x] >> 1) ^ -int(z[x] & 1)); }
template <bool NINE, class T> static void recon_wave(T* img, int S, int tw, int th, const uint16_t* z) {
    static thread_local std::vector<int16_t> Eb, Ob, Tb; const int RS = tw + 64;             // padded rows [-32, tw + 32)
    Eb.assign(size_t(RS) * 16, 0); Ob.resize(size_t(RS) * 16); Tb.assign(size_t(tw) + 2, 0);
    const v16s K = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, lane0 = K == 0;
    for (int y0 = 0; y0 < th; y0 += 16) {
        const int R = std::min(16, th - y0), steps = tw + R - 1;
        for (int k = 0; k < R; ++k) unzz16w(z + size_t(y0 + k) * tw, Eb.data() + size_t(k) * RS + 32, tw);
        for (int k = R; k < 16; ++k) std::fill(Eb.data() + size_t(k) * RS, Eb.data() + size_t(k + 1) * RS, int16_t(0));
        int16_t* Tp = Tb.data() + 1;                                                  // Tp[-1] = 0, Tp[tw] = 0
        if (y0) for (int x = 0; x < tw; ++x) Tp[x] = int16_t(img[size_t(y0 - 1) * S + x]);
        const bool first = y0 == 0;
        v16s P1 = {}, P2 = {};
        for (int t0 = 0; t0 < steps; t0 += 16) {
            v16s m[16];
            for (int k = 0; k < 16; ++k) memcpy(&m[k], Eb.data() + size_t(k) * RS + 32 + t0 - k, 32);
            tr16s(m);                                                                 // m[j] = residuals of step t0 + j
            for (int j = 0; j < 16; ++j) {
                const int t = t0 + j, tb = std::min(t, tw), tc = std::min(t - 1, tw);
                v16s a = P1;
                v16s b = SV(P1, VSPL(v16s, (int16_t)Tp[tb]), 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
                v16s c = SV(P2, VSPL(v16s, (int16_t)Tp[tc]), 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
                if (first) { b = lane0 ? a : b; c = lane0 ? a : c; }                   // tile row 0: pred = left
                const v16s x0 = K == (int16_t)t; a = x0 ? b : a; c = x0 ? b : c;      // x = 0: pred = up
                const v16s mn = a < b ? a : b, mx = a < b ? b : a; v16s p = a + b - c; p = p < mn ? mn : p; p = p > mx ? mx : p;
                v16s v = p + m[j];
                if (NINE) { v = v > (int16_t)255 ? v - (int16_t)512 : v; v = v < (int16_t)-255 ? v + (int16_t)512 : v; }
                else v = v & (int16_t)255;
                P2 = P1; P1 = v; m[j] = v;
            }
            tr16s(m);                                                                 // m[k] = row k at x = t0 - k ..
            for (int k = 0; k < 16; ++k) memcpy(Ob.data() + size_t(k) * RS + 32 + t0 - k, &m[k], 32);
        }
        for (int k = 0; k < R; ++k) { T* r = img + size_t(y0 + k) * S; const int16_t* o = Ob.data() + size_t(k) * RS + 32;
            for (int x = 0; x < tw; ++x) r[x] = T(o[x]); }
    }
}
#undef SV
#endif
static void tile_enc(const uint8_t* img, int S, int tw, int th, const Cfg& c, Out& o, std::vector<uint8_t>& tmp) {
    const size_t n = size_t(tw) * th, nb = (n + 15) / 16;
    static thread_local UBuf<uint16_t> zb, hb[8], plb;
    zb.need(nb * 16 + 16); uint16_t* z = zb.data(); std::fill(z + n, z + nb * 16 + 16, uint16_t(0));
    resid_row0(img, z, tw);
    for (int y = 1; y < th; ++y) resid_row(img + size_t(y) * S, img + size_t(y - 1) * S, z + size_t(y) * tw, tw);
    plb.need(nb * 8); uint16_t* pw = plb.data(); size_t hn[8] = {0};
    for (auto& h : hb) h.need(nb * 16);
    if (c.off == 2) {
        for (size_t i = 0; i < nb; ++i) {
            alignas(32) uint16_t hk[16];
            const int L = causal_Lt<2>(z, i, tw);
            block_split(z + i * 16, L, pw, hk);
            memcpy(hb[L].data() + hn[L], hk, 32); hn[L] += 16;
        }
    } else {
        for (size_t i = 0; i < nb; ++i) {
            alignas(32) uint16_t hk[16];
            const int L = causal_L(z, i, tw, c.off);
            block_split(z + i * 16, L, pw, hk);
            memcpy(hb[L].data() + hn[L], hk, 32); hn[L] += 16;
        }
    }
    for (int L = 0; L < 8; ++L) put_stream(o, hb[L].data(), hn[L], 256u >> L, tmp);
    o.raw(plb.data(), size_t(pw - plb.data()) * 2);
}
static void unzz_row(const uint16_t* __restrict z, int8_t* __restrict e, int w) { for (int x = 0; x < w; ++x) e[x] = int8_t(int(z[x] >> 1) ^ -int(z[x] & 1)); }
static void recon_row0(uint8_t* __restrict r, const int8_t* __restrict e, int w) { int v = 0; for (int x = 0; x < w; ++x) { v = uint8_t(v + e[x]); r[x] = uint8_t(v); } }
static void recon_row(uint8_t* __restrict r, const uint8_t* __restrict up, const int8_t* __restrict e, int w) {
    int a = uint8_t(up[0] + e[0]); r[0] = uint8_t(a);
    for (int x = 1; x < w; ++x) {
        const int b = up[x], c = up[x - 1], mn = a < b ? a : b, mx = a < b ? b : a;
        int p = a + b - c; p = p < mn ? mn : p; p = p > mx ? mx : p;
        a = uint8_t(p + e[x]); r[x] = uint8_t(a);
    }
}
static void tile_dec(In& in, uint8_t* img, int S, int tw, int th, const Cfg& c) {
    const size_t n = size_t(tw) * th, nb = (n + 15) / 16;
    static thread_local UBuf<uint16_t> zb, hb[8]; static thread_local std::vector<int8_t> e;
    zb.need(nb * 16 + 16); uint16_t* z = zb.data(); if (e.size() < size_t(tw)) e.resize(tw);
    size_t hn[8];
    for (int L = 0; L < 8; ++L) { hn[L] = get_stream(in, hb[L]); if (hn[L] & 15) throw std::runtime_error("high stream align"); }
    const uint8_t* pp = in.p; size_t hp[8] = {0};
    if (c.off == 2) {
        for (size_t i = 0; i < nb; ++i) {
            const int L = causal_Lt<2>(z, i, tw);
            if (hp[L] + 16 > hn[L] || pp + 2 * L > in.e) throw std::runtime_error("high stream");
            block_unpack(hb[L].data() + hp[L], L, pp, z + i * 16); hp[L] += 16;
        }
    } else {
        for (size_t i = 0; i < nb; ++i) {
            const int L = causal_L(z, i, tw, c.off);
            if (hp[L] + 16 > hn[L] || pp + 2 * L > in.e) throw std::runtime_error("high stream");
            block_unpack(hb[L].data() + hp[L], L, pp, z + i * 16); hp[L] += 16;
        }
    }
    in.p = pp;
    for (int L = 0; L < 8; ++L) if (hp[L] != hn[L]) throw std::runtime_error("high stream size");
#if defined(PLR_VEC) && !defined(PLN_ROWS)
    recon_wave<false>(img, S, tw, th, z);
#else
    for (int y = 0; y < th; ++y) {
        const uint16_t* zr = z + size_t(y) * tw;
        unzz_row(zr, e.data(), tw);
        if (y == 0) recon_row0(img, e.data(), tw); else recon_row(img + size_t(y) * S, img + size_t(y - 1) * S, e.data(), tw);
    }
#endif
}
struct Grid { int nx, ny, ts, W, H; int tx(int i) const { return (i % nx) * ts; } int ty(int i) const { return (i / nx) * ts; }
    int tw(int i) const { return std::min(ts, W - tx(i)); } int th(int i) const { return std::min(ts, H - ty(i)); } };
static Grid grid(int W, int H) { return Grid{(W + TILE - 1) / TILE, (H + TILE - 1) / TILE, TILE, W, H}; }

static void put_text_header(Out& o, const char* magic, int W, int H) {
    char buf[64];
    int n = std::snprintf(buf, sizeof buf, "%s\n%d %d\n", magic, W, H);
    if (n < 0 || n >= int(sizeof buf)) throw std::runtime_error("header");
    o.raw(buf, size_t(n));
}
// "PLN1\n" / "PLN2\n" then "W H\n". Returns the byte offset of the tile-length list.
static size_t read_text_header(const std::vector<uint8_t>& d, bool& rgb, int& W, int& H) {
    auto nl = std::find(d.begin(), d.end(), '\n');
    if (nl == d.end()) throw std::runtime_error("bad header");
    std::string magic(d.begin(), nl);
    if (magic != "PLN1" && magic != "PLN2") throw std::runtime_error("not PLN1/PLN2");
    rgb = magic == "PLN2";
    auto nl2 = std::find(nl + 1, d.end(), '\n');
    if (nl2 == d.end()) throw std::runtime_error("bad header");
    std::string dims(nl + 1, nl2);
    int w = 0, h = 0;
    if (std::sscanf(dims.c_str(), "%d %d", &w, &h) != 2 || w <= 0 || h <= 0) throw std::runtime_error("bad header");
    W = w; H = h;
    return size_t(nl2 - d.begin()) + 1;
}

std::vector<uint8_t> compress(const uint8_t* img, int W, int H, const Cfg& c) {
    Grid g = grid(W, H); const int nt = g.nx * g.ny; static thread_local std::vector<uint8_t> tmp;
    Out o; o.b.reserve(size_t(W) * H + size_t(nt) * 4 + 4096);
    put_text_header(o, "PLN1", W, H);
    const size_t lens = o.b.size(); o.b.resize(lens + size_t(nt) * 4);
    for (int i = 0; i < nt; ++i) {
        const size_t b = o.b.size(); const uint8_t* ti = img + size_t(g.ty(i)) * W + g.tx(i);
        tile_enc(ti, W, g.tw(i), g.th(i), c, o, tmp);
        const uint32_t l = uint32_t(o.b.size() - b); memcpy(&o.b[lens + size_t(i) * 4], &l, 4);
    }
    return std::move(o.b);
}
void decompress(const std::vector<uint8_t>& d, std::vector<uint8_t>& img, int& W, int& H) {
    bool rgb = false; size_t pos = read_text_header(d, rgb, W, H);
    if (rgb) throw std::runtime_error("not PLN1");
    Cfg c;
    Grid g = grid(W, H); const int nt = g.nx * g.ny; std::vector<uint32_t> len(nt);
    if (d.size() < pos + size_t(nt) * 4) throw std::runtime_error("truncated");
    for (auto& l : len) { memcpy(&l, d.data() + pos, 4); pos += 4; }
    img.resize(size_t(W) * H);
    for (int i = 0; i < nt; ++i) {
        if (d.size() < pos + len[i]) throw std::runtime_error("truncated");
        In ti{d.data() + pos, d.data() + pos + len[i]}; pos += len[i];
        tile_dec(ti, img.data() + size_t(g.ty(i)) * W + g.tx(i), W, g.tw(i), g.th(i), c); if (ti.p != ti.e) throw std::runtime_error("tile trailing bytes"); }
}

// Colour. U = R-G, V = B-G, Y = G + (U+V+2)>>2, then V -= U>>2.
// U,V are -255..255; residual -510..510 folds mod 512 into -256..255.
static inline int wrap9(int d) { return int(unsigned(d + 256) & 511u) - 256; }
static inline int fold9(int s) { if (s > 255) return s - 512; if (s < -255) return s + 512; return s; }
static inline uint16_t zz16(int e) { return uint16_t((e << 1) ^ (e >> 31)); }
static void resid16_row0(const int16_t* __restrict r, uint16_t* __restrict z, int w) {
    z[0] = zz16(r[0]);
    for (int x = 1; x < w; ++x) z[x] = zz16(wrap9(r[x] - r[x - 1]));
}
static void resid16_row(const int16_t* __restrict r, const int16_t* __restrict up, uint16_t* __restrict z, int w) {
    z[0] = zz16(wrap9(r[0] - up[0]));
    for (int x = 1; x < w; ++x) z[x] = zz16(wrap9(r[x] - med(r[x - 1], up[x], up[x - 1])));
}
static void tile_enc16(const int16_t* img, int tw, int th, const Cfg& c, Out& o, std::vector<uint8_t>& tmp) {
    const size_t n = size_t(tw) * th, nb = (n + 15) / 16;
    static thread_local UBuf<uint16_t> zb, hb[8], plb;
    zb.need(nb * 16 + 16); uint16_t* z = zb.data(); std::fill(z + n, z + nb * 16 + 16, uint16_t(0));
    resid16_row0(img, z, tw);
    for (int y = 1; y < th; ++y) resid16_row(img + size_t(y) * tw, img + size_t(y - 1) * tw, z + size_t(y) * tw, tw);
    plb.need(nb * 8); uint16_t* pw = plb.data(); size_t hn[8] = {0};
    for (auto& h : hb) h.need(nb * 16);
    if (c.off == 2) {
        for (size_t i = 0; i < nb; ++i) {
            alignas(32) uint16_t hk[16];
            const int L = causal_Lt<2>(z, i, tw);
            block_split(z + i * 16, L, pw, hk);
            memcpy(hb[L].data() + hn[L], hk, 32); hn[L] += 16;
        }
    } else {
        for (size_t i = 0; i < nb; ++i) {
            alignas(32) uint16_t hk[16];
            const int L = causal_L(z, i, tw, c.off);
            block_split(z + i * 16, L, pw, hk);
            memcpy(hb[L].data() + hn[L], hk, 32); hn[L] += 16;
        }
    }
    for (int L = 0; L < 8; ++L) put_stream(o, hb[L].data(), hn[L], 512u >> L, tmp);
    o.raw(plb.data(), size_t(pw - plb.data()) * 2);
}
static void recon16_row0(int16_t* r, const int16_t* e, int w) { int v = 0; for (int x = 0; x < w; ++x) { v = fold9(v + e[x]); r[x] = int16_t(v); } }
static void recon16_row(int16_t* r, const int16_t* up, const int16_t* e, int w) {
    int a = fold9(up[0] + e[0]); r[0] = int16_t(a);
    for (int x = 1; x < w; ++x) { a = fold9(med(a, up[x], up[x - 1]) + e[x]); r[x] = int16_t(a); }
}
static void tile_dec16(In& in, int16_t* img, int tw, int th, const Cfg& c) {
    const size_t n = size_t(tw) * th, nb = (n + 15) / 16;
    static thread_local UBuf<uint16_t> zb, hb[8]; static thread_local std::vector<int16_t> e;
    zb.need(nb * 16 + 16); uint16_t* z = zb.data(); if (e.size() < size_t(tw)) e.resize(tw);
    size_t hn[8];
    for (int L = 0; L < 8; ++L) { hn[L] = get_stream(in, hb[L]); if (hn[L] & 15) throw std::runtime_error("high stream align"); }
    const uint8_t* pp = in.p; size_t hp[8] = {0};
    if (c.off == 2) {
        for (size_t i = 0; i < nb; ++i) {
            const int L = causal_Lt<2>(z, i, tw);
            if (hp[L] + 16 > hn[L] || pp + 2 * L > in.e) throw std::runtime_error("high stream");
            block_unpack(hb[L].data() + hp[L], L, pp, z + i * 16); hp[L] += 16;
        }
    } else {
        for (size_t i = 0; i < nb; ++i) {
            const int L = causal_L(z, i, tw, c.off);
            if (hp[L] + 16 > hn[L] || pp + 2 * L > in.e) throw std::runtime_error("high stream");
            block_unpack(hb[L].data() + hp[L], L, pp, z + i * 16); hp[L] += 16;
        }
    }
    in.p = pp;
    for (int L = 0; L < 8; ++L) if (hp[L] != hn[L]) throw std::runtime_error("high stream size");
#if defined(PLR_VEC) && !defined(PLN_ROWS)
    recon_wave<true>(img, tw, tw, th, z);
#else
    for (int y = 0; y < th; ++y) {
        const uint16_t* zr = z + size_t(y) * tw;
        for (int x = 0; x < tw; ++x) e[x] = int16_t(int(zr[x] >> 1) ^ -int(zr[x] & 1));
        if (y == 0) recon16_row0(img + size_t(y) * tw, e.data(), tw);
        else recon16_row(img + size_t(y) * tw, img + size_t(y - 1) * tw, e.data(), tw);
    }
#endif
}
static void rgb_split(const uint8_t* rgb, int W, int tw, int th, std::vector<uint8_t>& Y, std::vector<int16_t>& U, std::vector<int16_t>& V) {
    Y.resize(size_t(tw) * th); U.resize(Y.size()); V.resize(Y.size());
    for (int y = 0; y < th; ++y) {
        const uint8_t* p = rgb + size_t(y) * W * 3;
        for (int x = 0; x < tw; ++x) {
            int r = p[3 * x], g = p[3 * x + 1], b = p[3 * x + 2];
            int u = r - g, v = b - g, yy = g + ((u + v + 2) >> 2); v -= u >> 2;
            if (unsigned(yy) > 255) throw std::runtime_error("Y out of range");
            Y[size_t(y) * tw + x] = uint8_t(yy); U[size_t(y) * tw + x] = int16_t(u); V[size_t(y) * tw + x] = int16_t(v);
        }
    }
}
static void rgb_join(uint8_t* rgb, int W, int tw, int th, const uint8_t* Y, const int16_t* U, const int16_t* V) {
    for (int y = 0; y < th; ++y) for (int x = 0; x < tw; ++x) {
        int yy = Y[size_t(y) * tw + x], u = U[size_t(y) * tw + x], v = V[size_t(y) * tw + x];
        v += u >> 2; yy -= (u + v + 2) >> 2;
        uint8_t* p = rgb + (size_t(y) * W + x) * 3;
        p[0] = uint8_t(u + yy); p[1] = uint8_t(yy); p[2] = uint8_t(v + yy);
    }
}
std::vector<uint8_t> compress_rgb(const uint8_t* rgb, int W, int H, const Cfg& c) {
    Grid g = grid(W, H); const int nt = g.nx * g.ny; static thread_local std::vector<uint8_t> tmp, Y; static thread_local std::vector<int16_t> U, V;
    Out o; o.b.reserve(size_t(W) * H * 3);
    put_text_header(o, "PLN2", W, H);
    const size_t lens = o.b.size(); o.b.resize(lens + size_t(nt) * 8);
    for (int i = 0; i < nt; ++i) {
        const size_t b = o.b.size();
        rgb_split(rgb + (size_t(g.ty(i)) * W + g.tx(i)) * 3, W, g.tw(i), g.th(i), Y, U, V);
        tile_enc(Y.data(), g.tw(i), g.tw(i), g.th(i), c, o, tmp);
        const uint32_t ny = uint32_t(o.b.size() - b);
        tile_enc16(U.data(), g.tw(i), g.th(i), c, o, tmp);
        const uint32_t nu = uint32_t(o.b.size() - b - ny);
        tile_enc16(V.data(), g.tw(i), g.th(i), c, o, tmp);
        const uint32_t nv = uint32_t(o.b.size() - b - ny - nu);
        if (ny >= (1u << 18) || nu >= (1u << 18) || nv >= (1u << 18)) throw std::runtime_error("plane does not fit in 18 bits");
        const uint64_t ent = uint64_t(ny) | uint64_t(nu) << 18 | uint64_t(nv) << 36;
        memcpy(&o.b[lens + size_t(i) * 8], &ent, 8);
    }
    return std::move(o.b);
}
void decompress_rgb(const std::vector<uint8_t>& d, std::vector<uint8_t>& rgb, int& W, int& H) {
    bool rgb_file = false; size_t pos = read_text_header(d, rgb_file, W, H);
    if (!rgb_file) throw std::runtime_error("not PLN2");
    Cfg c, uv;
    Grid g = grid(W, H); const int nt = g.nx * g.ny;
    if (d.size() < pos + size_t(nt) * 8) throw std::runtime_error("truncated");
    std::vector<uint64_t> ent(nt);
    for (auto& e : ent) { memcpy(&e, d.data() + pos, 8); pos += 8; }
    rgb.resize(size_t(W) * H * 3);
    static thread_local std::vector<uint8_t> Y; static thread_local std::vector<int16_t> U, V;
    for (int i = 0; i < nt; ++i) {
        if (ent[i] >> 54) throw std::runtime_error("index reserved bits");
        const uint32_t ny = uint32_t(ent[i] & 0x3ffff), nu = uint32_t(ent[i] >> 18 & 0x3ffff), nv = uint32_t(ent[i] >> 36 & 0x3ffff);
        if (d.size() < pos + ny + nu + nv) throw std::runtime_error("truncated");
        const int tw = g.tw(i), th = g.th(i);
        Y.resize(size_t(tw) * th); U.resize(Y.size()); V.resize(Y.size());
        auto plane = [&](uint32_t n, auto dec) {
            In ti{d.data() + pos, d.data() + pos + n}; pos += n;
            dec(ti);
            if (ti.p != ti.e) throw std::runtime_error("plane trailing bytes");
        };
        plane(ny, [&](In& ti) { tile_dec(ti, Y.data(), tw, tw, th, c); });
        plane(nu, [&](In& ti) { tile_dec16(ti, U.data(), tw, th, uv); });
        plane(nv, [&](In& ti) { tile_dec16(ti, V.data(), tw, th, uv); });
        rgb_join(rgb.data() + (size_t(g.ty(i)) * W + g.tx(i)) * 3, W, tw, th, Y.data(), U.data(), V.data());
    }
}
} // namespace palic2

static std::vector<uint8_t> read_netpbm(const std::string& fn, int& W, int& H, bool& rgb) {
    Netpbm nb(fn);
    if (nb.maxvalue != 255 || (nb.type != '5' && nb.type != '6')) throw std::runtime_error("need 8-bit P5 or P6");
    W = int(nb.width); H = int(nb.height); rgb = nb.type == '6';
    std::vector<uint8_t> px(size_t(W) * H * (rgb ? 3 : 1));
    nb.read(px.data(), px.size());
    return px;
}
static void write_netpbm(const std::string& fn, const std::vector<uint8_t>& img, int W, int H, bool rgb) {
    Netpbm out(fn, rgb ? '6' : '5', uint32_t(W), uint32_t(H), 255);
    out.write(img.data(), img.size());
}
static std::vector<uint8_t> read_bin(const std::string& fn) {
    FILE* fp = fopen(fn.c_str(), "rb"); if (!fp) return {};
    fseek(fp, 0, SEEK_END); std::vector<uint8_t> b(ftell(fp)); fseek(fp, 0, SEEK_SET);
    if (fread(b.data(), 1, b.size(), fp) != b.size()) b.clear(); fclose(fp); return b;
}

int main(int argc, char** argv) {
    using namespace palic2;
    if (argc < 3) { fprintf(stderr, "usage: %s c in.pgm|ppm out.pln (the input itself if it does not shrink) | d in.pln|pgm|ppm out.pgm|ppm | bench a.pgm|ppm|dir ... [-r N]\n", argv[0]); return 1; }
    std::string cmd = argv[1]; Cfg c; int reps = 3; std::vector<std::string> pos;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i];
        if (a == "-r" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (cmd == "bench" && std::filesystem::is_directory(a)) { std::vector<std::string> v;
            for (auto& e : std::filesystem::directory_iterator(a)) { auto x = e.path().extension();
                if (x == ".pgm" || x == ".ppm" || x == ".pnm") v.push_back(e.path().string()); }
            std::sort(v.begin(), v.end()); pos.insert(pos.end(), v.begin(), v.end()); }
        else pos.push_back(a); }
    try {
        if (cmd == "c" && pos.size() == 2) {
            int W, H; bool rgb; auto px = read_netpbm(pos[0], W, H, rgb);
            auto o = rgb ? compress_rgb(px.data(), W, H, c) : compress(px.data(), W, H, c);
            // not smaller than the input file: store the input file itself (a P5/P6 text header like ours);
            // the decoder sees "P5"/"P6" instead of "PLN" and just copies the file
            auto src = read_bin(pos[0]);
            if (!src.empty() && o.size() >= src.size()) o.swap(src);
            FILE* f = fopen(pos[1].c_str(), "wb"); if (!f) return 1; fwrite(o.data(), 1, o.size(), f); fclose(f); return 0; }
        if (cmd == "d" && pos.size() == 2) {
            auto d = read_bin(pos[0]); if (d.empty()) { fprintf(stderr, "cannot read %s\n", pos[0].c_str()); return 1; }
            if (d.size() >= 2 && d[0] == 'P' && (d[1] == '5' || d[1] == '6')) {       // stored original: copy
                FILE* f = fopen(pos[1].c_str(), "wb"); if (!f) return 1;
                const bool ok = fwrite(d.data(), 1, d.size(), f) == d.size(); fclose(f); return ok ? 0 : 1; }
            std::vector<uint8_t> img; int W, H; bool rgb = d.size() >= 4 && memcmp(d.data(), "PLN2", 4) == 0;
            if (rgb) decompress_rgb(d, img, W, H); else decompress(d, img, W, H);
            write_netpbm(pos[1], img, W, H, rgb); return 0; }
        if (cmd == "bench") {
            double te = 0, td = 0, tpx = 0, tpix = 0, tb = 0; bool ok = true;
            for (auto& fn : pos) { int W, H; bool rgb; std::vector<uint8_t> px;
                try { px = read_netpbm(fn, W, H, rgb); } catch (...) { printf("%s: skip\n", fn.c_str()); continue; }
                double be = 1e9, bd = 1e9; std::vector<uint8_t> o, r; int w2, h2;
                for (int k = 0; k < reps; ++k) { double a = now_s(); o = rgb ? compress_rgb(px.data(), W, H, c) : compress(px.data(), W, H, c); double b = now_s();
                    if (rgb) decompress_rgb(o, r, w2, h2); else decompress(o, r, w2, h2); double e = now_s(); be = std::min(be, b - a); bd = std::min(bd, e - b); }
                bool g = r == px; ok &= g; double n = double(W) * H * (rgb ? 3 : 1); tpx += n; tpix += double(W) * H; te += be; td += bd;
                const size_t fsz = size_t(std::filesystem::file_size(fn)), sz = std::min(o.size(), fsz);   // "c" stores the input if it does not shrink
                tb += sz;
                printf("%-16s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s  %s%s\n", std::filesystem::path(fn).filename().string().c_str(), W, H, sz, 8.0 * sz / (double(W) * H), n / be / 1e6, n / bd / 1e6, g ? "ok" : "FAIL", sz < o.size() ? "  (stored)" : ""); }
            printf("TOTAL %zu files %.4f bpp  enc %.1f dec %.1f MB/s  (tile %d)  %s\n", pos.size(), 8 * tb / tpix, tpx / te / 1e6, tpx / td / 1e6, TILE, ok ? "all ok" : "FAIL");
            return ok ? 0 : 3; }
    } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 2; }
    fprintf(stderr, "bad arguments\n"); return 1;
}
