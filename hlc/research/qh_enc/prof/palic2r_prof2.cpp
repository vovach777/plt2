// palic2r: PLN2 with the two-pass FSE replaced by one-pass coders. Files "PLR1"/"PLR2": static QH tables (qh.hpp,
// qh_tables.h; off 2; runs of zero blocks), fast mode "PLF1"/"PLF2": RLGR pairs (off 1). Both on one bit lane per tile
// (rlgr_lane.hpp). -DPLR_NO_QH: PLR with RLGR as before (off 1). The notes below describe the RLGR history.
// (old first line:) palic2r: PLN2 with the FSE streams replaced by interleaved one-pass RLGR.
// (The page description below is layout 0, the previous format; the format is now layout 3, see PLR_LAYOUT.)
// Every plane of a tile has 8 RLGR coders (one per L); they and one raw bit writer for the L bit planes share one
// page buffer per tile (rlgr_fast.hpp: the llrice RLGR and its reservation rule on a lean page buffer), so all
// streams are interleaved in coding order and written on the fly; the decoder reserves pages at the same points
// in the same order. Tile = var number of pages, pages (uint64). No histograms, no second pass. With -DPLR_LLRICE
// the coders are llrice's own classes (llrice/, the proof of concept). The rest (tiles, MED, causal L, planes,
// RCT, the wave decoder, stored original) is palic2.cpp as is, except off: PLR uses off = 1 (PLN uses 2), and PLR2
// codes chroma as one plane S = zU + zV plus a raw split of zU (PLF2 keeps U and V planes); see PLN1.md.
// --- original header of palic2.cpp:
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
#if defined(__AVX2__)
#include <immintrin.h>
#endif
extern "C" {
#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
size_t FSE_buildCTableU16(FSE_CTable* ct, const short* normalizedCounter, unsigned maxSymbolValue, unsigned tableLog);
}
#include "/tmp/fable_enc/palic-1/netpbm.hpp"
#ifndef PLR_FAST_MASK
#define PLR_FAST_MASK 0xffu                     // fast mode: pairs in every bin (experiments: other L sets)
#endif
#if defined(PLR_LLRICE)
#include "llrice/rlgr.hpp"
using REnc = RLGR::Encoder; using RDec = RLGR::Decoder; using RRawW = BitStream::Writer; using RRawR = BitStream::Reader;
#define RAW_PUT(w, n, v) (w)->put_bits(n, v)
#define RAW_GET16(r) ([&] { uint16_t m_ = uint16_t((r)->peek_n(16)); (r)->skip(16); return m_; }())
#define RAW_GET(r, n) ([&] { const int n_ = (n); uint32_t v_ = uint32_t((r)->peek_n(n_)); (r)->skip(n_); return v_; }())
#else
#include "/tmp/fable_enc/palic-2/rlgr_fast.hpp"
// PLR_LAYOUT: 3 = one bit lane per tile, run headers in holes (rlgr_lane.hpp; the format of PLR / PLF);
// experiments: 0 = all streams of a tile interleaved in one page buffer (rlgr_fast.hpp, the previous format);
// 1 = every coder and the raw writer has its own stream, coded on the fly; 2 = own streams, the symbols of each bin
// are collected first and every bin is RLGR-coded at the end of the tile in one tight loop (the decoder decodes
// every bin up front from its symbol count, then the block loop only reads arrays).
#ifndef PLR_LAYOUT
#define PLR_LAYOUT 3
#endif
#if PLR_LAYOUT == 3 && !defined(PLR_NO_QH) && !defined(PLR_QH)
#define PLR_QH 1                                // the format: static QH tables (qh.hpp) instead of RLGR, except PLF
#endif
#if PLR_LAYOUT == 3
#include "/tmp/fable_enc/palic-2/rlgr_lane.hpp"
#if defined(PLR_QH)
#include "/tmp/fable_enc/palic-2/qh.hpp"
#endif
using REnc = rlgrl::Encoder; using RDec = rlgrl::Decoder; using RRawW = rlgrl::Lane; using RRawR = rlgrl::LaneR;
#undef RAW_PUT
#undef RAW_GET16
#undef RAW_GET
#define RAW_PUT(w, n, v) (w)->put(n, v)
#define RAW_GET16(r) uint16_t((r)->get(16))
#define RAW_GET(r, n) (r)->get(n)
#elif PLR_LAYOUT == 2
struct BinEnc { std::vector<uint32_t> v; inline void put(uint32_t x) { v.push_back(x); } };
struct BinDec { std::vector<uint32_t> v; size_t i = 0; inline uint32_t get() { return v[i++]; } };
using REnc = BinEnc; using RDec = BinDec; using RRawW = rlgrf::Writer; using RRawR = rlgrf::Reader;
#else
using REnc = rlgrf::Encoder; using RDec = rlgrf::Decoder; using RRawW = rlgrf::Writer; using RRawR = rlgrf::Reader;
#endif
#define RAW_PUT(w, n, v) (w)->put(n, v)
#define RAW_GET16(r) uint16_t((r)->get(16))
#define RAW_GET(r, n) (r)->get(n)
#endif

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
#include <x86intrin.h>
static uint64_t g_t[16]; static const char* g_tn[16]={"rgb_split","tile ctor","Y resid","Y blocks","UV resid+S","UV blocks","finish","rest"};
#define TS(k) { const uint64_t t_=__rdtsc(); g_t[k]+=t_-g_t0; g_t0=t_; }
static uint64_t g_t0;
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static inline uint32_t bwz(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
static inline int med(int a, int b, int c) { int mn = std::min(a, b), mx = std::max(a, b), p = a + b - c; return p < mn ? mn : (p > mx ? mx : p); }
#ifndef PLR_FAST_OFF
#define PLR_FAST_OFF 1                          // PLF (RLGR pairs) keeps off 1
#endif
#ifndef PLR_OFF
#if defined(PLR_QH)
#define PLR_OFF 2                               // PLR with QH tables: off 2 (as FSE in PLN2)
#else
#define PLR_OFF 1
#endif
#endif
struct Cfg { int off = PLR_OFF; bool fast = false; };
static constexpr int TILE = 288;

static inline uint16_t zz8(int e) { e = int8_t(e); return uint16_t(uint8_t((unsigned(e) << 1) ^ unsigned(e >> 7))); }
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
#if defined(PLR_QH)
static uint32_t g_sa = 64;                      // the causal sum of the current block (context of the QH tables)
#endif
#if defined(PLR_DUMP)
static FILE* g_dump = nullptr; static uint32_t g_lastsum = 0;
#endif
static inline int block_L(const uint16_t* s, int off) {
#if defined(PLR_DUMP)
    { uint32_t sum = 0; for (int j = 0; j < 16; ++j) sum += s[j]; g_lastsum = sum; }
#endif
#if defined(PLN_SSE)
    const __m128i a0 = _mm_loadu_si128((const __m128i*)s), a1 = _mm_loadu_si128((const __m128i*)(s + 8));
    __m128i t = _mm_add_epi16(a0, a1); t = _mm_add_epi16(t, _mm_srli_si128(t, 8)); t = _mm_add_epi16(t, _mm_srli_si128(t, 4)); t = _mm_add_epi16(t, _mm_srli_si128(t, 2));
#if defined(PLR_QH)
    g_sa = uint32_t(_mm_extract_epi16(t, 0));
#endif
    return std::min(7, std::max(0, int(bwz(uint32_t(_mm_extract_epi16(t, 0)) >> 4)) - off));
#else
    uint32_t sum = 0; for (int j = 0; j < 16; ++j) sum += s[j];
#if defined(PLR_QH)
    g_sa = sum;
#endif
    return std::min(7, std::max(0, int(bwz(sum >> 4)) - off));
#endif
}
template<int Off>
static inline int block_Lt(const uint16_t* s) { return block_L(s, Off); }
template<int Off>
static inline int causal_Lt(const uint16_t* z, size_t i, int tw) {
    const size_t px = i * 16;
    if (tw >= 16 && px >= size_t(tw)) return block_Lt<Off>(z + px - tw);
#if defined(PLR_QH)
    if (!i) { g_sa = 64; return 2; }
#endif
    return i ? block_Lt<Off>(z + px - 16) : 2;
}
static inline int causal_L(const uint16_t* z, size_t i, int tw, int off) {
    const size_t px = i * 16;
    if (tw >= 16 && px >= size_t(tw)) return block_L(z + px - tw, off);
#if defined(PLR_DUMP)
    if (!i) g_lastsum = 0xffffffffu;
#endif
#if defined(PLR_QH)
    if (!i) { g_sa = 64; return 2; }
#endif
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
// Fast mode (files PLF1 / PLF2): the high parts of neighbours 2j, 2j+1 of a block go as one RLGR symbol a+b
// (llrice CODER_G_RB_SUM / Malvar's RLGR3 idea) plus `a` in truncated binary over the a+b+1 values: k = floor(log2(a+b+1))
// bits if a < u = 2^(k+1) - (a+b+1), else k+1 bits of a+u. Half the RLGR steps. pair_mask: bit L = pairs in bin L.
// -DPLR_RAWSPLIT: `a` in bw(a+b) bits as llrice does (experiment only, not a file format).
struct Sink {                                   // one plane: 8 RLGR coders + the shared raw writer
    REnc* e[8]; RRawW* raw; unsigned pair_mask = 0; int pid = 0;
    inline void block(int L, const uint16_t* hk, const uint16_t* m) {
        REnc& c = *e[L];
#if defined(PLR_DUMP)
        const uint64_t dump_h0 = raw->head;
#endif
        if (pair_mask >> L & 1)
        for (int j = 0; j < 16; j += 2) { const uint32_t s2 = uint32_t(hk[j]) + hk[j + 1]; c.put(s2);
            if (s2) { const uint32_t a = hk[j];
#if !defined(PLR_RAWSPLIT)
                const int k = 31 - __builtin_clz(s2 + 1); const uint32_t u = (2u << k) - (s2 + 1);
                raw->reserve(k + 1); if (a < u) { if (k) RAW_PUT(raw, k, a); } else RAW_PUT(raw, k + 1, a + u);
#else
                const int n = 32 - __builtin_clz(s2); raw->reserve(n); RAW_PUT(raw, n, a);
#endif
            } }
        else
#if defined(PLR_QH)
        { (void)c; qh::qh_enc_block(*raw, pid ? 1u : 0u, g_sa, L, hk, m); return; }
#else
        for (int j = 0; j < 16; ++j) c.put(hk[j]);
#endif
#if defined(PLR_DUMP)
        if (g_dump) { uint8_t hd[2] = {uint8_t(pid), uint8_t(L)}; uint32_t x[2] = {uint32_t(raw->head - dump_h0), g_lastsum};
            uint16_t zz[16]; for (int j = 0; j < 16; ++j) { uint32_t v = uint32_t(hk[j]) << L; for (int k = 0; k < L; ++k) v |= uint32_t((m[k] >> j) & 1u) << k; zz[j] = uint16_t(v); }
            fwrite(hd, 1, 2, g_dump); fwrite(x, 4, 2, g_dump); fwrite(zz, 2, 16, g_dump); }
#endif
        if (L) { raw->reserve(16 * L); for (int k = 0; k < L; ++k) RAW_PUT(raw, 16, m[k]); }
    }
};
struct Source {
    RDec* d[8]; RRawR* raw; unsigned pair_mask = 0; int pid = 0;
    inline void block(int L, uint16_t* z) {
        alignas(32) uint16_t h[16]; alignas(32) uint16_t m[8];
        RDec& c = *d[L];
#if defined(PLR_QH)
        if (!(pair_mask >> L & 1)) { (void)c; qh::qh_dec_block(*raw, pid ? 1u : 0u, g_sa, L, h, m);
            const uint8_t* pq = reinterpret_cast<const uint8_t*>(m); block_unpack(h, L, pq, z); return; }
#endif
        if (pair_mask >> L & 1)
        for (int j = 0; j < 16; j += 2) { const uint32_t s2 = c.get(); uint32_t a = 0;
#if !defined(PLR_RAWSPLIT)
            if (s2) { const int k = 31 - __builtin_clz(s2 + 1); const uint32_t u = (2u << k) - (s2 + 1); raw->reserve(k + 1);
                a = k ? RAW_GET(raw, k) : 0; if (a >= u) a = ((a << 1) | RAW_GET(raw, 1)) - u; }
#else
            if (s2) { const int n = 32 - __builtin_clz(s2); raw->reserve(n); a = RAW_GET(raw, n); }
#endif
            h[j] = uint16_t(a); h[j + 1] = uint16_t(s2 - a); }
        else for (int j = 0; j < 16; ++j) h[j] = uint16_t(c.get());
        if (L) { raw->reserve(16 * L); for (int k = 0; k < L; ++k) m[k] = RAW_GET16(raw); }
        const uint8_t* pp = reinterpret_cast<const uint8_t*>(m); block_unpack(h, L, pp, z);
    }
};
static thread_local Sink* g_sink; static thread_local Source* g_src;
#if defined(PLR_QH)
// Runs of zero blocks (QH only; not in the fast mode): a block whose causal sum sa is 0 first sends n = the number of
// all-zero blocks from it on (in block order, within the plane of the tile) as Exp-Golomb order 0; those n blocks
// are not coded at all (no letters, no planes, no splits). n = 0: the block is coded as usual.
static inline bool blk_zero(const uint16_t* b) { uint32_t o = 0; for (int j = 0; j < 16; ++j) o |= b[j]; return !o; }
static inline bool qz_enc(size_t& left, const uint16_t* z, size_t i, size_t nb) {
    if (g_sink->pair_mask) return false;
    if (left) { --left; return true; }
    if (g_sa) return false;
    size_t n = 0; while (i + n < nb && blk_zero(z + (i + n) * 16)) ++n;
    const uint64_t v = n + 1; const int b = int(64 - __builtin_clzll(v)); g_sink->raw->put64(2 * b - 1, v);
    if (n) { left = n - 1; return true; }
    return false;
}
static inline bool qz_dec(size_t& left, uint16_t* zb, size_t i, size_t nb) {
    if (g_src->pair_mask) return false;
    if (!left) {
        if (g_sa) return false;
        rlgrl::LaneR& r = *g_src->raw; const uint64_t w = r.window(); const int zc = w ? __builtin_clzll(w) : 64;
        if (zc > 30) throw std::runtime_error("zero run code");
        const uint64_t v = w >> (63 - 2 * zc); r.skip(2 * zc + 1); const size_t n = size_t(v - 1);
        if (!n) return false;
        if (n > nb - i) throw std::runtime_error("zero run too long");
        left = n;
    }
    --left; memset(zb, 0, 32); return true;
}
// The encoder loop of one plane (PLR): the same bits as QZ_ENC + block_split + Sink::block (+ tb_put), written
// through the register writer. a != nullptr: the S plane, a = zU, split after every coded block.
static inline void tb_put16(rlgrl::LaneW& w, const uint16_t* a, const uint16_t* S) {     // 16 x tb_put, no branch per symbol: 4 (AVX2: 8 if they fit) per put
#if defined(__AVX2__)
    for (int g = 0; g < 16; g += 8) {
        const __m256i s1 = _mm256_add_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(S + g))), _mm256_set1_epi32(1));
        const __m256i x = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(a + g)));
        const __m256i k = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(s1)), 23), _mm256_set1_epi32(127));   // floor(log2(s1)), s1 < 2^24
        const __m256i u = _mm256_sub_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(2), k), s1);
        const __m256i ge = _mm256_cmpgt_epi32(x, _mm256_sub_epi32(u, _mm256_set1_epi32(1)));                                             // x >= u
        uint64_t c0, c1; unsigned n0, n1;
        rlgrl::vl_join8(_mm256_add_epi32(x, _mm256_and_si256(u, ge)), _mm256_sub_epi32(k, ge), c0, n0, c1, n1);
        if (__builtin_expect(n0 + n1 <= 63, 1)) w.put(n0 + n1, c0 << n1 | c1); else { w.put(n0, c0); w.put(n1, c1); } }
    return;
#endif
    for (int g = 0; g < 16; g += 4) { uint64_t acc = 0; unsigned n = 0;
        for (int j = g; j < g + 4; ++j) { const uint32_t s1 = uint32_t(S[j]) + 1, k = uint32_t(31 - __builtin_clz(s1)), u = (2u << k) - s1, x = a[j], ge = x >= u;
            acc = (acc << (k + ge)) | (x + (u & (0u - ge))); n += k + ge; }
        w.put(n, acc); }
}
static void qh_plane_enc(const uint16_t* z, const uint16_t* a, size_t nb, int tw, int off) {
    rlgrl::LaneW w(*g_sink->raw); const uint32_t pt = g_sink->pid ? 1u : 0u; size_t left = 0;
    for (size_t i = 0; i < nb; ++i) {
        if (left) { --left; continue; }
        const int L = causal_L(z, i, tw, off); w.room();
        if (!g_sa) { size_t n = 0; while (i + n < nb && blk_zero(z + (i + n) * 16)) ++n;
            const uint64_t v = n + 1; const int b = int(64 - __builtin_clzll(v)); w.put(unsigned(2 * b - 1), v);
            if (n) { left = n - 1; continue; } }
        qh::qh_enc_block_w(w, pt, g_sa, L, z + i * 16);
        if (a) tb_put16(w, a + i * 16, z + i * 16);
    }
    w.sync();
}
#define QZ_ENC(z, i, nb) if (qz_enc(zl, z, i, nb)) continue
#define QZ_DEC(z, i, nb) if (qz_dec(zl, (z) + (i) * 16, i, nb)) continue
#else
#define QZ_ENC(z, i, nb)
#define QZ_DEC(z, i, nb)
#endif
static void tile_enc(const uint8_t* img, int S, int tw, int th, const Cfg& c, Out& o, std::vector<uint8_t>& tmp) {
    const size_t n = size_t(tw) * th, nb = (n + 15) / 16;
    static thread_local UBuf<uint16_t> zb, hb[8], plb;
    zb.need(nb * 16 + 16); uint16_t* z = zb.data(); std::fill(z + n, z + nb * 16 + 16, uint16_t(0));
    resid_row0(img, z, tw);
    for (int y = 1; y < th; ++y) resid_row(img + size_t(y) * S, img + size_t(y - 1) * S, z + size_t(y) * tw, tw);
#if defined(PLR_QH) && !defined(PLR_DUMP) && !defined(PLR_OLD_ENC)
    TS(2) if (!g_sink->pair_mask) { qh_plane_enc(z, nullptr, nb, tw, c.off); TS(3) return; }
#endif
    plb.need(nb * 8); uint16_t* pw = plb.data(); size_t hn[8] = {0};
    for (auto& h : hb) h.need(nb * 16);
    size_t zl = 0; (void)zl;
    if (c.off == 2) {
        for (size_t i = 0; i < nb; ++i) {
            alignas(32) uint16_t hk[16];
            const int L = causal_Lt<2>(z, i, tw); QZ_ENC(z, i, nb);
            block_split(z + i * 16, L, pw, hk);
            g_sink->block(L, hk, pw - L); (void)hn;
        }
    } else {
        for (size_t i = 0; i < nb; ++i) {
            alignas(32) uint16_t hk[16];
            const int L = causal_L(z, i, tw, c.off); QZ_ENC(z, i, nb);
            block_split(z + i * 16, L, pw, hk);
            g_sink->block(L, hk, pw - L); (void)hn;
        }
    }

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
    (void)in; (void)hb; size_t zl = 0; (void)zl;
    if (c.off == 2) {
        for (size_t i = 0; i < nb; ++i) {
            const int L = causal_Lt<2>(z, i, tw); QZ_DEC(z, i, nb);
            g_src->block(L, z + i * 16);
        }
    } else {
        for (size_t i = 0; i < nb; ++i) {
            const int L = causal_L(z, i, tw, c.off); QZ_DEC(z, i, nb);
            g_src->block(L, z + i * 16);
        }
    }

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
static size_t read_text_header(const std::vector<uint8_t>& d, bool& rgb, int& W, int& H, bool& fast) {
    auto nl = std::find(d.begin(), d.end(), '\n');
    if (nl == d.end()) throw std::runtime_error("bad header");
    std::string magic(d.begin(), nl);
    if (magic != "PLR1" && magic != "PLR2" && magic != "PLF1" && magic != "PLF2") throw std::runtime_error("not PLR1/2, PLF1/2");
    rgb = magic[3] == '2'; fast = magic[2] == 'F';
    auto nl2 = std::find(nl + 1, d.end(), '\n');
    if (nl2 == d.end()) throw std::runtime_error("bad header");
    std::string dims(nl + 1, nl2);
    int w = 0, h = 0;
    if (std::sscanf(dims.c_str(), "%d %d", &w, &h) != 2 || w <= 0 || h <= 0) throw std::runtime_error("bad header");
    W = w; H = h;
    return size_t(nl2 - d.begin()) + 1;
}

#if defined(PLR_LLRICE)
struct TileEnc {
    PagePool pool; std::vector<std::unique_ptr<REnc>> e; RRawW raw; Sink sk[3];
    explicit TileEnc(int planes, size_t px) : raw(pool) { pool.reserve(px / 4 + 64);
        for (int p = 0; p < planes; ++p) { for (int L = 0; L < 8; ++L) { e.emplace_back(new REnc(pool)); sk[p].e[L] = e.back().get(); } sk[p].raw = &raw; } }
    void finish(Out& o) { for (auto& x : e) x->flush(); raw.flush(); o.var(pool.size()); o.raw(pool.data(), pool.size() * 8); }
};
struct TileDec {
    PagePool pool; std::vector<std::unique_ptr<RDec>> d; std::unique_ptr<RRawR> raw; Source sr[3];
    TileDec(In& in, int planes) : pool(read_pages(in)) { raw.reset(new RRawR(pool));
        for (int p = 0; p < planes; ++p) { for (int L = 0; L < 8; ++L) { d.emplace_back(new RDec(pool)); sr[p].d[L] = d.back().get(); } sr[p].raw = raw.get(); } }
    static std::vector<uint64_t> read_pages(In& in) { size_t n = size_t(in.var()); std::vector<uint64_t> v(n); memcpy(v.data(), in.take(n * 8), n * 8); return v; }
};
#else
#if PLR_LAYOUT == 3
struct TileEnc {                                // one bit lane per tile for every coder and the raw bits
    rlgrl::Lane lane; std::vector<REnc> e; Sink sk[3];
    explicit TileEnc(int planes, size_t px) { lane.w.assign(px / 16 + 64, 0); e.reserve(24);
        for (int i = 0; i < planes * 8; ++i) e.emplace_back(&lane);
        for (int p = 0; p < planes; ++p) { sk[p].raw = &lane; for (int L = 0; L < 8; ++L) sk[p].e[L] = &e[size_t(p) * 8 + L]; } }
    void finish(Out& o) { for (auto& x : e) x.flush(); const size_t n = lane.words(); o.var(n); o.raw(lane.w.data(), n * 8);
#if defined(PLR_DUMP)
        if (g_dump) { uint8_t hd[2] = {255, 0}; uint32_t x[2] = {uint32_t(lane.head), uint32_t(n * 64)}; uint16_t z16[16] = {0};
            fwrite(hd, 1, 2, g_dump); fwrite(x, 4, 2, g_dump); fwrite(z16, 2, 16, g_dump); }
#endif
    }
};
struct TileDec {
    std::vector<uint64_t> words; rlgrl::LaneR lr; std::vector<RDec> d; Source sr[3];
    TileDec(In& in, int planes) { const size_t n = size_t(in.var()); words.assign(n + 2, 0); memcpy(words.data(), in.take(n * 8), n * 8);
        lr.w = words.data(); lr.nw = n + 2; d.reserve(24);
        for (int i = 0; i < planes * 8; ++i) d.emplace_back(&lr);
        for (int p = 0; p < planes; ++p) { sr[p].raw = &lr; for (int L = 0; L < 8; ++L) sr[p].d[L] = &d[size_t(p) * 8 + L]; } }
};
#elif PLR_LAYOUT == 0
struct TileEnc {                                // one page buffer per tile, shared by every coder of every plane
    rlgrf::PageBuf pb; std::vector<REnc> e; RRawW raw; Sink sk[3];
    explicit TileEnc(int planes, size_t px) : raw(&pb) { pb.w.resize(px / 4 + 64); e.reserve(24);
        for (int p = 0; p < planes; ++p) { for (int L = 0; L < 8; ++L) e.emplace_back(&pb); sk[p].raw = &raw; }
        for (int p = 0; p < planes; ++p) for (int L = 0; L < 8; ++L) sk[p].e[L] = &e[size_t(p) * 8 + L]; }
    void finish(Out& o) { for (auto& x : e) x.flush(); raw.flush(); o.var(pb.top); o.raw(pb.w.data(), pb.top * 8); }
};
struct TileDec {
    std::vector<uint64_t> pages; size_t top = 0; std::vector<RDec> d; std::unique_ptr<RRawR> raw; Source sr[3];
    TileDec(In& in, int planes) { const size_t n = size_t(in.var()); pages.resize(n); memcpy(pages.data(), in.take(n * 8), n * 8);
        raw.reset(new RRawR(pages.data(), &top, n)); d.reserve(24);
        for (int p = 0; p < planes; ++p) for (int L = 0; L < 8; ++L) d.emplace_back(pages.data(), &top, n);
        for (int p = 0; p < planes; ++p) { for (int L = 0; L < 8; ++L) sr[p].d[L] = &d[size_t(p) * 8 + L]; sr[p].raw = raw.get(); } }
};
#else
struct TileEnc {                                // own stream per coder (and for the raw writer)
    std::vector<rlgrf::PageBuf> pb; std::vector<REnc> e; RRawW raw; Sink sk[3]; int ns;
    explicit TileEnc(int planes, size_t px) : pb(size_t(planes) * 8 + 1), raw(&pb[size_t(planes) * 8]) { ns = planes * 8;
        for (auto& b : pb) b.w.resize(px / 64 + 64); e.reserve(24);
#if PLR_LAYOUT == 1
        for (int i = 0; i < ns; ++i) e.emplace_back(&pb[size_t(i)]);
#else
        for (int i = 0; i < ns; ++i) { e.emplace_back(); e.back().v.reserve(px / 4); }
#endif
        for (int p = 0; p < planes; ++p) { sk[p].raw = &raw; for (int L = 0; L < 8; ++L) sk[p].e[L] = &e[size_t(p) * 8 + L]; } }
    void finish(Out& o) {
        for (int i = 0; i < ns; ++i) {
#if PLR_LAYOUT == 1
            e[size_t(i)].flush();
#else
            o.var(e[size_t(i)].v.size()); rlgrf::Encoder c(&pb[size_t(i)]);
            for (const uint32_t v : e[size_t(i)].v) c.put(v);
            c.flush();
#endif
        }
        raw.flush();
        for (auto& b : pb) { o.var(b.top); o.raw(b.w.data(), b.top * 8); }
    }
};
struct TileDec {
    std::vector<uint64_t> pages; std::vector<size_t> top, off, cnt; std::vector<RDec> d; std::unique_ptr<RRawR> raw; Source sr[3];
    TileDec(In& in, int planes) { const int ns = planes * 8; cnt.resize(size_t(ns)); top.assign(size_t(ns) + 1, 0); off.resize(size_t(ns) + 2);
#if PLR_LAYOUT == 2
        for (auto& x : cnt) x = size_t(in.var());
#endif
        std::vector<std::pair<const uint8_t*, size_t>> st; size_t tot = 0;
        for (int i = 0; i <= ns; ++i) { const size_t n = size_t(in.var()); st.push_back({in.take(n * 8), n}); off[size_t(i)] = tot; tot += n; }
        pages.resize(tot); for (int i = 0; i <= ns; ++i) memcpy(pages.data() + off[size_t(i)], st[size_t(i)].first, st[size_t(i)].second * 8);
        raw.reset(new RRawR(pages.data() + off[size_t(ns)], &top[size_t(ns)], st[size_t(ns)].second)); d.reserve(24);
        for (int i = 0; i < ns; ++i) {
#if PLR_LAYOUT == 1
            d.emplace_back(pages.data() + off[size_t(i)], &top[size_t(i)], st[size_t(i)].second);
#else
            d.emplace_back(); rlgrf::Decoder c(pages.data() + off[size_t(i)], &top[size_t(i)], st[size_t(i)].second);
            auto& v = d.back().v; v.resize(cnt[size_t(i)]); for (auto& x : v) x = c.get();
#endif
        }
        for (int p = 0; p < planes; ++p) { sr[p].raw = raw.get(); for (int L = 0; L < 8; ++L) sr[p].d[L] = &d[size_t(p) * 8 + L]; } }
};
#endif
#endif
std::vector<uint8_t> compress(const uint8_t* img, int W, int H, const Cfg& c) {
    Grid g = grid(W, H); const int nt = g.nx * g.ny; static thread_local std::vector<uint8_t> tmp;
    Out o; o.b.reserve(size_t(W) * H + size_t(nt) * 4 + 4096);
    put_text_header(o, c.fast ? "PLF1" : "PLR1", W, H);
    const size_t lens = o.b.size(); o.b.resize(lens + size_t(nt) * 4);
    for (int i = 0; i < nt; ++i) {
        const size_t b = o.b.size(); const uint8_t* ti = img + size_t(g.ty(i)) * W + g.tx(i);
        TS(7) TileEnc te(1, size_t(g.tw(i)) * g.th(i)); te.sk[0].pair_mask = c.fast ? PLR_FAST_MASK : 0; g_sink = &te.sk[0]; TS(1)
        tile_enc(ti, W, g.tw(i), g.th(i), c, o, tmp); te.finish(o); TS(6)
        const uint32_t l = uint32_t(o.b.size() - b); memcpy(&o.b[lens + size_t(i) * 4], &l, 4);
    }
    return std::move(o.b);
}
void decompress(const std::vector<uint8_t>& d, std::vector<uint8_t>& img, int& W, int& H) {
    Cfg c; bool rgb = false; size_t pos = read_text_header(d, rgb, W, H, c.fast); if (c.fast) c.off = PLR_FAST_OFF;
    if (rgb) throw std::runtime_error("not PLR1/PLF1");
    Grid g = grid(W, H); const int nt = g.nx * g.ny; std::vector<uint32_t> len(nt);
    if (d.size() < pos + size_t(nt) * 4) throw std::runtime_error("truncated");
    for (auto& l : len) { memcpy(&l, d.data() + pos, 4); pos += 4; }
    img.resize(size_t(W) * H);
    for (int i = 0; i < nt; ++i) {
        if (d.size() < pos + len[i]) throw std::runtime_error("truncated");
        In ti{d.data() + pos, d.data() + pos + len[i]}; pos += len[i];
        TileDec td(ti, 1); td.sr[0].pair_mask = c.fast ? PLR_FAST_MASK : 0; g_src = &td.sr[0];
        tile_dec(ti, img.data() + size_t(g.ty(i)) * W + g.tx(i), W, g.tw(i), g.th(i), c); if (ti.p != ti.e) throw std::runtime_error("tile trailing bytes"); }
}

// Colour. U = R-G, V = B-G, Y = G + (U+V+2)>>2, then V -= U>>2.
// U,V are -255..255; residual -510..510 folds mod 512 into -256..255.
static inline int wrap9(int d) { return int(unsigned(d + 256) & 511u) - 256; }
static inline int fold9(int s) { if (s > 255) return s - 512; if (s < -255) return s + 512; return s; }
static inline uint16_t zz16(int e) { return uint16_t((unsigned(e) << 1) ^ unsigned(e >> 31)); }
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
    size_t zl = 0; (void)zl;
    if (c.off == 2) {
        for (size_t i = 0; i < nb; ++i) {
            alignas(32) uint16_t hk[16];
            const int L = causal_Lt<2>(z, i, tw); QZ_ENC(z, i, nb);
            block_split(z + i * 16, L, pw, hk);
            g_sink->block(L, hk, pw - L); (void)hn;
        }
    } else {
        for (size_t i = 0; i < nb; ++i) {
            alignas(32) uint16_t hk[16];
            const int L = causal_L(z, i, tw, c.off); QZ_ENC(z, i, nb);
            block_split(z + i * 16, L, pw, hk);
            g_sink->block(L, hk, pw - L); (void)hn;
        }
    }

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
    (void)in; (void)hb; size_t zl = 0; (void)zl;
    if (c.off == 2) {
        for (size_t i = 0; i < nb; ++i) {
            const int L = causal_Lt<2>(z, i, tw); QZ_DEC(z, i, nb);
            g_src->block(L, z + i * 16);
        }
    } else {
        for (size_t i = 0; i < nb; ++i) {
            const int L = causal_L(z, i, tw, c.off); QZ_DEC(z, i, nb);
            g_src->block(L, z + i * 16);
        }
    }

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
#ifndef PLR_UVOFF
#define PLR_UVOFF 0                             // added to off for the S plane
#endif
// PLR2 chroma (Vladimir's idea; PLF2 keeps the U and V planes): one chroma plane S = zU + zV (zigzag residuals of U and V at the same pixel, 0..1022) goes
// through the plane coder (causal L on S, L bit planes, RLGR on S >> L); after every block the raw writer gets zU of
// its 16 pixels in truncated binary over the S + 1 values (nothing when S = 0). zV = S - zU.
static inline void tb_put(RRawW* w, uint32_t a, uint32_t s2) {
    const int k = 31 - __builtin_clz(s2 + 1); const uint32_t u = (2u << k) - (s2 + 1);
    w->reserve(k + 1); if (a < u) { if (k) RAW_PUT(w, k, a); } else RAW_PUT(w, k + 1, a + u);
}
static inline uint32_t tb_get(RRawR* r, uint32_t s2) {
    const int k = 31 - __builtin_clz(s2 + 1); const uint32_t u = (2u << k) - (s2 + 1); r->reserve(k + 1);
    uint32_t a = k ? RAW_GET(r, k) : 0; if (a >= u) a = ((a << 1) | RAW_GET(r, 1)) - u; return a;
}
static void tile_enc_uv(const int16_t* U, const int16_t* V, int tw, int th, const Cfg& c) {
    const size_t n = size_t(tw) * th, nb = (n + 15) / 16;
    static thread_local UBuf<uint16_t> zu, zv, sb, plb;
    zu.need(nb * 16 + 16); zv.need(nb * 16 + 16); sb.need(nb * 16 + 16);
    uint16_t* a = zu.data(); uint16_t* b = zv.data(); uint16_t* S = sb.data();
    resid16_row0(U, a, tw); resid16_row0(V, b, tw);
    for (int y = 1; y < th; ++y) { resid16_row(U + size_t(y) * tw, U + size_t(y - 1) * tw, a + size_t(y) * tw, tw);
                                   resid16_row(V + size_t(y) * tw, V + size_t(y - 1) * tw, b + size_t(y) * tw, tw); }
    for (size_t p = 0; p < n; ++p) S[p] = uint16_t(a[p] + b[p]);
    std::fill(S + n, S + nb * 16 + 16, uint16_t(0)); std::fill(a + n, a + nb * 16 + 16, uint16_t(0));
#if defined(PLR_QH) && !defined(PLR_DUMP) && !defined(PLR_OLD_ENC)
    TS(4) if (!g_sink->pair_mask) { qh_plane_enc(S, a, nb, tw, c.off + PLR_UVOFF); TS(5) return; }
#endif
    plb.need(nb * 8); uint16_t* pw = plb.data(); size_t zl = 0; (void)zl;
    for (size_t i = 0; i < nb; ++i) {
        alignas(32) uint16_t hk[16];
        const int L = causal_L(S, i, tw, c.off + PLR_UVOFF); QZ_ENC(S, i, nb);
        block_split(S + i * 16, L, pw, hk);
        g_sink->block(L, hk, pw - L);
        for (int j = 0; j < 16; ++j) { const uint32_t s2 = S[i * 16 + j]; if (s2) tb_put(g_sink->raw, a[i * 16 + j], s2); }
    }
}
#if defined(PLR_QH)
static inline bool qz_dec_uv(size_t& zl, uint16_t* S, uint16_t* a, uint16_t* b, size_t i, size_t nb) {
    if (!qz_dec(zl, S + i * 16, i, nb)) return false;
    memset(a + i * 16, 0, 32); memset(b + i * 16, 0, 32); return true;
}
#else
static inline bool qz_dec_uv(size_t&, uint16_t*, uint16_t*, uint16_t*, size_t, size_t) { return false; }
#endif
static void tile_dec_uv(int16_t* U, int16_t* V, int tw, int th, const Cfg& c) {
    const size_t n = size_t(tw) * th, nb = (n + 15) / 16;
    static thread_local UBuf<uint16_t> zu, zv, sb;
    zu.need(nb * 16 + 16); zv.need(nb * 16 + 16); sb.need(nb * 16 + 16);
    uint16_t* a = zu.data(); uint16_t* b = zv.data(); uint16_t* S = sb.data(); size_t zl = 0; (void)zl;
    for (size_t i = 0; i < nb; ++i) {
        const int L = causal_L(S, i, tw, c.off + PLR_UVOFF);
        if (qz_dec_uv(zl, S, a, b, i, nb)) continue;
        g_src->block(L, S + i * 16);
        for (int j = 0; j < 16; ++j) { const size_t p = i * 16 + j; const uint32_t s2 = S[p];
            const uint32_t x = s2 ? tb_get(g_src->raw, s2) : 0; a[p] = uint16_t(x); b[p] = uint16_t(s2 - x); }
    }
#if defined(PLR_VEC) && !defined(PLN_ROWS)
    recon_wave<true>(U, tw, tw, th, a); recon_wave<true>(V, tw, tw, th, b);
#else
    static thread_local std::vector<int16_t> e; if (e.size() < size_t(tw)) e.resize(tw);
    for (int pl = 0; pl < 2; ++pl) { int16_t* img = pl ? V : U; const uint16_t* zz = pl ? b : a;
        for (int y = 0; y < th; ++y) { const uint16_t* zr = zz + size_t(y) * tw;
            for (int x = 0; x < tw; ++x) e[x] = int16_t(int(zr[x] >> 1) ^ -int(zr[x] & 1));
            if (y == 0) recon16_row0(img, e.data(), tw); else recon16_row(img + size_t(y) * tw, img + size_t(y - 1) * tw, e.data(), tw); } }
#endif
}
static void rgb_split(const uint8_t* rgb, int W, int tw, int th, std::vector<uint8_t>& Y, std::vector<int16_t>& U, std::vector<int16_t>& V) {
    Y.resize(size_t(tw) * th); U.resize(Y.size()); V.resize(Y.size()); int bad = 0;
    for (int y = 0; y < th; ++y) {
        const uint8_t* __restrict p = rgb + size_t(y) * W * 3;
        uint8_t* __restrict py = Y.data() + size_t(y) * tw; int16_t* __restrict pu = U.data() + size_t(y) * tw; int16_t* __restrict pv = V.data() + size_t(y) * tw;
        for (int x = 0; x < tw; ++x) {                 // no branch in the loop: it vectorises (the range check is after it)
            int r = p[3 * x], g = p[3 * x + 1], b = p[3 * x + 2];
            int u = r - g, v = b - g, yy = g + ((u + v + 2) >> 2); v -= u >> 2;
            bad |= yy; py[x] = uint8_t(yy); pu[x] = int16_t(u); pv[x] = int16_t(v);
        }
    }
    if (unsigned(bad) > 255) throw std::runtime_error("Y out of range");
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
    put_text_header(o, c.fast ? "PLF2" : "PLR2", W, H);
    const size_t lens = o.b.size(); o.b.resize(lens + size_t(nt) * 4);
    for (int i = 0; i < nt; ++i) {
        const size_t b = o.b.size();
        TS(7) rgb_split(rgb + (size_t(g.ty(i)) * W + g.tx(i)) * 3, W, g.tw(i), g.th(i), Y, U, V); TS(0)
        TileEnc te(3, size_t(g.tw(i)) * g.th(i) * 3); for (auto& k : te.sk) k.pair_mask = c.fast ? PLR_FAST_MASK : 0; TS(1)
        te.sk[1].pid = 1; te.sk[2].pid = 2; g_sink = &te.sk[0]; tile_enc(Y.data(), g.tw(i), g.tw(i), g.th(i), c, o, tmp);
        if (!c.fast) { g_sink = &te.sk[1]; tile_enc_uv(U.data(), V.data(), g.tw(i), g.th(i), c); }
        else {
        g_sink = &te.sk[1]; tile_enc16(U.data(), g.tw(i), g.th(i), c, o, tmp);
        g_sink = &te.sk[2]; tile_enc16(V.data(), g.tw(i), g.th(i), c, o, tmp); }
        te.finish(o); TS(6)
        const uint32_t l = uint32_t(o.b.size() - b); memcpy(&o.b[lens + size_t(i) * 4], &l, 4);
    }
    return std::move(o.b);
}
void decompress_rgb(const std::vector<uint8_t>& d, std::vector<uint8_t>& rgb, int& W, int& H) {
    Cfg c; bool rgb_file = false; size_t pos = read_text_header(d, rgb_file, W, H, c.fast); if (c.fast) c.off = PLR_FAST_OFF;
    if (!rgb_file) throw std::runtime_error("not PLR2/PLF2");
    Grid g = grid(W, H); const int nt = g.nx * g.ny; std::vector<uint32_t> len(nt);
    if (d.size() < pos + size_t(nt) * 4) throw std::runtime_error("truncated");
    for (auto& l : len) { memcpy(&l, d.data() + pos, 4); pos += 4; }
    rgb.resize(size_t(W) * H * 3);
    static thread_local std::vector<uint8_t> Y; static thread_local std::vector<int16_t> U, V;
    for (int i = 0; i < nt; ++i) {
        if (d.size() < pos + len[i]) throw std::runtime_error("truncated");
        In ti{d.data() + pos, d.data() + pos + len[i]}; pos += len[i];
        const int tw = g.tw(i), th = g.th(i);
        Y.resize(size_t(tw) * th); U.resize(Y.size()); V.resize(Y.size());
        TileDec td(ti, 3); for (auto& k : td.sr) k.pair_mask = c.fast ? PLR_FAST_MASK : 0;
        td.sr[1].pid = 1; td.sr[2].pid = 2; g_src = &td.sr[0]; tile_dec(ti, Y.data(), tw, tw, th, c);
        if (!c.fast) { g_src = &td.sr[1]; tile_dec_uv(U.data(), V.data(), tw, th, c); }
        else {
        g_src = &td.sr[1]; tile_dec16(ti, U.data(), tw, th, c);
        g_src = &td.sr[2]; tile_dec16(ti, V.data(), tw, th, c); }
        if (ti.p != ti.e) throw std::runtime_error("tile trailing bytes");
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
#if defined(PLR_QH)
    qh::init();
#endif
    if (argc < 3) { fprintf(stderr, "usage: %s c in.pgm|ppm out.plr [-f] (the input itself if it does not shrink) | d in.plr|pgm|ppm out.pgm|ppm | bench a.pgm|ppm|dir ... [-r N] [-f]  (-f: fast mode, pairs, files PLF1/PLF2)\n", argv[0]); return 1; }
    std::string cmd = argv[1]; Cfg c; int reps = 3; std::vector<std::string> pos;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i];
        if (a == "-r" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "-f") { c.fast = true; c.off = PLR_FAST_OFF; }
        else if (cmd == "bench" && std::filesystem::is_directory(a)) { std::vector<std::string> v;
            for (auto& e : std::filesystem::directory_iterator(a)) { auto x = e.path().extension();
                if (x == ".pgm" || x == ".ppm" || x == ".pnm") v.push_back(e.path().string()); }
            std::sort(v.begin(), v.end()); pos.insert(pos.end(), v.begin(), v.end()); }
        else pos.push_back(a); }
    try {
        if (cmd == "c" && pos.size() == 2) {
            int W, H; bool rgb; auto px = read_netpbm(pos[0], W, H, rgb);
#if defined(PLR_DUMP)
            if (const char* dp = getenv("PLR_DUMP_FILE")) { palic2::g_dump = fopen(dp, "wb"); int hdr[4] = {W, H, rgb ? 3 : 1, c.fast ? 1 : 0}; fwrite(hdr, 4, 4, palic2::g_dump); }
#endif
            auto o = rgb ? compress_rgb(px.data(), W, H, c) : compress(px.data(), W, H, c);
            // not smaller than the input file: store the input file itself (a P5/P6 text header like ours);
            // the decoder sees "P5"/"P6" instead of "PLN" and just copies the file
            auto src = read_bin(pos[0]);
            if (!src.empty() && o.size() >= src.size()) o.swap(src);
#if defined(RLGR_STATS) && !defined(PLR_LLRICE)
            fprintf(stderr, "STATS %zu %llu %llu %llu %llu %llu %d\n", o.size(), (unsigned long long)rlgrf::g_payload, (unsigned long long)rlgrf::g_waste_full, (unsigned long long)rlgrf::g_nfull, (unsigned long long)rlgrf::g_waste_flush, (unsigned long long)rlgrf::g_nruns, (W + 287) / 288 * ((H + 287) / 288));
            fprintf(stderr, "SIMB %llu %llu %llu\n", (unsigned long long)rlgrf::g_head, (unsigned long long)rlgrf::g_nshrink, (unsigned long long)rlgrf::g_lost_b);
#endif
#if defined(PLR_DUMP)
            if (palic2::g_dump) fclose(palic2::g_dump);
#endif
            FILE* f = fopen(pos[1].c_str(), "wb"); if (!f) return 1; fwrite(o.data(), 1, o.size(), f); fclose(f); return 0; }
        if (cmd == "d" && pos.size() == 2) {
            auto d = read_bin(pos[0]); if (d.empty()) { fprintf(stderr, "cannot read %s\n", pos[0].c_str()); return 1; }
            if (d.size() >= 2 && d[0] == 'P' && (d[1] == '5' || d[1] == '6')) {       // stored original: copy
                FILE* f = fopen(pos[1].c_str(), "wb"); if (!f) return 1;
                const bool ok = fwrite(d.data(), 1, d.size(), f) == d.size(); fclose(f); return ok ? 0 : 1; }
            std::vector<uint8_t> img; int W, H; bool rgb = d.size() >= 4 && d[0] == 'P' && d[1] == 'L' && d[3] == '2';
            if (rgb) decompress_rgb(d, img, W, H); else decompress(d, img, W, H);
            write_netpbm(pos[1], img, W, H, rgb); return 0; }
        if (cmd == "bench") {
            double te = 0, td = 0, tpx = 0, tpix = 0, tb = 0; bool ok = true;
            for (auto& fn : pos) { int W, H; bool rgb; std::vector<uint8_t> px;
                try { px = read_netpbm(fn, W, H, rgb); } catch (...) { printf("%s: skip\n", fn.c_str()); continue; }
                double be = 1e9, bd = 1e9; std::vector<uint8_t> o, r; int w2, h2;
                for (int k = 0; k < reps; ++k) { g_t0 = __rdtsc(); double a = now_s(); o = rgb ? compress_rgb(px.data(), W, H, c) : compress(px.data(), W, H, c); double b = now_s(); TS(7)
                    if (rgb) decompress_rgb(o, r, w2, h2); else decompress(o, r, w2, h2); double e = now_s(); be = std::min(be, b - a); bd = std::min(bd, e - b); }
                bool g = r == px; ok &= g; double n = double(W) * H * (rgb ? 3 : 1); tpx += n; tpix += double(W) * H; te += be; td += bd;
                const size_t fsz = size_t(std::filesystem::file_size(fn)), sz = std::min(o.size(), fsz);   // "c" stores the input if it does not shrink
                tb += sz;
                printf("%-16s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s  %s%s\n", std::filesystem::path(fn).filename().string().c_str(), W, H, sz, 8.0 * sz / (double(W) * H), n / be / 1e6, n / bd / 1e6, g ? "ok" : "FAIL", sz < o.size() ? "  (stored)" : ""); }
            { uint64_t tt=0; for (int k=0;k<8;++k) tt+=g_t[k]; for (int k=0;k<8;++k) printf("  %-12s %5.1f%%  %6.2f cyc/px\n", g_tn[k], 100.0*g_t[k]/tt, double(g_t[k])/reps/tpix); printf("  total %.2f cyc/px\n", double(tt)/reps/tpix); }
            printf("TOTAL %zu files %.4f bpp  enc %.1f dec %.1f MB/s  (tile %d)  %s\n", pos.size(), 8 * tb / tpix, tpx / te / 1e6, tpx / td / 1e6, TILE, ok ? "all ok" : "FAIL");
            return ok ? 0 : 3; }
    } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 2; }
    fprintf(stderr, "bad arguments\n"); return 1;
}
