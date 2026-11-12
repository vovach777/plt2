// qh.hpp - prototype "QH": static quad-Huffman coder for the high parts of a 16-block (tables from train_tables.py).
// A block: table = ctxmap[plane type][Q(sa)] (sa = the causal sum L is made from); 4 letters (4 symbols each, min(h,3), 2 bits
// per symbol, Huffman, <= ML bits, one lookup per letter); then for every symbol == 3 the excess h - 3 as Rice(k of the table)
// (escape QESC ones + 10 raw bits); then the L raw bit planes. Everything at the head of the one bit lane; no holes, no state.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#if defined(PLN_SSE)
#include <smmintrin.h>
#endif
#include "rlgr_lane.hpp"
#include "qh_tables.h"
namespace qh {
using rlgrl::Lane; using rlgrl::LaneR;
static inline uint32_t bw(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
// ---- QH tables
static int K, ML, NQ, QESC; static uint8_t ctxmap[2 * 68]; static uint8_t lens[64][256], rk[64];
static uint16_t enc_code[64][256];                 // canonical code, MSB first
static std::vector<uint16_t> dect;                  // [K << ML]: letter | len << 8 | (number of 3s) << 12 ... see below
static uint64_t expand[256]; static uint8_t escmask[256];
static uint16_t tail_lim[64];                       // table t: a tail code of h < tail_lim[t] has at most 15 bits
static inline uint32_t Qsa(uint32_t sa) {
    if (sa < 32) return sa;
    const uint32_t e = bw(sa); const uint32_t q = 32 + (e - 6) * 4 + ((sa >> (e - 3)) & 3); return q < 67 ? q : 67;
}
static void build_tables();
#if defined(QH_TABLES_ENV)
static void load_tables(const char* fn) {        // experiments only (env QH_TABLES with -DQH_TABLES_ENV)
    FILE* f = fopen(fn, "rb"); if (!f) { perror(fn); exit(1); }
    int hd[4]; if (fread(hd, 4, 4, f) != 4) exit(1);
    K = hd[0]; ML = hd[1]; NQ = hd[2]; QESC = hd[3];
    if (fread(ctxmap, 1, 2 * NQ, f) != size_t(2 * NQ)) exit(1);
    for (int k = 0; k < K; ++k) if (fread(lens[k], 1, 256, f) != 256) exit(1);
    if (fread(rk, 1, K, f) != size_t(K)) exit(1);
    fclose(f); build_tables();
}
#endif
static void load_builtin() {                       // the tables of the format (qh_tables.h)
    K = qh_tab::K; ML = qh_tab::ML; NQ = qh_tab::NQ; QESC = qh_tab::QESC;
    memcpy(ctxmap, qh_tab::ctxmap, sizeof qh_tab::ctxmap); memcpy(lens, qh_tab::lens, sizeof qh_tab::lens); memcpy(rk, qh_tab::rk, sizeof qh_tab::rk);
    build_tables();
}
static inline void init() { static const bool done = [] {
#if defined(QH_TABLES_ENV)
    if (const char* tf = getenv("QH_TABLES")) { load_tables(tf); return true; }
#endif
    load_builtin(); return true; }(); (void)done; }
static void build_tables() {
    dect.assign(size_t(K) << ML, 0);
    for (int k = 0; k < K; ++k) {
        std::vector<int> ord(256); for (int i = 0; i < 256; ++i) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return lens[k][a] < lens[k][b]; });
        uint32_t code = 0; int prev = lens[k][ord[0]];
        for (int i = 0; i < 256; ++i) { const int s = ord[i], l = lens[k][s]; code <<= (l - prev); prev = l;
            enc_code[k][s] = uint16_t(code);
            const uint32_t lo = code << (ML - l), n = 1u << (ML - l);
            for (uint32_t x = 0; x < n; ++x) dect[(size_t(k) << ML) + lo + x] = uint16_t(s | (l << 8));
            ++code; }
        if (code != (1u << prev)) { fprintf(stderr, "table %d: Kraft sum is not 1\n", k); exit(1); }
    }
    for (int k = 0; k < K; ++k) tail_lim[k] = uint16_t(3 + ((15 - rk[k]) << rk[k]));
    for (int s = 0; s < 256; ++s) { uint64_t e = 0; uint8_t m = 0;
        for (int j = 0; j < 4; ++j) { const uint32_t c = (s >> (2 * j)) & 3; e |= uint64_t(c) << (16 * j); if (c == 3) m |= uint8_t(1 << j); }
        expand[s] = e; escmask[s] = m; }
}
// ---- QH coder
static inline void qh_enc_block(Lane& s, uint32_t pt, uint32_t sa, int L, const uint16_t* bh, const uint16_t* bm) {
    const int t = ctxmap[pt * NQ + Qsa(sa)]; const uint32_t k = rk[t];
    uint64_t acc = 0; int n = 0;
    for (int q = 0; q < 4; ++q) { const uint16_t* h = bh + 4 * q;
        const uint32_t l = std::min<uint32_t>(h[0], 3) | std::min<uint32_t>(h[1], 3) << 2 | std::min<uint32_t>(h[2], 3) << 4 | std::min<uint32_t>(h[3], 3) << 6;
        const int ln = lens[t][l]; acc = (acc << ln) | enc_code[t][l]; n += ln; }
    s.put64(n, acc);                                   // 4 letters, at most 4 * 12 bits
    for (int j = 0; j < 16; ++j) if (bh[j] >= 3) { const uint32_t x = bh[j] - 3u, q = x >> k;
        if (q >= uint32_t(QESC)) { s.put64(QESC + 10, (((1ull << QESC) - 1) << 10) | x); }
        else s.put64(int(q + 1 + k), (((2ull << q) - 2) << k) | (x & ((1u << k) - 1))); }
    for (int p = 0; p < L; ++p) s.put(16, bm[p]);
}
// The same block through the register writer (rlgrl::LaneW), from the 16 symbols z (h = z >> L): letters by SIMD;
// tails without a branch per symbol while every tail code is short (h < tail_lim[t]: <= 15 bits): AVX2 = all 16
// joined, usually one put together with the letters; else one put per quad; the planes 3 per put.
// Bit for bit the output of qh_enc_block.
template <class W> static inline void qh_enc_tails_slow(W& s, uint32_t k, const uint16_t* bh) {
    for (int j = 0; j < 16; ++j) if (bh[j] >= 3) { const uint32_t x = bh[j] - 3u, q = x >> k;
        if (q >= uint32_t(QESC)) s.put(unsigned(QESC + 10), (((1ull << QESC) - 1) << 10) | x);
        else s.put(q + 1 + k, (((2ull << q) - 2) << k) | (x & ((1u << k) - 1))); }
}
// la, ln: the 4 letters, not written yet (they go first; one put for letters + tails when it all fits 63 bits)
template <class W> static inline void qh_enc_tails4(W& s, uint32_t k, const uint16_t* bh, uint64_t la, unsigned ln) {
#if defined(__AVX2__)
    const __m128i kk = _mm_cvtsi32_si128(int(k)); const __m256i two = _mm256_set1_epi32(2); uint64_t c[4]; unsigned n[4];
    for (int g = 0; g < 16; g += 8) { const __m256i h = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(bh + g)));
        const __m256i has = _mm256_cmpgt_epi32(h, two), x = _mm256_and_si256(_mm256_sub_epi32(h, _mm256_set1_epi32(3)), has), q = _mm256_srl_epi32(x, kk);
        const __m256i nn = _mm256_and_si256(_mm256_add_epi32(q, _mm256_set1_epi32(int(1 + k))), has);
        const __m256i v = _mm256_or_si256(_mm256_sll_epi32(_mm256_sub_epi32(_mm256_sllv_epi32(two, q), two), kk), _mm256_and_si256(x, _mm256_set1_epi32(int((1u << k) - 1))));
        rlgrl::vl_join8(v, nn, c[g / 4], n[g / 4], c[g / 4 + 1], n[g / 4 + 1]); }
    const unsigned nt = n[0] + n[1] + n[2] + n[3];
    if (__builtin_expect(nt <= 63, 1)) { const uint64_t ct = (((c[0] << n[1] | c[1]) << n[2] | c[2]) << n[3]) | c[3];
        if (__builtin_expect(ln + nt <= 63, 1)) s.put(ln + nt, la << nt | ct); else { s.put(ln, la); s.put(nt, ct); } }
    else { s.put(ln, la); s.put(n[0], c[0]); s.put(n[1], c[1]); s.put(n[2], c[2]); s.put(n[3], c[3]); }
    return;
#endif
    s.put(ln, la); const uint32_t km = (1u << k) - 1;
    for (int g = 0; g < 16; g += 4) { uint64_t acc = 0; unsigned n = 0;
        for (int j = g; j < g + 4; ++j) { const uint32_t h = bh[j], has = 0u - uint32_t(h >= 3), x = (h - 3) & has, q = x >> k, nn = (q + 1 + k) & has;
            acc = (acc << nn) | ((((2u << q) - 2) << k) | (x & km)); n += nn; }
        s.put(n, acc); }
}
#if defined(PLN_SSE)
template <class W> static inline void qh_enc_block_w(W& s, uint32_t pt, uint32_t sa, int L, const uint16_t* z) {
    const int t = ctxmap[pt * NQ + Qsa(sa)];
    const __m128i a0 = _mm_loadu_si128((const __m128i*)z), a1 = _mm_loadu_si128((const __m128i*)(z + 8)), sh = _mm_cvtsi32_si128(L);
    const __m128i h0 = _mm_srl_epi16(a0, sh), h1 = _mm_srl_epi16(a1, sh), three = _mm_set1_epi16(3);
    const __m128i c8 = _mm_packus_epi16(_mm_min_epu16(h0, three), _mm_min_epu16(h1, three));          // 16 x min(h, 3)
    const __m128i l4 = _mm_madd_epi16(_mm_maddubs_epi16(c8, _mm_set1_epi16(0x0401)), _mm_set1_epi32(0x00100001));   // 4 letters
    const uint32_t l0 = uint32_t(_mm_cvtsi128_si32(l4)), l1 = uint32_t(_mm_extract_epi32(l4, 1)), l2 = uint32_t(_mm_extract_epi32(l4, 2)), l3 = uint32_t(_mm_extract_epi32(l4, 3));
    const uint8_t* ln = lens[t]; const uint16_t* cd = enc_code[t];
    uint64_t acc = cd[l0]; acc = (acc << ln[l1]) | cd[l1]; acc = (acc << ln[l2]) | cd[l2]; acc = (acc << ln[l3]) | cd[l3];
    const unsigned nl = unsigned(ln[l0]) + ln[l1] + ln[l2] + ln[l3];
    if (_mm_movemask_epi8(_mm_cmpeq_epi8(c8, _mm_set1_epi8(3)))) {
        alignas(16) uint16_t bh[16]; _mm_store_si128((__m128i*)bh, h0); _mm_store_si128((__m128i*)(bh + 8), h1);
        const __m128i lim = _mm_set1_epi16(short(tail_lim[t] - 1));
        if (__builtin_expect(_mm_movemask_epi8(_mm_or_si128(_mm_cmpgt_epi16(h0, lim), _mm_cmpgt_epi16(h1, lim))) == 0, 1)) qh_enc_tails4(s, rk[t], bh, acc, nl);
        else { s.put(nl, acc); qh_enc_tails_slow(s, rk[t], bh); }
    } else s.put(nl, acc);
    if (L) {
        const __m128i lb = _mm_packus_epi16(_mm_and_si128(a0, _mm_set1_epi16(255)), _mm_and_si128(a1, _mm_set1_epi16(255)));
#define QH_PL(k) uint64_t(uint32_t(_mm_movemask_epi8(_mm_slli_epi16(lb, 7 - (k)))))
        const unsigned n1 = 16u * unsigned(L < 3 ? L : 3);
        s.put(n1, (QH_PL(0) << 32 | QH_PL(1) << 16 | QH_PL(2)) >> (48 - n1));
        if (L > 3) { const unsigned n2 = 16u * unsigned(L < 6 ? L - 3 : 3);
            s.put(n2, (QH_PL(3) << 32 | QH_PL(4) << 16 | QH_PL(5)) >> (48 - n2));
            if (L > 6) s.put(16, QH_PL(6)); }
#undef QH_PL
    }
}
#else
template <class W> static inline void qh_enc_block_w(W& s, uint32_t pt, uint32_t sa, int L, const uint16_t* z) {
    const int t = ctxmap[pt * NQ + Qsa(sa)]; uint16_t bh[16]; uint32_t hmax = 0, any = 0;
    for (int j = 0; j < 16; ++j) { bh[j] = uint16_t(z[j] >> L); hmax = std::max<uint32_t>(hmax, bh[j]); }
    uint64_t acc = 0; unsigned n = 0;
    for (int q = 0; q < 4; ++q) { const uint16_t* h = bh + 4 * q;
        const uint32_t l = std::min<uint32_t>(h[0], 3) | std::min<uint32_t>(h[1], 3) << 2 | std::min<uint32_t>(h[2], 3) << 4 | std::min<uint32_t>(h[3], 3) << 6;
        const unsigned ln = lens[t][l]; acc = (acc << ln) | enc_code[t][l]; n += ln; }
    any = hmax >= 3;
    if (any && hmax < tail_lim[t]) qh_enc_tails4(s, rk[t], bh, acc, n); else { s.put(n, acc); if (any) qh_enc_tails_slow(s, rk[t], bh); }
    for (int k = 0; k < L; ++k) { uint32_t m = 0; for (int j = 0; j < 16; ++j) m |= ((uint32_t(z[j]) >> k) & 1u) << j; s.put(16, m); }
}
#endif
__attribute__((always_inline)) static inline void qh_dec_block(LaneR& s, uint32_t pt, uint32_t sa, int L, uint16_t* h, uint16_t* m) {
    const int t = ctxmap[pt * NQ + Qsa(sa)]; const uint16_t* T = dect.data() + (size_t(t) << ML);
    uint64_t pos = s.head; uint64_t w = s.peek_at(pos); int used = 0; uint32_t em = 0;       // local window: w = bits from pos + used
    for (int q = 0; q < 4; ++q) { const uint32_t e = T[w >> (64 - ML)]; const int ln = int(e >> 8); const uint32_t l = e & 255;
        memcpy(h + 4 * q, &expand[l], 8); em |= uint32_t(escmask[l]) << (4 * q); w <<= ln; used += ln; }
    // tails: the only test per tail is "the code ends inside the window" (rare miss: refill and read it again)
    if (em) { const uint32_t k = rk[t];
        do { const int j = __builtin_ctz(em); em &= em - 1;
            int q = __builtin_clzll(~w | 1);                                                 // leading ones
            if (__builtin_expect((q >= QESC) | (used + q + 1 + int(k) > 64), 0)) {
                pos += uint64_t(used); w = s.peek_at(pos); used = 0; q = __builtin_clzll(~w | 1);
                if (q >= QESC) { h[j] = uint16_t(3 + uint32_t((w << QESC) >> 54)); w <<= QESC + 10; used += QESC + 10; continue; } }
            const uint32_t low = uint32_t((w << q) >> (63 - k)); const int n = q + 1 + int(k); h[j] = uint16_t(3 + ((uint32_t(q) << k) | low)); w <<= n; used += n;   // (w << q): the 0 that ends the ones, then the k low bits
        } while (em); }
    // planes: 4 per window at fixed places (m[p], p >= L: not used by the caller)
    if (L) { if ((L >= 2) | (used > 48)) { pos += uint64_t(used); w = s.peek_at(pos); used = 0; }      // L = 1 mostly fits the window
        m[0] = uint16_t(w >> 48); m[1] = uint16_t(w >> 32); m[2] = uint16_t(w >> 16); m[3] = uint16_t(w);
        if (L > 4) { pos += 64; w = s.peek_at(pos); m[4] = uint16_t(w >> 48); m[5] = uint16_t(w >> 32); m[6] = uint16_t(w >> 16); m[7] = uint16_t(w); used = 16 * (L - 4); }
        else used += 16 * L; }
    s.head = pos + uint64_t(used);
}
}  // namespace qh
