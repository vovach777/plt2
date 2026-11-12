// qhs.hpp - the screen mode of PLR (files PLS1 / PLS2, -s): QH letters with their own static tables (qhs_tables.h) over
// the Sparse-Raw split (Argon's zero-pinned shift), Ls and the table context from est = max(sa, sl) (Fable's MAX).
// A block of 16 residuals z: h = 0 if z == 0 else ((z - 1) >> Ls) + 1; 4 letters of min(h, 3) (Huffman, table
// ctxmap[plane type][Q(est)]); for every h >= 3 the tail x = h - 3: q = x >> k < 8: Rice(k) (q ones, a 0, k low bits);
// else 8 ones + Exp-Golomb order 2 of x - (8 << k); then, for every z > 0 in pixel order, the Ls low bits of z - 1.
// The decoder knows from the letters and tails which z are 0, so zeros cost no raw bits (the point of the mode:
// screen content = zeros and rare large residuals). Research note: ../hlc/research/ctx4d/PLN1.md (hybrid screen mode).
//
// Added in fast screen mode (Argon):
// 1. Cheap per-plane classifier `screen_plane_mode` (0 = PLR, 1 = borderline check against PLR, 2 = pure PLS).
// 2. Top-down deduplicated AVX2 8-variant evaluator `accum_block_bits8` with zero-shift & constant-3s shortcuts.
// 3. Vectorized `enc_block` (AVX2 letter packing + `vl_join8` for Rice tails and Sparse-Raw bits) and `nzmask`-driven `dec_block`.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "rlgr_lane.hpp"
#include "qhs_tables.h"
namespace qhs {
using qhs_tab::K; using qhs_tab::ML; using qhs_tab::NQ;
static constexpr int ESC = 8;                       // Rice quotients >= ESC: escape (ESC ones + EG2)
static inline uint32_t bw(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
static inline uint32_t Qsa(uint32_t sa) {           // = qh::Qsa
    if (sa < 32) return sa;
    const uint32_t e = bw(sa); const uint32_t q = 32 + (e - 6) * 4 + ((sa >> (e - 3)) & 3); return q < 67 ? q : 67;
}
static constexpr int NVAR = 8, VBITS = 3;
static inline int Ls_of(uint32_t est, int v, bool first) {
    if (v >= 4) return v - 1;
    return first ? 2 + v : std::min(7, std::max(0, int(bw(est >> 4)) - 2 + v));
}
static uint16_t enc_code[K][256]; static std::vector<uint16_t> dect;
static uint64_t expand[256]; static uint8_t escmask[256]; static uint8_t nzmask[256];
static uint8_t tail_len[7][1024];                   // precomputed exact Rice+EG2 tail bit lengths for k=0..6, h=0..1023
static uint16_t tail_lim[K];                        // 3 + (ESC << rk[t]): threshold where all tails fit short Rice (q < ESC)

static void init() { static const bool done = [] {
    dect.assign(size_t(K) << ML, 0);
    for (int k = 0; k < K; ++k) {
        const uint8_t* ln = qhs_tab::lens[k]; std::vector<int> ord(256); for (int i = 0; i < 256; ++i) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return ln[a] < ln[b]; });
        uint32_t code = 0; int prev = ln[ord[0]];
        for (int i = 0; i < 256; ++i) { const int s = ord[i], l = ln[s]; code <<= (l - prev); prev = l; enc_code[k][s] = uint16_t(code);
            const uint32_t lo = code << (ML - l), n = 1u << (ML - l);
            for (uint32_t x = 0; x < n; ++x) dect[(size_t(k) << ML) + lo + x] = uint16_t(s | (l << 8));
            ++code; }
        if (code != (1u << prev)) { fprintf(stderr, "qhs table %d: Kraft sum is not 1\n", k); exit(1); }
        tail_lim[k] = uint16_t(3 + (uint32_t(ESC) << qhs_tab::rk[k]));
    }
    for (int s = 0; s < 256; ++s) { uint64_t e = 0; uint8_t m = 0, nz = 0;
        for (int j = 0; j < 4; ++j) { const uint32_t c = (s >> (2 * j)) & 3; e |= uint64_t(c) << (16 * j);
            if (c == 3) m |= uint8_t(1 << j);
            if (c) nz |= uint8_t(1 << j); }
        expand[s] = e; escmask[s] = m; nzmask[s] = nz; }
    for (uint32_t k = 0; k <= 6; ++k) {
        for (uint32_t h = 0; h < 1024; ++h) {
            if (h < 3) { tail_len[k][h] = 0; continue; }
            const uint32_t x = h - 3, q = x >> k;
            tail_len[k][h] = uint8_t(q < uint32_t(ESC) ? q + 1 + k : uint32_t(ESC) + 2 * bw(((x - (uint32_t(ESC) << k)) >> 2) + 1) + 1);
        }
    }
    return true; }(); (void)done; }

// Cheap plane-level detector: distinguishes screen/text/render planes (PLS) from natural photos & noisy scans (PLR).
// Samples every 8th block (1/8 of blocks). Counts exact zeros in non-zero blocks (z0_nz), bonus from all-zero blocks
// (4 zero-pixel equivalents per empty block), and micro-noise (1 <= z <= 7).
// Both AVX2 and scalar branches use identical 2x units (+2 per pixel, +8 per all-zero block) for bit-exact determinism.
// Returns: 0 = use PLR immediately; 1 = borderline (compare sampled PLS cost vs PLR cost); 2 = unambiguous PLS.
static inline int screen_plane_mode(const uint16_t* z, size_t nb) {
    uint32_t z0_nz = 0, z0_blk = 0, n17_score = 0; bool any_nz = false;
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    for (size_t i = 0; i < nb; i += 8) {
        const __m256i zv = _mm256_loadu_si256((const __m256i*)(z + i * 16));
        if (_mm256_testz_si256(zv, zv)) { z0_blk += 8; continue; }
        any_nz = true;
        const uint32_t mz = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi16(zv, zero)));
        const uint32_t ml = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_srli_epi16(zv, 3), zero)));
        z0_nz     += uint32_t(__builtin_popcount(mz));
        n17_score += uint32_t(__builtin_popcount(ml ^ mz));
    }
#else
    for (size_t i = 0; i < nb; i += 8) {
        const uint16_t* b = z + i * 16;
        uint32_t o = 0; for (int j = 0; j < 16; ++j) o |= b[j];
        if (!o) { z0_blk += 8; continue; }
        any_nz = true;
        for (int j = 0; j < 16; ++j) {
            if (b[j] == 0) z0_nz += 2;
            else if (b[j] <= 7) n17_score += 2;
        }
    }
#endif
    if (!any_nz || (z0_nz + z0_blk <= n17_score)) return 0;
    return (z0_nz >= 3 * n17_score) ? 2 : 1;
}
static inline bool is_screen_plane(const uint16_t* z, size_t nb) { return screen_plane_mode(z, nb) != 0; }

template <class W> static inline void enc_tails_slow(W& s, uint32_t k, const uint16_t* h, uint32_t em) {
    while (em) {
        const int j = __builtin_ctz(em); em &= em - 1;
        const uint32_t x = h[j] - 3u, q = x >> k;
        if (q < uint32_t(ESC)) s.put(q + 1 + k, (((2ull << q) - 2) << k) | (x & ((1u << k) - 1)));
        else { const uint32_t v = x - (uint32_t(ESC) << k), m = (v >> 2) + 1, b = bw(m);
            s.put(unsigned(ESC) + 2 * b - 1 + 2, ((((1ull << ESC) - 1) << (2 * b - 1)) | m) << 2 | (v & 3)); }
    }
}

#if defined(__AVX2__)
template <class W> static inline void enc_tails_fast(W& s, uint32_t k, const uint16_t* bh, uint64_t la, unsigned ln) {
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
}

template <class W> static inline void enc_block(W& s, uint32_t pt, uint32_t est, int Ls, const uint16_t* z) {
    const int t = qhs_tab::ctxmap[pt * NQ + Qsa(est)]; const uint32_t k = qhs_tab::rk[t];
    const uint8_t* ln = qhs_tab::lens[t]; const uint16_t* cd = enc_code[t];
    const __m256i zv = _mm256_loadu_si256((const __m256i*)z);
    if (_mm256_testz_si256(zv, zv)) {
        const unsigned l0 = ln[0]; const uint64_t c0 = cd[0];
        uint64_t a = (c0 << l0) | c0; a = (a << (2 * l0)) | a;
        s.put(4 * l0, a); return;
    }
    const __m256i zero = _mm256_setzero_si256();
    const __m256i z_zero = _mm256_cmpeq_epi16(zv, zero);
    const __m256i one_nz = _mm256_andnot_si256(z_zero, _mm256_set1_epi16(1));
    const __m256i zm1 = _mm256_sub_epi16(zv, one_nz);
    const __m256i hv = _mm256_add_epi16(_mm256_srl_epi16(zm1, _mm_cvtsi32_si128(Ls)), one_nz);
    const __m128i h0 = _mm256_castsi256_si128(hv), h1 = _mm256_extracti128_si256(hv, 1), three = _mm_set1_epi16(3);
    const __m128i c8 = _mm_packus_epi16(_mm_min_epu16(h0, three), _mm_min_epu16(h1, three));
    const __m128i l4 = _mm_madd_epi16(_mm_maddubs_epi16(c8, _mm_set1_epi16(0x0401)), _mm_set1_epi32(0x00100001));
    const uint32_t l0 = uint32_t(_mm_cvtsi128_si32(l4)), l1 = uint32_t(_mm_extract_epi32(l4, 1)),
                   l2 = uint32_t(_mm_extract_epi32(l4, 2)), l3 = uint32_t(_mm_extract_epi32(l4, 3));
    uint64_t la = cd[l0]; la = (la << ln[l1]) | cd[l1]; la = (la << ln[l2]) | cd[l2]; la = (la << ln[l3]) | cd[l3];
    const unsigned nl = unsigned(ln[l0]) + ln[l1] + ln[l2] + ln[l3];
    const uint32_t em = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(c8, _mm_set1_epi8(3))));
    if (em) {
        alignas(32) uint16_t bh[16]; _mm256_store_si256((__m256i*)bh, hv);
        const __m256i lim = _mm256_set1_epi16(short(tail_lim[t] - 1));
        if (__builtin_expect(_mm256_testz_si256(_mm256_cmpgt_epi16(hv, lim), _mm256_set1_epi32(-1)), 1))
            enc_tails_fast(s, k, bh, la, nl);
        else { s.put(nl, la); enc_tails_slow(s, k, bh, em); }
    } else {
        s.put(nl, la);
    }
    if (Ls) {
        const __m256i raw16 = _mm256_and_si256(zm1, _mm256_set1_epi16(short((1u << Ls) - 1)));
        const __m256i len16 = _mm256_andnot_si256(z_zero, _mm256_set1_epi16(short(Ls)));
        uint64_t c0, c1, c2, c3; unsigned n0, n1, n2, n3;
        rlgrl::vl_join8(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw16)), _mm256_cvtepu16_epi32(_mm256_castsi256_si128(len16)), c0, n0, c1, n1);
        rlgrl::vl_join8(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw16, 1)), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(len16, 1)), c2, n2, c3, n3);
        const unsigned na = n0 + n1, nb = n2 + n3;
        const uint64_t ca = (c0 << n1) | c1, cb = (c2 << n3) | c3;
        if (__builtin_expect(na + nb <= 63, 1)) {
            if (na + nb) s.put(na + nb, (ca << nb) | cb);
        } else {
            s.put(na, ca); s.put(nb, cb);
        }
    }
}
#else
template <class W> static inline void enc_block(W& s, uint32_t pt, uint32_t est, int Ls, const uint16_t* z) {
    const int t = qhs_tab::ctxmap[pt * NQ + Qsa(est)]; const uint32_t k = qhs_tab::rk[t]; const uint8_t* ln = qhs_tab::lens[t];
    uint16_t h[16]; uint32_t hmax = 0;
    for (int j = 0; j < 16; ++j) { h[j] = z[j] ? uint16_t(((z[j] - 1u) >> Ls) + 1) : 0; hmax = std::max<uint32_t>(hmax, h[j]); }
    uint64_t acc = 0; unsigned n = 0;
    for (int q = 0; q < 4; ++q) { const uint16_t* g = h + 4 * q;
        const uint32_t l = std::min<uint32_t>(g[0], 3) | std::min<uint32_t>(g[1], 3) << 2 | std::min<uint32_t>(g[2], 3) << 4 | std::min<uint32_t>(g[3], 3) << 6;
        acc = (acc << ln[l]) | enc_code[t][l]; n += ln[l]; }
    s.put(n, acc);
    if (hmax >= 3) for (int j = 0; j < 16; ++j) if (h[j] >= 3) { const uint32_t x = h[j] - 3u, q = x >> k;
        if (q < uint32_t(ESC)) s.put(q + 1 + k, (((2ull << q) - 2) << k) | (x & ((1u << k) - 1)));
        else { const uint32_t v = x - (uint32_t(ESC) << k), m = (v >> 2) + 1, b = bw(m);
            s.put(unsigned(ESC) + 2 * b - 1 + 2, ((((1ull << ESC) - 1) << (2 * b - 1)) | m) << 2 | (v & 3)); } }
    if (Ls) { acc = 0; n = 0; const uint32_t mk = (1u << Ls) - 1;
        for (int j = 0; j < 16; ++j) if (z[j]) { if (n + Ls > 63) { s.put(n, acc); acc = 0; n = 0; }
            acc = (acc << Ls) | ((z[j] - 1u) & mk); n += Ls; }
        if (n) s.put(n, acc); }
}
#endif

// Evaluate all 8 variants for one non-zero block in a single deduplicated AVX2 pass (with Constant-1s and Constant-3s shortcuts)
static inline void accum_block_bits8(uint32_t pt, uint32_t est, bool first, const uint16_t* z, uint64_t* cost) {
    const int t = qhs_tab::ctxmap[pt * NQ + Qsa(est)]; const uint32_t k = qhs_tab::rk[t];
    const uint8_t* ln = qhs_tab::lens[t]; const uint8_t* tl = tail_len[k];
    const int l_v0 = Ls_of(est, 0, first), l_v1 = Ls_of(est, 1, first), l_v2 = Ls_of(est, 2, first), l_v3 = Ls_of(est, 3, first);
#if defined(__AVX2__)
    const __m256i zv = _mm256_loadu_si256((const __m256i*)z);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i z_zero = _mm256_cmpeq_epi16(zv, zero);
    const __m256i one_nz = _mm256_andnot_si256(z_zero, _mm256_set1_epi16(1));
    const __m256i zm1 = _mm256_sub_epi16(zv, one_nz);
    const __m128i three = _mm_set1_epi16(3), three8 = _mm_set1_epi8(3), m0401 = _mm_set1_epi16(0x0401), m0010 = _mm_set1_epi32(0x00100001);
    const __m128i c8_ones = _mm_packus_epi16(_mm256_castsi256_si128(one_nz), _mm256_extracti128_si256(one_nz, 1));
    const uint32_t nz_mask8 = uint32_t(_mm_movemask_epi8(_mm_cmpgt_epi8(c8_ones, _mm_setzero_si128())));
    const uint32_t nz = uint32_t(__builtin_popcount(nz_mask8));

    uint32_t mask = 0x78u | (1u << l_v0) | (1u << l_v1) | (1u << l_v2) | (1u << l_v3);
    uint32_t b_ls[8];
    alignas(32) uint16_t u[16]; bool u_stored = false;

    // Visit shifts from largest Ls down to smallest Ls to exploit Constant-3s when em == nz_mask8:
    while (mask) {
        const int Ls = 31 - __builtin_clz(mask); mask ^= (1u << Ls);
        const __m256i shifted = _mm256_srl_epi16(zm1, _mm_cvtsi32_si128(Ls));
        if (_mm256_testz_si256(shifted, shifted)) {
            const __m128i l4 = _mm_madd_epi16(_mm_maddubs_epi16(c8_ones, m0401), m0010);
            const uint32_t huff_ones = uint32_t(ln[uint32_t(_mm_cvtsi128_si32(l4))]) + ln[uint32_t(_mm_extract_epi32(l4, 1))] +
                                       ln[uint32_t(_mm_extract_epi32(l4, 2))] + ln[uint32_t(_mm_extract_epi32(l4, 3))];
            b_ls[Ls] = huff_ones + nz * uint32_t(Ls);
            continue;
        }
        const __m256i hv = _mm256_add_epi16(shifted, one_nz);
        const __m128i h0 = _mm256_castsi256_si128(hv), h1 = _mm256_extracti128_si256(hv, 1);
        const __m128i c8 = _mm_packus_epi16(_mm_min_epu16(h0, three), _mm_min_epu16(h1, three));
        const __m128i l4 = _mm_madd_epi16(_mm_maddubs_epi16(c8, m0401), m0010);
        const uint32_t huff_bits = uint32_t(ln[uint32_t(_mm_cvtsi128_si32(l4))]) + ln[uint32_t(_mm_extract_epi32(l4, 1))] +
                                   ln[uint32_t(_mm_extract_epi32(l4, 2))] + ln[uint32_t(_mm_extract_epi32(l4, 3))];
        uint32_t bits = huff_bits + nz * uint32_t(Ls);
        const uint32_t em = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(c8, three8)));
        if (em) {
            if (!u_stored) { _mm256_store_si256((__m256i*)u, zm1); u_stored = true; }
            uint32_t m = em;
            do { const int j = __builtin_ctz(m); m &= m - 1; bits += tl[(u[j] >> Ls) + 1]; } while (m);
            b_ls[Ls] = bits;
            // Constant-3s theorem: if all non-zero elements already have h >= 3 at Ls, they have h >= 3 at all Ls2 < Ls!
            if (em == nz_mask8) {
                while (mask) {
                    const int Ls2 = 31 - __builtin_clz(mask); mask ^= (1u << Ls2);
                    uint32_t b2 = huff_bits + nz * uint32_t(Ls2);
                    uint32_t m2 = nz_mask8;
                    do { const int j = __builtin_ctz(m2); m2 &= m2 - 1; b2 += tl[(u[j] >> Ls2) + 1]; } while (m2);
                    b_ls[Ls2] = b2;
                }
                break;
            }
        } else {
            b_ls[Ls] = bits;
        }
    }
    cost[0] += b_ls[l_v0];
    cost[1] += b_ls[l_v1];
    cost[2] += b_ls[l_v2];
    cost[3] += b_ls[l_v3];
    cost[4] += b_ls[3];
    cost[5] += b_ls[4];
    cost[6] += b_ls[5];
    cost[7] += b_ls[6];
#else
    for (int v = 0; v < NVAR; ++v) {
        const int Ls = Ls_of(est, v, first);
        uint32_t bits = 0, nz = 0, l = 0;
        for (int j = 0; j < 16; ++j) { const uint32_t h = z[j] ? ((z[j] - 1u) >> Ls) + 1 : 0; nz += z[j] != 0;
            l |= std::min<uint32_t>(h, 3) << (2 * (j & 3));
            if ((j & 3) == 3) { bits += ln[l]; l = 0; }
            if (h >= 3) bits += tl[h]; }
        cost[v] += bits + nz * uint32_t(Ls);
    }
#endif
}

static inline uint32_t block_bits(uint32_t pt, uint32_t est, int Ls, const uint16_t* z) {
    const int t = qhs_tab::ctxmap[pt * NQ + Qsa(est)]; const uint32_t k = qhs_tab::rk[t]; const uint8_t* ln = qhs_tab::lens[t];
    uint32_t bits = 0, nz = 0, l = 0;
    for (int j = 0; j < 16; ++j) { const uint32_t h = z[j] ? ((z[j] - 1u) >> Ls) + 1 : 0; nz += z[j] != 0;
        l |= std::min<uint32_t>(h, 3) << (2 * (j & 3));
        if ((j & 3) == 3) { bits += ln[l]; l = 0; }
        if (h >= 3) { const uint32_t x = h - 3, q = x >> k;
            bits += q < uint32_t(ESC) ? q + 1 + k : uint32_t(ESC) + 2 * bw(((x - (uint32_t(ESC) << k)) >> 2) + 1) + 1; } }
    return bits + nz * uint32_t(Ls);
}

static inline void dec_block(rlgrl::LaneR& r, uint32_t pt, uint32_t est, int Ls, uint16_t* z) {
    const int t = qhs_tab::ctxmap[pt * NQ + Qsa(est)]; const uint16_t* T = dect.data() + (size_t(t) << ML); const uint32_t k = qhs_tab::rk[t];
    uint64_t pos = r.head; uint64_t w = r.peek_at(pos); unsigned used = 0; uint32_t em = 0, nzm = 0; alignas(32) uint16_t h[16];
    for (int q = 0; q < 4; ++q) { const uint32_t e = T[w >> (64 - ML)]; const unsigned l = e >> 8; const uint32_t c = e & 255;
        memcpy(h + 4 * q, &expand[c], 8); em |= uint32_t(escmask[c]) << (4 * q); nzm |= uint32_t(nzmask[c]) << (4 * q); w <<= l; used += l; }
    pos += used;
    if (!nzm) { memset(z, 0, 32); r.head = pos; return; }
    while (em) { const int j = __builtin_ctz(em); em &= em - 1; w = r.peek_at(pos);
        const int q = w == ~0ull ? 64 : __builtin_clzll(~w);
        if (q < ESC) { const uint32_t low = k ? uint32_t((w << (q + 1)) >> (64 - k)) : 0; h[j] = uint16_t(3 + ((uint32_t(q) << k) | low)); pos += uint64_t(q) + 1 + k; }
        else { const uint64_t w2 = w << ESC; const int zc = __builtin_clzll(w2 | 1);
            if (zc > 15) throw std::runtime_error("qhs escape code");
            const uint32_t m = uint32_t(w2 >> (63 - 2 * zc)), lo = uint32_t((w2 << (2 * zc + 1)) >> 62);
            h[j] = uint16_t(3 + (uint32_t(ESC) << k) + (((m - 1) << 2) | lo)); pos += uint64_t(ESC) + 2 * zc + 3; } }
    if (!Ls) { memcpy(z, h, 32); r.head = pos; return; }
    memset(z, 0, 32);
    w = r.peek_at(pos); used = 0;
    do {
        const int j = __builtin_ctz(nzm); nzm &= nzm - 1;
        if (__builtin_expect(used + Ls > 64, 0)) { pos += used; w = r.peek_at(pos); used = 0; }
        const uint32_t lo = uint32_t(w >> (64 - Ls)); w <<= Ls; used += Ls;
        z[j] = uint16_t(((h[j] - 1u) << Ls) + lo + 1);
    } while (nzm);
    r.head = pos + used;
}
}  // namespace qhs
