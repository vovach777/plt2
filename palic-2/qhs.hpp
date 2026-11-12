// qhs.hpp - the screen mode of PLR (files PLS1 / PLS2, -s): QH letters with their own static tables (qhs_tables.h) over
// the Sparse-Raw split (Argon's zero-pinned shift), Ls and the table context from est = max(sa, sl) (Fable's MAX).
// A block of 16 residuals z: h = 0 if z == 0 else ((z - 1) >> Ls) + 1; 4 letters of min(h, 3) (Huffman, table
// ctxmap[plane type][Q(est)]); for every h >= 3 the tail x = h - 3: q = x >> k < 8: Rice(k) (q ones, a 0, k low bits);
// else 8 ones + Exp-Golomb order 2 of x - (8 << k); then, for every z > 0 in pixel order, the Ls low bits of z - 1.
// The decoder knows from the letters and tails which z are 0, so zeros cost no raw bits (the point of the mode:
// screen content = zeros and rare large residuals). Research note: ../hlc/research/ctx4d/PLN1.md (hybrid screen mode).
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
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
static inline int Ls_of(uint32_t est) { return std::min(7, std::max(0, int(bw(est >> 4)) - 2)); }
static uint16_t enc_code[K][256]; static std::vector<uint16_t> dect; static uint64_t expand[256]; static uint8_t escmask[256];
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
    }
    for (int s = 0; s < 256; ++s) { uint64_t e = 0; uint8_t m = 0;
        for (int j = 0; j < 4; ++j) { const uint32_t c = (s >> (2 * j)) & 3; e |= uint64_t(c) << (16 * j); if (c == 3) m |= uint8_t(1 << j); }
        expand[s] = e; escmask[s] = m; }
    return true; }(); (void)done; }
// encoder, through the register writer (rlgrl::LaneW); one block < 800 bits
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
// decoder: z = the 16 residuals of the block
static inline void dec_block(rlgrl::LaneR& r, uint32_t pt, uint32_t est, int Ls, uint16_t* z) {
    const int t = qhs_tab::ctxmap[pt * NQ + Qsa(est)]; const uint16_t* T = dect.data() + (size_t(t) << ML); const uint32_t k = qhs_tab::rk[t];
    uint64_t pos = r.head; uint64_t w = r.peek_at(pos); unsigned used = 0; uint32_t em = 0; uint16_t h[16];
    for (int q = 0; q < 4; ++q) { const uint32_t e = T[w >> (64 - ML)]; const unsigned l = e >> 8; const uint32_t c = e & 255;
        memcpy(h + 4 * q, &expand[c], 8); em |= uint32_t(escmask[c]) << (4 * q); w <<= l; used += l; }
    pos += used;
    while (em) { const int j = __builtin_ctz(em); em &= em - 1; w = r.peek_at(pos);
        const int q = w == ~0ull ? 64 : __builtin_clzll(~w);
        if (q < ESC) { const uint32_t low = k ? uint32_t((w << (q + 1)) >> (64 - k)) : 0; h[j] = uint16_t(3 + ((uint32_t(q) << k) | low)); pos += uint64_t(q) + 1 + k; }
        else { const uint64_t w2 = w << ESC; const int zc = __builtin_clzll(w2 | 1);
            if (zc > 15) throw std::runtime_error("qhs escape code");
            const uint32_t m = uint32_t(w2 >> (63 - 2 * zc)), lo = uint32_t((w2 << (2 * zc + 1)) >> 62);
            h[j] = uint16_t(3 + (uint32_t(ESC) << k) + (((m - 1) << 2) | lo)); pos += uint64_t(ESC) + 2 * zc + 3; } }
    if (!Ls) { memcpy(z, h, 32); r.head = pos; return; }
    w = r.peek_at(pos); used = 0;
    for (int j = 0; j < 16; ++j) {
        if (!h[j]) { z[j] = 0; continue; }
        if (used + Ls > 64) { pos += used; w = r.peek_at(pos); used = 0; }
        const uint32_t lo = uint32_t(w >> (64 - Ls)); w <<= Ls; used += Ls;
        z[j] = uint16_t(((h[j] - 1u) << Ls) + lo + 1); }
    r.head = pos + used;
}
}  // namespace qhs
