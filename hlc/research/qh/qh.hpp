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
#include "palic-2/rlgr_lane.hpp"
namespace qh {
using rlgrl::Lane; using rlgrl::LaneR;
static inline uint32_t bw(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
// ---- QH tables
static int K, ML, NQ, QESC; static uint8_t ctxmap[2 * 68]; static uint8_t lens[64][256], rk[64];
static uint16_t enc_code[64][256];                 // canonical code, MSB first
static std::vector<uint16_t> dect;                  // [K << ML]: letter | len << 8 | (number of 3s) << 12 ... see below
static uint64_t expand[256]; static uint8_t escmask[256];
static inline uint32_t Qsa(uint32_t sa) {
    if (sa < 32) return sa;
    const uint32_t e = bw(sa); const uint32_t q = 32 + (e - 6) * 4 + ((sa >> (e - 3)) & 3); return q < 67 ? q : 67;
}
static void load_tables(const char* fn) {
    FILE* f = fopen(fn, "rb"); if (!f) { perror(fn); exit(1); }
    int hd[4]; if (fread(hd, 4, 4, f) != 4) exit(1); K = hd[0]; ML = hd[1]; NQ = hd[2]; QESC = hd[3];
    if (fread(ctxmap, 1, 2 * NQ, f) != size_t(2 * NQ)) exit(1);
    for (int k = 0; k < K; ++k) if (fread(lens[k], 1, 256, f) != 256) exit(1);
    if (fread(rk, 1, K, f) != size_t(K)) exit(1); fclose(f);
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
__attribute__((always_inline)) static inline void qh_dec_block(LaneR& s, uint32_t pt, uint32_t sa, int L, uint16_t* h, uint16_t* m) {
    const int t = ctxmap[pt * NQ + Qsa(sa)]; const uint16_t* T = dect.data() + (size_t(t) << ML);
    uint64_t pos = s.head; uint64_t w = s.peek_at(pos); int used = 0; uint32_t em = 0;       // local window: w = bits from pos + used
    for (int q = 0; q < 4; ++q) { const uint32_t e = T[w >> (64 - ML)]; const int ln = int(e >> 8); const uint32_t l = e & 255;
        memcpy(h + 4 * q, &expand[l], 8); em |= uint32_t(escmask[l]) << (4 * q); w <<= ln; used += ln; }
    if (em) { const uint32_t k = rk[t];
        do { const int j = __builtin_ctz(em); em &= em - 1;
            if (__builtin_expect(used > 64 - (QESC + 10), 0)) { pos += uint64_t(used); w = s.peek_at(pos); used = 0; }
            const int q = __builtin_clzll(~w | 1);                                           // leading ones
            if (__builtin_expect(q >= QESC, 0)) { h[j] = uint16_t(3 + uint32_t((w << QESC) >> 54)); w <<= QESC + 10; used += QESC + 10; }
            else { const uint32_t low = uint32_t((w << (q + 1)) >> 1 >> (63 - k)); const int n = q + 1 + int(k); h[j] = uint16_t(3 + ((uint32_t(q) << k) | low)); w <<= n; used += n; }
        } while (em); }
    for (int p = 0; p < L; ++p) { if (used > 48) { pos += uint64_t(used); w = s.peek_at(pos); used = 0; } m[p] = uint16_t(w >> 48); w <<= 16; used += 16; }
    s.head = pos + uint64_t(used);
}
}  // namespace qh
