// planes_codec.cpp - experimental gray codec "PLN1" (8-bit P5): independent tiles (default 288 x 288, as HALIC),
// inside a tile PALIC-1 MED -> zigzag z (tile borders = image borders), then per 16-symbol block (raster order of
// the tile) the "noise" L_b = clamp(bw(mean z of the block) - off, 0, 7): the L_b low bits of the 16 symbols go raw
// as L_b bit planes (SSE movemask, 2 bytes each), the high part z >> L_b goes to one FSE4 stream per L_b value
// (8 streams = 8 contexts "noise of the block").
//   -m 0: plain FSE4 over z.
//   -m 1: L_b from the current block, sent as an FSE4 stream (alphabet 8).
//   Format and the "blocks across rows" rule with a worked 500 x 500 example: PLN1.md.
//   -m 2 (default, by Argon): NO L stream — causal L_b from the 16 symbols directly above (z - tw; row 0 or a tile narrower
//         than 16: the previous block; block 0: L = 2). Here off is part of the format (the decoder repeats the rule).
//   -m 3: as 2, but every row of the tile is padded with z = 0 to a multiple of 16: blocks never straddle rows, the
//         block above is always the same 16 columns, its L is kept from when it was coded. The padding is not coded
//         (a row's last block puts only its real symbols into the high stream; its planes stay 16 bits).
// File: "PLN1" | var W, H, tile, mode, off | u32 tile sizes | tiles (raster order of tiles).
// Tile (mode 1): L stream, 8 high streams, planes.  Tile (mode 2): 8 high streams, planes.  Tile (mode 0): one stream.
// usage: planes_codec c in.pgm out.pln [-m 0|1|2] [-ts N] [-off K]  |  d in.pln out.pgm
//        planes_codec bench a.pgm|dir ... [-r N] [-m 0|1|2] [-ts N] [-off K]
// build: clang++ -O3 -march=native -std=c++17 -I. -I../../../third_party/fse -I../../.. planes_codec.cpp <fse objs>
#define CTX4D_CODEC_NO_MAIN
#include "ctx4d_codec.cpp"
#include <chrono>
#if defined(PLN_SSE)
#include <smmintrin.h>
#endif

namespace pln {
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static inline uint32_t bwz(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
static inline int med(int a, int b, int c) { int mn = std::min(a, b), mx = std::max(a, b), p = a + b - c; return p < mn ? mn : (p > mx ? mx : p); }
static inline int pred(const uint8_t* r, int x, int y, int S) {     // r = row y of the tile, S = image stride
    if (y == 0) return x ? r[x - 1] : 0;
    if (x == 0) return r[-S];
    return med(r[x - 1], r[x - S], r[x - S - 1]);
}
static double g_t[6];   // profile: enc residual, blocks, FSE | dec FSE, unpack, MED
struct Cfg { int ts = 288, mode = 2, off = 2; };

// ---- residuals: plain loops, clang auto-vectorizes them (borders peeled off, no branch in the loop body)
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
// ---- 16-symbol blocks: clang vector types (ext_vector_type); planes = bit masks via bool vectors (pmovmskb)
#if defined(__clang__)
typedef uint16_t v16u __attribute__((ext_vector_type(16)));
typedef bool v16b __attribute__((ext_vector_type(16)));
static inline v16u ld16(const uint16_t* p) { v16u v; memcpy(&v, p, 32); return v; }
static inline void st16(uint16_t* p, v16u v) { memcpy(p, &v, 32); }
static inline uint16_t mask16(v16u v) { return __builtin_bit_cast(uint16_t, __builtin_convertvector(v != 0, v16b)); }
#endif
static inline int block_L(const uint16_t* s, int off) {
#if defined(PLN_SSE)
    const __m128i a0 = _mm_loadu_si128((const __m128i*)s), a1 = _mm_loadu_si128((const __m128i*)(s + 8));
    __m128i t = _mm_add_epi16(a0, a1); t = _mm_add_epi16(t, _mm_srli_si128(t, 8)); t = _mm_add_epi16(t, _mm_srli_si128(t, 4)); t = _mm_add_epi16(t, _mm_srli_si128(t, 2));
    return std::min(7, std::max(0, int(bwz(uint32_t(_mm_extract_epi16(t, 0)) >> 4)) - off));
#elif defined(__clang__)
    return std::min(7, std::max(0, int(bwz(uint32_t(__builtin_reduce_add(ld16(s))) >> 4)) - off));
#else
    uint32_t sum = 0; for (int j = 0; j < 16; ++j) sum += s[j];
    return std::min(7, std::max(0, int(bwz(sum >> 4)) - off));
#endif
}
static inline void block_split(const uint16_t* s, int L, uint16_t*& pw, uint16_t* hi) {
#if defined(PLN_SSE)
    const __m128i a0 = _mm_loadu_si128((const __m128i*)s), a1 = _mm_loadu_si128((const __m128i*)(s + 8));
    for (int k = 0; k < L; ++k) { const __m128i q = _mm_cvtsi32_si128(15 - k);
        *pw++ = uint16_t(_mm_movemask_epi8(_mm_packs_epi16(_mm_sll_epi16(a0, q), _mm_sll_epi16(a1, q)))); }
    const __m128i q = _mm_cvtsi32_si128(L);
    _mm_storeu_si128((__m128i*)hi, _mm_srl_epi16(a0, q)); _mm_storeu_si128((__m128i*)(hi + 8), _mm_srl_epi16(a1, q));
#elif defined(__clang__)
    const v16u a = ld16(s);
    for (int k = 0; k < L; ++k) *pw++ = mask16(a & uint16_t(1u << k));
    st16(hi, a >> uint16_t(L));
#else
    for (int k = 0; k < L; ++k) { uint16_t m = 0; for (int j = 0; j < 16; ++j) m |= uint16_t(((s[j] >> k) & 1u) << j); *pw++ = m; }
    for (int j = 0; j < 16; ++j) hi[j] = uint16_t(s[j] >> L);
#endif
}
static inline int block_planes(const uint16_t* s, int off, uint16_t*& pw, uint16_t* hi) {   // -> L
    const int L = block_L(s, off); block_split(s, L, pw, hi); return L;
}
static inline int causal_L(const uint16_t* z, size_t i, int tw, int off) {
    const size_t px = i * 16;                                   // the 16 symbols above must end before this block:
    if (tw >= 16 && px >= size_t(tw)) return block_L(z + px - tw, off);   // tw < 16 (narrow edge tiles) -> previous block
    return i ? block_L(z + px - 16, off) : 2;
}
static inline void block_unpack(const uint16_t* h, int L, const uint8_t*& pp, uint16_t* z) {
#if defined(__clang__)
    const v16u w = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    v16u v = ld16(h) << uint16_t(L);
    for (int k = 0; k < L; ++k) { uint16_t m; memcpy(&m, pp, 2); pp += 2; v |= (v16u)((v16u(m) & w) != 0) & uint16_t(1u << k); }
    st16(z, v);
#else
    for (int j = 0; j < 16; ++j) z[j] = uint16_t(h[j] << L);
    for (int k = 0; k < L; ++k) { uint16_t m; memcpy(&m, pp, 2); pp += 2; for (int j = 0; j < 16; ++j) z[j] |= uint16_t(((m >> j) & 1u) << k); }
#endif
}

static void tile_enc(const uint8_t* img, int S, int tw, int th, const Cfg& c, Out& o, std::vector<uint8_t>& tmp) {
    const int zs = c.mode == 3 ? (tw + 15) & ~15 : tw;                           // mode 3: rows padded with z = 0 to 16
    const size_t n = size_t(zs) * th, nb = (n + 15) / 16;
    static thread_local UBuf<uint16_t> zb, hb[8], lbb, plb;                       // grow-only, never zero-filled
    zb.need(nb * 16 + 16); uint16_t* z = zb.data(); std::fill(z + n, z + nb * 16 + 16, uint16_t(0));
    double t0 = now_s(); resid_row0(img, z, tw);
    for (int y = 1; y < th; ++y) resid_row(img + size_t(y) * S, img + size_t(y - 1) * S, z + size_t(y) * zs, tw);
    if (zs != tw) for (int y = 0; y < th; ++y) {
#if defined(PLN_PADCOPY)                                                          // padding = cyclic copy of the row's tail block
        uint16_t* zr = z + size_t(y) * zs; const int base = zs - 16, r = tw - base;
        for (int x = tw; x < zs; ++x) zr[x] = zr[base + (x - base) % r];
#else
        std::fill(z + size_t(y) * zs + tw, z + size_t(y + 1) * zs, uint16_t(0));
#endif
    }
    double t1 = now_s(); g_t[0] += t1 - t0;
    if (c.mode == 0) { put_stream(o, z, n, 256, tmp); g_t[2] += now_s() - t1; return; }
    plb.need(nb * 8); uint16_t* pw = plb.data(); size_t hn[8] = {0};
    for (auto& h : hb) h.need(nb * 16);
    if (c.mode == 1) {
        lbb.need(nb); uint16_t* Lb = lbb.data();
        for (size_t i = 0; i < nb; ++i) {
            alignas(32) uint16_t hk[16]; const int L = block_planes(z + i * 16, c.off, pw, hk); Lb[i] = uint16_t(L);
            memcpy(hb[L].data() + hn[L], hk, 32); hn[L] += 16;
        }
        double t2 = now_s(); g_t[1] += t2 - t1;
        put_stream(o, Lb, nb, 8, tmp);
        for (int L = 0; L < 8; ++L) put_stream(o, hb[L].data(), hn[L], 256u >> L, tmp);
        o.raw(plb.data(), size_t(pw - plb.data()) * 2); g_t[2] += now_s() - t2;
    } else {
        const size_t bpr = size_t(zs) / 16; lbb.need(nb); uint16_t* Lo = lbb.data();  // mode 3: own L of each block
        static thread_local std::vector<uint8_t> rawt; rawt.clear();
        for (size_t i = 0; i < nb; ++i) {
            alignas(32) uint16_t hk[16]; int L = 0;
            if (c.mode == 2) L = causal_L(z, i, tw, c.off);
            const int r = c.mode == 3 && i % bpr == bpr - 1 ? tw - int(bpr - 1) * 16 : 16;   // mode 3: padding is not coded
            if (c.mode == 3) { L = i >= bpr ? Lo[i - bpr] : (i ? Lo[i - 1] : 2); Lo[i] = uint16_t(block_L(z + i * 16, c.off)); }   // padding counts as 0 (better than the real mean)
#if defined(PLN_RAWTAIL)                                                          // mode 3: the row's last partial block stored raw
            if (r < 16) { for (int j = 0; j < r; ++j) rawt.push_back(uint8_t(z[i * 16 + j])); continue; }
#endif
            block_split(z + i * 16, L, pw, hk);
            memcpy(hb[L].data() + hn[L], hk, 32); hn[L] += size_t(r);
        }
        double t2 = now_s(); g_t[1] += t2 - t1;
        for (int L = 0; L < 8; ++L) put_stream(o, hb[L].data(), hn[L], 256u >> L, tmp);
        o.raw(rawt.data(), rawt.size());
        o.raw(plb.data(), size_t(pw - plb.data()) * 2); g_t[2] += now_s() - t2;
    }
}
// ---- decoder: unpack (vector), inverse zigzag per row (auto-vectorized), MED chain scalar (cmov, borders peeled)
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
#if defined(__clang__)
// ---- decoder MED as a wave over bands of 16 rows: lane k = row y0 + k, step t, lane k at x = t - k (skew 1 px per
// row: MED needs a, b, c only). a = own lane, previous step; b = lane k-1, previous step; c = lane k-1, two steps
// back; lane 0 takes b, c from the row above the band. Borders as PALIC-1: tile row 0: b = c = a (pred = left,
// 0 at x = 0); x = 0: a = c = b (pred = up).
// Skew in and out with 16 x 16 transposes: residual rows go to padded int16 rows (inverse zigzag, auto-vectorized);
// steps t0..t0+15 = transpose of the 16 rows loaded at x = t0 - k; the 16 results transposed back are 16 runs of
// row k at x = t0 - k .. t0 - k + 15 (out-of-tile x lands in the padding). Then int16 rows -> uint8 image rows.
typedef int16_t v16s __attribute__((ext_vector_type(16)));
static inline v16s ld16s(const int16_t* p) { v16s v; memcpy(&v, p, 32); return v; }
#define SV __builtin_shufflevector
static inline void tr16s(v16s* m) {                                            // in-place 16 x 16 transpose
    for (int i = 0; i < 16; i++) if (!(i & 8)) { v16s a = m[i], b = m[i + 8];
        m[i] = SV(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23); m[i + 8] = SV(a, b, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31); }
    for (int i = 0; i < 16; i++) if (!(i & 4)) { v16s a = m[i], b = m[i + 4];
        m[i] = SV(a, b, 0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27); m[i + 4] = SV(a, b, 4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31); }
    for (int i = 0; i < 16; i++) if (!(i & 2)) { v16s a = m[i], b = m[i + 2];
        m[i] = SV(a, b, 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29); m[i + 2] = SV(a, b, 2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31); }
    for (int i = 0; i < 16; i += 2) { v16s a = m[i], b = m[i + 1];
        m[i] = SV(a, b, 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30); m[i + 1] = SV(a, b, 1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31); }
}
static void unzz16(const uint16_t* __restrict z, int16_t* __restrict e, int w) { for (int x = 0; x < w; ++x) e[x] = int16_t(int(z[x] >> 1) ^ -int(z[x] & 1)); }
static void row8(const int16_t* __restrict o, uint8_t* __restrict r, int w) { for (int x = 0; x < w; ++x) r[x] = uint8_t(o[x]); }
static void recon_wave(uint8_t* img, int S, int tw, int th, const uint16_t* z, int zs) {
    static thread_local std::vector<int16_t> Eb, Ob, T; const int RS = tw + 64;            // padded row: [-32, tw + 32)
    Eb.assign(size_t(RS) * 16, 0); Ob.resize(size_t(RS) * 16); T.assign(size_t(tw) + 2, 0);
    const v16s K = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, lane0 = K == 0;
    for (int y0 = 0; y0 < th; y0 += 16) {
        const int R = std::min(16, th - y0), steps = tw + R - 1;
        for (int k = 0; k < R; ++k) unzz16(z + size_t(y0 + k) * zs, Eb.data() + size_t(k) * RS + 32, tw);
        for (int k = R; k < 16; ++k) std::fill(Eb.data() + size_t(k) * RS, Eb.data() + size_t(k + 1) * RS, int16_t(0));
        int16_t* Tp = T.data() + 1;                                                // T[-1] = 0, T[tw] = 0
        if (y0) for (int x = 0; x < tw; ++x) Tp[x] = img[size_t(y0 - 1) * S + x];
        const bool first = y0 == 0;
        v16s P1 = 0, P2 = 0;
        for (int t0 = 0; t0 < steps; t0 += 16) {
            v16s m[16];
            for (int k = 0; k < 16; ++k) m[k] = ld16s(Eb.data() + size_t(k) * RS + 32 + t0 - k);
            tr16s(m);                                                              // m[j] = residuals of step t0 + j
            for (int j = 0; j < 16; ++j) {
                const int t = t0 + j, tb = std::min(t, tw), tc = std::min(t - 1, tw);
                v16s a = P1;
                v16s b = SV(P1, (v16s)(int16_t)Tp[tb], 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
                v16s c = SV(P2, (v16s)(int16_t)Tp[tc], 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
                if (first) { b = lane0 ? a : b; c = lane0 ? a : c; }                // tile row 0: pred = left
                const v16s x0 = K == (int16_t)t; a = x0 ? b : a; c = x0 ? b : c;   // x = 0: pred = up
                const v16s mn = a < b ? a : b, mx = a < b ? b : a; v16s p = a + b - c; p = p < mn ? mn : p; p = p > mx ? mx : p;
                const v16s v = (p + m[j]) & 255; P2 = P1; P1 = v; m[j] = v;
            }
            tr16s(m);                                                              // m[k] = row k at x = t0 - k ..
            for (int k = 0; k < 16; ++k) memcpy(Ob.data() + size_t(k) * RS + 32 + t0 - k, &m[k], 32);
        }
        for (int k = 0; k < R; ++k) row8(Ob.data() + size_t(k) * RS + 32, img + size_t(y0 + k) * S, tw);
    }
}
#undef SV
#endif
static void tile_dec(In& in, uint8_t* img, int S, int tw, int th, const Cfg& c) {
    const int zs = c.mode == 3 ? (tw + 15) & ~15 : tw;
    const size_t n = size_t(zs) * th, nb = (n + 15) / 16;
    static thread_local UBuf<uint16_t> zb; static thread_local std::vector<int8_t> e; zb.need(nb * 16 + 16); uint16_t* z = zb.data(); if (e.size() < size_t(tw)) e.resize(size_t(tw));
    double t0 = now_s(), t1 = t0;
    if (c.mode == 0) { static thread_local UBuf<uint16_t> b; if (get_stream(in, b) != n) throw std::runtime_error("tile size"); memcpy(z, b.data(), n * 2); }
    else if (c.mode == 1) {
        static thread_local UBuf<uint16_t> lb, hb[8]; if (get_stream(in, lb) != nb) throw std::runtime_error("L size");
        size_t hn[8], np = 0; for (int L = 0; L < 8; ++L) hn[L] = get_stream(in, hb[L]);
        for (size_t i = 0; i < nb; ++i) { if (lb.data()[i] > 7) throw std::runtime_error("L value"); np += lb.data()[i]; }
        const uint8_t* pp = in.take(np * 2); size_t hp[8] = {0}; t1 = now_s(); g_t[3] += t1 - t0;
        for (size_t i = 0; i < nb; ++i) {
            const int L = lb.data()[i]; if (hp[L] + 16 > hn[L]) throw std::runtime_error("high stream");
            block_unpack(hb[L].data() + hp[L], L, pp, z + i * 16); hp[L] += 16;
        }
        for (int L = 0; L < 8; ++L) if (hp[L] != hn[L]) throw std::runtime_error("high stream size");
    } else {
        static thread_local UBuf<uint16_t> hb[8];
        size_t hn[8];
        for (int L = 0; L < 8; ++L) { hn[L] = get_stream(in, hb[L]);
            if (c.mode == 2 && (hn[L] & 15)) throw std::runtime_error("high stream align"); }
        const size_t bpr = size_t(zs) / 16, rl = size_t(tw) - (bpr - 1) * 16;             // mode 3: real symbols of a row's last block
        const uint8_t* pp = in.p; size_t hp[8] = {0}; t1 = now_s(); g_t[3] += t1 - t0;
#if defined(PLN_RAWTAIL)
        const uint8_t* rt = pp; if (c.mode == 3 && rl < 16) pp = in.take(rl * size_t(th)) + rl * size_t(th);
#endif
        static thread_local UBuf<uint16_t> lo; lo.need(nb); uint16_t* Lo = lo.data();
        for (size_t i = 0; i < nb; ++i) {
            const int L = c.mode == 2 ? causal_L(z, i, tw, c.off) : (i >= bpr ? Lo[i - bpr] : (i ? Lo[i - 1] : 2));
            const size_t r = c.mode == 3 && i % bpr == bpr - 1 ? rl : 16;
#if defined(PLN_RAWTAIL)
            if (r < 16) { for (size_t j = 0; j < 16; ++j) z[i * 16 + j] = j < r ? *rt++ : 0; Lo[i] = uint16_t(block_L(z + i * 16, c.off)); continue; }
#endif
            if (hp[L] + r > hn[L] || pp + 2 * L > in.e) throw std::runtime_error("high stream");
            uint16_t* h = hb[L].data() + hp[L];
            if (r < 16) { alignas(32) uint16_t t[16] = {0}; memcpy(t, h, r * 2);
#if defined(PLN_PADCOPY)
                for (size_t j = r; j < 16; ++j) t[j] = t[j % r];                             // same copy as the encoder (planes carry its bits)
                block_unpack(t, L, pp, z + i * 16); }
#else
                block_unpack(t, L, pp, z + i * 16); for (size_t j = r; j < 16; ++j) z[i * 16 + j] = 0; }
#endif
            else block_unpack(h, L, pp, z + i * 16);
            hp[L] += r;
            if (c.mode == 3) Lo[i] = uint16_t(block_L(z + i * 16, c.off));
        }
        in.p = pp;
        for (int L = 0; L < 8; ++L) if (hp[L] != hn[L]) throw std::runtime_error("high stream size");
    }
    double t2 = now_s(); g_t[4] += t2 - t1;
#if defined(__clang__) && !defined(PLN_ROWS)
    recon_wave(img, S, tw, th, z, zs);
#else
    for (int y = 0; y < th; ++y) {
        const uint16_t* zr = z + size_t(y) * zs;
        unzz_row(zr, e.data(), tw);                                                       // inverse zigzag, vectorized
        if (y == 0) recon_row0(img, e.data(), tw); else recon_row(img + size_t(y) * S, img + size_t(y - 1) * S, e.data(), tw);
    }
#endif
    g_t[5] += now_s() - t2;
}
struct Grid { int nx, ny, ts, W, H; int tx(int i) const { return (i % nx) * ts; } int ty(int i) const { return (i / nx) * ts; }
    int tw(int i) const { return std::min(ts, W - tx(i)); } int th(int i) const { return std::min(ts, H - ty(i)); } };
static Grid grid(int W, int H, int ts) { return Grid{(W + ts - 1) / ts, (H + ts - 1) / ts, ts, W, H}; }

std::vector<uint8_t> compress(const uint8_t* img, int W, int H, const Cfg& c) {
    Grid g = grid(W, H, c.ts); const int nt = g.nx * g.ny; static thread_local std::vector<uint8_t> tmp;
    Out o; o.b.reserve(size_t(W) * H + size_t(nt) * 4 + 4096);                 // tiles go straight into the output
    o.raw("PLN1", 4); o.var(W); o.var(H); o.var(c.ts); o.var(c.mode); o.var(c.off < 0 ? 15 : c.off);          // mode 2: used by the decoder; mode 1: informational (15 = per tile)
    const size_t lens = o.b.size(); o.b.resize(lens + size_t(nt) * 4);
    for (int i = 0; i < nt; ++i) {
        const size_t b = o.b.size(); const uint8_t* ti = img + size_t(g.ty(i)) * W + g.tx(i);
        if (c.off >= 0 || c.mode != 1) tile_enc(ti, W, g.tw(i), g.th(i), c, o, tmp);
        else {                                                                  // -off -1 (mode 1): best off 0..4 per tile (encoder only)
            static thread_local Out best, cur; best.b.clear(); Cfg cc = c;
            for (int off = 0; off <= 4; ++off) { cc.off = off; cur.b.clear(); tile_enc(ti, W, g.tw(i), g.th(i), cc, cur, tmp);
                if (best.b.empty() || cur.b.size() < best.b.size()) std::swap(best.b, cur.b); }
            o.raw(best.b.data(), best.b.size());
        }
        const uint32_t l = uint32_t(o.b.size() - b); memcpy(&o.b[lens + size_t(i) * 4], &l, 4);
    }
    return std::move(o.b);
}
void decompress(const std::vector<uint8_t>& d, std::vector<uint8_t>& img, int& W, int& H) {
    In in{d.data(), d.data() + d.size()}; if (d.size() < 4 || memcmp(in.take(4), "PLN1", 4)) throw std::runtime_error("not PLN1");
    Cfg c; W = int(in.var()); H = int(in.var()); c.ts = int(in.var()); c.mode = int(in.var()); c.off = int(in.var());
    if (W <= 0 || H <= 0 || c.ts < 1 || c.mode > 3) throw std::runtime_error("bad header");
    Grid g = grid(W, H, c.ts); const int nt = g.nx * g.ny; std::vector<uint32_t> len(nt);
    for (auto& l : len) memcpy(&l, in.take(4), 4);
    img.resize(size_t(W) * H);                                                  // every pixel is written
    for (int i = 0; i < nt; ++i) { In ti{in.take(len[i]), nullptr}; ti.e = ti.p + len[i];
        tile_dec(ti, img.data() + size_t(g.ty(i)) * W + g.tx(i), W, g.tw(i), g.th(i), c); if (ti.p != ti.e) throw std::runtime_error("tile trailing bytes"); }
}
static bool read_pgm(const char* fn, std::vector<uint8_t>& px, int& W, int& H) {
    FILE* f = fopen(fn, "rb"); if (!f) return false; int mx;
    if (fscanf(f, "P5 %d %d %d", &W, &H, &mx) != 3 || mx != 255) { fclose(f); return false; }
    fgetc(f); px.resize(size_t(W) * H); bool ok = fread(px.data(), 1, px.size(), f) == px.size(); fclose(f); return ok;
}
}  // namespace pln

int main(int argc, char** argv) {
    using namespace pln;
    if (argc < 3) { fprintf(stderr, "usage: %s c in.pgm out.pln [-m 0|1|2] [-ts N] [-off K] | d in.pln out.pgm | bench a.pgm|dir ... [-r N] [-m] [-ts] [-off]\n", argv[0]); return 1; }
    std::string cmd = argv[1]; Cfg c; int reps = 3; std::vector<std::string> pos;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) c.mode = atoi(argv[++i]); else if (a == "-ts" && i + 1 < argc) c.ts = atoi(argv[++i]);
        else if (a == "-off" && i + 1 < argc) c.off = atoi(argv[++i]); else if (a == "-r" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (cmd == "bench" && std::filesystem::is_directory(a)) { std::vector<std::string> v;
            for (auto& e : std::filesystem::directory_iterator(a)) if (e.path().extension() == ".pgm") v.push_back(e.path().string());
            std::sort(v.begin(), v.end()); pos.insert(pos.end(), v.begin(), v.end()); }
        else pos.push_back(a); }
    if (c.ts < 16 || c.mode < 0 || c.mode > 3 || c.off < -1 || (c.off < 0 && c.mode != 1) || c.off > 14) { fprintf(stderr, "bad options\n"); return 1; }
    try {
        if (cmd == "c" && pos.size() == 2) { std::vector<uint8_t> px; int W, H; if (!read_pgm(pos[0].c_str(), px, W, H)) { fprintf(stderr, "need 8-bit P5\n"); return 1; }
            auto o = compress(px.data(), W, H, c); FILE* f = fopen(pos[1].c_str(), "wb"); if (!f) return 1; fwrite(o.data(), 1, o.size(), f); fclose(f); return 0; }
        if (cmd == "d" && pos.size() == 2) { auto d = read_file(pos[0].c_str()); std::vector<uint8_t> img; int W, H; decompress(d, img, W, H);
            FILE* f = fopen(pos[1].c_str(), "wb"); if (!f) return 1; fprintf(f, "P5\n%d %d\n255\n", W, H); fwrite(img.data(), 1, img.size(), f); fclose(f); return 0; }
        if (cmd == "bench") {
            double te = 0, td = 0, tpx = 0, tb = 0; bool ok = true;
            for (auto& fn : pos) { std::vector<uint8_t> px; int W, H; if (!read_pgm(fn.c_str(), px, W, H)) { printf("%s: skip\n", fn.c_str()); continue; }
                double be = 1e9, bd = 1e9; std::vector<uint8_t> o, r; int w2, h2;
                for (int k = 0; k < reps; ++k) { double a = now_s(); o = compress(px.data(), W, H, c); double b = now_s(); decompress(o, r, w2, h2); double e = now_s(); be = std::min(be, b - a); bd = std::min(bd, e - b); }
                bool g = r == px; ok &= g; double n = double(W) * H; tpx += n; tb += o.size(); te += be; td += bd;
                printf("%-16s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s  %s\n", std::filesystem::path(fn).filename().string().c_str(), W, H, o.size(), 8 * o.size() / n, n / be / 1e6, n / bd / 1e6, g ? "ok" : "FAIL"); }
            for (double& t : g_t) t /= reps;
            printf("ns/px  enc: residual %.2f  blocks %.2f  FSE %.2f | dec: FSE %.2f  unpack %.2f  MED %.2f\n", g_t[0] * 1e9 / tpx, g_t[1] * 1e9 / tpx, g_t[2] * 1e9 / tpx, g_t[3] * 1e9 / tpx, g_t[4] * 1e9 / tpx, g_t[5] * 1e9 / tpx);
            printf("TOTAL %zu files %.4f bpp  enc %.1f dec %.1f MB/s  (mode %d tile %d off %d)  %s\n", pos.size(), 8 * tb / tpx, tpx / te / 1e6, tpx / td / 1e6, c.mode, c.ts, c.off, ok ? "all ok" : "FAIL");
            return ok ? 0 : 3;
        }
    } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 2; }
    fprintf(stderr, "bad arguments\n"); return 1;
}