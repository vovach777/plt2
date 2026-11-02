// PLT2 decoder for WebAssembly, freestanding (no libc / libc++): system clang --target=wasm32 + wasm-ld.
// Decodes the default layout (SPLIT=3: one buffer per strip, RC Y/U/V + one shared bypass stream), A31 or A15,
// model cluster64 or cluster64 x Enb. Single thread (one strip after another). Bit-exact with plt2.cpp.
// SIMD: -DPLT_WASM_SIMD -msimd128 -> CDF update and letter search on wasm v128 (i16x8); otherwise scalar.
// Exports: plt_alloc(n) -> ptr (bump arena), plt_reset(), plt_decode(in, n, out_rgb) -> 0 ok / <0 error,
//          plt_dims(in) -> (W << 16 | H) is not enough for big images, so plt_width/plt_height(in).
#include <stdint.h>
#include <stddef.h>
#ifdef PLT_WASM_SIMD
#include <wasm_simd128.h>
#endif
#define EXPORT extern "C" __attribute__((visibility("default")))
#define AI __attribute__((always_inline)) inline

namespace std { using ::uint8_t; using ::uint16_t; using ::uint32_t; using ::uint64_t; using ::int8_t; using ::int16_t; using ::int32_t; }
#include "../plt2_tables.h"
using namespace plt2_tab;

// ------------------------------------------------------------------ arena
extern "C" unsigned char __heap_base;
static uintptr_t g_top = 0;
static void* arena(size_t n) {
    if (!g_top) g_top = (uintptr_t)&__heap_base;
    uintptr_t p = (g_top + 63) & ~uintptr_t(63);
    uintptr_t end = p + n; size_t have = __builtin_wasm_memory_size(0) * 65536u;
    if (end > have) { if (__builtin_wasm_memory_grow(0, (end - have + 65535) / 65536) == size_t(-1)) return nullptr; }
    g_top = end; return (void*)p;
}
EXPORT void* plt_alloc(uint32_t n) { return arena(n); }
EXPORT void plt_reset() { g_top = 0; }
extern "C" void* memcpy(void* d, const void* s, size_t n) { auto* a = (uint8_t*)d; auto* b = (const uint8_t*)s; while (n--) *a++ = *b++; return d; }
extern "C" void* memset(void* d, int c, size_t n) { auto* a = (uint8_t*)d; while (n--) *a++ = uint8_t(c); return d; }

// ------------------------------------------------------------------ static helpers
static int32_t G1[1024], G2[1024], G3[1024];
static uint16_t BW8[512];
static bool g_init = false;
static int bw(int v) { return v > 0 ? 32 - __builtin_clz(unsigned(v)) : 0; }
alignas(64) static uint16_t T31le[32], T31gt[32], T15le[16], T15gt[16];
static void init_static() {
    for (int g = -511; g <= 512; ++g) { int lv = bw(g < 0 ? -g : g) * ((g > 0) - (g < 0)) + 16; G1[g + 511] = lv * 1089; G2[g + 511] = lv * 33; G3[g + 511] = lv; }
    for (int v = 0; v < 512; ++v) { int b = bw(v); BW8[v] = uint16_t(b > 8 ? 8 : b); }
    for (int i = 0; i < 32; ++i) { T31le[i] = uint16_t((i < 31 ? i : 31) * 2); T31gt[i] = uint16_t(32768 - (31 - i > 0 ? 31 - i : 0) * 2); }
    for (int i = 0; i < 16; ++i) { T15le[i] = uint16_t((i < 15 ? i : 15) * 2); T15gt[i] = uint16_t(32768 - (15 - i > 0 ? 15 - i : 0) * 2); }
    g_init = true;
}

// ------------------------------------------------------------------ alphabets / CDF
template <int A> struct Al;
template <> struct Al<31> { static constexpr int NL = 31, LANES = 32; static AI const uint16_t* lo(int g) { return LO31[g]; } static AI const uint8_t* raw(int g) { return RAW31[g]; }
    static AI const int8_t* sign(int g) { return SIGN31[g]; } static AI const uint16_t* cdf(bool e, int C) { return e ? &CDFE_31[C][0][0] : &CDF1_31[C][0][0]; }
    static AI const uint16_t* le() { return T31le; } static AI const uint16_t* gt() { return T31gt; } };
template <> struct Al<15> { static constexpr int NL = 15, LANES = 16; static AI const uint16_t* lo(int g) { return LO15[g]; } static AI const uint8_t* raw(int g) { return RAW15[g]; }
    static AI const int8_t* sign(int g) { return SIGN15[g]; } static AI const uint16_t* cdf(bool e, int C) { return e ? &CDFE_15[C][0][0] : &CDF1_15[C][0][0]; }
    static AI const uint16_t* le() { return T15le; } static AI const uint16_t* gt() { return T15gt; } };

static AI int rate_of(uint16_t& cnt) {
    int r = 6 + (cnt > 15) + (cnt > 31) + (cnt > 63) + (cnt > 127) + (cnt > 255) + (cnt > 511);
    cnt += cnt < 1024; return r < 8 ? r : 8;
}
template <int A>
static AI void cdf_update(uint16_t* c, uint16_t& cnt, int l) {
    int rate = rate_of(cnt);
#ifdef PLT_WASM_SIMD
    const v128_t idx = wasm_i16x8_make(0, 1, 2, 3, 4, 5, 6, 7), vl = wasm_i16x8_splat(short(l));
    for (int h = 0; h < Al<A>::LANES; h += 8) {
        v128_t v = wasm_v128_load(c + h);
        v128_t gt = wasm_i16x8_gt(wasm_i16x8_add(idx, wasm_i16x8_splat(short(h))), vl);
        v128_t up = wasm_u16x8_shr(wasm_i16x8_sub(wasm_v128_load(Al<A>::gt() + h), v), rate);
        v128_t dn = wasm_u16x8_shr(wasm_i16x8_sub(v, wasm_v128_load(Al<A>::le() + h)), rate);
        wasm_v128_store(c + h, wasm_v128_bitselect(wasm_i16x8_add(v, up), wasm_i16x8_sub(v, dn), gt));
    }
#else
    for (int i = 1; i <= l; ++i) c[i] = uint16_t(c[i] - (uint16_t(c[i] - Al<A>::le()[i]) >> rate));
    for (int i = l + 1; i < Al<A>::NL; ++i) c[i] = uint16_t(c[i] + (uint16_t(Al<A>::gt()[i] - c[i]) >> rate));
#endif
}
template <int A>
static AI int cdf_find(const uint16_t* c, uint32_t v) {
#ifdef PLT_WASM_SIMD
    v128_t vv = wasm_i16x8_splat(short(v)); int n = 0;
    for (int h = 0; h < Al<A>::LANES; h += 8) n += __builtin_popcount(wasm_i16x8_bitmask(wasm_u16x8_le(wasm_v128_load(c + h), vv)));
    return n - 1;
#else
    int l = 0;
    if constexpr (A == 31) l += (c[l + 16] <= v) ? 16 : 0;
    l += (c[l + 8] <= v) ? 8 : 0; l += (c[l + 4] <= v) ? 4 : 0; l += (c[l + 2] <= v) ? 2 : 0; l += (c[l + 1] <= v) ? 1 : 0;
    return l;
#endif
}

// ------------------------------------------------------------------ range decoder (hybrid carry) + bit reader
constexpr uint32_t TOP = 1u << 24;
struct RcDec {
    uint64_t low = 0; uint32_t range = 0xFFFFFFFFu, code = 0;
    AI void init(const uint8_t* b, uint32_t& nx) { for (int i = 0; i < 4; ++i) code = (code << 8) | b[nx++]; }
    AI void step(uint32_t c, uint32_t f, uint32_t r, const uint8_t* b, uint32_t& nx) {
        code -= c * r; low += uint64_t(c) * r; range = f * r;
        while (range < TOP) {
            code = (code << 8) | b[nx++]; low <<= 8; range <<= 8;
            if (__builtin_expect((low >> 32) == 0xFFFFFFFFu, 0)) { uint64_t lim = (1ull << 32) - (low & 0xFFFFFFFFu); if (range > lim) range = uint32_t(lim); }
        }
    }
};
struct BitR {
    uint64_t cur = 0; int avail = 0;
    AI uint32_t get(const uint8_t* b, uint32_t& nx, int n) {
        if (avail >= n) { uint32_t v = uint32_t(cur) & ((1u << n) - 1); cur >>= n; avail -= n; return v; }
        uint64_t w; __builtin_memcpy(&w, b + nx, 8); nx += 8;
        uint32_t v = uint32_t(cur | (w << avail)) & ((1u << n) - 1);
        int k = n - avail; cur = w >> k; avail = 64 - k; return v;
    }
};

// ------------------------------------------------------------------ per-component state
template <int A, int C, bool ENB>
struct Comp {
    static constexpr int G = C ? 1 : 0, NCTX = ENB ? 576 : 64;
    uint16_t* cdf; uint16_t* cnt; int16_t* ring; uint16_t *e0, *e1; int w;
    int16_t* P; const int16_t* Q; uint16_t* Ec; const uint16_t* Ep;
    bool init(int w_) {
        w = w_; cdf = (uint16_t*)arena(size_t(NCTX) * Al<A>::LANES * 2); cnt = (uint16_t*)arena(NCTX * 2);
        ring = (int16_t*)arena(4 * size_t(w)); e0 = (uint16_t*)arena(2 * size_t(w)); e1 = (uint16_t*)arena(2 * size_t(w));
        if (!cdf || !cnt || !ring || !e0 || !e1) return false;
        __builtin_memcpy(cdf, Al<A>::cdf(ENB, C), size_t(NCTX) * Al<A>::LANES * 2);
        memset(cnt, 0, NCTX * 2); memset(e0, 0, 2 * size_t(w)); memset(e1, 0, 2 * size_t(w)); return true;
    }
    AI void row(int y) { P = ring + (y & 1) * w; Q = y ? ring + ((y - 1) & 1) * w : P; Ec = (y & 1) ? e1 : e0; Ep = (y & 1) ? e0 : e1; }
    AI void px(const uint8_t* b, uint32_t& nx, RcDec& D, BitR& R, int x, int y, bool ad) {
        constexpr int MID = C == 0 ? 128 : 0;
        int a, bb, c, d, el = 0, et = 0;
        if (y == 0) { a = x ? P[x - 1] : MID; bb = c = d = a; if (x) el = Ec[x - 1]; }
        else { bb = Q[x]; if (x == 0) { a = c = bb; } else { a = P[x - 1]; c = Q[x - 1]; el = Ec[x - 1]; } d = (x + 1 < w) ? Q[x + 1] : bb; et = Ep[x]; }
        int mn = a < bb ? a : bb, mx = a < bb ? bb : a, pr = a + bb - c; pr = pr < mn ? mn : (pr > mx ? mx : pr);
        int m = LUT[C][G1[a - c + 511] + G2[c - bb + 511] + G3[d - bb + 511]];
        int ctx = ENB ? (m & 63) * 9 + (el > et ? el : et) : (m & 63), s = (m & 128) ? -1 : 1;
        uint16_t* cd = cdf + ctx * Al<A>::LANES;
        uint32_t r = D.range >> 15, v = D.code / r; v = v < 32767 ? v : 32767;
        int l = cdf_find<A>(cd, v);
        D.step(cd[l], cd[l + 1] - cd[l], r, b, nx);
        if (ad) cdf_update<A>(cd, cnt[ctx], l);
        int av = Al<A>::lo(G)[l];
        if (int nb = Al<A>::raw(G)[l]) av += int(R.get(b, nx, nb));
        Ec[x] = BW8[av];
        int vv = pr + s * Al<A>::sign(G)[l] * av;
        P[x] = int16_t(C == 0 ? (vv & 255) : (((vv + 256) & 511) - 256));
    }
};
// out: pixel (x, y) of the strip goes to out[(y * stride + ox + x) * BPP]; BPP 3 = RGB, 4 = RGBA (alpha 255)
template <int A, bool ENB, int BPP>
static bool dec_strip(const uint8_t* b, uint8_t* out, int stride, int ox, int w, int H, bool ad) {
    uintptr_t mark = g_top;
    Comp<A, 0, ENB> cy; Comp<A, 1, ENB> cu; Comp<A, 2, ENB> cv;
    if (!cy.init(w) || !cu.init(w) || !cv.init(w)) return false;
    uint32_t n = 0; RcDec D0, D1, D2; BitR R;
    D0.init(b, n); D1.init(b, n); D2.init(b, n);
    for (int y = 0; y < H; ++y) {
        cy.row(y); cu.row(y); cv.row(y);
        for (int x = 0; x < w; ++x) { cy.px(b, n, D0, R, x, y, ad); cu.px(b, n, D1, R, x, y, ad); cv.px(b, n, D2, R, x, y, ad); }
        uint8_t* p = out + (size_t(y) * stride + ox) * BPP;
        for (int x = 0; x < w; ++x) {
            int u = cu.P[x], v = cv.P[x], g = cy.P[x] - ((u + v) >> 2);
            p[BPP * x] = uint8_t(v + g); p[BPP * x + 1] = uint8_t(g); p[BPP * x + 2] = uint8_t(u + g); if (BPP == 4) p[BPP * x + 3] = 255;
        }
    }
    g_top = mark; return true;
}
static uint32_t get32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24; }
EXPORT uint32_t plt_width(const uint8_t* in) { return get32(in + 4); }
EXPORT uint32_t plt_height(const uint8_t* in) { return get32(in + 8); }
struct Hdr { int W, H, model, alpha, n, q, r; uint32_t off0; bool ad;
    int x(int i) const { return n == 1 ? 0 : 64 * (i * q + (i < r ? i : r)); }           // same geometry as plt2.cpp
    int w(int i) const { return i + 1 < n ? 64 * (q + (i < r)) : W - x(i); } };
static int parse(const uint8_t* in, uint32_t size, Hdr& h) {
    if (!g_init) init_static();
    if (size < 16 || in[0] != 'P' || in[1] != 'L' || in[2] != 'T' || in[3] != '2') return -1;
    h.W = int(get32(in + 4)); h.H = int(get32(in + 8)); h.model = in[12]; h.alpha = (in[13] & 1) ? 15 : 31; h.ad = !(in[13] & 2);
    if (h.W <= 0 || h.H <= 0 || h.model > 1 || in[14] != 3) return -2;
    if (in[13] & 4) return -6;                                     // custom (-T) tables: not built into this module
    { int req = in[15], m64 = h.W / 64 < 1 ? 1 : h.W / 64; h.n = req ? (req < m64 ? req : m64) : (h.W / 512 < 1 ? 1 : h.W / 512); } h.q = h.n > 1 ? (h.W / 64) / h.n : 0; h.r = h.n > 1 ? (h.W / 64) % h.n : 0; h.off0 = 16 + 4 * uint32_t(h.n);
    return 0;
}
template <int BPP>
static int strip(const uint8_t* in, uint32_t size, const Hdr& h, int i, uint8_t* out, int stride, int ox) {
    uint32_t off = h.off0; for (int k = 0; k < i; ++k) off += get32(in + 16 + 4 * k);
    uint32_t len = get32(in + 16 + 4 * i); if (off + len > size) return -3;
    int w = h.w(i); const uint8_t* b = in + off; bool ok;
    if (h.alpha == 15) ok = h.model ? dec_strip<15, true, BPP>(b, out, stride, ox, w, h.H, h.ad) : dec_strip<15, false, BPP>(b, out, stride, ox, w, h.H, h.ad);
    else               ok = h.model ? dec_strip<31, true, BPP>(b, out, stride, ox, w, h.H, h.ad) : dec_strip<31, false, BPP>(b, out, stride, ox, w, h.H, h.ad);
    return ok ? 0 : -4;
}
EXPORT int plt_decode(const uint8_t* in, uint32_t size, uint8_t* rgb) {      // whole image, RGB
    Hdr h; if (int e = parse(in, size, h)) return e;
    for (int i = 0; i < h.n; ++i) if (int e = strip<3>(in, size, h, i, rgb, h.W, h.x(i))) return e;
    return 0;
}
EXPORT int plt_strips(const uint8_t* in, uint32_t size) { Hdr h; return parse(in, size, h) ? 0 : h.n; }
EXPORT int plt_strip_x(const uint8_t* in, uint32_t size, int i) { Hdr h; return parse(in, size, h) ? -1 : h.x(i); }
EXPORT int plt_strip_w(const uint8_t* in, uint32_t size, int i) { Hdr h; return parse(in, size, h) ? -1 : h.w(i); }
// one strip straight to RGBA (w x H x 4), ready for ImageData; strips are independent -> one per Worker
EXPORT int plt_decode_strip_rgba(const uint8_t* in, uint32_t size, int i, uint8_t* rgba) {
    Hdr h; if (int e = parse(in, size, h)) return e; if (i < 0 || i >= h.n) return -5;
    int w = h.w(i);
    return strip<4>(in, size, h, i, rgba, w, 0);
}
