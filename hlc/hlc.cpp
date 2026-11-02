// hlc — HALIC-class codec built on PALIC-1: one colour transform, independent 256x256 tiles, one FSE table per
// plane-tile, in-place decode.
//
//   RGB -> RCT (V = R-G, U = B-G, Y = G + ((U+V)>>2); the same as HALIC's mode 0 Y,Co,Cg with Cg = -U), SIMD 16 px
//   -> tiles of 256 x 256 (edge tiles smaller), independent (prediction restarts at the tile border)
//   -> per plane-tile predictor (byte): 0 MED, 1 (2(a+b)-c)/3, 2 (a+b+1)>>1, 3 none; chosen by sum of bit_width(zz)
//   -> residual wrapped (Y mod 256, U/V mod 512), zigzag -> FSE (Collet): byte FSE when max zz < 256, else FSE-U16
//   decoder: FSE writes straight into the tile buffer (raster order = buffer order), the predictor restores the
//   plane in place (left neighbour in a register), the inverse RCT writes interleaved RGB (SSSE3, 16 px).
//
//   hlc c in.ppm out.hlc [-t threads]     hlc d in.hlc out.ppm [-t threads]     hlc bench [-t threads] a.ppm ...
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
#include "fseU16.h"
unsigned FSE_isError(size_t code);
size_t FSE_compress2(void* dst, size_t dstCapacity, const void* src, size_t srcSize, unsigned maxSymbolValue, unsigned tableLog);
size_t FSE_decompress(void* dst, size_t dstCapacity, const void* cSrc, size_t cSrcSize);
}

namespace hlc {
constexpr int TS = 256;                                         // tile side
static inline uint32_t bw(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
static inline uint16_t zz(int e) { return uint16_t(e >= 0 ? 2 * e : -2 * e - 1); }
static inline int unzz(uint32_t z) { return int(z >> 1) ^ -int(z & 1); }
static inline int med(int a, int b, int c) { int mn = std::min(a, b), mx = std::max(a, b); return std::clamp(a + b - c, mn, mx); }
template <int M> static inline int pred(int a, int b, int c) {
    if constexpr (M == 0) return med(a, b, c);
    else if constexpr (M == 1) return (2 * (a + b) - c) / 3;
    else if constexpr (M == 2) return (a + b + 1) >> 1;
    else return 0;
}
template <bool N9> static inline int wrape(int e) { return N9 ? ((e + 256) & 511) - 256 : int(int8_t(e)); }
template <bool N9> static inline int wrapv(int v) { return N9 ? ((v + 256) & 511) - 256 : (v & 255); }

struct Img { int W = 0, H = 0; std::vector<uint8_t> rgb; };
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

// ------------------------------------------------------------------ prediction over a tile plane (stride = w)
// borders: row 0 -> left neighbour (first sample: 0); column 0 -> top; last column: d unused (MED needs a, b, c only)
template <int M, bool N9> static void residuals(const int16_t* p, int w, int h, uint16_t* out) {
    for (int y = 0; y < h; ++y) {
        const int16_t* r = p + size_t(y) * w; const int16_t* u = r - w; uint16_t* o = out + size_t(y) * w;
        if (y == 0) { o[0] = zz(wrape<N9>(r[0])); for (int x = 1; x < w; ++x) o[x] = zz(wrape<N9>(r[x] - r[x - 1])); continue; }
        o[0] = zz(wrape<N9>(r[0] - u[0]));
        for (int x = 1; x < w; ++x) o[x] = zz(wrape<N9>(r[x] - pred<M>(r[x - 1], u[x], u[x - 1])));
    }
}
template <int M, bool N9> static long cost(const int16_t* p, int w, int h) {             // every 2nd row
    long s = 0;
    for (int y = 1; y < h; y += 2) { const int16_t* r = p + size_t(y) * w; const int16_t* u = r - w;
        for (int x = 1; x < w; ++x) s += bw(zz(wrape<N9>(r[x] - pred<M>(r[x - 1], u[x], u[x - 1])))); }
    return s;
}
// in place: on entry p[] holds zigzag residuals (as int16), on exit the plane
template <int M, bool N9> static void restore(int16_t* p, int w, int h) {
    for (int y = 0; y < h; ++y) {
        int16_t* r = p + size_t(y) * w; const int16_t* u = r - w;
        int a;
        if (y == 0) { a = wrapv<N9>(unzz(uint16_t(r[0]))); r[0] = int16_t(a); for (int x = 1; x < w; ++x) { a = wrapv<N9>(a + unzz(uint16_t(r[x]))); r[x] = int16_t(a); } continue; }
        a = wrapv<N9>(u[0] + unzz(uint16_t(r[0]))); r[0] = int16_t(a);
        for (int x = 1; x < w; ++x) { a = wrapv<N9>(pred<M>(a, u[x], u[x - 1]) + unzz(uint16_t(r[x]))); r[x] = int16_t(a); }
    }
}
template <bool N9> static void residuals_m(int m, const int16_t* p, int w, int h, uint16_t* o) { switch (m) { case 0: residuals<0, N9>(p, w, h, o); break; case 1: residuals<1, N9>(p, w, h, o); break; case 2: residuals<2, N9>(p, w, h, o); break; default: residuals<3, N9>(p, w, h, o); } }
template <bool N9> static long cost_m(int m, const int16_t* p, int w, int h) { switch (m) { case 0: return cost<0, N9>(p, w, h); case 1: return cost<1, N9>(p, w, h); case 2: return cost<2, N9>(p, w, h); default: return cost<3, N9>(p, w, h); } }
template <bool N9> static void restore_m(int m, int16_t* p, int w, int h) { switch (m) { case 0: restore<0, N9>(p, w, h); break; case 1: restore<1, N9>(p, w, h); break; case 2: restore<2, N9>(p, w, h); break; default: restore<3, N9>(p, w, h); } }

// ------------------------------------------------------------------ FSE stream: mode u8 (0 raw u16, 1 rle, 2 FSE-U16, 3 FSE-8, 4 raw u8), [var csize][data]
static void put_stream(Out& o, const uint16_t* s, size_t n, std::vector<uint8_t>& tmp, std::vector<uint8_t>& b8) {
    unsigned mx = *std::max_element(s, s + n); tmp.resize(n * 2 + 4096);
    if (mx < 256) {
        b8.resize(n); for (size_t i = 0; i < n; ++i) b8[i] = uint8_t(s[i]);
        size_t c = FSE_compress2(tmp.data(), tmp.size(), b8.data(), n, mx ? mx : 1, 12);
        if (!FSE_isError(c) && c == 1) { o.u8(1); o.var(s[0]); return; }
        if (!FSE_isError(c) && c > 1 && c < n) { o.u8(3); o.var(c); o.raw(tmp.data(), c); return; }
        o.u8(4); o.raw(b8.data(), n); return;
    }
    size_t c = FSE_compressU16(tmp.data(), tmp.size(), s, n, mx, 0);
    if (!FSE_isError(c) && c == 1) { o.u8(1); o.var(s[0]); return; }
    if (FSE_isError(c) || c == 0 || c >= n * 2) { o.u8(0); o.raw(s, n * 2); return; }
    o.u8(2); o.var(c); o.raw(tmp.data(), c);
}
// decode straight into the tile buffer (int16 holding zigzag values)
static void get_stream(In& in, int16_t* dst, size_t n, std::vector<uint8_t>& b8) {
    uint8_t mode = in.u8();
    if (mode == 1) { int16_t v = int16_t(in.var()); std::fill(dst, dst + n, v); return; }
    if (mode == 0) { std::memcpy(dst, in.take(n * 2), n * 2); return; }
    if (mode == 4) { const uint8_t* b = in.take(n); for (size_t i = 0; i < n; ++i) dst[i] = b[i]; return; }
    size_t c = in.var(); const uint8_t* src = in.take(c); size_t r;
    if (mode == 3) { b8.resize(n + 16); r = FSE_decompress(b8.data(), n, src, c); for (size_t i = 0; i < n; ++i) dst[i] = b8[i]; }
    else r = FSE_decompressU16((unsigned short*)dst, n, src, c);
    if (FSE_isError(r) || r != n) throw std::runtime_error("entropy stream");
}

template <class F> static void parallel_for(int n, int threads, F f) {
    if (threads <= 1 || n <= 1) { for (int i = 0; i < n; ++i) f(i); return; }
    std::atomic<int> next{0}; std::vector<std::thread> th;
    for (int t = 0; t < std::min(threads, n); ++t) th.emplace_back([&] { for (int i; (i = next.fetch_add(1)) < n;) f(i); });
    for (auto& t : th) t.join();
}

// file: "HLC1" var W var H, then per tile (raster order): var size, payload = 3 x [u8 predictor][stream]
std::vector<uint8_t> compress(const Img& im, int threads, int* pstat = nullptr) {
    const int W = im.W, H = im.H, tx = (W + TS - 1) / TS, ty = (H + TS - 1) / TS, nt = tx * ty;
    std::vector<std::vector<uint8_t>> tiles(nt); std::vector<std::array<int, 3>> ms(nt);
    parallel_for(nt, threads, [&](int t) {
        int x0 = (t % tx) * TS, y0 = (t / tx) * TS, w = std::min(TS, W - x0), h = std::min(TS, H - y0);
        thread_local std::vector<int16_t> P[3]; thread_local std::vector<uint16_t> R; thread_local std::vector<uint8_t> tmp, b8;
        for (auto& p : P) p.resize(size_t(w) * h); R.resize(size_t(w) * h);
        for (int y = 0; y < h; ++y) rct_row(im.rgb.data() + (size_t(y0 + y) * W + x0) * 3, w, P[0].data() + size_t(y) * w, P[1].data() + size_t(y) * w, P[2].data() + size_t(y) * w);
        Out o;
        for (int c = 0; c < 3; ++c) {
            bool n9 = c < 2; int best = 0; long bc = -1;
            for (int m = 0; m < 4; ++m) { long k = n9 ? cost_m<true>(m, P[c].data(), w, h) : cost_m<false>(m, P[c].data(), w, h); if (bc < 0 || k < bc) { bc = k; best = m; } }
            if (n9) residuals_m<true>(best, P[c].data(), w, h, R.data()); else residuals_m<false>(best, P[c].data(), w, h, R.data());
            o.u8(uint8_t(best)); ms[t][c] = best; put_stream(o, R.data(), size_t(w) * h, tmp, b8);
        }
        tiles[t] = std::move(o.b);
    });
    Out f; f.raw("HLC1", 4); f.var(W); f.var(H);
    for (int t = 0; t < nt; ++t) { f.var(tiles[t].size()); f.raw(tiles[t].data(), tiles[t].size()); if (pstat) for (int c = 0; c < 3; ++c) pstat[ms[t][c]]++; }
    return std::move(f.b);
}

Img decompress(const std::vector<uint8_t>& data, int threads) {
    In in{data.data(), data.data() + data.size()};
    if (std::memcmp(in.take(4), "HLC1", 4)) throw std::runtime_error("not HLC1");
    Img im; im.W = int(in.var()); im.H = int(in.var()); const int W = im.W, H = im.H, tx = (W + TS - 1) / TS, ty = (H + TS - 1) / TS, nt = tx * ty;
    im.rgb.resize(size_t(W) * H * 3 + 64);
    std::vector<const uint8_t*> tp(nt); std::vector<size_t> tn(nt);
    for (int t = 0; t < nt; ++t) { tn[t] = in.var(); tp[t] = in.take(tn[t]); }
    parallel_for(nt, threads, [&](int t) {
        int x0 = (t % tx) * TS, y0 = (t / tx) * TS, w = std::min(TS, W - x0), h = std::min(TS, H - y0); size_t n = size_t(w) * h;
        thread_local std::vector<int16_t> P[3]; thread_local std::vector<uint8_t> b8;
        In ti{tp[t], tp[t] + tn[t]};
        for (int c = 0; c < 3; ++c) {
            P[c].resize(n + 16); int m = ti.u8(); if (m > 3) throw std::runtime_error("bad predictor");
            get_stream(ti, P[c].data(), n, b8);
            if (c < 2) restore_m<true>(m, P[c].data(), w, h); else restore_m<false>(m, P[c].data(), w, h);
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
    if (argc < 3) { std::fprintf(stderr, "usage: hlc c in.ppm out.hlc [-t N] | hlc d in.hlc out.ppm [-t N] | hlc bench [-t N] a.ppm ...\n"); return 1; }
    std::string mode = argv[1]; int threads = 1; std::vector<const char*> pos;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i]; if (a == "-t" && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i])); else pos.push_back(argv[i]); }
    try {
        if (mode == "c" && pos.size() == 2) { auto o = compress(read_ppm(pos[0]), threads); FILE* f = std::fopen(pos[1], "wb"); std::fwrite(o.data(), 1, o.size(), f); std::fclose(f); return 0; }
        if (mode == "d" && pos.size() == 2) { Img im = decompress(slurp(pos[0]), threads); FILE* f = std::fopen(pos[1], "wb"); std::fprintf(f, "P6\n%d %d\n255\n", im.W, im.H); std::fwrite(im.rgb.data(), 1, im.rgb.size(), f); std::fclose(f); return 0; }
        if (mode == "bench") {
            double te = 0, td = 0; size_t px = 0, by = 0; bool ok = true; int ps[4] = {0, 0, 0, 0};
            for (auto fn : pos) {
                Img im = read_ppm(fn); double be = 1e9, bd = 1e9; std::vector<uint8_t> o; Img r;
                for (int k = 0; k < 3; ++k) {
                    auto t0 = std::chrono::steady_clock::now(); o = compress(im, threads, k ? nullptr : ps); auto t1 = std::chrono::steady_clock::now(); r = decompress(o, threads); auto t2 = std::chrono::steady_clock::now();
                    be = std::min(be, std::chrono::duration<double>(t1 - t0).count()); bd = std::min(bd, std::chrono::duration<double>(t2 - t1).count());
                }
                bool g = r.rgb == im.rgb; ok &= g; size_t n = size_t(im.W) * im.H; px += n; by += o.size(); te += be; td += bd;
                std::printf("%-28s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s %s\n", fn, im.W, im.H, o.size(), 8.0 * o.size() / n, 3e-6 * n / be, 3e-6 * n / bd, g ? "ok" : "FAIL");
            }
            std::printf("TOTAL %zu files  %.4f bpp  enc %.1f  dec %.1f MB/s (RGB)  t=%d  predictors MED %d 3pt %d avg %d none %d  %s\n", pos.size(), 8.0 * by / px, 3e-6 * px / te, 3e-6 * px / td, threads, ps[0], ps[1], ps[2], ps[3], ok ? "all ok" : "FAIL");
            return ok ? 0 : 3;
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    std::fprintf(stderr, "bad arguments\n"); return 1;
}
