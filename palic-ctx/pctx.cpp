// pctx — PALIC with activity contexts, HALIC-like block pipeline (prototype).
//
//   RGB -> RCT planes (Y = G + ((U+V)>>2), U = B-G, V = R-G)  ->  blocks of 200 rows (HALIC rule: rest < 101
//   rows glued to the last block; -b N for experiments, 0 = whole image)  ->  per plane-block predictor (one byte:
//   0 MED, 1 (2(a+b)-c)/3, 2 (a+b+1)>>1; chosen by min sum of bit_width(zz))  ->  per pixel, per plane: residual e,
//   wrapped (Y mod 256, U/V mod 512), zz = zigzag(e) is the symbol itself (Y < 256, U/V < 512);
//   class = #thresholds <= act,  act = |a-c| + |c-b| + |b-d|  (own plane):
//       Y:  1, 3, 10, 18, 35        U, V:  1, 4, 8, 16, 30          (6 classes each; from the CLIC sample)
//   streams per block, FSE (Collet's, the engine HALIC uses), one table per stream: byte FSE (tableLog 12) when all
//   symbols of the stream are < 256 (HALIC's "FSE-256" path, bit 19), FSE-U16 otherwise (-16 forces U16):
//       class 0 of Y, U and V together (flat pixels)      classes 1..5 of Y       classes 1..5 of U + V (shared)
//   Prediction runs across block borders (only the statistics reset per block).
//
//   pctx c in.ppm out.pcx [-b rows] [-P mask]      pctx d in.pcx out.ppm      pctx bench [-b rows] [-P mask] a.ppm ...
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include "fseU16.h"
unsigned FSE_isError(size_t code);
size_t FSE_compress(void* dst, size_t dstCapacity, const void* src, size_t srcSize);           // byte FSE (zstd)
size_t FSE_compress2(void* dst, size_t dstCapacity, const void* src, size_t srcSize, unsigned maxSymbolValue, unsigned tableLog);
size_t FSE_decompress(void* dst, size_t dstCapacity, const void* cSrc, size_t cSrcSize);
}
static bool g_fse8 = true; static unsigned g_tl8 = 12;                                 // byte FSE when the stream fits in 0..255 (HALIC "FSE-256")

namespace pctx {
constexpr int NCL = 6, NSTR = 11;
static int g_rows = 200;                                    // block height (HALIC: 200)
static const int THY[5] = {1, 3, 10, 18, 35}, THC[5] = {1, 4, 8, 16, 30};
static uint8_t CLS[2][1024];                       // act (capped 1023) -> class, [0] Y, [1] chroma
static void init_cls() { for (int t = 0; t < 2; ++t) for (int a = 0; a < 1024; ++a) { int k = 0; for (int i = 0; i < 5; ++i) k += a >= (t ? THC : THY)[i]; CLS[t][a] = uint8_t(k); } }
static inline uint32_t bw(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
static inline int med(int a, int b, int c) { int mn = std::min(a, b), mx = std::max(a, b); return std::clamp(a + b - c, mn, mx); }
// predictor per plane-block (HALIC has 0..2; 3 is a candidate for HALIC's "(prev + c)/2")
//   0 MED   1 (2(a+b) - c) / 3 (trunc, as imul 0x55555556)   2 (a+b+1)>>1   3 (a+c+1)>>1
static inline int pred(int m, int a, int b, int c) {
    switch (m) { case 0: return med(a, b, c); case 1: return (2 * (a + b) - c) / 3; case 2: return (a + b + 1) >> 1; default: return (a + c + 1) >> 1; }
}
static int g_pmask = 7;                                    // allowed predictors (bench option -P mask)

// ------------------------------------------------------------------ byte output helpers
struct Out {
    std::vector<uint8_t> b;
    void u8(uint8_t v) { b.push_back(v); }
    void var(uint64_t v) { while (v >= 128) { b.push_back(uint8_t(v | 128)); v >>= 7; } b.push_back(uint8_t(v)); }
    void raw(const void* p, size_t n) { b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n); }
};
struct In {
    const uint8_t* p; const uint8_t* e;
    uint8_t u8() { if (p >= e) throw std::runtime_error("truncated"); return *p++; }
    uint64_t var() { uint64_t v = 0; int s = 0; for (;;) { uint8_t c = u8(); v |= uint64_t(c & 127) << s; if (!(c & 128)) return v; s += 7; } }
    const uint8_t* take(size_t n) { if (size_t(e - p) < n) throw std::runtime_error("truncated"); const uint8_t* r = p; p += n; return r; }
};

// symbol stream (u16 = whole zigzag residual): var(n), mode u8 (0 raw u16, 1 rle, 2 FSE-U16), [var csize][data]
// symbol stream: var(n), mode u8: 0 raw u16, 1 rle, 2 FSE-U16, 3 FSE-8 (all symbols < 256, as HALIC's 256 path)
static void put_stream(Out& o, const std::vector<uint16_t>& s, std::vector<uint8_t>& tmp) {
    o.var(s.size()); if (s.empty()) return;
    unsigned mx = *std::max_element(s.begin(), s.end());
    tmp.resize(s.size() * 2 + 4096); size_t c;
    if (g_fse8 && mx < 256) {
        static thread_local std::vector<uint8_t> b; b.resize(s.size()); for (size_t i = 0; i < s.size(); ++i) b[i] = uint8_t(s[i]);
        c = FSE_compress2(tmp.data(), tmp.size(), b.data(), b.size(), mx ? mx : 1, g_tl8);
        if (!FSE_isError(c) && c > 1 && c < s.size()) { o.u8(3); o.var(c); o.raw(tmp.data(), c); return; }
    }
    if (!(g_fse8 && mx < 256) || FSE_isError(c) || c > 1) c = FSE_compressU16(tmp.data(), tmp.size(), s.data(), s.size(), mx ? mx : 1, 0);
    if (!FSE_isError(c) && c == 1) { o.u8(1); o.var(s[0]); }
    else if (FSE_isError(c) || c == 0 || c >= s.size() * 2) { o.u8(0); o.raw(s.data(), s.size() * 2); }
    else { o.u8(2); o.var(c); o.raw(tmp.data(), c); }
}
static double g_tent = 0;
static void get_stream(In& in, std::vector<uint16_t>& s) {
    size_t n = in.var(); s.resize(n + 8); if (!n) return;
    uint8_t mode = in.u8();
    if (mode == 1) { uint16_t v = uint16_t(in.var()); std::fill(s.begin(), s.begin() + n, v); return; }
    if (mode == 0) { std::memcpy(s.data(), in.take(n * 2), n * 2); return; }
    size_t c = in.var(); const uint8_t* src = in.take(c); size_t r;
    auto t0 = std::chrono::steady_clock::now();
    if (mode == 3) { static thread_local std::vector<uint8_t> b; b.resize(n + 8); r = FSE_decompress(b.data(), n, src, c); for (size_t i = 0; i < n; ++i) s[i] = b[i]; }
    else r = FSE_decompressU16(s.data(), n, src, c);
    g_tent += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (FSE_isError(r) || r != n) throw std::runtime_error("entropy stream");
}

// ------------------------------------------------------------------ image and planes
struct Img { int W = 0, H = 0; std::vector<uint8_t> rgb; };
static Img read_ppm(const char* fn) {
    FILE* f = std::fopen(fn, "rb"); if (!f) throw std::runtime_error("open");
    auto tok = [&]() { int c; std::string s; while ((c = std::fgetc(f)) != EOF) { if (c == '#') { while ((c = std::fgetc(f)) != '\n' && c != EOF) {} continue; } if (isspace(c)) { if (!s.empty()) break; } else s += char(c); } return s; };
    if (tok() != "P6") throw std::runtime_error("P6 only");
    Img im; im.W = std::stoi(tok()); im.H = std::stoi(tok()); if (std::stoi(tok()) != 255) throw std::runtime_error("maxval 255 only");
    im.rgb.resize(size_t(im.W) * im.H * 3); if (std::fread(im.rgb.data(), 1, im.rgb.size(), f) != im.rgb.size()) throw std::runtime_error("short");
    std::fclose(f); return im;
}
static std::vector<int> block_rows(int H, int R) {                  // HALIC rule for R = 200: rest < R/2+1 glued
    std::vector<int> r; if (R <= 0 || R >= H) { r.push_back(H); return r; }
    int q = H / R, rem = H % R, glue = rem < R / 2 + 1, n = q + 1 - glue;
    for (int i = 0; i < n - 1; ++i) r.push_back(R); r.push_back(glue ? rem + R : rem); return r;
}

// per plane state: two rows (cur, up) of int16 values
struct Plane { std::vector<int16_t> r0, r1; int16_t* cur; int16_t* up; void init(int W) { r0.assign(W + 2, 0); r1.assign(W + 2, 0); cur = r0.data() + 1; up = r1.data() + 1; } void swap() { std::swap(cur, up); } };

// neighbours with borders (same rule in encoder and decoder)
static inline void nbr(const Plane& P, int x, int y, int W, int& a, int& b, int& c, int& d) {
    if (y == 0) { a = x ? P.cur[x - 1] : 0; b = c = d = a; return; }
    b = P.up[x]; if (x == 0) { a = c = b; } else { a = P.cur[x - 1]; c = P.up[x - 1]; } d = x + 1 < W ? P.up[x + 1] : b;
}

std::vector<uint8_t> compress(const Img& im, int* pstat = nullptr) {
    const int W = im.W, H = im.H; init_cls();
    Out o; o.raw("PCX3", 4); o.var(W); o.var(H); o.var(g_rows);
    std::vector<int16_t> PL[3]; for (auto& p : PL) p.resize(size_t(W) * H);
    for (size_t i = 0, n = size_t(W) * H; i < n; ++i) { int R = im.rgb[3 * i], G = im.rgb[3 * i + 1], B = im.rgb[3 * i + 2], u = B - G, v = R - G;
        PL[0][i] = int16_t(G + ((u + v) >> 2)); PL[1][i] = int16_t(u); PL[2][i] = int16_t(v); }
    auto nb = [&](const int16_t* p, int x, int y, int& a, int& b, int& c, int& d) {
        const int16_t* cur = p + size_t(y) * W;
        if (y == 0) { a = x ? cur[x - 1] : 0; b = c = d = a; return; }
        const int16_t* up = cur - W; b = up[x]; if (x == 0) { a = c = b; } else { a = cur[x - 1]; c = up[x - 1]; } d = x + 1 < W ? up[x + 1] : b; };
    auto wrap = [](int c, int e) { return c ? ((e + 256) & 511) - 256 : int(int8_t(e)); };
    std::vector<uint16_t> S[NSTR]; std::vector<uint8_t> tmp; int y0 = 0;
    for (int rows : block_rows(H, g_rows)) {
        int mode[3];
        for (int c = 0; c < 3; ++c) {                                   // pick the predictor: min sum of bit_width(zz)
            long best = -1; mode[c] = 0;
            for (int m = 0; m < 4; ++m) { if (!(g_pmask >> m & 1)) continue; long cost = 0;
                for (int y = y0; y < y0 + rows; ++y) for (int x = 0; x < W; ++x) { int a, b, cc, d; nb(PL[c].data(), x, y, a, b, cc, d);
                    int e = wrap(c, PL[c][size_t(y) * W + x] - pred(m, a, b, cc)); cost += bw(uint32_t(e >= 0 ? 2 * e : -2 * e - 1)); }
                if (best < 0 || cost < best) { best = cost; mode[c] = m; } }
            o.u8(uint8_t(mode[c])); if (pstat) pstat[mode[c]]++;
        }
        for (auto& s : S) s.clear();
        for (int y = y0; y < y0 + rows; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < 3; ++c) {
                    int a, b, cc, d; nb(PL[c].data(), x, y, a, b, cc, d);
                    int act = std::abs(a - cc) + std::abs(cc - b) + std::abs(b - d);
                    int k = CLS[c != 0][act < 1023 ? act : 1023];
                    int e = wrap(c, PL[c][size_t(y) * W + x] - pred(mode[c], a, b, cc));
                    S[k == 0 ? 0 : (c ? 5 + k : k)].push_back(uint16_t(e >= 0 ? 2 * e : -2 * e - 1));
                }
        y0 += rows;
        for (auto& s : S) put_stream(o, s, tmp);
    }
    return std::move(o.b);
}

Img decompress(const std::vector<uint8_t>& data) {
    In in{data.data(), data.data() + data.size()}; init_cls();
    if (std::memcmp(in.take(4), "PCX3", 4)) throw std::runtime_error("not PCX3");
    Img im; im.W = int(in.var()); im.H = int(in.var()); int brows = int(in.var()); const int W = im.W, H = im.H;
    im.rgb.resize(size_t(W) * H * 3);
    Plane P[3]; for (auto& p : P) p.init(W);
    std::vector<uint16_t> S[NSTR]; int y = 0;
    for (int rows : block_rows(H, brows)) {
        int mode[3]; for (int c = 0; c < 3; ++c) { mode[c] = in.u8(); if (mode[c] > 3) throw std::runtime_error("bad predictor"); }
        for (auto& s : S) get_stream(in, s);
        const uint16_t* sp[NSTR]; for (int i = 0; i < NSTR; ++i) sp[i] = S[i].data();
        for (int yy = 0; yy < rows; ++yy, ++y) {
            uint8_t* rgb = im.rgb.data() + size_t(y) * W * 3;
            for (int x = 0; x < W; ++x) {
                int val[3];
                for (int c = 0; c < 3; ++c) {
                    int a, b, cc, d; nbr(P[c], x, y, W, a, b, cc, d);
                    int act = std::abs(a - cc) + std::abs(cc - b) + std::abs(b - d);
                    int k = CLS[c != 0][act < 1023 ? act : 1023];
                    uint32_t z = *sp[k == 0 ? 0 : (c ? 5 + k : k)]++;
                    int e = int(z >> 1) ^ -int(z & 1), v = pred(mode[c], a, b, cc) + e;
                    v = c ? ((v + 256) & 511) - 256 : (v & 255);
                    P[c].cur[x] = int16_t(v); val[c] = v;
                }
                int g = val[0] - ((val[1] + val[2]) >> 2);
                rgb[3 * x] = uint8_t(val[2] + g); rgb[3 * x + 1] = uint8_t(g); rgb[3 * x + 2] = uint8_t(val[1] + g);
            }
            for (auto& p : P) p.swap();
        }
    }
    return im;
}
}  // namespace pctx

static std::vector<uint8_t> slurp(const char* fn) { FILE* f = std::fopen(fn, "rb"); if (!f) throw std::runtime_error("open"); std::vector<uint8_t> v; uint8_t b[1 << 16]; size_t k; while ((k = std::fread(b, 1, sizeof b, f))) v.insert(v.end(), b, b + k); std::fclose(f); return v; }
int main(int argc, char** argv) {
    using namespace pctx;
    if (argc < 3) { std::fprintf(stderr, "usage: pctx c in.ppm out.pcx [-b rows] [-P mask] | pctx d in.pcx out.ppm | pctx bench [-b rows] [-P mask] a.ppm ...\n"); return 1; }
    std::string mode = argv[1]; std::vector<const char*> pos;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i]; if (a == "-b" && i + 1 < argc) g_rows = std::atoi(argv[++i]); else if (a == "-16") g_fse8 = false; else if (a == "-T" && i + 1 < argc) g_tl8 = unsigned(std::clamp(std::atoi(argv[++i]), 5, 12)); else if (a == "-P" && i + 1 < argc) g_pmask = std::atoi(argv[++i]) & 15; else pos.push_back(argv[i]); }
    if (!g_pmask) g_pmask = 1;
    try {
        if (mode == "c" && pos.size() == 2) { auto o = compress(read_ppm(pos[0])); FILE* f = std::fopen(pos[1], "wb"); std::fwrite(o.data(), 1, o.size(), f); std::fclose(f); return 0; }
        if (mode == "d" && pos.size() == 2) { Img im = decompress(slurp(pos[0])); FILE* f = std::fopen(pos[1], "wb"); std::fprintf(f, "P6\n%d %d\n255\n", im.W, im.H); std::fwrite(im.rgb.data(), 1, im.rgb.size(), f); std::fclose(f); return 0; }
        if (mode == "bench") {
            double te = 0, td = 0; size_t px = 0, by = 0; bool ok = true; int ps[4] = {0, 0, 0, 0};
            for (auto fn : pos) {
                Img im = read_ppm(fn); double be = 1e9, bd = 1e9; std::vector<uint8_t> o; Img r;
                for (int k = 0; k < 3; ++k) {
                    auto t0 = std::chrono::steady_clock::now(); o = compress(im, k ? nullptr : ps); auto t1 = std::chrono::steady_clock::now(); r = decompress(o); auto t2 = std::chrono::steady_clock::now();
                    be = std::min(be, std::chrono::duration<double>(t1 - t0).count()); bd = std::min(bd, std::chrono::duration<double>(t2 - t1).count());
                }
                bool g = r.rgb == im.rgb; ok &= g; size_t n = size_t(im.W) * im.H; px += n; by += o.size(); te += be; td += bd;
                std::printf("%-28s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s %s\n", fn, im.W, im.H, o.size(), 8.0 * o.size() / n, 3e-6 * n / be, 3e-6 * n / bd, g ? "ok" : "FAIL");
            }
            std::printf("entropy decode share: %.1f%% of decode time (3 reps)\n", 100.0 * g_tent / (3 * td) / 1.0);
            std::printf("TOTAL %zu files  %.4f bpp  enc %.1f  dec %.1f MB/s (RGB)  b=%d  P=%d  predictors used: MED %d  3pt %d  (a+b)/2 %d  (a+c)/2 %d  %s\n", pos.size(), 8.0 * by / px, 3e-6 * px / te, 3e-6 * px / td, g_rows, g_pmask, ps[0], ps[1], ps[2], ps[3], ok ? "all ok" : "FAIL");
            return ok ? 0 : 3;
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    std::fprintf(stderr, "bad arguments\n"); return 1;
}
