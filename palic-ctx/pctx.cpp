// pctx — PALIC with activity contexts on the HALIC pipeline (format PCX5).
//
//   HALIC 0.7.2 pipeline (reverse-engineered with REA): blocks of 200 rows (rest < 101 glued to the last block),
//   tiles of 300 columns (HALIC's working buffer has rows of 300 samples; rest < 151 glued, -w 0 = full width);
//   per tile a colour mode = one of 17 triples of signals R, G, B, Y = (R+2G+B)>>2, Co = R-G, Cg = G-B, D = R-B
//   (mode 0 = Y,Co,Cg, modes 1..16 = the 16 independent triples of R,G,B,Co,Cg,D); per plane-block a predictor
//   (0 MED, 1 (2(a+b)-c)/3, 2 (a+b+1)>>1); residual wrapped (8-bit signals mod 256, differences mod 512), zigzag;
//   FSE (Collet's, as HALIC): byte path when the stream is < 256, U16 otherwise; raw / RLE fallbacks.
//   Our additions over HALIC:
//     -k 1  activity contexts: class = #thresholds <= |a-c| + |c-b| + |b-d|; plane 0 -> 5 streams, planes 1+2 -> 5
//           shared streams, class 0 shared by all planes (11 streams per block). -k 0 = one stream per plane (HALIC).
//     -C 2  colour mode chosen by MED cost (sum of bit_width(zz), every 2nd row); -C 1 = HALIC's sum|dx| with 2 %
//           hysteresis; -C 0 = always Y,Co,Cg.
//     -I 0  prediction continues across tile borders (neighbours converted to the tile's signals);
//           -I 1 = independent tiles (HALIC).
//   Defaults: -k 1 -C 2 -I 0 -w 300.   HALIC-like:  -k 0 -C 1 -I 1.
//
//   pctx c in.ppm out.pcx [-k 0|1] [-C 0|1|2] [-I 0|1] [-w cols] [-b rows] [-P mask]   pctx d in.pcx out.ppm   pctx bench [opts] a.ppm ...
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
// symbol stream: var(n), mode u8: 0 raw u16, 1 rle, 2 FSE-U16, 3 FSE-8 (all symbols < 256, as HALIC's 256 path), 4 raw u8
static void put_stream(Out& o, const std::vector<uint16_t>& s, std::vector<uint8_t>& tmp) {
    o.var(s.size()); if (s.empty()) return;
    unsigned mx = *std::max_element(s.begin(), s.end());
    tmp.resize(s.size() * 2 + 4096); size_t c;
    if (g_fse8 && mx < 256) {
        static thread_local std::vector<uint8_t> b; b.resize(s.size() + 16); for (size_t i = 0; i < s.size(); ++i) b[i] = uint8_t(s[i]);
        c = FSE_compress2(tmp.data(), tmp.size(), b.data(), s.size(), mx ? mx : 1, g_tl8);
        if (!FSE_isError(c) && c > 1 && c < s.size()) { o.u8(3); o.var(c); o.raw(tmp.data(), c); return; }
        if (!FSE_isError(c) && (c == 0 || c >= s.size())) { o.u8(4); o.raw(b.data(), s.size()); return; }   // incompressible bytes
    }
    if (!(g_fse8 && mx < 256) || FSE_isError(c) || c > 1) c = FSE_compressU16(tmp.data(), tmp.size(), s.data(), s.size(), mx ? mx : 1, 0);
    if (!FSE_isError(c) && c == 1) { o.u8(1); o.var(s[0]); }
    else if (FSE_isError(c) || c == 0 || c >= s.size() * 2) {
        if (mx < 256) { o.u8(4); for (uint16_t v : s) o.u8(uint8_t(v)); }    // raw bytes
        else { o.u8(0); o.raw(s.data(), s.size() * 2); } }                   // raw u16
    else { o.u8(2); o.var(c); o.raw(tmp.data(), c); }
}
static double g_tent = 0;
static void get_stream(In& in, std::vector<uint16_t>& s) {
    size_t n = in.var(); s.resize(n + 8); if (!n) return;
    uint8_t mode = in.u8();
    if (mode == 1) { uint16_t v = uint16_t(in.var()); std::fill(s.begin(), s.begin() + n, v); return; }
    if (mode == 0) { std::memcpy(s.data(), in.take(n * 2), n * 2); return; }
    if (mode == 4) { const uint8_t* b = in.take(n); for (size_t i = 0; i < n; ++i) s[i] = b[i]; return; }
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

// ------------------------------------------------------------------ HALIC colour modes: 17 triples of signals
// signals: 0 R, 1 G, 2 B, 3 Y = (R+2G+B)>>2, 4 Co = R-G, 5 Cg = G-B, 6 D = R-B   (Co + Cg = D)
static const uint8_t TRI[17][3] = {{3, 4, 5}, {0, 1, 2}, {0, 1, 6}, {0, 1, 5}, {0, 2, 4}, {0, 2, 5}, {1, 2, 4}, {1, 2, 6}, {0, 4, 6},
                                   {1, 4, 6}, {2, 4, 6}, {0, 4, 5}, {1, 4, 5}, {2, 4, 5}, {0, 6, 5}, {1, 6, 5}, {2, 6, 5}};
static inline int sig(int id, int R, int G, int B) {
    switch (id) { case 0: return R; case 1: return G; case 2: return B; case 3: return (R + 2 * G + B) >> 2; case 4: return R - G; case 5: return G - B; default: return R - B; }
}
static inline void inv(int m, int p, int q, int r, int& R, int& G, int& B) {
    switch (m) {
    case 0: G = p - ((q - r) >> 2); R = G + q; B = G - r; break;
    case 1: R = p; G = q; B = r; break;            case 2: R = p; G = q; B = p - r; break;
    case 3: R = p; G = q; B = q - r; break;        case 4: R = p; B = q; G = p - r; break;
    case 5: R = p; B = q; G = q + r; break;        case 6: G = p; B = q; R = p + r; break;
    case 7: G = p; B = q; R = q + r; break;        case 8: R = p; G = p - q; B = p - r; break;
    case 9: G = p; R = p + q; B = R - r; break;    case 10: B = p; R = p + r; G = R - q; break;
    case 11: R = p; G = p - q; B = G - r; break;   case 12: G = p; R = p + q; B = p - r; break;
    case 13: B = p; G = p + r; R = G + q; break;   case 14: R = p; B = p - q; G = B + r; break;
    case 15: G = p; B = p - r; R = B + q; break;   default: B = p; R = p + q; G = p + r; break;
    }
}
static inline bool nine(int id) { return id >= 4; }                         // differences are 9-bit (mod 512)
static inline int wrapv(bool n9, int v) { return n9 ? ((v + 256) & 511) - 256 : (v & 255); }
static inline int wrape(bool n9, int e) { return n9 ? ((e + 256) & 511) - 256 : int(int8_t(e)); }
static int g_ctx = 1, g_indep = 0, g_cmode = 2;   // -k contexts (0 = one table per plane, as HALIC), -I independent tiles (HALIC: 1), -C colour mode: 0 fixed YCoCg, 1 HALIC sum|dx|, 2 MED cost
static int g_tw = 300;                             // -w tile width (HALIC working buffer: rows of 300 samples); 0 = full width

static std::vector<int> spans(int n, int R) {      // 200-row blocks / 300-column tiles, rest < R/2+1 glued (HALIC rule for rows)
    std::vector<int> r; if (R <= 0 || R >= n) { r.push_back(n); return r; }
    int q = n / R, rem = n % R, glue = rem < R / 2 + 1, k = q + 1 - glue;
    for (int i = 0; i < k - 1; ++i) r.push_back(R); r.push_back(glue ? rem + R : rem); return r;
}
static inline int stream_of(int ctx, int plane, int k) { return !ctx ? plane : (k == 0 ? 0 : (plane ? 5 + k : k)); }

// Tile working buffer (as HALIC: row stride = tile width + border). Interior (x, y) at (y + 1) * S + x + 1;
// row 0 and column 0 are the border (row above the tile, column left of it), filled only when the neighbour
// exists and tiles are not independent.
struct Tile {
    int tw = 0, th = 0, S = 0; bool top = false, left = false;
    std::vector<int16_t> buf[7];
    void init(int w, int h, int nsig) { tw = w; th = h; S = w + 2; for (int i = 0; i < nsig; ++i) buf[i].assign(size_t(S) * (h + 1), 0); }
    int16_t* at(int i, int x, int y) { return buf[i].data() + size_t(y + 1) * S + x + 1; }
};
// neighbours: inside the tile always; across the tile border only if the border was filled
static inline void nbt(const int16_t* p, int S, int x, int y, int tw, bool top, bool left, int& a, int& b, int& c, int& d) {
    bool hu = y > 0 || top, hl = x > 0 || left;
    if (!hu) { a = hl ? p[-1] : 0; b = c = d = a; return; }
    b = p[-S]; if (!hl) { a = c = b; } else { a = p[-1]; c = p[-S - 1]; } d = x + 1 < tw ? p[-S + 1] : b;
}

struct Enc1 { int cm, mode[3]; };
std::vector<uint8_t> compress(const Img& im, int* pstat = nullptr, int* mstat = nullptr) {
    const int W = im.W, H = im.H; init_cls();
    Out o; o.raw("PCX5", 4); o.var(W); o.var(H); o.var(g_rows); o.var(g_tw); o.u8(uint8_t(g_ctx | g_indep << 1));
    auto pix = [&](int x, int y, int id) { const uint8_t* q = im.rgb.data() + (size_t(y) * W + x) * 3; return sig(id, q[0], q[1], q[2]); };
    std::vector<uint16_t> S[NSTR]; std::vector<uint8_t> tmp; Tile T; int y0 = 0;
    std::vector<int> cols = spans(W, g_tw);
    for (int rows : spans(H, g_rows)) {
        int x0 = 0;
        for (int tw : cols) {
            T.init(tw, rows, 7); T.top = !g_indep && y0 > 0; T.left = !g_indep && x0 > 0;
            for (int id = 0; id < 7; ++id) {
                for (int y = 0; y < rows; ++y) for (int x = 0; x < tw; ++x) *T.at(id, x, y) = int16_t(pix(x0 + x, y0 + y, id));
                if (T.top) for (int x = -1; x < tw; ++x) if (x0 + x >= 0) *T.at(id, x, -1) = int16_t(pix(x0 + x, y0 - 1, id));
                if (T.left) for (int y = 0; y < rows; ++y) *T.at(id, -1, y) = int16_t(pix(x0 - 1, y0 + y, id));
            }
            auto cost_of = [&](int id, int m, int step) {
                long cost = 0; bool n9 = nine(id);
                for (int y = 0; y < rows; y += step) for (int x = 0; x < tw; ++x) {
                    const int16_t* p = T.at(id, x, y); int a, b, cc, d; nbt(p, T.S, x, y, tw, T.top, T.left, a, b, cc, d);
                    int e = wrape(n9, *p - pred(m, a, b, cc)); cost += bw(uint32_t(e >= 0 ? 2 * e : -2 * e - 1)); }
                return cost; };
            int cm = 0;
            if (g_cmode == 1) {                                        // HALIC: sum |dx|, 2 % hysteresis
                long cost[7] = {0};
                for (int id = 0; id < 7; ++id) for (int y = 0; y < rows; ++y) for (int x = 1; x < tw; ++x) cost[id] += std::abs(*T.at(id, x, y) - *T.at(id, x - 1, y));
                long best = cost[3] + cost[4] + cost[5];
                for (int m = 1; m < 17; ++m) { long c = cost[TRI[m][0]] + cost[TRI[m][1]] + cost[TRI[m][2]]; if (c * 50 < best * 49) { best = c; cm = m; } }
            } else if (g_cmode == 2) {
                long cost[7]; for (int id = 0; id < 7; ++id) cost[id] = cost_of(id, 0, 2);
                long best = cost[3] + cost[4] + cost[5];
                for (int m = 1; m < 17; ++m) { long c = cost[TRI[m][0]] + cost[TRI[m][1]] + cost[TRI[m][2]]; if (c * 50 < best * 49) { best = c; cm = m; } }
            }
            o.u8(uint8_t(cm)); if (mstat) mstat[cm]++;
            int mode[3];
            for (int c = 0; c < 3; ++c) {
                long best = -1; mode[c] = 0;
                for (int m = 0; m < 4; ++m) { if (!(g_pmask >> m & 1)) continue; long cost = cost_of(TRI[cm][c], m, 2); if (best < 0 || cost < best) { best = cost; mode[c] = m; } }
                o.u8(uint8_t(mode[c])); if (pstat) pstat[mode[c]]++;
            }
            for (auto& s : S) s.clear();
            for (int y = 0; y < rows; ++y)
                for (int x = 0; x < tw; ++x)
                    for (int c = 0; c < 3; ++c) {
                        int id = TRI[cm][c]; bool n9 = nine(id); const int16_t* p = T.at(id, x, y);
                        int a, b, cc, d; nbt(p, T.S, x, y, tw, T.top, T.left, a, b, cc, d);
                        int k = 0;
                        if (g_ctx) { int act = std::abs(a - cc) + std::abs(cc - b) + std::abs(b - d); k = CLS[n9][act < 1023 ? act : 1023]; }
                        int e = wrape(n9, *p - pred(mode[c], a, b, cc));
                        S[stream_of(g_ctx, c, k)].push_back(uint16_t(e >= 0 ? 2 * e : -2 * e - 1));
                    }
            for (int i = 0; i < (g_ctx ? NSTR : 3); ++i) put_stream(o, S[i], tmp);
            x0 += tw;
        }
        y0 += rows;
    }
    return std::move(o.b);
}

Img decompress(const std::vector<uint8_t>& data) {
    In in{data.data(), data.data() + data.size()}; init_cls();
    if (std::memcmp(in.take(4), "PCX5", 4)) throw std::runtime_error("not PCX5");
    Img im; im.W = int(in.var()); im.H = int(in.var()); int brows = int(in.var()), btw = int(in.var()); uint8_t fl = in.u8(); const int W = im.W, H = im.H;
    const int ctx = fl & 1, indep = fl >> 1 & 1;
    im.rgb.resize(size_t(W) * H * 3);
    std::vector<uint16_t> S[NSTR]; Tile T; int y0 = 0;
    std::vector<int> cols = spans(W, btw);
    for (int rows : spans(H, brows)) {
        int x0 = 0;
        for (int tw : cols) {
            int cm = in.u8(); if (cm > 16) throw std::runtime_error("bad colour mode");
            int mode[3]; for (int c = 0; c < 3; ++c) { mode[c] = in.u8(); if (mode[c] > 3) throw std::runtime_error("bad predictor"); }
            for (int i = 0; i < (ctx ? NSTR : 3); ++i) get_stream(in, S[i]);
            const uint16_t* sp[NSTR]; for (int i = 0; i < NSTR; ++i) sp[i] = S[i].data();
            T.init(tw, rows, 3); T.top = !indep && y0 > 0; T.left = !indep && x0 > 0;
            bool N9[3]; for (int c = 0; c < 3; ++c) N9[c] = nine(TRI[cm][c]);
            auto rgbp = [&](int x, int y) { return im.rgb.data() + (size_t(y) * W + x) * 3; };
            for (int c = 0; c < 3; ++c) {                                 // borders from the decoded picture, in this tile's signals
                if (T.top) for (int x = -1; x < tw; ++x) if (x0 + x >= 0) { const uint8_t* q = rgbp(x0 + x, y0 - 1); *T.at(c, x, -1) = int16_t(sig(TRI[cm][c], q[0], q[1], q[2])); }
                if (T.left) for (int y = 0; y < rows; ++y) { const uint8_t* q = rgbp(x0 - 1, y0 + y); *T.at(c, -1, y) = int16_t(sig(TRI[cm][c], q[0], q[1], q[2])); }
            }
            for (int y = 0; y < rows; ++y) {
                uint8_t* out = rgbp(x0, y0 + y);
                for (int x = 0; x < tw; ++x) {
                    int val[3];
                    for (int c = 0; c < 3; ++c) {
                        int16_t* p = T.at(c, x, y); int a, b, cc, d; nbt(p, T.S, x, y, tw, T.top, T.left, a, b, cc, d);
                        int k = 0;
                        if (ctx) { int act = std::abs(a - cc) + std::abs(cc - b) + std::abs(b - d); k = CLS[N9[c]][act < 1023 ? act : 1023]; }
                        uint32_t z = *sp[stream_of(ctx, c, k)]++;
                        int e = int(z >> 1) ^ -int(z & 1), v = wrapv(N9[c], pred(mode[c], a, b, cc) + e);
                        *p = int16_t(v); val[c] = v;
                    }
                    int R, G, B; inv(cm, val[0], val[1], val[2], R, G, B);
                    out[3 * x] = uint8_t(R); out[3 * x + 1] = uint8_t(G); out[3 * x + 2] = uint8_t(B);
                }
            }
            x0 += tw;
        }
        y0 += rows;
    }
    return im;
}
}  // namespace pctx

static std::vector<uint8_t> slurp(const char* fn) { FILE* f = std::fopen(fn, "rb"); if (!f) throw std::runtime_error("open"); std::vector<uint8_t> v; uint8_t b[1 << 16]; size_t k; while ((k = std::fread(b, 1, sizeof b, f))) v.insert(v.end(), b, b + k); std::fclose(f); return v; }
int main(int argc, char** argv) {
    using namespace pctx;
    if (argc < 3) { std::fprintf(stderr, "usage: pctx c in.ppm out.pcx [-k 0|1] [-C 0|1|2] [-I 0|1] [-b rows] [-P mask] | pctx d in.pcx out.ppm | pctx bench [opts] a.ppm ...\n"); return 1; }
    std::string mode = argv[1]; std::vector<const char*> pos;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i]; if (a == "-b" && i + 1 < argc) g_rows = std::atoi(argv[++i]); else if (a == "-16") g_fse8 = false; else if (a == "-T" && i + 1 < argc) g_tl8 = unsigned(std::clamp(std::atoi(argv[++i]), 5, 12)); else if (a == "-P" && i + 1 < argc) g_pmask = std::atoi(argv[++i]) & 15; else if (a == "-w" && i + 1 < argc) g_tw = std::atoi(argv[++i]); else if (a == "-k" && i + 1 < argc) g_ctx = std::atoi(argv[++i]) ? 1 : 0; else if (a == "-I" && i + 1 < argc) g_indep = std::atoi(argv[++i]) ? 1 : 0; else if (a == "-C" && i + 1 < argc) g_cmode = std::clamp(std::atoi(argv[++i]), 0, 2); else pos.push_back(argv[i]); }
    if (!g_pmask) g_pmask = 1;
    try {
        if (mode == "c" && pos.size() == 2) { auto o = compress(read_ppm(pos[0])); FILE* f = std::fopen(pos[1], "wb"); std::fwrite(o.data(), 1, o.size(), f); std::fclose(f); return 0; }
        if (mode == "d" && pos.size() == 2) { Img im = decompress(slurp(pos[0])); FILE* f = std::fopen(pos[1], "wb"); std::fprintf(f, "P6\n%d %d\n255\n", im.W, im.H); std::fwrite(im.rgb.data(), 1, im.rgb.size(), f); std::fclose(f); return 0; }
        if (mode == "bench") {
            double te = 0, td = 0; size_t px = 0, by = 0; bool ok = true; int ps[4] = {0, 0, 0, 0}, ms[17] = {0};
            for (auto fn : pos) {
                Img im = read_ppm(fn); double be = 1e9, bd = 1e9; std::vector<uint8_t> o; Img r;
                for (int k = 0; k < 3; ++k) {
                    auto t0 = std::chrono::steady_clock::now(); o = compress(im, k ? nullptr : ps, k ? nullptr : ms); auto t1 = std::chrono::steady_clock::now(); r = decompress(o); auto t2 = std::chrono::steady_clock::now();
                    be = std::min(be, std::chrono::duration<double>(t1 - t0).count()); bd = std::min(bd, std::chrono::duration<double>(t2 - t1).count());
                }
                bool g = r.rgb == im.rgb; ok &= g; size_t n = size_t(im.W) * im.H; px += n; by += o.size(); te += be; td += bd;
                std::printf("%-28s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s %s\n", fn, im.W, im.H, o.size(), 8.0 * o.size() / n, 3e-6 * n / be, 3e-6 * n / bd, g ? "ok" : "FAIL");
            }
            std::printf("colour modes used:"); for (int m = 0; m < 17; ++m) if (ms[m]) std::printf(" %d:%d", m, ms[m]); std::printf("\n");
            std::printf("TOTAL %zu files  %.4f bpp  enc %.1f  dec %.1f MB/s (RGB)  k=%d I=%d C=%d w=%d b=%d  P=%d  predictors used: MED %d  3pt %d  (a+b)/2 %d  (a+c)/2 %d  %s\n", pos.size(), 8.0 * by / px, 3e-6 * px / te, 3e-6 * px / td, g_ctx, g_indep, g_cmode, g_tw, g_rows, g_pmask, ps[0], ps[1], ps[2], ps[3], ok ? "all ok" : "FAIL");
            return ok ? 0 : 3;
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    std::fprintf(stderr, "bad arguments\n"); return 1;
}
