// llrans — PALIC-1 as is, without contexts, on three planes: RCT (R-G, B-G, Y) -> MED -> zigzag ->
// symbol = bit_width(zz) (A10: 0..9) + (s-1) raw mantissa bits -> semi-static window of 2^SB symbols
// (histogram per window, packed like PALIC-1) -> 4-way interleaved rANS64 (32-bit renorm), raw bits
// go into the rANS state itself (PutBits). States are carried across windows: one flush per plane.
//
//   llrans c in.ppm out.lrn [-w 12..15] [-t 1|3]      llrans d in.lrn out.ppm [-t 1|3]
#include <algorithm>
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
#include "rgb.hpp"

namespace llrans {
constexpr int NSYM = 10;                      // bit_width of zz < 512
constexpr uint64_t L = 1ull << 31;
static inline uint32_t bw(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }

// ------------------------------------------------------------------ rANS64, words collected in reverse
struct Enc {
    uint64_t x = L;
    inline void renorm(uint64_t xmax, std::vector<uint32_t>& o) { if (x >= xmax) { o.push_back(uint32_t(x)); x >>= 32; } }
    inline void put(uint32_t start, uint32_t freq, int sb, std::vector<uint32_t>& o) {
        renorm(((L >> sb) << 32) * freq, o); x = ((x / freq) << sb) + (x % freq) + start;
    }
    // division-free put (ryg_rans Rans64EncSymbol): q = mulhi(x, rcp) >> shift == x / freq
    inline void put_sym(const struct Sym& s, std::vector<uint32_t>& o);
    inline void bits(uint32_t v, int n, std::vector<uint32_t>& o) {     // n <= 16
        renorm(((L >> 16) << 32) << (16 - n), o); x = (x << n) | v;
    }
};
static inline uint64_t mulhi64(uint64_t a, uint64_t b) {
#if defined(_MSC_VER) && !defined(__clang__)
    return __umulh(a, b);
#else
    return uint64_t((unsigned __int128)a * b >> 64);
#endif
}
struct Sym { uint64_t rcp, xmax; uint32_t bias, cmpl; int shift; };
static inline void sym_init(Sym& s, uint32_t start, uint32_t freq, int sb) {
    s.xmax = ((L >> sb) << 32) * freq; s.cmpl = (1u << sb) - freq;
    if (freq < 2) { s.rcp = ~0ull; s.shift = 0; s.bias = start + (1u << sb) - 1; return; }
    int sh = 0; while (freq > (1u << sh)) ++sh;
    uint64_t x0 = freq - 1, x1 = 1ull << (sh + 31), t1 = x1 / freq; x0 += (x1 % freq) << 32; uint64_t t0 = x0 / freq;
    s.rcp = t0 + (t1 << 32); s.shift = sh - 1; s.bias = start;
}
inline void Enc::put_sym(const Sym& s, std::vector<uint32_t>& o) {
    renorm(s.xmax, o); uint64_t q = mulhi64(x, s.rcp) >> s.shift; x = x + s.bias + q * s.cmpl;
}
struct Dec {
    uint64_t x;
    inline void renorm(const uint32_t*& p) { if (x < L) x = (x << 32) | *p++; }
};

struct Hist { uint32_t start[NSYM + 1], freq[NSYM]; alignas(16) uint16_t bound[16]; };

// histogram of one window: entries 0..NSYM-2 while symbols remain, the last one implied (PALIC-1 packing)
static void enc_hist(const Hist& h, int sb, Enc& e, std::vector<uint32_t>& o) {
    int m = 0; while (m < NSYM - 1 && (int(h.start[m]) < (1 << sb))) ++m;     // entries 0..m-1 are coded
    for (int i = m - 1; i >= 0; --i) {
        uint32_t f = h.freq[i], th = (1u << sb) - h.start[i];
        if (f) e.bits(f, int(bw(th)), o);
        e.bits(f == 0, 1, o);
    }
}
static inline uint32_t dbits(Dec& d, int n, const uint32_t*& p) { uint32_t v = uint32_t(d.x) & ((1u << n) - 1); d.x >>= n; d.renorm(p); return v; }
static void dec_hist(Hist& h, int sb, Dec& d, const uint32_t*& p) {
    uint32_t th = 1u << sb, cum = 0;
    for (int i = 0; i < NSYM - 1; ++i) {
        h.start[i] = cum; uint32_t f = 0;
        if (th) { if (!dbits(d, 1, p)) f = dbits(d, int(bw(th)), p); }
        h.freq[i] = f; cum += f; th -= f;
    }
    h.start[NSYM - 1] = cum; h.freq[NSYM - 1] = th;
    for (int i = 0; i < 16; ++i) h.bound[i] = uint16_t((i < NSYM - 1 ? h.start[i + 1] : (1u << sb)) ^ 0x8000);   // biased: signed compare = unsigned
}

// ------------------------------------------------------------------ one plane: zz[] -> words
static std::vector<uint32_t> enc_plane(const std::vector<uint16_t>& zz, int sb) {
    const size_t W = size_t(1) << sb, n = zz.size(), nw = (n + W - 1) / W;
    std::vector<uint32_t> o; o.reserve(n / 2 + 64); Enc R[4];
    std::vector<uint8_t> sym(W);
    for (size_t w = nw; w-- > 0;) {
        Hist h{}; size_t b = w * W;
        for (size_t i = 0; i < W; ++i) { uint32_t s = b + i < n ? bw(zz[b + i]) : 0; sym[i] = uint8_t(s); h.freq[s]++; }
        for (int i = 0, c = 0; i < NSYM; ++i) { h.start[i] = c; c += h.freq[i]; }
        Sym S[NSYM]; for (int i = 0; i < NSYM; ++i) if (h.freq[i]) sym_init(S[i], h.start[i], h.freq[i], sb);
        for (size_t i = W; i-- > 0;) {
            Enc X = R[3]; uint32_t s = sym[i], v = b + i < n ? zz[b + i] : 0;
            if (s > 1) X.bits(v - (1u << (s - 1)), int(s - 1), o);
            X.put_sym(S[s], o);
            R[3] = R[2]; R[2] = R[1]; R[1] = R[0]; R[0] = X;
        }
        enc_hist(h, sb, R[0], o);
    }
    for (int i = 3; i >= 0; --i) { o.push_back(uint32_t(R[i].x)); o.push_back(uint32_t(R[i].x >> 32)); }
    std::reverse(o.begin(), o.end());
    return o;
}
static void dec_plane(const uint32_t* p, size_t n, int sb, uint16_t* zz) {
    const size_t W = size_t(1) << sb, nw = (n + W - 1) / W; Dec R[4];
    for (auto& r : R) { r.x = (uint64_t(p[0]) << 32) | p[1]; p += 2; }
    Hist h;
    for (size_t w = 0; w < nw; ++w) {
        dec_hist(h, sb, R[0], p);
        const uint32_t mask = uint32_t(W - 1); size_t b = w * W, e = std::min(W, n - b);
        const __m128i b0 = _mm_load_si128((const __m128i*)h.bound), b1 = _mm_load_si128((const __m128i*)(h.bound + 8));
        for (size_t i = 0; i < W; ++i) {
            Dec X = R[0]; uint32_t r = uint32_t(X.x) & mask;
            __m128i rv = _mm_set1_epi16(short(r ^ 0x8000));          // s = #{bounds <= r}
            uint32_t m = uint32_t(_mm_movemask_epi8(_mm_packs_epi16(_mm_cmpgt_epi16(b0, rv), _mm_cmpgt_epi16(b1, rv))));
            uint32_t s = 16 - __builtin_popcount(m);
            X.x = h.freq[s] * (X.x >> sb) + r - h.start[s]; X.renorm(p);
            uint32_t v = s;
            if (s > 1) v = dbits(X, int(s - 1), p) + (1u << (s - 1));
            if (i < e) zz[b + i] = uint16_t(v);
            R[0] = R[1]; R[1] = R[2]; R[2] = R[3]; R[3] = X;
        }
    }
}

// ------------------------------------------------------------------ RCT rows (SIMD 16 px blocks, tail block overlapped)
static void rct_row(const uint8_t* rgb, int W, int16_t* u, int16_t* v, int16_t* y) {
    if (W < 16) { for (int x = 0; x < W; ++x) { int R = rgb[3 * x], G = rgb[3 * x + 1], B = rgb[3 * x + 2]; u[x] = int16_t(R - G); v[x] = int16_t(B - G); y[x] = int16_t(G + ((R - G + B - G) >> 2)); } return; }
    for (int x = 0; x < W; x += 16) { int xs = std::min(x, W - 16);   // last block overlaps the previous one (both in RGB and planes)
        conv::convert_to_plannar_line_8to16_48(rgb + 3 * xs, u + xs, v + xs, y + xs); }
}
alignas(16) static uint8_t SHUF[3][3][16];      // [out16 block][channel][byte] pshufb masks for R,G,B -> RGB interleave
static void init_shuf() { for (int o = 0; o < 3; ++o) for (int c = 0; c < 3; ++c) for (int i = 0; i < 16; ++i) { int j = 16 * o + i; SHUF[o][c][i] = j % 3 == c ? uint8_t(j / 3) : 0x80; } }
static inline void irct16(const int16_t* u, const int16_t* v, const int16_t* y, uint8_t* out) {
    __m128i R8, G8, B8;
    __m128i u0 = _mm_loadu_si128((const __m128i*)u), u1 = _mm_loadu_si128((const __m128i*)(u + 8));
    __m128i v0 = _mm_loadu_si128((const __m128i*)v), v1 = _mm_loadu_si128((const __m128i*)(v + 8));
    __m128i y0 = _mm_loadu_si128((const __m128i*)y), y1 = _mm_loadu_si128((const __m128i*)(y + 8));
    __m128i g0 = _mm_sub_epi16(y0, _mm_srai_epi16(_mm_add_epi16(u0, v0), 2)), g1 = _mm_sub_epi16(y1, _mm_srai_epi16(_mm_add_epi16(u1, v1), 2));
    const __m128i ff = _mm_set1_epi16(255);
    R8 = _mm_packus_epi16(_mm_and_si128(_mm_add_epi16(u0, g0), ff), _mm_and_si128(_mm_add_epi16(u1, g1), ff));
    G8 = _mm_packus_epi16(_mm_and_si128(g0, ff), _mm_and_si128(g1, ff));
    B8 = _mm_packus_epi16(_mm_and_si128(_mm_add_epi16(v0, g0), ff), _mm_and_si128(_mm_add_epi16(v1, g1), ff));
    for (int o = 0; o < 3; ++o)
        _mm_storeu_si128((__m128i*)(out + 16 * o), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(R8, _mm_load_si128((const __m128i*)SHUF[o][0])),
                                                                             _mm_shuffle_epi8(G8, _mm_load_si128((const __m128i*)SHUF[o][1]))),
                                                                _mm_shuffle_epi8(B8, _mm_load_si128((const __m128i*)SHUF[o][2]))));
}
static void irct_row(const int16_t* u, const int16_t* v, const int16_t* y, int W, uint8_t* rgb) {
    if (W < 16) { for (int x = 0; x < W; ++x) { int g = y[x] - ((u[x] + v[x]) >> 2); rgb[3 * x] = uint8_t(u[x] + g); rgb[3 * x + 1] = uint8_t(g); rgb[3 * x + 2] = uint8_t(v[x] + g); } return; }
    for (int x = 0; x < W; x += 16) { int xs = std::min(x, W - 16); irct16(u + xs, v + xs, y + xs, rgb + 3 * xs); }
}

// ------------------------------------------------------------------ MED per plane; plane 0,1: 9-bit (R-G, B-G), plane 2: 8-bit (Y)
static inline int med(int a, int b, int c) { int mn = std::min(a, b), mx = std::max(a, b); return std::clamp(a + b - c, mn, mx); }
template <bool NINE> static inline uint16_t zz_of(int e) { e = NINE ? ((e + 256) & 511) - 256 : int(int8_t(e)); return uint16_t(e >= 0 ? 2 * e : -2 * e - 1); }

struct Img { int W, H; std::vector<uint8_t> rgb; };
static Img read_ppm(const char* fn) {
    FILE* f = std::fopen(fn, "rb"); if (!f) throw std::runtime_error("open");
    auto tok = [&]() { int c; std::string s; while ((c = std::fgetc(f)) != EOF) { if (c == '#') { while ((c = std::fgetc(f)) != '\n' && c != EOF) {} continue; } if (isspace(c)) { if (!s.empty()) break; } else s += char(c); } return s; };
    if (tok() != "P6") throw std::runtime_error("P6 only");
    Img im; im.W = std::stoi(tok()); im.H = std::stoi(tok()); if (std::stoi(tok()) != 255) throw std::runtime_error("maxval 255 only");
    im.rgb.resize(size_t(im.W) * im.H * 3); if (std::fread(im.rgb.data(), 1, im.rgb.size(), f) != im.rgb.size()) throw std::runtime_error("short");
    std::fclose(f); return im;
}

std::vector<uint8_t> compress(const Img& im, int sb, int threads) {
    const int W = im.W, H = im.H; const size_t n = size_t(W) * H;
    std::vector<uint16_t> zz[3]; for (auto& z : zz) z.resize(n);
    std::vector<int16_t> pl[3][2]; for (auto& p : pl) for (auto& r : p) r.assign(W, 0);
    for (int y = 0; y < H; ++y) {
        int16_t* P[3]; const int16_t* Q[3];
        for (int c = 0; c < 3; ++c) { P[c] = pl[c][y & 1].data(); Q[c] = pl[c][(y + 1) & 1].data(); }
        rct_row(im.rgb.data() + size_t(y) * W * 3, W, P[0], P[1], P[2]);
        for (int c = 0; c < 3; ++c) {
            uint16_t* z = zz[c].data() + size_t(y) * W; const int16_t *cur = P[c], *up = Q[c];
            for (int x = 0; x < W; ++x) {
                int pr;
                if (y == 0) pr = x ? cur[x - 1] : 0; else if (x == 0) pr = up[0]; else pr = med(cur[x - 1], up[x], up[x - 1]);
                z[x] = c < 2 ? zz_of<true>(cur[x] - pr) : zz_of<false>(cur[x] - pr);
            }
        }
    }
    std::vector<uint32_t> s[3];
    if (threads >= 3) { std::thread t[3]; for (int c = 0; c < 3; ++c) t[c] = std::thread([&, c] { s[c] = enc_plane(zz[c], sb); }); for (auto& x : t) x.join(); }
    else for (int c = 0; c < 3; ++c) s[c] = enc_plane(zz[c], sb);
    std::vector<uint8_t> o = {'L', 'R', 'N', '1', uint8_t(W), uint8_t(W >> 8), uint8_t(H), uint8_t(H >> 8), uint8_t(sb)};
    for (auto& v : s) for (int k = 0; k < 4; ++k) o.push_back(uint8_t(v.size() >> (8 * k)));
    for (auto& v : s) { size_t b = o.size(); o.resize(b + v.size() * 4); std::memcpy(o.data() + b, v.data(), v.size() * 4); }
    return o;
}
Img decompress(const std::vector<uint8_t>& d, int threads) {
    if (d.size() < 21 || std::memcmp(d.data(), "LRN1", 4)) throw std::runtime_error("not LRN1");
    Img im; im.W = d[4] | d[5] << 8; im.H = d[6] | d[7] << 8; int sb = d[8]; const int W = im.W, H = im.H; const size_t n = size_t(W) * H;
    size_t len[3], off = 21; for (int c = 0; c < 3; ++c) len[c] = d[9 + 4 * c] | d[10 + 4 * c] << 8 | d[11 + 4 * c] << 16 | size_t(d[12 + 4 * c]) << 24;
    std::vector<uint32_t> w[3]; std::vector<uint16_t> zz[3];
    for (int c = 0; c < 3; ++c) { w[c].assign(len[c] + 8, 0); std::memcpy(w[c].data(), d.data() + off, len[c] * 4); off += len[c] * 4; zz[c].resize(n); }
    if (threads >= 3) { std::thread t[3]; for (int c = 0; c < 3; ++c) t[c] = std::thread([&, c] { dec_plane(w[c].data(), n, sb, zz[c].data()); }); for (auto& x : t) x.join(); }
    else for (int c = 0; c < 3; ++c) dec_plane(w[c].data(), n, sb, zz[c].data());
    im.rgb.resize(n * 3 + 48);
    std::vector<int16_t> pl[3][2]; for (auto& p : pl) for (auto& r : p) r.assign(W + 16, 0);
    for (int y = 0; y < H; ++y) {
        int16_t* P[3];
        for (int c = 0; c < 3; ++c) {
            P[c] = pl[c][y & 1].data(); const int16_t* up = pl[c][(y + 1) & 1].data(); const uint16_t* z = zz[c].data() + size_t(y) * W; int16_t* cur = P[c];
            for (int x = 0; x < W; ++x) {
                int pr; if (y == 0) pr = x ? cur[x - 1] : 0; else if (x == 0) pr = up[0]; else pr = med(cur[x - 1], up[x], up[x - 1]);
                int e = int(z[x] >> 1) ^ -int(z[x] & 1), v = pr + e;
                cur[x] = int16_t(c < 2 ? ((v + 256) & 511) - 256 : (v & 255));
            }
        }
        irct_row(P[0], P[1], P[2], W, im.rgb.data() + size_t(y) * W * 3);
    }
    im.rgb.resize(n * 3); return im;
}
}  // namespace llrans

static std::vector<uint8_t> slurp(const char* fn) { FILE* f = std::fopen(fn, "rb"); if (!f) throw std::runtime_error("open"); std::vector<uint8_t> v; uint8_t b[1 << 16]; size_t k; while ((k = std::fread(b, 1, sizeof b, f))) v.insert(v.end(), b, b + k); std::fclose(f); return v; }
int main(int argc, char** argv) {
    using namespace llrans; init_shuf();
    if (argc < 4) { std::fprintf(stderr, "usage: llrans c in.ppm out.lrn [-w 12..15] [-t 1|3] | llrans d in.lrn out.ppm [-t 1|3] | llrans bench [-w N] [-t T] in.ppm...\n"); return 1; }
    std::string mode = argv[1]; int sb = 12, threads = 1; std::vector<const char*> pos;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i]; if (a == "-w" && i + 1 < argc) sb = std::atoi(argv[++i]); else if (a == "-t" && i + 1 < argc) threads = std::atoi(argv[++i]); else pos.push_back(argv[i]); }
    sb = std::clamp(sb, 12, 15);
    try {
        if (mode == "c" && pos.size() == 2) { auto o = compress(read_ppm(pos[0]), sb, threads); FILE* f = std::fopen(pos[1], "wb"); std::fwrite(o.data(), 1, o.size(), f); std::fclose(f); return 0; }
        if (mode == "d" && pos.size() == 2) { Img im = decompress(slurp(pos[0]), threads); FILE* f = std::fopen(pos[1], "wb"); std::fprintf(f, "P6\n%d %d\n255\n", im.W, im.H); std::fwrite(im.rgb.data(), 1, im.rgb.size(), f); std::fclose(f); return 0; }
        if (mode == "bench") {
            double te = 0, td = 0; size_t px = 0, by = 0; bool ok = true;
            for (auto fn : pos) {
                Img im = read_ppm(fn); double be = 1e9, bd = 1e9; std::vector<uint8_t> o; Img r;
                for (int k = 0; k < 3; ++k) {
                    auto t0 = std::chrono::steady_clock::now(); o = compress(im, sb, threads); auto t1 = std::chrono::steady_clock::now(); r = decompress(o, threads); auto t2 = std::chrono::steady_clock::now();
                    be = std::min(be, std::chrono::duration<double>(t1 - t0).count()); bd = std::min(bd, std::chrono::duration<double>(t2 - t1).count());
                }
                bool g = r.rgb == im.rgb; ok &= g; size_t n = size_t(im.W) * im.H; px += n; by += o.size(); te += be; td += bd;
                std::printf("%-28s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s %s\n", fn, im.W, im.H, o.size(), 8.0 * o.size() / n, 3e-6 * n / be, 3e-6 * n / bd, g ? "ok" : "FAIL");
            }
            std::printf("TOTAL %zu files  %.4f bpp  enc %.1f  dec %.1f MB/s (RGB)  w=%d t=%d  %s\n", pos.size(), 8.0 * by / px, 3e-6 * px / te, 3e-6 * px / td, sb, threads, ok ? "all ok" : "FAIL");
            return ok ? 0 : 3;
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    std::fprintf(stderr, "bad arguments\n"); return 1;
}
