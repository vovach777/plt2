// planes_codec.cpp - experimental gray codec "PLN1" (8-bit P5): independent tiles (default 288 x 288, as HALIC),
// inside a tile PALIC-1 MED -> zigzag z (tile borders = image borders), then per 16-symbol block (raster order of
// the tile) the "noise" L_b = clamp(bw(mean z of the block) - off, 0, 7): the L_b low bits of the 16 symbols go raw
// as L_b bit planes (SSE movemask, 2 bytes each), the high part z >> L_b goes to one FSE4 stream per L_b value
// (8 streams = 8 contexts "noise of the block"), L_b itself through FSE4 (alphabet 8).  -m 0: plain FSE4 over z.
// File: "PLN1" | var W, H, tile, mode, off | u32 tile sizes | tiles (raster order of tiles).
// Tile (mode 1): L stream, 8 high streams, planes.  Tile (mode 0): one stream.  (streams: ctx4d_codec put_stream)
// usage: planes_codec c in.pgm out.pln [-m 0|1] [-ts N] [-off K]  |  d in.pln out.pgm
//        planes_codec bench a.pgm|dir ... [-r N] [-m 0|1] [-ts N] [-off K]
// build: clang++ -O3 -march=native -std=c++17 -I. -I../../../third_party/fse -I../../.. planes_codec.cpp <fse objs>
#define CTX4D_CODEC_NO_MAIN
#include "ctx4d_codec.cpp"
#include <smmintrin.h>
#include <chrono>

namespace pln {
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static inline uint32_t bwz(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
static inline int med(int a, int b, int c) { int mn = std::min(a, b), mx = std::max(a, b), p = a + b - c; return p < mn ? mn : (p > mx ? mx : p); }
static inline int pred(const uint8_t* r, int x, int y, int S) {     // r = row y of the tile, S = image stride
    if (y == 0) return x ? r[x - 1] : 0;
    if (x == 0) return r[-S];
    return med(r[x - 1], r[x - S], r[x - S - 1]);
}
struct Cfg { int ts = 288, mode = 1, off = 1; };

static void tile_enc(const uint8_t* img, int S, int tw, int th, const Cfg& c, Out& o, std::vector<uint8_t>& tmp) {
    const size_t n = size_t(tw) * th, nb = (n + 15) / 16;
    static thread_local std::vector<uint16_t> z, hi[8], Lb, pl; z.assign(nb * 16, 0);
    for (int y = 0; y < th; ++y) { const uint8_t* r = img + size_t(y) * S;
        for (int x = 0; x < tw; ++x) { int e = int8_t(r[x] - pred(r, x, y, S)); z[size_t(y) * tw + x] = uint16_t(e >= 0 ? 2 * e : -2 * e - 1); } }
    if (c.mode == 0) { put_stream(o, z.data(), n, 256, tmp); return; }
    Lb.resize(nb); pl.resize(nb * 8); uint16_t* pw = pl.data(); for (auto& h : hi) h.resize(nb * 16); size_t hn[8] = {0};
    for (size_t i = 0; i < nb; ++i) {
        const uint16_t* s = &z[i * 16];
        const __m128i a0 = _mm_loadu_si128((const __m128i*)s), a1 = _mm_loadu_si128((const __m128i*)(s + 8));
        __m128i t = _mm_add_epi16(a0, a1); t = _mm_add_epi16(t, _mm_srli_si128(t, 8)); t = _mm_add_epi16(t, _mm_srli_si128(t, 4)); t = _mm_add_epi16(t, _mm_srli_si128(t, 2));
        const uint32_t sum = uint32_t(_mm_extract_epi16(t, 0));               // 16 x 255 < 65536
        int L = std::min(7, std::max(0, int(bwz(sum >> 4)) - c.off)); Lb[i] = uint16_t(L);
        for (int k = 0; k < L; ++k) { const __m128i q = _mm_cvtsi32_si128(15 - k);
            *pw++ = uint16_t(_mm_movemask_epi8(_mm_packs_epi16(_mm_sll_epi16(a0, q), _mm_sll_epi16(a1, q)))); }
        const __m128i q = _mm_cvtsi32_si128(L); uint16_t* h = &hi[L][hn[L]]; hn[L] += 16;
        _mm_storeu_si128((__m128i*)h, _mm_srl_epi16(a0, q)); _mm_storeu_si128((__m128i*)(h + 8), _mm_srl_epi16(a1, q));
    }
    put_stream(o, Lb.data(), nb, 8, tmp);
    for (int L = 0; L < 8; ++L) put_stream(o, hi[L].data(), hn[L], 256u >> L, tmp);
    o.raw(pl.data(), size_t(pw - pl.data()) * 2);
}
static void tile_dec(In& in, uint8_t* img, int S, int tw, int th, const Cfg& c) {
    const size_t n = size_t(tw) * th, nb = (n + 15) / 16;
    static thread_local std::vector<uint16_t> z; z.resize(nb * 16);
    if (c.mode == 0) { static thread_local UBuf<uint16_t> b; if (get_stream(in, b) != n) throw std::runtime_error("tile size"); memcpy(z.data(), b.data(), n * 2); }
    else {
        static thread_local UBuf<uint16_t> lb, hb[8]; if (get_stream(in, lb) != nb) throw std::runtime_error("L size");
        size_t hn[8], np = 0; for (int L = 0; L < 8; ++L) hn[L] = get_stream(in, hb[L]);
        for (size_t i = 0; i < nb; ++i) { if (lb.data()[i] > 7) throw std::runtime_error("L value"); np += lb.data()[i]; }
        const uint8_t* pp = in.take(np * 2); size_t hp[8] = {0};
        const __m128i bit0 = _mm_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128), bit1 = _mm_slli_epi16(bit0, 8);
        for (size_t i = 0; i < nb; ++i) {
            const int L = lb.data()[i]; if (hp[L] + 16 > hn[L]) throw std::runtime_error("high stream");
            const uint16_t* h = hb[L].data() + hp[L]; hp[L] += 16; const __m128i q = _mm_cvtsi32_si128(L);
            __m128i v0 = _mm_sll_epi16(_mm_loadu_si128((const __m128i*)h), q), v1 = _mm_sll_epi16(_mm_loadu_si128((const __m128i*)(h + 8)), q);
            for (int k = 0; k < L; ++k) {                                 // plane k: lane j gets bit j of the mask, as bit k
                uint16_t m; memcpy(&m, pp, 2); pp += 2; const __m128i mv = _mm_set1_epi16(short(m)), one = _mm_set1_epi16(short(1 << k));
                v0 = _mm_or_si128(v0, _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(mv, bit0), bit0), one));
                v1 = _mm_or_si128(v1, _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(mv, bit1), bit1), one));
            }
            _mm_storeu_si128((__m128i*)&z[i * 16], v0); _mm_storeu_si128((__m128i*)&z[i * 16 + 8], v1);
        }
        for (int L = 0; L < 8; ++L) if (hp[L] != hn[L]) throw std::runtime_error("high stream size");
    }
    for (int y = 0; y < th; ++y) { uint8_t* r = img + size_t(y) * S;
        for (int x = 0; x < tw; ++x) { uint32_t v = z[size_t(y) * tw + x]; int e = int(v >> 1) ^ -int(v & 1); r[x] = uint8_t(pred(r, x, y, S) + e); } }
}
struct Grid { int nx, ny, ts, W, H; int tx(int i) const { return (i % nx) * ts; } int ty(int i) const { return (i / nx) * ts; }
    int tw(int i) const { return std::min(ts, W - tx(i)); } int th(int i) const { return std::min(ts, H - ty(i)); } };
static Grid grid(int W, int H, int ts) { return Grid{(W + ts - 1) / ts, (H + ts - 1) / ts, ts, W, H}; }

std::vector<uint8_t> compress(const uint8_t* img, int W, int H, const Cfg& c) {
    Grid g = grid(W, H, c.ts); const int nt = g.nx * g.ny; std::vector<std::vector<uint8_t>> t(nt); std::vector<uint8_t> tmp;
    for (int i = 0; i < nt; ++i) { Out o; tile_enc(img + size_t(g.ty(i)) * W + g.tx(i), W, g.tw(i), g.th(i), c, o, tmp); t[i].swap(o.b); }
    Out o; o.raw("PLN1", 4); o.var(W); o.var(H); o.var(c.ts); o.var(c.mode); o.var(c.off);
    for (auto& b : t) { uint32_t l = uint32_t(b.size()); o.raw(&l, 4); }
    for (auto& b : t) o.raw(b.data(), b.size());
    return o.b;
}
void decompress(const std::vector<uint8_t>& d, std::vector<uint8_t>& img, int& W, int& H) {
    In in{d.data(), d.data() + d.size()}; if (d.size() < 4 || memcmp(in.take(4), "PLN1", 4)) throw std::runtime_error("not PLN1");
    Cfg c; W = int(in.var()); H = int(in.var()); c.ts = int(in.var()); c.mode = int(in.var()); c.off = int(in.var());
    if (W <= 0 || H <= 0 || c.ts < 1 || c.mode > 1) throw std::runtime_error("bad header");
    Grid g = grid(W, H, c.ts); const int nt = g.nx * g.ny; std::vector<uint32_t> len(nt);
    for (auto& l : len) memcpy(&l, in.take(4), 4);
    img.assign(size_t(W) * H, 0);
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
    if (argc < 3) { fprintf(stderr, "usage: %s c in.pgm out.pln [-m 0|1] [-ts N] [-off K] | d in.pln out.pgm | bench a.pgm|dir ... [-r N] [-m] [-ts] [-off]\n", argv[0]); return 1; }
    std::string cmd = argv[1]; Cfg c; int reps = 3; std::vector<std::string> pos;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) c.mode = atoi(argv[++i]); else if (a == "-ts" && i + 1 < argc) c.ts = atoi(argv[++i]);
        else if (a == "-off" && i + 1 < argc) c.off = atoi(argv[++i]); else if (a == "-r" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (cmd == "bench" && std::filesystem::is_directory(a)) { std::vector<std::string> v;
            for (auto& e : std::filesystem::directory_iterator(a)) if (e.path().extension() == ".pgm") v.push_back(e.path().string());
            std::sort(v.begin(), v.end()); pos.insert(pos.end(), v.begin(), v.end()); }
        else pos.push_back(a); }
    if (c.ts < 16 || c.mode < 0 || c.mode > 1 || c.off < 0) { fprintf(stderr, "bad options\n"); return 1; }
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
            printf("TOTAL %zu files %.4f bpp  enc %.1f dec %.1f MB/s  (mode %d tile %d off %d)  %s\n", pos.size(), 8 * tb / tpx, tpx / te / 1e6, tpx / td / 1e6, c.mode, c.ts, c.off, ok ? "all ok" : "FAIL");
            return ok ? 0 : 3;
        }
    } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 2; }
    fprintf(stderr, "bad arguments\n"); return 1;
}
