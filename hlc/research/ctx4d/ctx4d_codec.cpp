// ctx4d_codec.cpp - prototype codec: 4-D context (ctx4d_vec kernels, vector types) + FSE (hlc's 4-state FSE4).
// Encoder: RCT -> per plane context + symbol (raster, vector kernel) -> symbols into 16 streams in decoder order
//          (band of 16 rows, step t = x + 2k, plane Y U V, lane) -> FSE per unit -> file.
// Decoder: file -> per unit FSE into 16 stream buffers -> wavefront decode of Y, U, V (three chains in one t loop)
//          -> inverse RCT. Streams 0..7 Y, 8..15 shared by U and V.
// Unit (one FSE run per stream): whole bands, at least 256 compute blocks of 128 x 16 px (W = 2048: 16 bands).
// Format "C4D1": var W, var H, then per unit and stream: var n, payload (mode u8: 0 raw u16, 1 rle var, 6 FSE4 var c).
// usage:  ctx4d_codec c in.ppm out.c4d groups.csv | d in.c4d out.ppm groups.csv | bench groups.csv a.ppm|dir ... [-r N] [-max N] [-s]
#define CTX4D_LANES 16                       // the format: bands of 16 rows
#define CTX4D_NO_MAIN
#include "ctx4d_vec.cpp"
#include <stdexcept>
#include <string>
#include <filesystem>
#include <algorithm>
extern "C" {
#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
size_t FSE_buildCTableU16(FSE_CTable* ct, const short* normalizedCounter, unsigned maxSymbolValue, unsigned tableLog);
}

static inline uint32_t bw(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
struct Out { std::vector<uint8_t> b; void u8(uint8_t v) { b.push_back(v); } void var(uint64_t v) { while (v >= 128) { b.push_back(uint8_t(v | 128)); v >>= 7; } b.push_back(uint8_t(v)); } void raw(const void* p, size_t n) { b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n); } };
struct In { const uint8_t* p; const uint8_t* e;
    uint8_t u8() { if (p >= e) throw std::runtime_error("truncated"); return *p++; }
    uint64_t var() { uint64_t v = 0; int s = 0; for (;;) { uint8_t c = u8(); v |= uint64_t(c & 127) << s; if (!(c & 128)) return v; s += 7; } }
    const uint8_t* take(size_t n) { if (size_t(e - p) < n) throw std::runtime_error("truncated"); const uint8_t* r = p; p += n; return r; } };

// ---------------- FSE4 (copied from hlc.cpp: 4 interleaved states, one bitstream, own decoder) ----------------
static size_t fse4_enc(uint8_t* dst, size_t cap, const uint16_t* s, size_t n, unsigned mx, const unsigned* cnt) {
    const unsigned tcap = std::min(12u, std::max(9u, bw(uint32_t(n)) - 1 - 4));
    const unsigned tl = FSE_optimalTableLog(tcap, n, mx); short norm[1024];
    if (FSE_isError(FSE_normalizeCount(norm, tl, cnt, n, mx))) return 0;
    size_t h = FSE_writeNCount(dst, cap, norm, mx, tl); if (FSE_isError(h)) return 0;
    static std::vector<FSE_CTable> ct(FSE_CTABLE_SIZE_U32(13, 1023));
    if (FSE_isError(FSE_buildCTableU16(ct.data(), norm, mx, tl))) return 0;
    BIT_CStream_t bc; if (FSE_isError(BIT_initCStream(&bc, dst + h, cap - h))) return 0;
    FSE_CState_t st[4]; const size_t L = std::min<size_t>(n, 4); size_t i = n;
    for (size_t j = 0; j < L; ++j) { --i; FSE_initCState2(&st[i & 3], ct.data(), s[i]); }
    while (i > 0 && (i & 3)) { --i; FSE_encodeSymbol(&bc, &st[i & 3], s[i]); if ((i & 3) == 0) BIT_flushBits(&bc); }
    if (i >= 4) {
        const FSE_symbolCompressionTransform* const tt = (const FSE_symbolCompressionTransform*)st[0].symbolTT;
        const U16* const stt = (const U16*)st[0].stateTable;
        uint32_t v0 = uint32_t(st[0].value), v1 = uint32_t(st[1].value), v2 = uint32_t(st[2].value), v3 = uint32_t(st[3].value);
        size_t box = bc.bitContainer; unsigned bp = bc.bitPos; char* bptr = bc.ptr; char* const bend = bc.endPtr; const unsigned* const msk = BIT_mask;
#define C4_ENC1(v, sym) { const FSE_symbolCompressionTransform T = tt[sym]; const uint32_t nb = (v + T.deltaNbBits) >> 16; \
                          box |= size_t(v & msk[nb]) << bp; bp += nb; v = stt[uint32_t((v >> nb) + uint32_t(T.deltaFindState))]; }
        for (; i >= 4; i -= 4) {
            C4_ENC1(v3, s[i - 1]) C4_ENC1(v2, s[i - 2]) C4_ENC1(v1, s[i - 3]) C4_ENC1(v0, s[i - 4])
            memcpy(bptr, &box, sizeof box); const unsigned nbB = bp >> 3; bptr += nbB; box >>= nbB * 8; bp &= 7;
        }
#undef C4_ENC1
        if (bptr > bend) return 0;
        bc.bitContainer = box; bc.bitPos = bp; bc.ptr = bptr;
        st[0].value = ptrdiff_t(v0); st[1].value = ptrdiff_t(v1); st[2].value = ptrdiff_t(v2); st[3].value = ptrdiff_t(v3);
    }
    while (i > 0) { --i; FSE_encodeSymbol(&bc, &st[i & 3], s[i]); if ((i & 3) == 0) BIT_flushBits(&bc); }
    for (size_t k = L; k-- > 0;) FSE_flushCState(&bc, &st[k]);
    size_t c = BIT_closeCStream(&bc); return c ? h + c : 0;
}
static void fse4_dec(uint16_t* out, size_t n, const uint8_t* src, size_t c) {
    short norm[1024]; unsigned mx = 1023, tl;
    size_t h = FSE_readNCount(norm, &mx, &tl, src, c); if (FSE_isError(h) || tl > 13 || tl < 1) throw std::runtime_error("ncount");
    static std::vector<uint32_t> T(1 << 13); static std::vector<uint16_t> nxt(1024);
    const uint32_t ts = 1u << tl, mask = ts - 1, step = (ts >> 1) + (ts >> 3) + 3; uint32_t hi = ts - 1;
    for (unsigned s = 0; s <= mx; ++s) { if (norm[s] == -1) { T[hi--] = s; nxt[s] = 1; } else nxt[s] = uint16_t(norm[s]); }
    uint32_t pos = 0;
    for (unsigned s = 0; s <= mx; ++s) for (int k = 0; k < norm[s]; ++k) { T[pos] = s; pos = (pos + step) & mask; while (pos > hi) pos = (pos + step) & mask; }
    if (pos) throw std::runtime_error("ncount spread");
    for (uint32_t u = 0; u < ts; ++u) { uint32_t s = T[u], ns = nxt[s]++, nb = tl - bw(ns) + 1; T[u] = (((ns << nb) - ts) & 0xffff) | nb << 16 | s << 20; }
    BIT_DStream_t bd; if (FSE_isError(BIT_initDStream(&bd, src + h, c - h))) throw std::runtime_error("bitstream");
    const size_t L = std::min<size_t>(n, 4); uint32_t st[4] = {0, 0, 0, 0};
    for (size_t k = 0; k < L; ++k) { st[k] = uint32_t(BIT_readBits(&bd, tl)); BIT_reloadDStream(&bd); }
    const uint32_t* t = T.data(); size_t i = 0;
#define C4_DEC1(k) { uint32_t e = t[st[k]]; out[i + k] = uint16_t(e >> 20); st[k] = (e & 0xffff) + uint32_t(BIT_readBits(&bd, (e >> 16) & 15)); }
    for (; i + 4 <= n; i += 4) { BIT_reloadDStream(&bd); C4_DEC1(0) C4_DEC1(1) C4_DEC1(2) C4_DEC1(3) }
    BIT_reloadDStream(&bd); for (size_t k = 0; i < n; ++i, ++k) { uint32_t e = t[st[k]]; out[i] = uint16_t(e >> 20); st[k] = (e & 0xffff) + uint32_t(BIT_readBits(&bd, (e >> 16) & 15)); }
#undef C4_DEC1
}
static void put_stream(Out& o, const uint16_t* s, size_t n, std::vector<uint8_t>& tmp, unsigned A) {
    o.var(n); if (!n) return;
    static unsigned cnt[512], c4[4][512]; memset(c4, 0, sizeof c4); unsigned mx = 0; size_t i = 0;
    for (; i + 4 <= n; i += 4) { c4[0][s[i]]++; c4[1][s[i + 1]]++; c4[2][s[i + 2]]++; c4[3][s[i + 3]]++; }
    for (; i < n; ++i) c4[0][s[i]]++;
    for (unsigned v = 0; v < A; ++v) { cnt[v] = c4[0][v] + c4[1][v] + c4[2][v] + c4[3][v]; if (cnt[v]) mx = v; }
    if (cnt[s[0]] == n) { o.u8(1); o.var(s[0]); return; }
    tmp.resize(n * 2 + 4096); size_t c = fse4_enc(tmp.data(), tmp.size(), s, n, mx, cnt);
    if (!c || c >= n * 2) { o.u8(0); o.raw(s, n * 2); return; }
    o.u8(6); o.var(c); o.raw(tmp.data(), c);
}
static size_t get_stream(In& in, std::vector<uint16_t>& dst) {
    size_t n = in.var(); dst.resize(n + 2); if (!n) return 0;
    uint8_t mode = in.u8();
    if (mode == 1) { uint16_t v = uint16_t(in.var()); std::fill(dst.begin(), dst.begin() + n, v); }
    else if (mode == 0) memcpy(dst.data(), in.take(n * 2), n * 2);
    else if (mode == 6) { size_t c = in.var(); fse4_dec(dst.data(), n, in.take(c), c); }
    else throw std::runtime_error("stream mode");
    return n;
}

static bool g_serial = false;                 // -s: planes one after another inside a unit (no interleave)
static int bands_per_unit(int W) { int bpb = (W + 127) / 128; return std::max(1, (256 + bpb - 1) / bpb); }
struct Times { double rct = 0, ctx = 0, scat = 0, fse = 0, dfse = 0, dwave = 0, irct = 0; };
static double msnow() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<uint8_t> encode(const uint8_t* rgb, int W, int H, Times& tm) {
    size_t N = (size_t)W * H; std::vector<int16_t> P[3]; for (auto& p : P) p.resize(N);
    double t0 = msnow();
    for (size_t i = 0; i < N; i++) { int R = rgb[3 * i], Gc = rgb[3 * i + 1], B = rgb[3 * i + 2]; P[0][i] = (int16_t)((R + 2 * Gc + B) >> 2); P[1][i] = (int16_t)(B - Gc); P[2][i] = (int16_t)(R - Gc); }
    double t1 = msnow();
    std::vector<uint16_t> pk(N * 3);
    encode_plane_vec<8>(P[0].data(), pk.data(), W, H, 0); encode_plane_vec<7>(P[1].data(), pk.data() + N, W, H, 1); encode_plane_vec<7>(P[2].data(), pk.data() + 2 * N, W, H, 2);
    double t2 = msnow(); tm.rct += t1 - t0; tm.ctx += t2 - t1;
    Out o; o.raw("C4D1", 4); o.var(W); o.var(H);
    const int BU = bands_per_unit(W) * LN; std::vector<uint16_t> st[16]; std::vector<uint8_t> tmp;
    for (int u0 = 0; u0 < H; u0 += BU) {
        double s0 = msnow();
        for (auto& v : st) v.clear();
        auto put = [&](int k, int y0, int t) { const uint16_t* pl = pk.data() + k * N;
            for (int l = 0; l < LN; l++) { int x = t - 2 * l, y = y0 + l; if (x < 0 || x >= W || y >= H) continue;
                uint16_t p = pl[(size_t)y * W + x]; st[p & 15].push_back(p >> 4); } };
        if (g_serial) { for (int k = 0; k < 3; k++) for (int y0 = u0; y0 < std::min(H, u0 + BU); y0 += LN) for (int t = 0; t < W + 2 * (LN - 1); t++) put(k, y0, t); }
        else for (int y0 = u0; y0 < std::min(H, u0 + BU); y0 += LN) for (int t = 0; t < W + 2 * (LN - 1); t++) for (int k = 0; k < 3; k++) put(k, y0, t);
        double s1 = msnow();
        for (int s = 0; s < 16; s++) put_stream(o, st[s].data(), st[s].size(), tmp, s < 8 ? 256 : 512);
        tm.scat += s1 - s0; tm.fse += msnow() - s1;
    }
    return o.b;
}

static void decode(const uint8_t* buf, size_t len, std::vector<uint8_t>& rgb, int& W, int& H, Times& tm) {
    In in{buf, buf + len}; if (len < 4 || memcmp(in.take(4), "C4D1", 4)) throw std::runtime_error("not C4D1");
    W = (int)in.var(); H = (int)in.var(); size_t N = (size_t)W * H;
    std::vector<int16_t> D[3]; for (auto& d : D) d.resize(N);
    pdY.init(D[0].data(), W, H, 0, nullptr, nullptr, nullptr); pdU.init(D[1].data(), W, H, 1, nullptr, nullptr, nullptr); pdV.init(D[2].data(), W, H, 2, nullptr, nullptr, nullptr);
    std::vector<uint16_t> sb[16]; const uint16_t *SY[8], *SUV[8];
    pdY.Sp = SY; pdU.Sp = SUV; pdV.Sp = SUV;                                   // U and V share streams 8..15
    const int BU = bands_per_unit(W) * LN;
    for (int u0 = 0; u0 < H; u0 += BU) {
        double f0 = msnow();
        for (int s = 0; s < 16; s++) get_stream(in, sb[s]);
        for (int g = 0; g < 8; g++) { SY[g] = sb[g].data(); SUV[g] = sb[8 + g].data(); }
        double f1 = msnow();
        if (g_serial) {
            for (int y0 = u0; y0 < std::min(H, u0 + BU); y0 += LN) { pdY.begin(y0); for (int t = 0; t < pdY.T; t++) pdY.step(t); pdY.end(y0); }
            for (int y0 = u0; y0 < std::min(H, u0 + BU); y0 += LN) { pdU.begin(y0); for (int t = 0; t < pdU.T; t++) pdU.step(t); pdU.end(y0); }
            for (int y0 = u0; y0 < std::min(H, u0 + BU); y0 += LN) { pdV.begin(y0); for (int t = 0; t < pdV.T; t++) pdV.step(t); pdV.end(y0); }
        } else for (int y0 = u0; y0 < std::min(H, u0 + BU); y0 += LN) {
            pdY.begin(y0); pdU.begin(y0); pdV.begin(y0);
            for (int t = 0; t < pdY.T; t++) { pdY.step(t); pdU.step(t); pdV.step(t); }
            pdY.end(y0); pdU.end(y0); pdV.end(y0);
        }
        tm.dfse += f1 - f0; tm.dwave += msnow() - f1;
    }
    double r0 = msnow(); rgb.resize(N * 3);
    for (size_t i = 0; i < N; i++) { int Y = D[0][i], U = D[1][i], V2 = D[2][i], Gc = Y - ((U + V2) >> 2);
        rgb[3 * i] = (uint8_t)(V2 + Gc); rgb[3 * i + 1] = (uint8_t)Gc; rgb[3 * i + 2] = (uint8_t)(U + Gc); }
    tm.irct += msnow() - r0;
}

static bool read_ppm(const char* f, std::vector<uint8_t>& rgb, int& W, int& H) {
    FILE* fp = fopen(f, "rb"); int mx; if (!fp) return false;
    if (fscanf(fp, "P6 %d %d %d", &W, &H, &mx) != 3 || mx != 255) { fclose(fp); return false; }
    fgetc(fp); rgb.resize((size_t)W * H * 3); bool ok = fread(rgb.data(), 1, rgb.size(), fp) == rgb.size(); fclose(fp); return ok;
}
static std::vector<uint8_t> read_file(const char* f) { std::vector<uint8_t> b; FILE* fp = fopen(f, "rb"); if (!fp) return b;
    fseek(fp, 0, SEEK_END); b.resize(ftell(fp)); fseek(fp, 0, SEEK_SET); if (fread(b.data(), 1, b.size(), fp) != b.size()) b.clear(); fclose(fp); return b; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s c in.ppm out.c4d groups.csv | d in.c4d out.ppm groups.csv | bench groups.csv a.ppm|dir ... [-r N] [-max N] [-s]\n", argv[0]); return 1; }
    std::string cmd = argv[1];
    try {
        if (cmd == "c" && argc >= 5) {
            if (!load_groups(argv[4])) { fprintf(stderr, "bad groups CSV\n"); return 1; }
            std::vector<uint8_t> rgb; int W, H; if (!read_ppm(argv[2], rgb, W, H)) { fprintf(stderr, "need a binary P6 PPM (8 bit)\n"); return 1; }
            Times tm; auto b = encode(rgb.data(), W, H, tm); FILE* fp = fopen(argv[3], "wb"); fwrite(b.data(), 1, b.size(), fp); fclose(fp);
            printf("%dx%d -> %zu B, %.4f bpp\n", W, H, b.size(), 8.0 * b.size() / ((double)W * H)); return 0;
        }
        if (cmd == "d" && argc >= 5) {
            if (!load_groups(argv[4])) { fprintf(stderr, "bad groups CSV\n"); return 1; }
            auto b = read_file(argv[2]); std::vector<uint8_t> rgb; int W, H; Times tm; decode(b.data(), b.size(), rgb, W, H, tm);
            FILE* fp = fopen(argv[3], "wb"); fprintf(fp, "P6\n%d %d\n255\n", W, H); fwrite(rgb.data(), 1, rgb.size(), fp); fclose(fp); return 0;
        }
        if (cmd == "bench") {
            if (!load_groups(argv[2])) { fprintf(stderr, "bad groups CSV\n"); return 1; }
            int reps = 5, mx = 0; std::vector<std::string> names;
            for (int i = 3; i < argc; i++) {
                if (!strcmp(argv[i], "-r") && i + 1 < argc) { reps = atoi(argv[++i]); continue; }
                if (!strcmp(argv[i], "-max") && i + 1 < argc) { mx = atoi(argv[++i]); continue; }
                if (!strcmp(argv[i], "-s")) { g_serial = true; continue; }
                if (std::filesystem::is_directory(argv[i])) {                 // a folder: its *.ppm / *.pnm, sorted
                    std::vector<std::string> d;
                    for (auto& e : std::filesystem::directory_iterator(argv[i])) { auto x = e.path().extension().string();
                        if (x == ".ppm" || x == ".pnm" || x == ".PPM" || x == ".PNM") d.push_back(e.path().string()); }
                    std::sort(d.begin(), d.end()); names.insert(names.end(), d.begin(), d.end());
                } else names.push_back(argv[i]);
            }
            if (mx > 0 && (int)names.size() > mx) names.resize(mx);
            std::vector<const char*> files; for (auto& n : names) files.push_back(n.c_str());
            double tb = 0, tpx = 0, te = 0, td = 0; Times T; bool allok = true;
            for (auto f : files) {
                std::vector<uint8_t> rgb; int W, H; if (!read_ppm(f, rgb, W, H)) { printf("%s: skip\n", f); continue; }
                double be = 1e30, bd = 1e30; std::vector<uint8_t> b, out; Times bt, bdt;
                for (int r = 0; r < reps; r++) {
                    Times t1; double a = msnow(); b = encode(rgb.data(), W, H, t1); double m = msnow();
                    Times t2; int w2, h2; decode(b.data(), b.size(), out, w2, h2, t2); double e = msnow();
                    if (m - a < be) { be = m - a; bt = t1; } if (e - m < bd) { bd = e - m; bdt = t2; }
                }
                bool ok = out == rgb; allok &= ok; double px = (double)W * H;
                printf("%-60s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s %s\n", f, W, H, b.size(), 8.0 * b.size() / px, 3 * px / be / 1e3, 3 * px / bd / 1e3, ok ? "ok" : "MISMATCH");
                tb += b.size(); tpx += px; te += be; td += bd;
                T.rct += bt.rct; T.ctx += bt.ctx; T.scat += bt.scat; T.fse += bt.fse; T.dfse += bdt.dfse; T.dwave += bdt.dwave; T.irct += bdt.irct;
            }
            printf("TOTAL %zu files %.4f bpp  enc %.1f dec %.1f MB/s (image bytes)  %s\n", files.size(), 8.0 * tb / tpx, 3 * tpx / te / 1e3, 3 * tpx / td / 1e3, allok ? "all ok" : "MISMATCH");
            printf("ns/px  enc: RCT %.2f  context+symbol %.2f  wave scatter %.2f  FSE %.2f | dec: FSE %.2f  wave %.2f  inverse RCT %.2f\n",
                   T.rct * 1e6 / tpx, T.ctx * 1e6 / tpx, T.scat * 1e6 / tpx, T.fse * 1e6 / tpx, T.dfse * 1e6 / tpx, T.dwave * 1e6 / tpx, T.irct * 1e6 / tpx);
            return allok ? 0 : 2;
        }
    } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 3; }
    fprintf(stderr, "bad command\n"); return 1;
}
