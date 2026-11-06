// ctx4d_blk.cpp - experimental codec "C4B1": independent compute blocks (default 128 x 16), fused in L1 on both sides.
// Every block is coded as a separate image (the 4-D context of ctx4d_ref.cpp, block borders = image borders), so
// blocks share nothing but the FSE chunks. 16 blocks of the same shape run at once, one block per vector lane: all
// lanes sit at the same (x, y) of their block, a step needs no lane shifts, no masks, no wave.
//   encoder, per row y of a unit of 16 blocks: RGB of the 16 block rows -> RCT -> [x][lane] row buffers (two rows
//            per plane, L1) -> context + symbol for Y, U, V -> symbols straight into the 16 streams.
//   decoder, per row y: Y, U, V from the streams (same kernel backwards) -> inverse RCT straight into the RGB image.
//   Only FSE (histograms + coding of a chunk) is a separate pass. Working set per unit: 3 planes x 2 rows x
//   (w + 2) x 16 lanes x 2 B (~25 KB for w = 128) + the 16 RGB rows.
// Layout: blocks TW x TH, the last column / row absorbs the remainder (w < 2 TW, h < 2 TH; a small image is one
// block). Units = up to 16 blocks of one shape: inner, right column, bottom row, corner.
// Symbol order: unit, row y, plane Y U V, x, lane. Streams 0..7 Y, 8..15 shared by U and V (U row first).
// A chunk (one FSE run per stream) = whole units; flush after a unit when one stream reaches CHUNK symbols.
// Format "C4B1": var W, H, TW, TH; per chunk: var units, 16 streams (put_stream of ctx4d_codec).
// usage:  ctx4d_blk c in.ppm out.c4b groups.csv [-tw N] [-th N] [-cb N] | d in.c4b out.ppm groups.csv
//         ctx4d_blk bench groups.csv a.ppm|dir ... [-r N] [-max N] [-tw N] [-th N] [-cb N]
#define CTX4D_CODEC_NO_MAIN
#include "ctx4d_codec.cpp"

static const int NLB = CTX4D_LANES;              // blocks per unit (= lanes; the format needs 16)
static_assert(CTX4D_LANES == 16, "C4B1 is defined for 16 lanes");
static int TW = 128, TH = 16;
struct BT { double blk = 0, fse = 0, dfse = 0, dblk = 0; };

struct Unit { int w, h; std::vector<int> x, y; };
static std::vector<Unit> layout(int W, int H, int tw, int th) {
    const int nx = std::max(1, W / tw), ny = std::max(1, H / th), wl = W - tw * (nx - 1), hl = H - th * (ny - 1);
    std::vector<Unit> U;
    auto add = [&](int w, int h, int xa, int xb, int ya, int yb) {
        Unit u{w, h, {}, {}};
        for (int by = ya; by < yb; by++) for (int bx = xa; bx < xb; bx++) {
            u.x.push_back(bx * tw); u.y.push_back(by * th);
            if ((int)u.x.size() == NLB) { U.push_back(u); u.x.clear(); u.y.clear(); }
        }
        if (!u.x.empty()) U.push_back(u);
    };
    if (nx > 1 && ny > 1) add(tw, th, 0, nx - 1, 0, ny - 1);
    if (ny > 1) add(wl, th, nx - 1, nx, 0, ny - 1);
    if (nx > 1) add(tw, hl, 0, nx - 1, ny - 1, ny);
    add(wl, hl, nx - 1, nx, ny - 1, ny);
    return U;
}

// row buffers of one unit: plane k, row parity j; layout [x + 1][lane] (halo at x = -1 and x = w)
struct Rows {
    int w = 0; std::vector<int16_t> P[3][2], Z[3][2];
    void reset(int w_) { w = w_; const size_t n = (size_t)(w + 2) * NLB;
        for (int k = 0; k < 3; k++) for (int j = 0; j < 2; j++) { P[k][j].assign(n, 0); Z[k][j].assign(n, 0); } }
};

// one row of one plane; ENC: x known -> z out to the streams; else z from the streams -> x
template <int SH, bool ENC>
static inline void row_kernel(Rows& R, int k, int y, int nl, uint16_t** wp, const uint16_t** rp) {
    const int t8 = k ? 1 : 0, w = R.w, cur = y & 1;
    int16_t* Pc = R.P[k][cur].data(); int16_t* Pu = R.P[k][cur ^ 1].data();
    int16_t* Zc = R.Z[k][cur].data(); const int16_t* Zu = R.Z[k][cur ^ 1].data();
    const V e0 = splat(EDGE[t8][0] - 1), e1 = splat(EDGE[t8][1] - 1), e2 = splat(EDGE[t8][2] - 1);
    const V v0 = splat(VT[t8][0] - 1), v1 = splat(VT[t8][1] - 1), one = splat(1), c24 = splat(24), c9 = splat(9);
    const uint8_t* Gt = G[t8];
    if (y == 0) { memset(Pu, 0, (size_t)(w + 2) * NLB * 2); memset(R.Z[k][cur ^ 1].data(), 0, (size_t)(w + 2) * NLB * 2); memset(Pc, 0, NLB * 2); }
    else { memcpy(Pu, Pu + NLB, NLB * 2); memcpy(Pu + (size_t)(w + 1) * NLB, Pu + (size_t)w * NLB, NLB * 2); memcpy(Pc, Pu + NLB, NLB * 2); }
    alignas(64) int16_t ca[NLB], za[NLB] = {};
    for (int x = 0; x < w; x++) {
        const int16_t* pu = Pu + (size_t)x * NLB;                          // pu = c, pu + NLB = b, pu + 2 NLB = d
        V a = ld(Pc + (size_t)x * NLB), c = ld(pu), b = ld(pu + NLB), d = ld(pu + 2 * NLB);
        V ga = c - a, gb = c - b, gd = d - b;
        V qa = q7v(ga, e0, e1, e2), qb = q7v(gb, e0, e1, e2), qd = q7v(gd, e0, e1, e2);
        V sec = qb + c24 - ((qa << 3) - qa);
        const int16_t* zu = Zu + (size_t)x * NLB;
        V vv = vmax(vmax(ld(zu), ld(zu + NLB)), ld(zu + 2 * NLB));
        V lev = -((V)(vv > v0) + (V)(vv > v1));
        V cell = (sec << 4) + (sec << 2) + sec + qd + qd + qd + lev + c9;
        st(ca, cell);
        V f = vmin(vmax(ga + gb, vmin(ga, gb)), vmax(ga, gb));
        V sgb = (V)(gb < 0) - (V)(gb > 0), s = (ga + ga + sgb) | one;
        int16_t* px = Pc + (size_t)(x + 1) * NLB;
        if (ENC) {
            V e = vnegif(ld(px) - c + f, s);
            e = (V)(e << SH) >> SH;
            V z = (e + e) ^ (e >> 15);
            st(Zc + (size_t)(x + 1) * NLB, z); st(za, z);
            for (int i = 0; i < nl; i++) *wp[Gt[ca[i]]]++ = (uint16_t)za[i];
        } else {
            for (int i = 0; i < nl; i++) za[i] = (int16_t)*rp[Gt[ca[i]]]++;
            V z = ld(za);
            V e = (V)((VU)z >> 1) ^ -(z & one);
            V val = c - f + vnegif(e, s);
            val = (V)(val << SH) >> SH;
            if (SH == 8) val &= splat(0xFF);
            st(px, val); st(Zc + (size_t)(x + 1) * NLB, z);
        }
    }
}

#ifndef BLK_PALIC_RGB
#define BLK_PALIC_RGB 1                                                         // 0: own shufflevector deinterleave
#endif
#include "../../../palic-1/rgb.hpp"                                             // conv::convert_to_plannar_line_8to16_48 (its 16to8 is broken, unused)
// ---- RGB -> RCT -> [x][lane], vector types: per lane 48 B = 16 px deinterleaved (two shuffles per colour),
// RCT on 16 x i16, then a 16 x 16 transpose (4 butterfly rounds) -> 16 rows of [lane]. Tail (w % 16 px): scalar.
typedef uint8_t U8 __attribute__((vector_size(16)));
static inline U8 ldu8(const uint8_t* p) { U8 v; memcpy(&v, p, 16); return v; }
#define SV __builtin_shufflevector
static inline void deint(const uint8_t* s, V& Y, V& Uo, V& Vo) {
    U8 a0 = ldu8(s), a1 = ldu8(s + 16), a2 = ldu8(s + 32);
    U8 r = SV(SV(a0, a1, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0), a2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);
    U8 g = SV(SV(a0, a1, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0), a2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30);
    U8 b = SV(SV(a0, a1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0), a2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);
    V R = __builtin_convertvector(r, V), G = __builtin_convertvector(g, V), B = __builtin_convertvector(b, V);
    Y = (R + G + G + B) >> 2; Uo = B - G; Vo = R - G;
}
static inline void reint(uint8_t* d, V Y, V Uo, V Vo) {                     // inverse RCT, 16 px -> 48 B RGB
    V Gv = Y - ((Uo + Vo) >> 2), Rv = Vo + Gv, Bv = Uo + Gv;
    U8 r = __builtin_convertvector(Rv, U8), g = __builtin_convertvector(Gv, U8), b = __builtin_convertvector(Bv, U8);
    U8 o0 = SV(SV(r, g, 0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5), b, 0, 1, 16, 3, 4, 17, 6, 7, 18, 9, 10, 19, 12, 13, 20, 15);
    U8 o1 = SV(SV(r, g, 21, 0, 6, 22, 0, 7, 23, 0, 8, 24, 0, 9, 25, 0, 10, 26), b, 0, 21, 2, 3, 22, 5, 6, 23, 8, 9, 24, 11, 12, 25, 14, 15);
    U8 o2 = SV(SV(r, g, 0, 11, 27, 0, 12, 28, 0, 13, 29, 0, 14, 30, 0, 15, 31, 0), b, 26, 1, 2, 27, 4, 5, 28, 7, 8, 29, 10, 11, 30, 13, 14, 31);
    memcpy(d, &o0, 16); memcpy(d + 16, &o1, 16); memcpy(d + 32, &o2, 16);
}
static inline void tr16(V* m) {                                               // in-place 16 x 16 transpose
    for (int i = 0; i < 16; i++) if (!(i & 8)) { V a = m[i], b = m[i + 8];
        m[i] = SV(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23); m[i + 8] = SV(a, b, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31); }
    for (int i = 0; i < 16; i++) if (!(i & 4)) { V a = m[i], b = m[i + 4];
        m[i] = SV(a, b, 0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27); m[i + 4] = SV(a, b, 4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31); }
    for (int i = 0; i < 16; i++) if (!(i & 2)) { V a = m[i], b = m[i + 2];
        m[i] = SV(a, b, 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29); m[i + 2] = SV(a, b, 2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31); }
    for (int i = 0; i < 16; i += 2) { V a = m[i], b = m[i + 1];
        m[i] = SV(a, b, 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30); m[i + 1] = SV(a, b, 1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31); }
}
#undef SV
static void rct_rows(const uint8_t* const* src, int nl, int w, int16_t* PY, int16_t* PU, int16_t* PV) {
    V my[NLB], mu[NLB], mv[NLB];
    const int wv = w & ~15;
    for (int x0 = 0; x0 < wv; x0 += 16) {
#if BLK_PALIC_RGB                                                               // PALIC-1 rgb.hpp: V, U, Y planar
        for (int i = 0; i < nl; i++) { alignas(32) int16_t tv[16], tu[16], ty[16];
            conv::convert_to_plannar_line_8to16_48(src[i] + 3 * x0, tv, tu, ty); my[i] = ld(ty); mu[i] = ld(tu); mv[i] = ld(tv); }
#else
        for (int i = 0; i < nl; i++) deint(src[i] + 3 * x0, my[i], mu[i], mv[i]);
#endif
        for (int i = nl; i < NLB; i++) my[i] = mu[i] = mv[i] = splat(0);
        tr16(my); tr16(mu); tr16(mv);
        for (int j = 0; j < 16; j++) { const size_t o = (size_t)(x0 + j) * NLB; st(PY + o, my[j]); st(PU + o, mu[j]); st(PV + o, mv[j]); }
    }
    for (int i = 0; i < nl; i++) { const uint8_t* s = src[i];
        for (int x = wv; x < w; x++) { int r = s[3 * x], g = s[3 * x + 1], b = s[3 * x + 2]; size_t j = (size_t)x * NLB + i;
            PY[j] = (int16_t)((r + 2 * g + b) >> 2); PU[j] = (int16_t)(b - g); PV[j] = (int16_t)(r - g); } }
}

static std::vector<uint8_t> enc_blk(const uint8_t* rgb, int W, int H, BT& tm) {
    const std::vector<Unit> U = layout(W, H, TW, TH);
    Out o; o.raw("C4B1", 4); o.var(W); o.var(H); o.var(TW); o.var(TH);
    size_t umax = 0; for (auto& u : U) umax = std::max(umax, (size_t)u.w * u.h * u.x.size() * 2);
    static UBuf<uint16_t> sb[16]; const size_t cap = CHUNK + umax + 64; for (auto& q : sb) q.need(cap);
    uint16_t* wp[16]; for (int i = 0; i < 16; i++) wp[i] = sb[i].data();
    std::vector<uint8_t> tmp; long units = 0; Rows R;
    auto flush = [&]() {
        double f0 = msnow(); o.var((uint64_t)units);
        for (int g = 0; g < 16; g++) put_stream(o, sb[g].data(), wp[g] - sb[g].data(), g < 8 ? 256 : 512, tmp);
        for (int i = 0; i < 16; i++) wp[i] = sb[i].data(); units = 0; tm.fse += msnow() - f0;
    };
    auto full = [&]() { for (int g = 0; g < 16; g++) if (size_t(wp[g] - sb[g].data()) >= CHUNK) return true; return false; };
    for (auto& u : U) {
        double b0 = msnow(); const int nl = (int)u.x.size(), w = u.w; R.reset(w);
        for (int y = 0; y < u.h; y++) {
            const int cur = y & 1;
            int16_t* PY = R.P[0][cur].data() + NLB; int16_t* PU = R.P[1][cur].data() + NLB; int16_t* PV = R.P[2][cur].data() + NLB;
            const uint8_t* src[NLB];
            for (int i = 0; i < nl; i++) src[i] = rgb + ((size_t)(u.y[i] + y) * W + u.x[i]) * 3;
            rct_rows(src, nl, w, PY, PU, PV);
            row_kernel<8, true>(R, 0, y, nl, wp, nullptr);
            row_kernel<7, true>(R, 1, y, nl, wp + 8, nullptr);
            row_kernel<7, true>(R, 2, y, nl, wp + 8, nullptr);
        }
        tm.blk += msnow() - b0; units++; if (full()) flush();
    }
    if (units) flush();
    return o.b;
}

static void dec_blk(const uint8_t* buf, size_t len, std::vector<uint8_t>& rgb, int& W, int& H, BT& tm) {
    In in{buf, buf + len}; if (len < 4 || memcmp(in.take(4), "C4B1", 4)) throw std::runtime_error("not C4B1");
    W = (int)in.var(); H = (int)in.var(); const int tw = (int)in.var(), th = (int)in.var();
    if (W <= 0 || H <= 0 || tw <= 0 || th <= 0) throw std::runtime_error("bad header");
    rgb.resize((size_t)W * H * 3);
    const std::vector<Unit> U = layout(W, H, tw, th); const int nunits = (int)U.size();
    static UBuf<uint16_t> sb[16]; const uint16_t* rp[16]; Rows R; int u = 0;
    while (u < nunits) {
        double f0 = msnow(); int cn = (int)in.var(); if (cn <= 0 || u + cn > nunits) throw std::runtime_error("chunk units");
        for (int s = 0; s < 16; s++) { get_stream(in, sb[s]); rp[s] = sb[s].data(); }
        tm.dfse += msnow() - f0;
        double b0 = msnow();
        for (int j = u; j < u + cn; j++) {
            const Unit& q = U[j]; const int nl = (int)q.x.size(), w = q.w; R.reset(w);
            for (int y = 0; y < q.h; y++) {
                row_kernel<8, false>(R, 0, y, nl, nullptr, rp);
                row_kernel<7, false>(R, 1, y, nl, nullptr, rp + 8);
                row_kernel<7, false>(R, 2, y, nl, nullptr, rp + 8);
                const int cur = y & 1;
                const int16_t* PY = R.P[0][cur].data() + NLB; const int16_t* PU = R.P[1][cur].data() + NLB; const int16_t* PV = R.P[2][cur].data() + NLB;
                uint8_t* dl[NLB]; for (int i = 0; i < nl; i++) dl[i] = rgb.data() + ((size_t)(q.y[i] + y) * W + q.x[i]) * 3;
                const int wv = w & ~15;
                for (int x0 = 0; x0 < wv; x0 += 16) {                          // [x][lane] -> transpose -> 48 B RGB per lane
                    V my[NLB], mu[NLB], mv[NLB];
                    for (int j = 0; j < 16; j++) { const size_t o = (size_t)(x0 + j) * NLB; my[j] = ld(PY + o); mu[j] = ld(PU + o); mv[j] = ld(PV + o); }
                    tr16(my); tr16(mu); tr16(mv);
                    for (int i = 0; i < nl; i++) reint(dl[i] + 3 * x0, my[i], mu[i], mv[i]);
                }
                for (int i = 0; i < nl; i++) {                                 // inverse RCT straight into the image (tail)
                    uint8_t* d = dl[i];
                    for (int x = wv; x < w; x++) { size_t k = (size_t)x * NLB + i; int Y = PY[k], Uu = PU[k], Vv = PV[k], g = Y - ((Uu + Vv) >> 2);
                        d[3 * x] = (uint8_t)(Vv + g); d[3 * x + 1] = (uint8_t)g; d[3 * x + 2] = (uint8_t)(Uu + g); }
                }
            }
        }
        tm.dblk += msnow() - b0; u += cn;
    }
}

static bool opt(int argc, char** argv, int& i) {
    if (!strcmp(argv[i], "-tw") && i + 1 < argc) { TW = atoi(argv[++i]); return true; }
    if (!strcmp(argv[i], "-th") && i + 1 < argc) { TH = atoi(argv[++i]); return true; }
    if (!strcmp(argv[i], "-cb") && i + 1 < argc) { CHUNK = (size_t)atol(argv[++i]); return true; }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s c in.ppm out.c4b groups.csv [-tw N] [-th N] [-cb N] | d in.c4b out.ppm groups.csv | bench groups.csv a.ppm|dir ... [-r N] [-max N] [-tw N] [-th N] [-cb N]\n", argv[0]); return 1; }
    std::string cmd = argv[1];
    try {
        if (cmd == "c" && argc >= 5) {
            for (int i = 5; i < argc; i++) opt(argc, argv, i);
            if (TW < 1 || TH < 1) { fprintf(stderr, "bad block size\n"); return 1; }
            if (!load_groups(argv[4])) { fprintf(stderr, "bad groups CSV\n"); return 1; }
            std::vector<uint8_t> rgb; int W, H; if (!read_ppm(argv[2], rgb, W, H)) { fprintf(stderr, "need a binary P6 PPM (8 bit)\n"); return 1; }
            BT tm; auto b = enc_blk(rgb.data(), W, H, tm); FILE* fp = fopen(argv[3], "wb"); if (!fp) return 1; fwrite(b.data(), 1, b.size(), fp); fclose(fp);
            printf("%dx%d -> %zu B, %.4f bpp (blocks %dx%d)\n", W, H, b.size(), 8.0 * b.size() / ((double)W * H), TW, TH); return 0;
        }
        if (cmd == "d" && argc >= 5) {
            if (!load_groups(argv[4])) { fprintf(stderr, "bad groups CSV\n"); return 1; }
            auto b = read_file(argv[2]); if (b.empty()) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
            std::vector<uint8_t> rgb; int W, H; BT tm; dec_blk(b.data(), b.size(), rgb, W, H, tm);
            FILE* fp = fopen(argv[3], "wb"); if (!fp) return 1; fprintf(fp, "P6\n%d %d\n255\n", W, H); fwrite(rgb.data(), 1, rgb.size(), fp); fclose(fp); return 0;
        }
        if (cmd == "bench") {
            if (!load_groups(argv[2])) { fprintf(stderr, "bad groups CSV\n"); return 1; }
            int reps = 3, mx = 0; std::vector<std::string> names;
            for (int i = 3; i < argc; i++) {
                if (opt(argc, argv, i)) continue;
                if (!strcmp(argv[i], "-r") && i + 1 < argc) { reps = atoi(argv[++i]); continue; }
                if (!strcmp(argv[i], "-max") && i + 1 < argc) { mx = atoi(argv[++i]); continue; }
                if (std::filesystem::is_directory(argv[i])) { std::vector<std::string> d;
                    for (auto& e : std::filesystem::directory_iterator(argv[i])) { auto x = e.path().extension().string();
                        if (x == ".ppm" || x == ".pnm" || x == ".PPM" || x == ".PNM") d.push_back(e.path().string()); }
                    std::sort(d.begin(), d.end()); names.insert(names.end(), d.begin(), d.end()); } else names.push_back(argv[i]);
            }
            if (TW < 1 || TH < 1) { fprintf(stderr, "bad block size\n"); return 1; }
            if (mx > 0 && (int)names.size() > mx) names.resize(mx);
            double tb = 0, tpx = 0, te = 0, td = 0; BT T; bool allok = true; int nf = 0;
            for (auto& f : names) {
                std::vector<uint8_t> rgb; int W, H; if (!read_ppm(f.c_str(), rgb, W, H)) { printf("%s: skip\n", f.c_str()); continue; }
                double be = 1e30, bd = 1e30; std::vector<uint8_t> b, out; BT bt, bdt;
                for (int r = 0; r < reps; r++) {
                    BT t1; double a = msnow(); b = enc_blk(rgb.data(), W, H, t1); double m = msnow();
                    BT t2; int w2, h2; dec_blk(b.data(), b.size(), out, w2, h2, t2); double e = msnow();
                    if (m - a < be) { be = m - a; bt = t1; } if (e - m < bd) { bd = e - m; bdt = t2; }
                }
                bool ok = out == rgb; allok &= ok; double px = (double)W * H; nf++;
                printf("%-14s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s  %s\n", std::filesystem::path(f).filename().string().c_str(), W, H, b.size(), 8.0 * b.size() / px, 3 * px / be / 1e3, 3 * px / bd / 1e3, ok ? "ok" : "MISMATCH");
                tb += b.size(); tpx += px; te += be; td += bd;
                T.blk += bt.blk; T.fse += bt.fse; T.dfse += bdt.dfse; T.dblk += bdt.dblk;
            }
            printf("TOTAL %d files %.4f bpp  enc %.1f dec %.1f MB/s  blocks %dx%d  %s\n", nf, 8.0 * tb / tpx, 3 * tpx / te / 1e3, 3 * tpx / td / 1e3, TW, TH, allok ? "all ok" : "MISMATCH");
            printf("ns/px  enc: blocks (RGB+RCT+context+scatter) %.2f  FSE %.2f | dec: FSE %.2f  blocks (context+inverse RCT) %.2f\n",
                   T.blk * 1e6 / tpx, T.fse * 1e6 / tpx, T.dfse * 1e6 / tpx, T.dblk * 1e6 / tpx);
            return allok ? 0 : 2;
        }
    } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 3; }
    fprintf(stderr, "bad command\n"); return 1;
}
