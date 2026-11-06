// ctx4d_tiles.cpp - experiment: independent 128 x 64 tiles, no wavefront. Every tile is coded as a separate image
// (same 4-D context, borders of the tile = image borders). The decoder runs 16 tiles at once, one tile per lane, raster
// inside the tile: all lanes sit at the same (x, y), so a step needs no lane shifts and no border masks (halos),
// and the working set is two rows x 16 lanes per plane (~8 KB). Partial tiles (right / bottom edge) go scalar.
// Streams: 16 as C4D2 (Y 0..7, U/V 8..15 shared), symbol order = (group of 16 full tiles: plane, y, x, lane),
// then the partial tiles one by one (plane, y, x). Flush after a group / partial tile when one stream >= CHUNK.
// usage:  ctx4d_tiles bench groups.csv a.ppm|dir ... [-r N] [-cb N]
#define CTX4D_CODEC_NO_MAIN
#include "ctx4d_codec.cpp"

static const int TW = 128, TH = 64, TL = 16;     // tile size, tiles per vector
struct TT { double ctx = 0, scat = 0, fse = 0, dfse = 0, dsimd = 0, dscal = 0, out = 0; };

// ---------- scalar decoder of one tile (reference, for the partial tiles) ----------
static void dec_tile_scalar(int16_t* dst, int dstride, int w, int h, int k, const uint16_t** Sp) {
    int t = k ? 1 : 0, nb = k ? 9 : 8; std::vector<int> Zup(w, 0), Zc(w, 0); std::vector<int16_t> Q((size_t)w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            auto at = [&](int xx, int yy) { return (int)Q[(size_t)yy * w + xx]; };
            int b = y ? at(x, y - 1) : 0, a = x ? at(x - 1, y) : b, c = (x && y) ? at(x - 1, y - 1) : b;
            int d = y ? (x + 1 < w ? at(x + 1, y - 1) : b) : 0, ga = c - a, gb = c - b, gd = d - b;
            int sector = (3 - q7s(ga, t)) * 7 + (q7s(gb, t) + 3), dcell = q7s(gd, t) + 3, v = 0;
            if (y) v = std::max({x ? Zup[x - 1] : 0, Zup[x], x + 1 < w ? Zup[x + 1] : 0});
            int level = (v >= VT[t][0]) + (v >= VT[t][1]), cell = (sector * 7 + dcell) * 3 + level;
            int s = ga ? (ga > 0 ? 1 : -1) : (gb ? (gb > 0 ? 1 : -1) : 1);
            int f = std::clamp(ga + gb, std::min(ga, gb), std::max(ga, gb));
            int z = *Sp[G[t][cell]]++, e = (z & 1) ? -((z + 1) >> 1) : (z >> 1);
            int m = 1 << nb, val = (c - f + s * e) & (m - 1); if (k) val = val >= m / 2 ? val - m : val;
            Q[(size_t)y * w + x] = (int16_t)val; Zc[x] = z;
        }
        std::swap(Zup, Zc);
    }
    for (int y = 0; y < h; y++) memcpy(dst + (size_t)y * dstride, &Q[(size_t)y * w], w * 2);
}

// ---------- SIMD decoder: 16 full tiles, lane = tile ----------
static const int TWMAX = 2 * TW - 1, THMAX = 2 * TH - 1;                     // edge blocks absorb the remainder
template <int SH>
static void dec_tiles_simd(int16_t* plane, int W, int tw, int th, const int* tx, const int* ty, int nt, int k, const uint16_t** Sp) {
    const int t8 = k ? 1 : 0;
    const V e0 = splat(EDGE[t8][0] - 1), e1 = splat(EDGE[t8][1] - 1), e2 = splat(EDGE[t8][2] - 1), v0 = splat(VT[t8][0] - 1), v1 = splat(VT[t8][1] - 1);
    const V one = splat(1), c24 = splat(24), c9 = splat(9);
    const uint8_t* Gt = G[t8];
    static V R[2][TWMAX + 2], Zr[2][TWMAX + 2];                                // two rows (+ halo at -1 and tw)
    alignas(64) int16_t ca[16], za[16]; const int lanes = nt;
    for (int y = 0; y < th; y++) {
        V* Pc = R[y & 1] + 1; V* Pu = R[(y & 1) ^ 1] + 1; V* Zc = Zr[y & 1] + 1; V* Zu = Zr[(y & 1) ^ 1] + 1;
        if (y == 0) { for (int x = -1; x <= tw; x++) { Pu[x] = splat(0); Zu[x] = splat(0); } Pc[-1] = splat(0); }
        else { Pu[-1] = Pu[0]; Pu[tw] = Pu[tw - 1]; Zu[-1] = splat(0); Zu[tw] = splat(0); Pc[-1] = Pu[0]; }   // x = 0: a = c = b, right: d = b
        for (int x = 0; x < tw; x++) {
            V a = Pc[x - 1], b = Pu[x], c = Pu[x - 1], d = Pu[x + 1];
            V ga = c - a, gb = c - b, gd = d - b;
            V qa = q7v(ga, e0, e1, e2), qb = q7v(gb, e0, e1, e2), qd = q7v(gd, e0, e1, e2);
            V sec = qb + c24 - ((qa << 3) - qa);
            V vv = vmax(vmax(Zu[x - 1], Zu[x]), Zu[x + 1]);
            V lev = -((V)(vv > v0) + (V)(vv > v1));
            V cell = (sec << 4) + (sec << 2) + sec + qd + qd + qd + lev + c9;
            st(ca, cell);
            for (int i = 0; i < lanes; i++) za[i] = (int16_t)*Sp[Gt[ca[i]]]++;
            V z = ld(za);
            V f = vmin(vmax(ga + gb, vmin(ga, gb)), vmax(ga, gb));
            V sgb = (V)(gb < 0) - (V)(gb > 0), s = (ga + ga + sgb) | one;
            V e = (V)((VU)z >> 1) ^ -(z & one);
            V val = c - f + vnegif(e, s);
            val = (V)(val << SH) >> SH;
            if (SH == 8) val &= splat(0xFF);
            Pc[x] = val; Zc[x] = z;
        }
        for (int i = 0; i < lanes; i++) {                                     // row y of every tile -> plane
            int16_t* dst = plane + (size_t)(ty[i] + y) * W + tx[i];
            for (int x = 0; x < tw; x++) dst[x] = Pc[x][i];
        }
    }
}

// blocks: 128 x 64, the last column / row absorbs the remainder (width up to 255, height up to 127; a small image
// is one block). Units = groups of up to 16 blocks of the same shape: inner, right column, bottom row, corner.
struct Unit { int w, h; std::vector<int> x, y; };
static std::vector<Unit> layout(int W, int H) {
    const int nx = std::max(1, W / TW), ny = std::max(1, H / TH), wl = W - TW * (nx - 1), hl = H - TH * (ny - 1);
    std::vector<Unit> U;
    auto add = [&](int w, int h, int xa, int xb, int ya, int yb) {        // block columns [xa, xb), rows [ya, yb)
        Unit u{w, h, {}, {}};
        for (int by = ya; by < yb; by++) for (int bx = xa; bx < xb; bx++) {
            u.x.push_back(bx * TW); u.y.push_back(by * TH);
            if ((int)u.x.size() == TL) { U.push_back(u); u.x.clear(); u.y.clear(); }
        }
        if (!u.x.empty()) U.push_back(u);
    };
    if (nx > 1 && ny > 1) add(TW, TH, 0, nx - 1, 0, ny - 1);                  // inner
    if (ny > 1) add(wl, TH, nx - 1, nx, 0, ny - 1);                           // right column
    if (nx > 1) add(TW, hl, 0, nx - 1, ny - 1, ny);                           // bottom row
    add(wl, hl, nx - 1, nx, ny - 1, ny);                                      // corner
    return U;
}

static std::vector<uint8_t> enc_tiles(const uint8_t* rgb, int W, int H, TT& tm) {
    const size_t N = (size_t)W * H; std::vector<int16_t> P[3]; for (auto& q : P) q.resize(N);
    for (size_t i = 0; i < N; i++) { int R = rgb[3 * i], Gc = rgb[3 * i + 1], B = rgb[3 * i + 2]; P[0][i] = (int16_t)((R + 2 * Gc + B) >> 2); P[1][i] = (int16_t)(B - Gc); P[2][i] = (int16_t)(R - Gc); }
    const std::vector<Unit> U = layout(W, H);
    double c0 = msnow();
    std::vector<std::vector<uint16_t>> pk; std::vector<int16_t> tb((size_t)TWMAX * THMAX);
    for (auto& u : U) for (size_t l = 0; l < u.x.size(); l++) for (int k = 0; k < 3; k++) {
        for (int y = 0; y < u.h; y++) memcpy(&tb[(size_t)y * u.w], &P[k][(size_t)(u.y[l] + y) * W + u.x[l]], u.w * 2);
        pk.emplace_back((size_t)u.w * u.h);
        if (k) encode_plane_vec<7>(tb.data(), pk.back().data(), u.w, u.h, k); else encode_plane_vec<8>(tb.data(), pk.back().data(), u.w, u.h, k);
    }
    tm.ctx += msnow() - c0;
    Out o; o.raw("C4T1", 4); o.var(W); o.var(H);
    static UBuf<uint16_t> sb[24], cat; const size_t cap = CHUNK + (size_t)TWMAX * THMAX * TL + 64; for (auto& q : sb) q.need(cap); cat.need(2 * cap);
    uint16_t* wp[24]; for (int i = 0; i < 24; i++) wp[i] = sb[i].data();
    std::vector<uint8_t> tmp; long units = 0;
    auto flush = [&]() {
        double f0 = msnow(); o.var((uint64_t)units);
        for (int g = 0; g < 8; g++) put_stream(o, sb[g].data(), wp[g] - sb[g].data(), 256, tmp);
        for (int g = 0; g < 8; g++) { size_t nu = wp[8 + g] - sb[8 + g].data(), nv = wp[16 + g] - sb[16 + g].data();
            memcpy(cat.data(), sb[8 + g].data(), nu * 2); memcpy(cat.data() + nu, sb[16 + g].data(), nv * 2); put_stream(o, cat.data(), nu + nv, 512, tmp); }
        for (int i = 0; i < 24; i++) wp[i] = sb[i].data(); units = 0; tm.fse += msnow() - f0;
    };
    auto full = [&]() { for (int g = 0; g < 8; g++) if (size_t(wp[g] - sb[g].data()) >= CHUNK || size_t((wp[8 + g] - sb[8 + g].data()) + (wp[16 + g] - sb[16 + g].data())) >= CHUNK) return true; return false; };
    size_t base = 0;
    for (auto& u : U) {                                                       // unit: plane, then pixel (raster), then lane
        double s0 = msnow(); const int nl = (int)u.x.size(), npx = u.w * u.h;
        for (int k = 0; k < 3; k++) { uint16_t** w = wp + k * 8;
            for (int px = 0; px < npx; px++) for (int l = 0; l < nl; l++) { uint16_t p = pk[base + l * 3 + k][px]; *w[p & 7]++ = uint16_t(p >> 4); } }
        base += nl * 3; tm.scat += msnow() - s0; units++; if (full()) flush();
    }
    if (units) flush();
    return o.b;
}

static void dec_tiles(const uint8_t* buf, size_t len, std::vector<uint8_t>& rgb, int& W, int& H, TT& tm) {
    In in{buf, buf + len}; if (len < 4 || memcmp(in.take(4), "C4T1", 4)) throw std::runtime_error("not C4T1");
    W = (int)in.var(); H = (int)in.var(); const size_t N = (size_t)W * H;
    std::vector<int16_t> D[3]; for (auto& d : D) d.resize(N);
    const std::vector<Unit> U = layout(W, H); const int nunits = (int)U.size();
    static UBuf<uint16_t> sb[16]; const uint16_t *SY[8], *SUV[8]; int u = 0;
    // a chunk = cn units (groups of 16 full tiles, then partial tiles); inside it plane-major: Y of all its units,
    // U of all its units, V of all its units (U and V share streams 8..15, U symbols first)
    while (u < nunits) {
        double f0 = msnow(); int cn = (int)in.var(); if (cn <= 0 || u + cn > nunits) throw std::runtime_error("chunk units");
        for (int s = 0; s < 16; s++) get_stream(in, sb[s]);
        for (int g = 0; g < 8; g++) { SY[g] = sb[g].data(); SUV[g] = sb[8 + g].data(); }
        tm.dfse += msnow() - f0;
        for (int k = 0; k < 3; k++) {
            const uint16_t** Sp = k ? SUV : SY;
            for (int j = u; j < u + cn; j++) {
                double s0 = msnow(); const Unit& q = U[j];
                if (k) dec_tiles_simd<7>(D[k].data(), W, q.w, q.h, q.x.data(), q.y.data(), (int)q.x.size(), k, Sp);
                else dec_tiles_simd<8>(D[k].data(), W, q.w, q.h, q.x.data(), q.y.data(), (int)q.x.size(), k, Sp);
                tm.dsimd += msnow() - s0;
            }
        }
        u += cn;
    }
    double r0 = msnow(); rgb.resize(N * 3);
    for (size_t i = 0; i < N; i++) { int Y = D[0][i], U = D[1][i], V2 = D[2][i], Gc = Y - ((U + V2) >> 2); rgb[3 * i] = (uint8_t)(V2 + Gc); rgb[3 * i + 1] = (uint8_t)Gc; rgb[3 * i + 2] = (uint8_t)(U + Gc); }
    tm.out += msnow() - r0;
}

int main(int argc, char** argv) {
    if (argc < 3 || strcmp(argv[1], "bench")) { fprintf(stderr, "usage: %s bench groups.csv a.ppm|dir ... [-r N] [-cb N]\n", argv[0]); return 1; }
    if (!load_groups(argv[2])) { fprintf(stderr, "bad groups CSV\n"); return 1; }
    int reps = 3; std::vector<std::string> names;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "-r") && i + 1 < argc) { reps = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "-cb") && i + 1 < argc) { CHUNK = (size_t)atol(argv[++i]); continue; }
        if (std::filesystem::is_directory(argv[i])) { std::vector<std::string> d;
            for (auto& e : std::filesystem::directory_iterator(argv[i])) { auto x = e.path().extension().string(); if (x == ".ppm" || x == ".pnm") d.push_back(e.path().string()); }
            std::sort(d.begin(), d.end()); names.insert(names.end(), d.begin(), d.end()); } else names.push_back(argv[i]);
    }
    double tb = 0, tpx = 0, te = 0, td = 0, fpx = 0; TT T; bool allok = true;
    try {
        for (auto& f : names) {
            std::vector<uint8_t> rgb; int W, H; if (!read_ppm(f.c_str(), rgb, W, H)) { printf("%s: skip\n", f.c_str()); continue; }
            double be = 1e30, bd = 1e30; std::vector<uint8_t> b, out; TT bt, bdt;
            for (int r = 0; r < reps; r++) {
                TT t1; double a = msnow(); b = enc_tiles(rgb.data(), W, H, t1); double m = msnow();
                TT t2; int w2, h2; dec_tiles(b.data(), b.size(), out, w2, h2, t2); double e = msnow();
                if (m - a < be) { be = m - a; bt = t1; } if (e - m < bd) { bd = e - m; bdt = t2; }
            }
            bool ok = out == rgb; allok &= ok; double px = (double)W * H; double full = px;
            printf("%-12s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s  %s\n", std::filesystem::path(f).filename().string().c_str(), W, H, b.size(), 8.0 * b.size() / px, 3 * px / be / 1e3, 3 * px / bd / 1e3, ok ? "ok" : "MISMATCH");
            tb += b.size(); tpx += px; te += be; td += bd; fpx += full;
            T.ctx += bt.ctx; T.scat += bt.scat; T.fse += bt.fse; T.dfse += bdt.dfse; T.dsimd += bdt.dsimd; T.dscal += bdt.dscal; T.out += bdt.out;
        }
    } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 3; }
    printf("TOTAL %zu files %.4f bpp  enc %.1f dec %.1f MB/s  %s\n", names.size(), 8.0 * tb / tpx, 3 * tpx / te / 1e3, 3 * tpx / td / 1e3, allok ? "all ok" : "MISMATCH");
    printf("ns/px enc: context %.2f scatter %.2f FSE %.2f | dec: FSE %.2f  tiles (SIMD, all blocks) %.2f  inverse RCT %.2f\n",
           T.ctx * 1e6 / tpx, T.scat * 1e6 / tpx, T.fse * 1e6 / tpx, T.dfse * 1e6 / tpx, T.dsimd * 1e6 / fpx, T.out * 1e6 / tpx);
    return allok ? 0 : 2;
}
