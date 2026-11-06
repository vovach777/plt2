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
template <int SH>
static void dec_tiles_simd(int16_t* plane, int W, const int* tx, const int* ty, int nt, int k, const uint16_t** Sp) {
    const int t8 = k ? 1 : 0;
    const V e0 = splat(EDGE[t8][0] - 1), e1 = splat(EDGE[t8][1] - 1), e2 = splat(EDGE[t8][2] - 1), v0 = splat(VT[t8][0] - 1), v1 = splat(VT[t8][1] - 1);
    const V one = splat(1), c24 = splat(24), c9 = splat(9);
    const uint8_t* Gt = G[t8];
    alignas(64) V R[2][TW + 2], Zr[2][TW + 2];                                   // two rows (+ halo at -1 and TW)
    alignas(64) int16_t ca[16], za[16]; const int lanes = nt;
    for (int y = 0; y < TH; y++) {
        V* Pc = R[y & 1] + 1; V* Pu = R[(y & 1) ^ 1] + 1; V* Zc = Zr[y & 1] + 1; V* Zu = Zr[(y & 1) ^ 1] + 1;
        if (y == 0) { for (int x = -1; x <= TW; x++) { Pu[x] = splat(0); Zu[x] = splat(0); } Pc[-1] = splat(0); }
        else { Pu[-1] = Pu[0]; Pu[TW] = Pu[TW - 1]; Zu[-1] = splat(0); Zu[TW] = splat(0); Pc[-1] = Pu[0]; }   // x = 0: a = c = b, right: d = b
        for (int x = 0; x < TW; x++) {
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
            for (int x = 0; x < TW; x++) dst[x] = Pc[x][i];
        }
    }
}

struct Layout { std::vector<int> fx, fy, px, py, pw, ph; };                 // full tiles, partial tiles
static Layout layout(int W, int H) {
    Layout L;
    for (int y = 0; y < H; y += TH) for (int x = 0; x < W; x += TW) {
        int w = std::min(TW, W - x), h = std::min(TH, H - y);
        if (w == TW && h == TH) { L.fx.push_back(x); L.fy.push_back(y); } else { L.px.push_back(x); L.py.push_back(y); L.pw.push_back(w); L.ph.push_back(h); }
    }
    return L;
}

static std::vector<uint8_t> enc_tiles(const uint8_t* rgb, int W, int H, TT& tm) {
    const size_t N = (size_t)W * H; std::vector<int16_t> P[3]; for (auto& q : P) q.resize(N);
    for (size_t i = 0; i < N; i++) { int R = rgb[3 * i], Gc = rgb[3 * i + 1], B = rgb[3 * i + 2]; P[0][i] = (int16_t)((R + 2 * Gc + B) >> 2); P[1][i] = (int16_t)(B - Gc); P[2][i] = (int16_t)(R - Gc); }
    Layout L = layout(W, H); const int nf = (int)L.fx.size(), np = (int)L.px.size();
    // context per tile (tile = separate image) -> packed symbols per tile, raster
    double c0 = msnow();
    std::vector<std::vector<uint16_t>> pkf(nf * 3), pkp(np * 3); std::vector<int16_t> tb((size_t)TW * TH);
    auto enc_tile = [&](int x0, int y0, int w, int h, int k, std::vector<uint16_t>& out) {
        for (int y = 0; y < h; y++) memcpy(&tb[(size_t)y * w], &P[k][(size_t)(y0 + y) * W + x0], w * 2);
        out.resize((size_t)w * h);
        if (k) encode_plane_vec<7>(tb.data(), out.data(), w, h, k); else encode_plane_vec<8>(tb.data(), out.data(), w, h, k);
    };
    for (int i = 0; i < nf; i++) for (int k = 0; k < 3; k++) enc_tile(L.fx[i], L.fy[i], TW, TH, k, pkf[i * 3 + k]);
    for (int i = 0; i < np; i++) for (int k = 0; k < 3; k++) enc_tile(L.px[i], L.py[i], L.pw[i], L.ph[i], k, pkp[i * 3 + k]);
    tm.ctx += msnow() - c0;
    Out o; o.raw("C4T1", 4); o.var(W); o.var(H);
    static UBuf<uint16_t> sb[24], cat; for (auto& q : sb) q.need(CHUNK + 3 * TW * TH * TL + 64); cat.need(2 * (CHUNK + 3 * TW * TH * TL + 64));
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
    for (int g0 = 0; g0 < nf; g0 += TL) {                                    // groups of 16 full tiles
        double s0 = msnow(); int nl = std::min(TL, nf - g0);
        for (int k = 0; k < 3; k++) { uint16_t** w = wp + k * 8;
            for (int px = 0; px < TW * TH; px++) for (int l = 0; l < nl; l++) { uint16_t p = pkf[(g0 + l) * 3 + k][px]; *w[p & 7]++ = uint16_t(p >> 4); } }
        tm.scat += msnow() - s0; units++; if (full()) flush();
    }
    for (int i = 0; i < np; i++) {                                           // partial tiles
        double s0 = msnow();
        for (int k = 0; k < 3; k++) { uint16_t** w = wp + k * 8; for (uint16_t p : pkp[i * 3 + k]) *w[p & 7]++ = uint16_t(p >> 4); }
        tm.scat += msnow() - s0; units++; if (full()) flush();
    }
    if (units) flush();
    return o.b;
}

static void dec_tiles(const uint8_t* buf, size_t len, std::vector<uint8_t>& rgb, int& W, int& H, TT& tm) {
    In in{buf, buf + len}; if (len < 4 || memcmp(in.take(4), "C4T1", 4)) throw std::runtime_error("not C4T1");
    W = (int)in.var(); H = (int)in.var(); const size_t N = (size_t)W * H;
    std::vector<int16_t> D[3]; for (auto& d : D) d.resize(N);
    Layout L = layout(W, H); const int nf = (int)L.fx.size(), np = (int)L.px.size(), ng = (nf + TL - 1) / TL, nunits = ng + np;
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
                double s0 = msnow();
                if (j < ng) { int g0 = j * TL, nl = std::min(TL, nf - g0);
                    if (k) dec_tiles_simd<7>(D[k].data(), W, &L.fx[g0], &L.fy[g0], nl, k, Sp); else dec_tiles_simd<8>(D[k].data(), W, &L.fx[g0], &L.fy[g0], nl, k, Sp);
                    tm.dsimd += msnow() - s0; }
                else { int i = j - ng; dec_tile_scalar(D[k].data() + (size_t)L.py[i] * W + L.px[i], W, L.pw[i], L.ph[i], k, Sp); tm.dscal += msnow() - s0; }
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
            bool ok = out == rgb; allok &= ok; double px = (double)W * H; Layout L = layout(W, H); double full = (double)L.fx.size() * TW * TH;
            printf("%-12s %5dx%-5d %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s  full tiles %.0f%% %s\n", std::filesystem::path(f).filename().string().c_str(), W, H, b.size(), 8.0 * b.size() / px, 3 * px / be / 1e3, 3 * px / bd / 1e3, 100 * full / px, ok ? "ok" : "MISMATCH");
            tb += b.size(); tpx += px; te += be; td += bd; fpx += full;
            T.ctx += bt.ctx; T.scat += bt.scat; T.fse += bt.fse; T.dfse += bdt.dfse; T.dsimd += bdt.dsimd; T.dscal += bdt.dscal; T.out += bdt.out;
        }
    } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 3; }
    printf("TOTAL %zu files %.4f bpp  enc %.1f dec %.1f MB/s  %s\n", names.size(), 8.0 * tb / tpx, 3 * tpx / te / 1e3, 3 * tpx / td / 1e3, allok ? "all ok" : "MISMATCH");
    printf("ns/px enc: context %.2f scatter %.2f FSE %.2f | dec: FSE %.2f  SIMD tiles %.2f ns per full-tile px  scalar partial tiles %.2f ns per partial px  inverse RCT %.2f\n",
           T.ctx * 1e6 / tpx, T.scat * 1e6 / tpx, T.fse * 1e6 / tpx, T.dfse * 1e6 / tpx, T.dsimd * 1e6 / fpx, T.dscal * 1e6 / std::max(1.0, tpx - fpx), T.out * 1e6 / tpx);
    return allok ? 0 : 2;
}
