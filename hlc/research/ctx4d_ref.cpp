// ctx4d_ref.cpp - plain scalar reference of the planned HLC5 4-D context + symbol construction (no SIMD, no tricks).
// Purpose: an isolated, readable spec to vectorise (SSE/SSSE3 on Ivy Bridge, or clang vector types) and to check
// a vectorised version against. Raster order; the decoder wave / skew is NOT here (it only reorders the same symbols).
//
// Per plane (Y 8 bit, U/V 9 bit, RCT: Y = (R + 2G + B) >> 2, U = B - G, V = R - G), per pixel x with neighbours
//      c b d
//      a x
// borders: x = 0: a = c = b;  y = 0: b = c = d = 0 (and a = b at x = 0);  x = W-1: d = b.
//   ga = c - a, gb = c - b, gd = d - b                         three gradients
//   sector = (3 - q(ga)) * 7 + (q(gb) + 3)                     0..48, q = signed q7 with edges Y 1 3 10, U/V 1 3 7
//   dcell  = q(gd) + 3                                         0..6
//   v      = max(z[y-1][x-1], z[y-1][x], z[y-1][x+1])         final symbols of the row above (0 outside)
//   level  = (v >= t1) + (v >= t2)                             Y 8 32, U/V 3 8 (zigzag scale, CLIC calibration)
//   cell   = (sector * 7 + dcell) * 3 + level                  0..1028
//   group  = G[plane type][cell]                               8 groups (free map from llcomp_q7d --e3 --v max3)
//   stream = plane type * 8 + group                            Y 0..7, U/V 8..15
//   s      = sign(ga), else sign(gb), else +1                  tiny mirror sign
//   f      = clamp(ga + gb, min(ga, gb), max(ga, gb))          MED: p = c - f (identical to the classic MED)
//   e      = wrap(s * (x - c + f))                             wrap to 8 / 9 bits (signed)
//   z      = zigzag(e)                                         0..255 / 0..511
//   packed = (z << 4) | stream                                 13 bits: max() over packed values = max over z
// The decoder recomputes everything except z from already decoded data, reads z from stream[stream], inverts.
//
// build:  g++ -O2 -std=c++17 ctx4d_ref.cpp -o ctx4d_ref
// usage:  ctx4d_ref image.ppm groups.csv      (groups.csv = q7d_max3_groups.csv; rows with groups == 8 are used)
// prints per-stream counts and zero-order entropy, and checks the round trip (decoder rebuilds every pixel).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

static const int EDGE[2][3] = {{1, 3, 10}, {1, 3, 7}};   // q7 edges per plane type (0 = Y, 1 = U/V)
static const int VT[2][2]   = {{8, 32}, {3, 8}};         // error-level thresholds on v (zigzag scale)
static uint8_t G[2][1029];                               // cell -> group 0..7

static int q7(int x, int t) {                            // signed: sign(x) * #(|x| >= edge)
    int a = x < 0 ? -x : x, q = (a >= EDGE[t][0]) + (a >= EDGE[t][1]) + (a >= EDGE[t][2]);
    return x < 0 ? -q : q;
}
static int sgn(int x) { return (x > 0) - (x < 0); }
static int wrapv(int e, int nb) { int m = 1 << nb; e &= m - 1; return e >= m / 2 ? e - m : e; }
static int zz(int e) { return e >= 0 ? 2 * e : -2 * e - 1; }
static int unzz(int z) { return (z & 1) ? -((z + 1) >> 1) : (z >> 1); }

struct Ctx { int stream, s, c, f; };
// context of pixel (x, y) from the plane P (only already known pixels are read) and the symbol row above Zup
static Ctx context(const std::vector<int>& P, const std::vector<int>& Zup, int W, int x, int y, int t) {
    auto at = [&](int xx, int yy) { return P[(size_t)yy * W + xx]; };
    int b = y ? at(x, y - 1) : 0;
    int a = x ? at(x - 1, y) : b;
    int c = (x && y) ? at(x - 1, y - 1) : b;
    int d = y ? (x + 1 < W ? at(x + 1, y - 1) : b) : 0;
    int ga = c - a, gb = c - b, gd = d - b;
    int sector = (3 - q7(ga, t)) * 7 + (q7(gb, t) + 3), dcell = q7(gd, t) + 3;
    int v = 0;
    if (y) {
        int zl = x ? Zup[x - 1] : 0, zc = Zup[x], zr = x + 1 < W ? Zup[x + 1] : 0;
        v = zl > zc ? zl : zc; v = v > zr ? v : zr;
    }
    int level = (v >= VT[t][0]) + (v >= VT[t][1]);
    int cell = (sector * 7 + dcell) * 3 + level;
    int s = ga ? sgn(ga) : (gb ? sgn(gb) : 1);
    int lo = ga < gb ? ga : gb, hi = ga < gb ? gb : ga, f = ga + gb;
    f = f < lo ? lo : (f > hi ? hi : f);
    return {t * 8 + G[t][cell], s, c, f};
}

static bool load_groups(const char* path) {
    FILE* fp = fopen(path, "r"); if (!fp) return false;
    char line[512]; int n[2] = {0, 0};
    if (!fgets(line, sizeof line, fp)) return false;                       // header
    while (fgets(line, sizeof line, fp)) {
        char plane[8]; int t1, t2, groups, sector, dcell, level, group; long samples;
        if (sscanf(line, "%7[^,],%d,%d,%d,%d,%d,%d,%d,%ld", plane, &t1, &t2, &groups, &sector, &dcell, &level, &group, &samples) != 9) continue;
        if (groups != 8) continue;
        int t = plane[0] == 'Y' ? 0 : 1, cell = ((sector - 1) * 7 + dcell) * 3 + level;   // CSV sector is 1..49
        G[t][cell] = (uint8_t)group; n[t]++;
        if (t1 != VT[t][0] || t2 != VT[t][1]) fprintf(stderr, "note: CSV thresholds %d %d for %s differ from VT\n", t1, t2, plane);
    }
    fclose(fp); return n[0] == 1029 && n[1] == 1029;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s image.ppm groups.csv\n", argv[0]); return 1; }
    if (!load_groups(argv[2])) { fprintf(stderr, "bad groups CSV\n"); return 1; }
    FILE* fp = fopen(argv[1], "rb"); int W, H, mx;
    if (!fp || fscanf(fp, "P6 %d %d %d", &W, &H, &mx) != 3) { fprintf(stderr, "need a binary P6 PPM\n"); return 1; }
    fgetc(fp); std::vector<uint8_t> rgb((size_t)W * H * 3);
    if (fread(rgb.data(), 1, rgb.size(), fp) != rgb.size()) return 1;
    fclose(fp);

    std::vector<int> P[3];                                                 // Y, U, V
    for (auto& p : P) p.resize((size_t)W * H);
    for (size_t i = 0; i < (size_t)W * H; i++) {
        int R = rgb[3 * i], Gc = rgb[3 * i + 1], B = rgb[3 * i + 2];
        P[0][i] = (R + 2 * Gc + B) >> 2; P[1][i] = B - Gc; P[2][i] = R - Gc;
    }
    // ---- encoder: per plane, raster; symbols go to 16 streams (in raster order here; the wave only reorders)
    std::vector<uint16_t> stream[16];
    std::vector<uint16_t> packed((size_t)W * H * 3);                      // (z << 4) | stream, what a vector stage would emit
    for (int k = 0; k < 3; k++) {
        int t = k ? 1 : 0, nb = k ? 9 : 8;
        std::vector<int> Zup(W, 0), Zcur(W, 0);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                Ctx C = context(P[k], Zup, W, x, y, t);
                int e = wrapv(C.s * (P[k][(size_t)y * W + x] - C.c + C.f), nb), z = zz(e);
                Zcur[x] = z; stream[C.stream].push_back((uint16_t)z);
                packed[((size_t)k * H + y) * W + x] = (uint16_t)((z << 4) | C.stream);
            }
            std::swap(Zup, Zcur);
        }
    }
    double bits = 0; size_t total = 0;
    printf("stream  symbols    share   H(bits/sym)\n");
    for (int s = 0; s < 16; s++) {
        std::vector<size_t> h(512, 0); for (auto z : stream[s]) h[z]++;
        double hb = 0; for (auto c : h) if (c) hb -= c * std::log2((double)c / stream[s].size());
        bits += hb; total += stream[s].size();
        printf("%4d %10zu  %6.2f%%  %8.4f\n", s, stream[s].size(), 100.0 * stream[s].size() / ((size_t)W * H * 3), stream[s].size() ? hb / stream[s].size() : 0.0);
    }
    printf("total %.4f bpp (zero-order entropy of the 16 streams, no tables)\n", bits / ((double)W * H));
    // ---- decoder: rebuild the planes from the streams only (contexts from decoded data), compare
    size_t rd[16] = {0}; bool ok = true;
    for (int k = 0; k < 3 && ok; k++) {
        int t = k ? 1 : 0, nb = k ? 9 : 8;
        std::vector<int> Q((size_t)W * H, 0), Zup(W, 0), Zcur(W, 0);
        for (int y = 0; y < H && ok; y++) {
            for (int x = 0; x < W; x++) {
                Ctx C = context(Q, Zup, W, x, y, t);
                int z = stream[C.stream][rd[C.stream]++], e = C.s * unzz(z);  // s = +-1: s * e' = e
                int val = C.c - C.f + e;
                int m = 1 << nb; val = k ? wrapv(val, nb) : (val & (m - 1));   // Y 0..255, U/V -256..255 window
                Q[(size_t)y * W + x] = val; Zcur[x] = z;
                if (val != P[k][(size_t)y * W + x]) { printf("MISMATCH plane %d at %d,%d\n", k, x, y); ok = false; break; }
            }
            std::swap(Zup, Zcur);
        }
    }
    printf(ok ? "round trip OK\n" : "round trip FAILED\n");
    return ok ? 0 : 2;
}
