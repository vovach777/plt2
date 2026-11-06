// ctx4d_vec.cpp - the ctx4d encoder kernel and wavefront decoder written with GCC/Clang vector types only
// (vector_size(32) int16 x 16, plain operators, __builtin_shufflevector for the lane shift) - no intrinsics, no SIMDe.
// The compiler picks the ISA (-march): AVX2, SSE, NEON ... Same spec and checks as ctx4d_enc.cpp / ctx4d_dec.cpp
// (bit-exact against the scalar reference, decoder round trip). Needs gcc >= 12 or clang >= 10.
// usage:  ctx4d_vec image.ppm groups.csv [reps]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>

typedef int16_t  V  __attribute__((vector_size(32)));
typedef uint16_t VU __attribute__((vector_size(32)));
static const int EDGE[2][3] = {{1, 3, 10}, {1, 3, 7}};
static const int VT[2][2]   = {{8, 32}, {3, 8}};
static uint8_t G[2][1029];
static const int LN = 16;

static bool load_groups(const char* path) {
    FILE* fp = fopen(path, "r"); if (!fp) return false;
    char line[512]; int n[2] = {0, 0};
    if (!fgets(line, sizeof line, fp)) { fclose(fp); return false; }
    while (fgets(line, sizeof line, fp)) {
        char plane[8]; int t1, t2, groups, sector, dcell, level, group; long samples;
        if (sscanf(line, "%7[^,],%d,%d,%d,%d,%d,%d,%d,%ld", plane, &t1, &t2, &groups, &sector, &dcell, &level, &group, &samples) != 9) continue;
        if (groups != 8) continue;
        int t = plane[0] == 'Y' ? 0 : 1; G[t][((sector - 1) * 7 + dcell) * 3 + level] = (uint8_t)group; n[t]++;
    }
    fclose(fp); return n[0] == 1029 && n[1] == 1029;
}

// ---------------- scalar reference encoder + wave scatter (as ctx4d_dec.cpp) ----------------
static int q7s(int x, int t) { int a = x < 0 ? -x : x, q = (a >= EDGE[t][0]) + (a >= EDGE[t][1]) + (a >= EDGE[t][2]); return x < 0 ? -q : q; }
static void encode_plane_ref(const int16_t* P, uint16_t* out, int W, int H, int k) {
    int t = k ? 1 : 0, nb = k ? 9 : 8; std::vector<int> Zup(W, 0), Zc(W, 0);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            auto at = [&](int xx, int yy) { return (int)P[(size_t)yy * W + xx]; };
            int b = y ? at(x, y - 1) : 0, a = x ? at(x - 1, y) : b, c = (x && y) ? at(x - 1, y - 1) : b;
            int d = y ? (x + 1 < W ? at(x + 1, y - 1) : b) : 0, ga = c - a, gb = c - b, gd = d - b;
            int sector = (3 - q7s(ga, t)) * 7 + (q7s(gb, t) + 3), dcell = q7s(gd, t) + 3, v = 0;
            if (y) { v = std::max({x ? Zup[x - 1] : 0, Zup[x], x + 1 < W ? Zup[x + 1] : 0}); }
            int level = (v >= VT[t][0]) + (v >= VT[t][1]), cell = (sector * 7 + dcell) * 3 + level;
            int s = ga ? (ga > 0 ? 1 : -1) : (gb ? (gb > 0 ? 1 : -1) : 1);
            int f = std::clamp(ga + gb, std::min(ga, gb), std::max(ga, gb));
            int m = 1 << nb, e = (s * (at(x, y) - c + f)) & (m - 1); e = e >= m / 2 ? e - m : e;
            int z = e >= 0 ? 2 * e : -2 * e - 1; Zc[x] = z; out[(size_t)y * W + x] = (uint16_t)((z << 4) | (t * 8 + G[t][cell]));
        }
        std::swap(Zup, Zc);
    }
}
static void scatter_wave(const uint16_t* packed, int W, int H, std::vector<uint16_t>* st) {
    for (int y0 = 0; y0 < H; y0 += LN)
        for (int t = 0; t < W + 2 * (LN - 1); t++)
            for (int k = 0; k < LN; k++) {
                int x = t - 2 * k, y = y0 + k;
                if (x < 0 || x >= W || y >= H) continue;
                uint16_t p = packed[(size_t)y * W + x]; st[p & 15].push_back(p >> 4);
            }
}

// ---------------- vector-type helpers ----------------
static inline V ld(const int16_t* p) { V v; memcpy(&v, p, sizeof v); return v; }
static inline void st(int16_t* p, V v) { memcpy(p, &v, sizeof v); }
static inline V splat(int x) { return (V){} + (int16_t)x; }
#if defined(__clang__) && __has_builtin(__builtin_elementwise_max)          // clang >= 14: direct pabsw / pmaxsw / pminsw
static inline V vabs(V g) { return __builtin_elementwise_abs(g); }
static inline V vmax(V a, V b) { return __builtin_elementwise_max(a, b); }
static inline V vmin(V a, V b) { return __builtin_elementwise_min(a, b); }
#else
static inline V vabs(V g) { V m = g >> 15; return (g ^ m) - m; }
static inline V vmax(V a, V b) { V m = a > b; return (a & m) | (b & ~m); }
static inline V vmin(V a, V b) { V m = a < b; return (a & m) | (b & ~m); }
#endif
static inline V vsel(V m, V a, V b) { return (a & m) | (b & ~m); }              // m ? a : b (m lanes 0 / -1)
static inline V vnegif(V e, V s) { V m = s >> 15; return (e ^ m) - m; }        // s < 0 ? -e : e
static inline V q7v(V g, V e0, V e1, V e2) {                                     // e* = edge - 1
    V a = vabs(g), nq = (V)(a > e0) + (V)(a > e1) + (V)(a > e2);              // 0 .. -3
    return vnegif(-nq, g);
}
// lane k <- v[k-1], lane 0 <- ins
static inline V shl1(V v, int16_t ins) {
    V w = splat(ins);                                                          // only w[0] is used
    return __builtin_shufflevector(v, w, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
}

// ---------------- encoder kernel (raster, 16 px per step) ----------------
template <int SH>
static void encode_plane_vec(const int16_t* __restrict P, uint16_t* __restrict out, int W, int H, int k) {
    const int t = k ? 1 : 0;
    const V e0 = splat(EDGE[t][0] - 1), e1 = splat(EDGE[t][1] - 1), e2 = splat(EDGE[t][2] - 1), v0 = splat(VT[t][0] - 1), v1 = splat(VT[t][1] - 1);
    const V one = splat(1), c24 = splat(24), c9 = splat(9), sb = splat(t * 8);
    const uint8_t* Gt = G[t];
    const int L = 32, Wp = (W + 15) & ~15, stride = Wp + 2 * L;
    std::vector<int16_t> buf((size_t)stride * 4, 0);
    int16_t *Pu = buf.data() + L, *Pc = Pu + stride, *Zu = Pc + stride, *Zc = Zu + stride;
    for (int y = 0; y < H; y++) {
        memcpy(Pc, P + (size_t)y * W, W * sizeof(int16_t));
        if (y == 0) { memset(Pu - 1, 0, (Wp + 2) * 2); memset(Zu - 1, 0, (Wp + 2) * 2); Pc[-1] = 0; }
        else { Pc[-1] = Pu[0]; Pu[-1] = Pu[0]; Pu[W] = Pu[W - 1]; Zu[-1] = 0; Zu[W] = 0; }
        uint16_t* orow = out + (size_t)y * W;
        for (int x = 0; x < W; x += 16) {
            V vx = ld(Pc + x), va = ld(Pc + x - 1), vb = ld(Pu + x), vc = ld(Pu + x - 1), vd = ld(Pu + x + 1);
            V ga = vc - va, gb = vc - vb, gd = vd - vb;
            V qa = q7v(ga, e0, e1, e2), qb = q7v(gb, e0, e1, e2), qd = q7v(gd, e0, e1, e2);
            V sec = qb + c24 - ((qa << 3) - qa);
            V vv = vmax(vmax(ld(Zu + x - 1), ld(Zu + x)), ld(Zu + x + 1));
            V lev = -((V)(vv > v0) + (V)(vv > v1));
            V cell = (sec << 4) + (sec << 2) + sec + qd + qd + qd + lev + c9;
            alignas(32) int16_t ca[16], ga16[16];                                 // element loops through arrays, not v[i]
            st(ca, cell); for (int i = 0; i < 16; i++) ga16[i] = Gt[ca[i]];
            V grp = ld(ga16);
            V f = vmin(vmax(ga + gb, vmin(ga, gb)), vmax(ga, gb));
            V sgb = (V)(gb < 0) - (V)(gb > 0);                                      // sign(gb)
            V s = (ga + ga + sgb) | one;
            V e = vnegif(vx - vc + f, s);
            e = (V)(e << SH) >> SH;
            V z = (e + e) ^ (e >> 15);
            st(Zc + x, z);
            V pk = (z << 4) | (sb + grp);
            if (x + 16 <= W) memcpy(orow + x, &pk, 32); else memcpy(orow + x, &pk, (W - x) * 2);
        }
        std::swap(Pu, Pc); std::swap(Zu, Zc);
    }
}

// ---------------- wavefront decoder ----------------
template <int SH>
static void decode_plane_vec(int16_t* __restrict out, int W, int H, int k, const uint16_t** S) {
    const int t8 = k ? 1 : 0;
    const V e0 = splat(EDGE[t8][0] - 1), e1 = splat(EDGE[t8][1] - 1), e2 = splat(EDGE[t8][2] - 1), v0 = splat(VT[t8][0] - 1), v1 = splat(VT[t8][1] - 1);
    const V one = splat(1), c24 = splat(24), c9 = splat(9), zero = splat(0);
    const V lane2k = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}, lanek = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const V vW = splat(W), vWm1 = splat(W - 1), m1 = splat(-1);
    const uint8_t* Gt = G[t8];
    const int T = W + 2 * (LN - 1), PAD = 40;
    std::vector<int16_t> topP(W + 2 * PAD, 0), topZ(W + 2 * PAD, 0), wave((size_t)LN * T), wavez((size_t)LN * T);
    int16_t* tp = topP.data() + PAD; int16_t* tz = topZ.data() + PAD;
    for (int y0 = 0; y0 < H; y0 += LN) {
        const V lanes_ok = splat(std::min(LN, H - y0)) > lanek;
        V P1 = zero, Z1 = zero;
        V Dp = shl1(P1, tp[0]), Bp = shl1(P1, tp[-1]), Dz = shl1(Z1, tz[0]), Bz = shl1(Z1, tz[-1]);
        for (int t = 0; t < T; t++) {
            V d = shl1(P1, tp[t + 1]), b = Dp, c = Bp, zr = shl1(Z1, tz[t + 1]), zc = Dz, zl = Bz;
            Bp = b; Dp = d; Bz = zc; Dz = zr;
            V xv = splat(t) - lane2k;
            V x0 = xv == zero, xr = xv == vWm1, valid = (V)(xv > m1) & (V)(vW > xv) & lanes_ok;
            V a = vsel(x0, b, P1); c = vsel(x0, b, c); d = vsel(xr, b, d); zl = zl & ~x0; zr = zr & ~xr;
            V ga = c - a, gb = c - b, gd = d - b;
            V qa = q7v(ga, e0, e1, e2), qb = q7v(gb, e0, e1, e2), qd = q7v(gd, e0, e1, e2);
            V sec = qb + c24 - ((qa << 3) - qa);
            V lev = -((V)(vmax(vmax(zl, zc), zr) > v0) + (V)(vmax(vmax(zl, zc), zr) > v1));
            V cell = ((sec << 4) + (sec << 2) + sec + qd + qd + qd + lev + c9) & valid;
            alignas(32) int16_t ca[16], va16[16], za[16];
            st(ca, cell); st(va16, valid);
            for (int i = 0; i < 16; i++) za[i] = va16[i] ? (int16_t)*S[t8 * 8 + Gt[ca[i]]]++ : 0;
            V z = ld(za);
            V f = vmin(vmax(ga + gb, vmin(ga, gb)), vmax(ga, gb));
            V sgb = (V)(gb < 0) - (V)(gb > 0), s = (ga + ga + sgb) | one;
            V e = (V)((VU)z >> 1) ^ -(z & one);
            V val = c - f + vnegif(e, s);
            val = (V)(val << SH) >> SH;
            if (SH == 8) val &= splat(0xFF);
            st(&wave[(size_t)t * LN], val); st(&wavez[(size_t)t * LN], z);
            P1 = val; Z1 = z;
        }
        int rows = std::min(LN, H - y0);
        for (int kk = 0; kk < rows; kk++) {
            int16_t* row = out + (size_t)(y0 + kk) * W; const int16_t* src = &wave[2 * kk * LN + kk];
            for (int x = 0; x < W; x++) row[x] = src[(size_t)x * LN];
        }
        if (rows == LN) { const int16_t *sp = &wave[2 * (LN - 1) * LN + LN - 1], *sz = &wavez[2 * (LN - 1) * LN + LN - 1];
                          for (int x = 0; x < W; x++) { tp[x] = sp[(size_t)x * LN]; tz[x] = sz[(size_t)x * LN]; } }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s image.ppm groups.csv [reps]\n", argv[0]); return 1; }
    if (!load_groups(argv[2])) { fprintf(stderr, "bad groups CSV\n"); return 1; }
    int reps = argc > 3 ? atoi(argv[3]) : 10;
    FILE* fp = fopen(argv[1], "rb"); int W, H, mx;
    if (!fp || fscanf(fp, "P6 %d %d %d", &W, &H, &mx) != 3) { fprintf(stderr, "need a binary P6 PPM\n"); return 1; }
    fgetc(fp); size_t N = (size_t)W * H; std::vector<uint8_t> rgb(N * 3);
    if (fread(rgb.data(), 1, rgb.size(), fp) != rgb.size()) return 1;
    fclose(fp);
    std::vector<int16_t> P[3], D[3]; std::vector<uint16_t> ref(N * 3), vec(N * 3), stv[3][16];
    for (int k = 0; k < 3; k++) { P[k].resize(N); D[k].assign(N, -999); }
    for (size_t i = 0; i < N; i++) { int R = rgb[3 * i], Gc = rgb[3 * i + 1], B = rgb[3 * i + 2]; P[0][i] = (R + 2 * Gc + B) >> 2; P[1][i] = B - Gc; P[2][i] = R - Gc; }
    for (int k = 0; k < 3; k++) { encode_plane_ref(P[k].data(), ref.data() + k * N, W, H, k); scatter_wave(ref.data() + k * N, W, H, stv[k]); }
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    double be = 1e30, bd = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t0 = now();
        for (int k = 0; k < 3; k++) { if (k) encode_plane_vec<7>(P[k].data(), vec.data() + k * N, W, H, k); else encode_plane_vec<8>(P[k].data(), vec.data(), W, H, k); }
        auto t1 = now();
        for (int k = 0; k < 3; k++) { const uint16_t* S[16]; for (int s = 0; s < 16; s++) S[s] = stv[k][s].data();
                                      if (k) decode_plane_vec<7>(D[k].data(), W, H, k, S); else decode_plane_vec<8>(D[k].data(), W, H, k, S); }
        auto t2 = now(); be = std::min(be, ms(t0, t1)); bd = std::min(bd, ms(t1, t2));
    }
    size_t bad_e = 0, bad_d = 0;
    for (size_t i = 0; i < N * 3; i++) bad_e += ref[i] != vec[i];
    for (int k = 0; k < 3; k++) for (size_t i = 0; i < N; i++) bad_d += D[k][i] != P[k][i];
    printf("[vector types, %s] %dx%d: encoder %.2f ns/px (%s), decoder %.2f ns/px (%s)\n",
#if defined(__AVX2__)
           "AVX2",
#elif defined(__AVX__)
           "AVX1",
#elif defined(__SSE4_1__)
           "SSE4",
#elif defined(__ARM_NEON)
           "NEON",
#else
           "generic",
#endif
           W, H, be * 1e6 / N, bad_e ? "FAIL" : "bit-exact", bd * 1e6 / N, bad_d ? "FAIL" : "round trip OK");
    return bad_e || bad_d ? 2 : 0;
}
