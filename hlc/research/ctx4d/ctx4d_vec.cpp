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

#include <utility>
#if defined(__AVX512VBMI__)
#include <immintrin.h>
#endif
#ifndef CTX4D_LANES
  #if defined(__AVX512BW__)
    #define CTX4D_LANES 32                       // 512-bit vectors
  #else
    #define CTX4D_LANES 16                       // 256-bit vectors (2 x 128 on SSE / NEON)
  #endif
#endif
static const int NL = CTX4D_LANES;               // int16 lanes per vector = pixels per encoder step = rows per decoder band
typedef int16_t  V  __attribute__((vector_size(2 * CTX4D_LANES)));
typedef uint16_t VU __attribute__((vector_size(2 * CTX4D_LANES)));
static const int EDGE[2][3] = {{1, 3, 10}, {1, 3, 7}};
static const int VT[2][2]   = {{8, 32}, {3, 8}};
static uint8_t G[2][1029];
static const int LN = NL;

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
template <size_t... I> static inline V shl1_(V v, V w, std::index_sequence<I...>) { return __builtin_shufflevector(v, w, NL, (int)I...); }
static inline V shl1(V v, int16_t ins) { return shl1_(v, splat(ins), std::make_index_sequence<CTX4D_LANES - 1>{}); }
#if defined(__AVX512VBMI__) && CTX4D_LANES == 32
// group lookup in registers: 1029-byte table as 9 x 128 bytes, vpermi2b by the low 7 bits of the cell,
// the high bits (0..8) pick the table slice: 9 vpermi2b + 9 compares + 9 masked moves per 32 cells, no memory gather
struct GTab { __m512i a[9], b[9]; };
static inline void gtab_init(GTab& T, const uint8_t* g) {
    alignas(64) uint8_t buf[9 * 128] = {0}; memcpy(buf, g, 1029);
    for (int j = 0; j < 9; j++) { T.a[j] = _mm512_load_si512(buf + 128 * j); T.b[j] = _mm512_load_si512(buf + 128 * j + 64); }
}
static inline V glookup(const GTab& T, V cell) {
    __m512i c = (__m512i)cell;                                          // 32 x u16 cells
    __m512i lo = _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_and_si512(c, _mm512_set1_epi16(127))));
    __m512i hi = _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(c, 7)));
    __m512i r = _mm512_setzero_si512();
    for (int j = 0; j < 9; j++)
        r = _mm512_mask_mov_epi8(r, _mm512_cmpeq_epi8_mask(hi, _mm512_set1_epi8((char)j)), _mm512_permutex2var_epi8(T.a[j], lo, T.b[j]));
    return (V)_mm512_cvtepu8_epi16(_mm512_castsi512_si256(r));
}
#define CTX4D_VBMI 1
#if defined(__AVX512VPOPCNTDQ__)
#define CTX4D_VREAD 1
#endif
#endif
static inline V iota(int mul) { alignas(64) int16_t a[CTX4D_LANES]; for (int i = 0; i < NL; i++) a[i] = (int16_t)(i * mul); return ld(a); }

// ---------------- encoder kernel (raster, 16 px per step) ----------------
template <int SH>
static void encode_plane_vec(const int16_t* __restrict P, uint16_t* __restrict out, int W, int H, int k) {
    const int t = k ? 1 : 0;
    const V e0 = splat(EDGE[t][0] - 1), e1 = splat(EDGE[t][1] - 1), e2 = splat(EDGE[t][2] - 1), v0 = splat(VT[t][0] - 1), v1 = splat(VT[t][1] - 1);
    const V one = splat(1), c24 = splat(24), c9 = splat(9), sb = splat(t * 8);
    const uint8_t* Gt = G[t];
#ifdef CTX4D_VBMI
    GTab GT; gtab_init(GT, Gt);
#endif
    const int L = 2 * NL, Wp = (W + NL - 1) / NL * NL, stride = Wp + 2 * L;
    std::vector<int16_t> buf((size_t)stride * 4, 0);
    int16_t *Pu = buf.data() + L, *Pc = Pu + stride, *Zu = Pc + stride, *Zc = Zu + stride;
    for (int y = 0; y < H; y++) {
        memcpy(Pc, P + (size_t)y * W, W * sizeof(int16_t));
        if (y == 0) { memset(Pu - 1, 0, (Wp + 2) * 2); memset(Zu - 1, 0, (Wp + 2) * 2); Pc[-1] = 0; }
        else { Pc[-1] = Pu[0]; Pu[-1] = Pu[0]; Pu[W] = Pu[W - 1]; Zu[-1] = 0; Zu[W] = 0; }
        uint16_t* orow = out + (size_t)y * W;
        for (int x = 0; x < W; x += NL) {
            V vx = ld(Pc + x), va = ld(Pc + x - 1), vb = ld(Pu + x), vc = ld(Pu + x - 1), vd = ld(Pu + x + 1);
            V ga = vc - va, gb = vc - vb, gd = vd - vb;
            V qa = q7v(ga, e0, e1, e2), qb = q7v(gb, e0, e1, e2), qd = q7v(gd, e0, e1, e2);
            V sec = qb + c24 - ((qa << 3) - qa);
            V vv = vmax(vmax(ld(Zu + x - 1), ld(Zu + x)), ld(Zu + x + 1));
            V lev = -((V)(vv > v0) + (V)(vv > v1));
            V cell = (sec << 4) + (sec << 2) + sec + qd + qd + qd + lev + c9;
#ifdef CTX4D_VBMI
            V grp = glookup(GT, cell);
#else
            alignas(64) int16_t ca[CTX4D_LANES], ga16[CTX4D_LANES];               // element loops through arrays, not v[i]
            st(ca, cell); for (int i = 0; i < NL; i++) ga16[i] = Gt[ca[i]];
            V grp = ld(ga16);
#endif
            V f = vmin(vmax(ga + gb, vmin(ga, gb)), vmax(ga, gb));
            V sgb = (V)(gb < 0) - (V)(gb > 0);                                      // sign(gb)
            V s = (ga + ga + sgb) | one;
            V e = vnegif(vx - vc + f, s);
            e = (V)(e << SH) >> SH;
            V z = (e + e) ^ (e >> 15);
            st(Zc + x, z);
            V pk = (z << 4) | (sb + grp);
            if (x + NL <= W) memcpy(orow + x, &pk, sizeof pk); else memcpy(orow + x, &pk, (W - x) * 2);
        }
        std::swap(Pu, Pc); std::swap(Zu, Zc);
    }
}

// ---------------- wavefront decoder ----------------
// state of one plane; step(t) advances it by one wave step. The driver runs one plane at a time (serial) or
// the three planes inside one t loop (three independent dependency chains for the out-of-order core).
template <int SH>
struct PlaneDec {
    int W, H, k, t8, T;
    V e0, e1, e2, v0, v1, one, c24, c9, zero, lane2k, lanek, vW, vWm1, m1, lanes_ok, P1, Z1, Dp, Bp, Dz, Bz;
    const uint8_t* Gt; const uint16_t* S8[8]; int16_t* out;
#ifdef CTX4D_VBMI
    GTab GT;
#endif
#if defined(CTX4D_VREAD)
    const void* sbuf; alignas(64) int32_t off[16]; __m512i PRE0, PRE1;
#endif
    std::vector<int16_t> topP, topZ, wave, wavez; int16_t *tp, *tz;
    void init(int16_t* o, int W_, int H_, int k_, const uint16_t** S, const uint16_t* sbuf_in, const int32_t* off_in) {
        out = o; W = W_; H = H_; k = k_; t8 = k ? 1 : 0; T = W + 2 * (LN - 1);
        e0 = splat(EDGE[t8][0] - 1); e1 = splat(EDGE[t8][1] - 1); e2 = splat(EDGE[t8][2] - 1); v0 = splat(VT[t8][0] - 1); v1 = splat(VT[t8][1] - 1);
        one = splat(1); c24 = splat(24); c9 = splat(9); zero = splat(0); lane2k = iota(2); lanek = iota(1); vW = splat(W); vWm1 = splat(W - 1); m1 = splat(-1);
        Gt = G[t8]; for (int g = 0; g < 8; g++) S8[g] = S[t8 * 8 + g];
#ifdef CTX4D_VBMI
        gtab_init(GT, Gt);
#endif
#if defined(CTX4D_VREAD)
        sbuf = sbuf_in; memset(off, 0, sizeof off); memcpy(off, off_in, 8 * sizeof(int32_t));
        alignas(64) uint32_t pre[32]; for (int i = 0; i < 32; i++) pre[i] = i ? (uint32_t)((1ull << i) - 1) : 0u;
        PRE0 = _mm512_load_si512(pre); PRE1 = _mm512_load_si512(pre + 16);
#else
        (void)sbuf_in; (void)off_in;
#endif
        const int PAD = 2 * LN + 8;
        topP.assign(W + 2 * PAD, 0); topZ.assign(W + 2 * PAD, 0); wave.resize((size_t)LN * T); wavez.resize((size_t)LN * T);
        tp = topP.data() + PAD; tz = topZ.data() + PAD;
    }
    void begin(int y0) {
        lanes_ok = splat(std::min(LN, H - y0)) > lanek; P1 = zero; Z1 = zero;
        Dp = shl1(P1, tp[0]); Bp = shl1(P1, tp[-1]); Dz = shl1(Z1, tz[0]); Bz = shl1(Z1, tz[-1]);
    }
    __attribute__((always_inline)) inline void step(int t) {
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
#if defined(CTX4D_VREAD)
        V grp = glookup(GT, cell);
        __mmask32 kv = _mm512_cmpneq_epi16_mask((__m512i)valid, _mm512_setzero_si512());
        alignas(64) uint32_t km[16] = {0};
        for (int g = 0; g < 8; g++) km[g] = _mm512_mask_cmpeq_epi16_mask(kv, (__m512i)grp, _mm512_set1_epi16((short)g));
        __m512i KM = _mm512_load_si512(km), OFF = _mm512_load_si512(off);
        __m512i g0 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256((__m512i)grp)), g1 = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64((__m512i)grp, 1));
        __m512i r0 = _mm512_popcnt_epi32(_mm512_and_si512(_mm512_permutexvar_epi32(g0, KM), PRE0));
        __m512i r1 = _mm512_popcnt_epi32(_mm512_and_si512(_mm512_permutexvar_epi32(g1, KM), PRE1));
        __m512i a0 = _mm512_add_epi32(_mm512_permutexvar_epi32(g0, OFF), r0), a1 = _mm512_add_epi32(_mm512_permutexvar_epi32(g1, OFF), r1);
        __m512i s0 = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), (__mmask16)kv, a0, sbuf, 2);
        __m512i s1 = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), (__mmask16)(kv >> 16), a1, sbuf, 2);
        V z = (V)_mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(s0)), _mm512_cvtepi32_epi16(s1), 1);
        for (int g = 0; g < 8; g++) off[g] += __builtin_popcount(km[g]);
#else
        alignas(64) int16_t ca[CTX4D_LANES], va16[CTX4D_LANES], za[CTX4D_LANES];
        st(ca, cell); st(va16, valid);
        for (int i = 0; i < NL; i++) za[i] = va16[i] ? (int16_t)*S8[Gt[ca[i]]]++ : 0;
        V z = ld(za);
#endif
        V f = vmin(vmax(ga + gb, vmin(ga, gb)), vmax(ga, gb));
        V sgb = (V)(gb < 0) - (V)(gb > 0), s = (ga + ga + sgb) | one;
        V e = (V)((VU)z >> 1) ^ -(z & one);
        V val = c - f + vnegif(e, s);
        val = (V)(val << SH) >> SH;
        if (SH == 8) val &= splat(0xFF);
        st(&wave[(size_t)t * LN], val); st(&wavez[(size_t)t * LN], z);
        P1 = val; Z1 = z;
    }
    void end(int y0) {
        int rows = std::min(LN, H - y0);
#ifndef NO_UNSKEW
        for (int kk = 0; kk < rows; kk++) {
            int16_t* row = out + (size_t)(y0 + kk) * W; const int16_t* src = &wave[2 * kk * LN + kk];
            for (int x = 0; x < W; x++) row[x] = src[(size_t)x * LN];
        }
#endif
        if (rows == LN) { const int16_t *sp = &wave[2 * (LN - 1) * LN + LN - 1], *sz = &wavez[2 * (LN - 1) * LN + LN - 1];
                          for (int x = 0; x < W; x++) { tp[x] = sp[(size_t)x * LN]; tz[x] = sz[(size_t)x * LN]; } }
    }
};
struct DecIn { int16_t* out; const uint16_t* S[16]; const uint16_t* sbuf; const int32_t* off; };
static PlaneDec<8> pdY; static PlaneDec<7> pdU, pdV;
static void decode_serial(DecIn* in, int W, int H) {                          // one plane after another
    pdY.init(in[0].out, W, H, 0, in[0].S, in[0].sbuf, in[0].off);
    for (int y0 = 0; y0 < H; y0 += LN) { pdY.begin(y0); for (int t = 0; t < pdY.T; t++) pdY.step(t); pdY.end(y0); }
    pdU.init(in[1].out, W, H, 1, in[1].S, in[1].sbuf, in[1].off);
    for (int y0 = 0; y0 < H; y0 += LN) { pdU.begin(y0); for (int t = 0; t < pdU.T; t++) pdU.step(t); pdU.end(y0); }
    pdV.init(in[2].out, W, H, 2, in[2].S, in[2].sbuf, in[2].off);
    for (int y0 = 0; y0 < H; y0 += LN) { pdV.begin(y0); for (int t = 0; t < pdV.T; t++) pdV.step(t); pdV.end(y0); }
}
static void decode_interleaved(DecIn* in, int W, int H) {                     // Y, U, V steps inside one t loop
    pdY.init(in[0].out, W, H, 0, in[0].S, in[0].sbuf, in[0].off);
    pdU.init(in[1].out, W, H, 1, in[1].S, in[1].sbuf, in[1].off);
    pdV.init(in[2].out, W, H, 2, in[2].S, in[2].sbuf, in[2].off);
    for (int y0 = 0; y0 < H; y0 += LN) {
        pdY.begin(y0); pdU.begin(y0); pdV.begin(y0);
        for (int t = 0; t < pdY.T; t++) { pdY.step(t); pdU.step(t); pdV.step(t); }
        pdY.end(y0); pdU.end(y0); pdV.end(y0);
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
    // the 8 streams of each plane's type concatenated (for the vector read), start offsets, 2 elements padding
    std::vector<uint16_t> comb[3]; int32_t coff[3][8];
    for (int k = 0; k < 3; k++) { int t8 = k ? 1 : 0;
        for (int g = 0; g < 8; g++) { coff[k][g] = (int32_t)comb[k].size(); comb[k].insert(comb[k].end(), stv[k][t8 * 8 + g].begin(), stv[k][t8 * 8 + g].end()); }
        comb[k].resize(comb[k].size() + 2, 0); }
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    double be = 1e30, bd = 1e30, bi = 1e30;
    DecIn in[3];
    for (int k = 0; k < 3; k++) { in[k].out = D[k].data(); for (int s = 0; s < 16; s++) in[k].S[s] = stv[k][s].data(); in[k].sbuf = comb[k].data(); in[k].off = coff[k]; }
    for (int r = 0; r < reps; r++) {
        auto t0 = now();
        for (int k = 0; k < 3; k++) { if (k) encode_plane_vec<7>(P[k].data(), vec.data() + k * N, W, H, k); else encode_plane_vec<8>(P[k].data(), vec.data(), W, H, k); }
        auto t1 = now(); decode_serial(in, W, H);
        auto t2 = now(); decode_interleaved(in, W, H);
        auto t3 = now(); be = std::min(be, ms(t0, t1)); bd = std::min(bd, ms(t1, t2)); bi = std::min(bi, ms(t2, t3));
    }
    size_t bad_s = 0; decode_serial(in, W, H); for (int k = 0; k < 3; k++) for (size_t i = 0; i < N; i++) bad_s += D[k][i] != P[k][i];
    for (int k = 0; k < 3; k++) D[k].assign(N, -999);
    decode_interleaved(in, W, H);
    size_t bad_e = 0, bad_d = 0;
    for (size_t i = 0; i < N * 3; i++) bad_e += ref[i] != vec[i];
    for (int k = 0; k < 3; k++) for (size_t i = 0; i < N; i++) bad_d += D[k][i] != P[k][i];
    bad_d += bad_s;
    printf("[vector types x%d, %s] %dx%d: encoder %.2f ns/px (%s), decoder serial %.2f / 3 planes interleaved %.2f ns/px (%s)\n", NL,
#if defined(CTX4D_VREAD)
           "AVX-512 + VBMI group lookup + vector stream read",
#elif defined(CTX4D_VBMI)
           "AVX-512 + VBMI group lookup",
#elif defined(__AVX512BW__)
           "AVX-512",
#elif defined(__AVX2__)
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
           W, H, be * 1e6 / N, bad_e ? "FAIL" : "bit-exact", bd * 1e6 / N, bi * 1e6 / N, bad_d ? "FAIL" : "round trip OK");
    return bad_e || bad_d ? 2 : 0;
}
