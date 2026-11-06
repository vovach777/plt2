// ctx4d_codec.cpp - prototype codec: 4-D context (ctx4d_vec kernels, vector types) + FSE (hlc's 4-state FSE4).
// Encoder: RCT -> per plane context + symbol (raster, vector kernel) -> symbols into 16 streams in decoder order
//          (band of 16 rows, step t = x + 2k, plane Y U V, lane) -> FSE per unit -> file.
// Decoder: file -> per unit FSE into 16 stream buffers -> wavefront decode of Y, U, V (three chains in one t loop)
//          -> inverse RCT. Streams 0..7 Y, 8..15 shared by U and V.
// Unit (one FSE run per stream): whole bands, at least 256 compute blocks of 128 x 16 px (W = 2048: 16 bands).
// Format "C4D1": var W, var H, then per unit and stream: var n, payload (mode u8: 0 raw u16, 1 rle var, 6 FSE4 var c).
// usage:  ctx4d_codec c in.ppm out.c4d groups.csv | d in.c4d out.ppm groups.csv | bench groups.csv a.ppm|dir ... [-r N] [-max N] [-cb N] [-planes]
#define CTX4D_LANES 16                       // the format: bands of 16 rows
#define CTX4D_NO_MAIN
#include "ctx4d_vec.cpp"
#include <stdexcept>
#include <string>
#include <filesystem>
#include <memory>
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

struct Times { double rct = 0, ctx = 0, scat = 0, fse = 0, dfse = 0, dwave = 0, irct = 0, hist = 0, dstep = 0, dend = 0, dtab = 0; };
static Times* g_tm = nullptr;
static double msnow();
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
    double tb0 = msnow();
    short norm[1024]; unsigned mx = 1023, tl;
    size_t h = FSE_readNCount(norm, &mx, &tl, src, c); if (FSE_isError(h) || tl > 13 || tl < 1) throw std::runtime_error("ncount");
    static std::vector<uint32_t> T(1 << 13); static std::vector<uint16_t> nxt(1024);
    const uint32_t ts = 1u << tl, mask = ts - 1, step = (ts >> 1) + (ts >> 3) + 3; uint32_t hi = ts - 1;
    for (unsigned s = 0; s <= mx; ++s) { if (norm[s] == -1) { T[hi--] = s; nxt[s] = 1; } else nxt[s] = uint16_t(norm[s]); }
    uint32_t pos = 0;
    for (unsigned s = 0; s <= mx; ++s) for (int k = 0; k < norm[s]; ++k) { T[pos] = s; pos = (pos + step) & mask; while (pos > hi) pos = (pos + step) & mask; }
    if (pos) throw std::runtime_error("ncount spread");
    for (uint32_t u = 0; u < ts; ++u) { uint32_t s = T[u], ns = nxt[s]++, nb = tl - bw(ns) + 1; T[u] = (((ns << nb) - ts) & 0xffff) | nb << 16 | s << 20; }
    if (g_tm) g_tm->dtab += msnow() - tb0;
    BIT_DStream_t bd; if (FSE_isError(BIT_initDStream(&bd, src + h, c - h))) throw std::runtime_error("bitstream");
    const size_t L = std::min<size_t>(n, 4); uint32_t st[4] = {0, 0, 0, 0};
    for (size_t k = 0; k < L; ++k) { st[k] = uint32_t(BIT_readBits(&bd, tl)); BIT_reloadDStream(&bd); }
    const uint32_t* t = T.data(); size_t i = 0;
#define C4_DEC1(k) { uint32_t e = t[st[k]]; out[i + k] = uint16_t(e >> 20); st[k] = (e & 0xffff) + uint32_t(BIT_readBits(&bd, (e >> 16) & 15)); }
    for (; i + 4 <= n; i += 4) { BIT_reloadDStream(&bd); C4_DEC1(0) C4_DEC1(1) C4_DEC1(2) C4_DEC1(3) }
    BIT_reloadDStream(&bd); for (size_t k = 0; i < n; ++i, ++k) { uint32_t e = t[st[k]]; out[i] = uint16_t(e >> 20); st[k] = (e & 0xffff) + uint32_t(BIT_readBits(&bd, (e >> 16) & 15)); }
#undef C4_DEC1
}
template <class T> struct UBuf {                                             // uninitialised growable buffer
    std::unique_ptr<T[]> p; size_t n = 0;
    void need(size_t m) { if (m > n) { p.reset(new T[m]); n = m; } }
    T* data() { return p.get(); }
};
static void put_stream(Out& o, const uint16_t* s, size_t n, unsigned A, std::vector<uint8_t>& tmp) {
    o.var(n); if (!n) return;
    double h0 = msnow();
    static unsigned cnt[512], c4[4][512]; memset(c4, 0, sizeof c4); unsigned mx = 0; size_t i = 0;   // 4 sub-histograms
    for (; i + 4 <= n; i += 4) { c4[0][s[i]]++; c4[1][s[i + 1]]++; c4[2][s[i + 2]]++; c4[3][s[i + 3]]++; }
    for (; i < n; ++i) c4[0][s[i]]++;
    for (unsigned v = 0; v < A; ++v) { cnt[v] = c4[0][v] + c4[1][v] + c4[2][v] + c4[3][v]; if (cnt[v]) mx = v; }
    if (g_tm) g_tm->hist += msnow() - h0;
    if (cnt[s[0]] == n) { o.u8(1); o.var(s[0]); return; }
    tmp.resize(n * 2 + 4096); size_t c = fse4_enc(tmp.data(), tmp.size(), s, n, mx, cnt);
    if (!c || c >= n * 2) { o.u8(0); o.raw(s, n * 2); return; }
    o.u8(6); o.var(c); o.raw(tmp.data(), c);
}
static size_t get_stream(In& in, UBuf<uint16_t>& dst) {
    size_t n = in.var(); dst.need(n + 2); if (!n) return 0;
    uint8_t mode = in.u8();
    if (mode == 1) { uint16_t v = uint16_t(in.var()); std::fill(dst.data(), dst.data() + n, v); }
    else if (mode == 0) memcpy(dst.data(), in.take(n * 2), n * 2);
    else if (mode == 6) { size_t c = in.var(); fse4_dec(dst.data(), n, in.take(c), c); }
    else throw std::runtime_error("stream mode");
    return n;
}

// Chunking: the symbols of one compute block (128 x 16 px) are put into the streams; after the block, if one stream
// (Y g, or U g + V g: they share stream 8 + g) holds >= CHUNK symbols, every stream is flushed. The chunk boundary is a
// wave step: after block x0..x0+127 of a band all steps t < x0 + 128 are complete (pixel x = t - 2k <= x0 + 127), the
// last block of a band completes the band. Chunk header: var number of wave steps (global step = band * T + t).
// Inside a chunk the planes are not interleaved: Y, then U, then V (streams 8..15 = U symbols, then V symbols).
static size_t CHUNK = 65536; static const size_t SLACK = 4096;   // -cb N: flush threshold (symbols in one stream)   // a block range adds <= (128 + 2 LN) * 16 symbols per plane
static double msnow() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<uint8_t> encode_planes(const uint8_t* rgb, int W, int H, Times& tm) {
    g_tm = &tm;
    const size_t N = (size_t)W * H; static UBuf<int16_t> P[3]; static UBuf<uint16_t> pk; for (auto& q : P) q.need(N); pk.need(N * 3);
    double t0 = msnow();
    int16_t *P0 = P[0].data(), *P1 = P[1].data(), *P2 = P[2].data();
    for (size_t i = 0; i < N; i++) { int R = rgb[3 * i], Gc = rgb[3 * i + 1], B = rgb[3 * i + 2]; P0[i] = (int16_t)((R + 2 * Gc + B) >> 2); P1[i] = (int16_t)(B - Gc); P2[i] = (int16_t)(R - Gc); }
    double t1 = msnow();
    encode_plane_vec<8>(P0, pk.data(), W, H, 0); encode_plane_vec<7>(P1, pk.data() + N, W, H, 1); encode_plane_vec<7>(P2, pk.data() + 2 * N, W, H, 2);
    double t2 = msnow(); tm.rct += t1 - t0; tm.ctx += t2 - t1;
    Out o; o.raw("C4D2", 4); o.var(W); o.var(H);
    const int T = W + 2 * (LN - 1);
    static UBuf<uint16_t> sb[24], cat;                                       // staging: Y g, U g, V g
    for (auto& q : sb) q.need(CHUNK + SLACK); cat.need(2 * (CHUNK + SLACK));
    uint16_t* wp[24]; for (int i = 0; i < 24; i++) wp[i] = sb[i].data();
    std::vector<uint8_t> tmp; long steps = 0;
    auto flush = [&]() {
        double f0 = msnow();
        o.var((uint64_t)steps);
        for (int g = 0; g < 8; g++) put_stream(o, sb[g].data(), wp[g] - sb[g].data(), 256, tmp);
        for (int g = 0; g < 8; g++) {
            size_t nu = wp[8 + g] - sb[8 + g].data(), nv = wp[16 + g] - sb[16 + g].data();
            memcpy(cat.data(), sb[8 + g].data(), nu * 2); memcpy(cat.data() + nu, sb[16 + g].data(), nv * 2);
            put_stream(o, cat.data(), nu + nv, 512, tmp);
        }
        for (int i = 0; i < 24; i++) wp[i] = sb[i].data();
        steps = 0; tm.fse += msnow() - f0;
    };
    for (int y0 = 0; y0 < H; y0 += LN) {
        const int rows = std::min(LN, H - y0); int tdone = 0;
        for (int x0 = 0; x0 < W; x0 += 128) {
            const int tf = x0 + 128 >= W ? T : x0 + 128;
            double s0 = msnow();
            for (int k = 0; k < 3; k++) {
                const uint16_t* pl = pk.data() + k * N; uint16_t** w = wp + k * 8;
                const uint16_t* rp[LN]; for (int l = 0; l < LN; l++) rp[l] = pl + (size_t)(y0 + std::min(l, rows - 1)) * W - 2 * l;
                for (int t = tdone; t < tf; t++) {
                    if (rows == LN && t >= 2 * (LN - 1) && t < W) {           // interior: all 16 lanes inside the image
                        for (int l = 0; l < LN; l++) { uint16_t p = rp[l][t]; *w[p & 7]++ = uint16_t(p >> 4); }
                    } else {
                        for (int l = 0; l < rows; l++) { int x = t - 2 * l; if (x < 0 || x >= W) continue;
                            uint16_t p = rp[l][t]; *w[p & 7]++ = uint16_t(p >> 4); }
                    }
                }
            }
            steps += tf - tdone; tdone = tf; tm.scat += msnow() - s0;
            bool full = false;
            for (int g = 0; g < 8; g++) full |= size_t(wp[g] - sb[g].data()) >= CHUNK || size_t((wp[8 + g] - sb[8 + g].data()) + (wp[16 + g] - sb[16 + g].data())) >= CHUNK;
            if (full) flush();
        }
    }
    if (steps) flush();
    return o.b;
}

// ---------------- fused block encoder: one pass per compute block (128 x 16) in L1 ----------------
// Band of 16 rows; block bx covers, in row k, the columns [x0 - k, x0 + 128 - k) (a parallelogram: the error axis
// needs the z of the top-right neighbour, which then is already computed one row up). Per row segment: RGB -> RCT
// into the band buffer, context + symbol into packed (z << 4 | stream) and z; after the block the complete wave steps
// t < x0 + 128 go into the streams, then the flush check. Same symbols, order and chunks as encode_planes.
static inline V vsplat(int x) { return splat(x); }
template <int SH>
static void ctx_seg(const int16_t* Pc, const int16_t* Pu, const int16_t* Zu, int16_t* Zc, uint16_t* Kc, int xs, int xe, int W, int t, bool top) {
    const V e0 = splat(EDGE[t][0] - 1), e1 = splat(EDGE[t][1] - 1), e2 = splat(EDGE[t][2] - 1), v0 = splat(VT[t][0] - 1), v1 = splat(VT[t][1] - 1);
    const V one = splat(1), c24 = splat(24), c9 = splat(9), sb = splat(t * 8), lanes = iota(1), zero = splat(0);
    const uint8_t* Gt = G[t];
    for (int x = xs; x < xe; x += NL) {
        V vx = ld(Pc + x), va = ld(Pc + x - 1), vb = ld(Pu + x), vc = ld(Pu + x - 1), vd = ld(Pu + x + 1);
        V zl = ld(Zu + x - 1), zc = ld(Zu + x), zr = ld(Zu + x + 1);
        if (top) { vb = zero; vc = zero; vd = zero; zl = zero; zc = zero; zr = zero; }          // image row 0
        if (x == 0 || x + NL > W - 1) {                                       // border lanes: x = 0: a = c = b, zl = 0; x = W-1: d = b, zr = 0
            V xv = splat(x) + lanes, m0 = xv == zero, mr = xv == splat(W - 1);
            va = vsel(m0, vb, va); vc = vsel(m0, vb, vc); zl = zl & ~m0; vd = vsel(mr, vb, vd); zr = zr & ~mr;
        }
        V ga = vc - va, gb = vc - vb, gd = vd - vb;
        V qa = q7v(ga, e0, e1, e2), qb = q7v(gb, e0, e1, e2), qd = q7v(gd, e0, e1, e2);
        V sec = qb + c24 - ((qa << 3) - qa);
        V vv = vmax(vmax(zl, zc), zr);
        V lev = -((V)(vv > v0) + (V)(vv > v1));
        V cell = (sec << 4) + (sec << 2) + sec + qd + qd + qd + lev + c9;
        alignas(64) int16_t ca[CTX4D_LANES], ga16[CTX4D_LANES]; st(ca, cell);
        for (int i = 0; i < NL; i++) ga16[i] = Gt[ca[i]];
        V grp = ld(ga16);
        V f = vmin(vmax(ga + gb, vmin(ga, gb)), vmax(ga, gb));
        V sgb = (V)(gb < 0) - (V)(gb > 0), s = (ga + ga + sgb) | one;
        V e = vnegif(vx - vc + f, s);
        e = (V)(e << SH) >> SH;
        V z = (e + e) ^ (e >> 15);
        st(Zc + x, z); st((int16_t*)Kc + x, (z << 4) | (sb + grp));
    }
}
static std::vector<uint8_t> encode_fused(const uint8_t* rgb, int W, int H, Times& tm) {
    g_tm = &tm;
    const int T = W + 2 * (LN - 1), PADL = 32, STR = W + 2 * PADL + NL;      // band rows: index -1 .. 15, halo left/right
    static UBuf<int16_t> Pb[3], Zb[3]; static UBuf<uint16_t> Kb[3];
    for (int k = 0; k < 3; k++) { Pb[k].need((size_t)STR * (LN + 1)); Zb[k].need((size_t)STR * (LN + 1)); Kb[k].need((size_t)STR * LN); }
    auto P = [&](int k, int r) { return Pb[k].data() + (size_t)(r + 1) * STR + PADL; };   // r = -1 .. 15
    auto Z = [&](int k, int r) { return Zb[k].data() + (size_t)(r + 1) * STR + PADL; };
    auto K = [&](int k, int r) { return Kb[k].data() + (size_t)r * STR + PADL; };
    for (int k = 0; k < 3; k++) { memset(Pb[k].data(), 0, (size_t)STR * 2); memset(Zb[k].data(), 0, (size_t)STR * 2); }   // row -1 of band 0
    Out o; o.raw("C4D2", 4); o.var(W); o.var(H);
    static UBuf<uint16_t> sb[24], cat; for (auto& q : sb) q.need(CHUNK + SLACK); cat.need(2 * (CHUNK + SLACK));
    uint16_t* wp[24]; for (int i = 0; i < 24; i++) wp[i] = sb[i].data();
    std::vector<uint8_t> tmp; long steps = 0;
    auto flush = [&]() {
        double f0 = msnow(); o.var((uint64_t)steps);
        for (int g = 0; g < 8; g++) put_stream(o, sb[g].data(), wp[g] - sb[g].data(), 256, tmp);
        for (int g = 0; g < 8; g++) { size_t nu = wp[8 + g] - sb[8 + g].data(), nv = wp[16 + g] - sb[16 + g].data();
            memcpy(cat.data(), sb[8 + g].data(), nu * 2); memcpy(cat.data() + nu, sb[16 + g].data(), nv * 2); put_stream(o, cat.data(), nu + nv, 512, tmp); }
        for (int i = 0; i < 24; i++) wp[i] = sb[i].data(); steps = 0; tm.fse += msnow() - f0;
    };
    for (int y0 = 0; y0 < H; y0 += LN) {
        const int rows = std::min(LN, H - y0); int tdone = 0;
        const int nblk = (W + 127) / 128;
        for (int bx = 0; bx < nblk; bx++) {
            const int x0 = bx * 128; const bool last = bx == nblk - 1;
            double b0 = msnow();
            for (int r = 0; r < rows; r++) {
                const int xs = bx ? x0 - r : 0, xe = last ? W : x0 + 128 - r;
                const uint8_t* src = rgb + ((size_t)(y0 + r) * W) * 3;
                int16_t *Y = P(0, r), *U = P(1, r), *Vv = P(2, r);
                for (int x = xs; x < xe; x++) { int R = src[3 * x], Gc = src[3 * x + 1], B = src[3 * x + 2]; Y[x] = (int16_t)((R + 2 * Gc + B) >> 2); U[x] = (int16_t)(B - Gc); Vv[x] = (int16_t)(R - Gc); }
                const bool top = y0 + r == 0;
                ctx_seg<8>(P(0, r), P(0, r - 1), Z(0, r - 1), Z(0, r), K(0, r), xs, xe, W, 0, top);
                ctx_seg<7>(P(1, r), P(1, r - 1), Z(1, r - 1), Z(1, r), K(1, r), xs, xe, W, 1, top);
                ctx_seg<7>(P(2, r), P(2, r - 1), Z(2, r - 1), Z(2, r), K(2, r), xs, xe, W, 1, top);
            }
            double b1 = msnow();
            const int tf = last ? T : x0 + 128;
            for (int k = 0; k < 3; k++) {
                uint16_t** w = wp + k * 8; const uint16_t* rp[LN]; for (int l = 0; l < LN; l++) rp[l] = K(k, std::min(l, rows - 1)) - 2 * l;
                for (int t = tdone; t < tf; t++) {
                    if (rows == LN && t >= 2 * (LN - 1) && t < W) { for (int l = 0; l < LN; l++) { uint16_t p = rp[l][t]; *w[p & 7]++ = uint16_t(p >> 4); } }
                    else for (int l = 0; l < rows; l++) { int x = t - 2 * l; if (x < 0 || x >= W) continue; uint16_t p = rp[l][t]; *w[p & 7]++ = uint16_t(p >> 4); }
                }
            }
            steps += tf - tdone; tdone = tf; double b2 = msnow(); tm.ctx += b1 - b0; tm.scat += b2 - b1;
            bool full = false;
            for (int g = 0; g < 8; g++) full |= size_t(wp[g] - sb[g].data()) >= CHUNK || size_t((wp[8 + g] - sb[8 + g].data()) + (wp[16 + g] - sb[16 + g].data())) >= CHUNK;
            if (full) flush();
        }
        for (int k = 0; k < 3; k++) { memcpy(P(k, -1) - PADL, P(k, rows - 1) - PADL, (size_t)STR * 2); memcpy(Z(k, -1) - PADL, Z(k, rows - 1) - PADL, (size_t)STR * 2); }
    }
    if (steps) flush();
    return o.b;
}
static bool g_planes = false;                                                  // -planes: the old per-plane encoder
static std::vector<uint8_t> encode(const uint8_t* rgb, int W, int H, Times& tm) { return g_planes ? encode_planes(rgb, W, H, tm) : encode_fused(rgb, W, H, tm); }

template <class PD> static void run_steps(PD& pd, int& band, int& t, long rem, double& tend) {
    while (rem > 0) {
        if (t == 0) pd.begin(band * LN);
        long run = std::min<long>(rem, pd.T - t);
        for (long i = 0; i < run; i++) pd.step(t + (int)i);
        t += (int)run; rem -= run;
        if (t == pd.T) { double e0 = msnow(); pd.end(band * LN); tend += msnow() - e0; band++; t = 0; }
    }
}
static void decode(const uint8_t* buf, size_t len, std::vector<uint8_t>& rgb, int& W, int& H, Times& tm) {
    g_tm = &tm;
    In in{buf, buf + len}; if (len < 4 || memcmp(in.take(4), "C4D2", 4)) throw std::runtime_error("not C4D2");
    W = (int)in.var(); H = (int)in.var(); const size_t N = (size_t)W * H;
    static UBuf<int16_t> D[3]; for (auto& d : D) d.need(N);
    pdY.init(D[0].data(), W, H, 0, nullptr, nullptr, nullptr); pdU.init(D[1].data(), W, H, 1, nullptr, nullptr, nullptr); pdV.init(D[2].data(), W, H, 2, nullptr, nullptr, nullptr);
    static UBuf<uint16_t> sb[16]; const uint16_t *SY[8], *SUV[8];
    pdY.Sp = SY; pdU.Sp = SUV; pdV.Sp = SUV;                                   // U and V share streams 8..15
    const long total = (long)((H + LN - 1) / LN) * pdY.T; long done = 0;
    int bY = 0, tY = 0, bU = 0, tU = 0, bV = 0, tV = 0; double tend = 0;
    while (done < total) {
        double f0 = msnow();
        long steps = (long)in.var(); if (steps <= 0 || done + steps > total) throw std::runtime_error("chunk steps");
        for (int s = 0; s < 16; s++) get_stream(in, sb[s]);
        for (int g = 0; g < 8; g++) { SY[g] = sb[g].data(); SUV[g] = sb[8 + g].data(); }
        double f1 = msnow();
        run_steps(pdY, bY, tY, steps, tend); run_steps(pdU, bU, tU, steps, tend); run_steps(pdV, bV, tV, steps, tend);
        done += steps; tm.dfse += f1 - f0; tm.dwave += msnow() - f1;
    }
    tm.dend += tend;
    double r0 = msnow(); rgb.resize(N * 3);
    const int16_t *D0 = D[0].data(), *D1 = D[1].data(), *D2 = D[2].data();
    for (size_t i = 0; i < N; i++) { int Y = D0[i], U = D1[i], V2 = D2[i], Gc = Y - ((U + V2) >> 2);
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

#ifndef CTX4D_CODEC_NO_MAIN
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s c in.ppm out.c4d groups.csv | d in.c4d out.ppm groups.csv | bench groups.csv a.ppm|dir ... [-r N] [-max N] [-cb N] [-planes]\n", argv[0]); return 1; }
    std::string cmd = argv[1];
    try {
        if (cmd == "c" && argc >= 5) {
            if (argc >= 6 && !strcmp(argv[5], "-planes")) g_planes = true;
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
                if (!strcmp(argv[i], "-cb") && i + 1 < argc) { CHUNK = (size_t)atol(argv[++i]); continue; }
                if (!strcmp(argv[i], "-planes")) { g_planes = true; continue; }
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
                T.rct += bt.rct; T.ctx += bt.ctx; T.scat += bt.scat; T.fse += bt.fse; T.hist += bt.hist; T.dfse += bdt.dfse; T.dwave += bdt.dwave; T.irct += bdt.irct; T.dend += bdt.dend; T.dtab += bdt.dtab;
            }
            printf("TOTAL %zu files %.4f bpp  enc %.1f dec %.1f MB/s (image bytes)  %s\n", files.size(), 8.0 * tb / tpx, 3 * tpx / te / 1e3, 3 * tpx / td / 1e3, allok ? "all ok" : "MISMATCH");
            printf("ns/px  enc: RCT %.2f  context+symbol %.2f  wave scatter %.2f  FSE %.2f (histograms %.2f) | dec: FSE %.2f (tables %.2f, decoding %.2f)  wave %.2f (steps %.2f, unskew+top %.2f)  inverse RCT %.2f\n",
                   T.rct * 1e6 / tpx, T.ctx * 1e6 / tpx, T.scat * 1e6 / tpx, T.fse * 1e6 / tpx, T.hist * 1e6 / tpx,
                   T.dfse * 1e6 / tpx, T.dtab * 1e6 / tpx, (T.dfse - T.dtab) * 1e6 / tpx, T.dwave * 1e6 / tpx, (T.dwave - T.dend) * 1e6 / tpx, T.dend * 1e6 / tpx, T.irct * 1e6 / tpx);
            return allok ? 0 : 2;
        }
    } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 3; }
    fprintf(stderr, "bad command\n"); return 1;
}
#endif
