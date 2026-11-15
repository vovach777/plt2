// QHK (RLK v9): adaptive run-length + canonical Huffman + grouped bypass plane coder.
// Native port of plrk_v9_fix1.ts (Argon RLK v9 + Fable init fix) in PLR/PLS style (Clang/GCC vector extensions).
#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include "rlgr_lane.hpp"
#include "qh.hpp"
#include "qhk_tables.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifndef PLR_AINLINE
#define PLR_AINLINE inline __attribute__((always_inline))
#endif
#ifndef PLR_NOINLINE
#define PLR_NOINLINE __attribute__((noinline))
#endif
#ifndef PLR_LIKELY
#define PLR_LIKELY(x)   __builtin_expect(!!(x), 1)
#endif
#ifndef PLR_UNLIKELY
#define PLR_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

// Vector types matching palic2r.cpp:287 (Clang ext_vector_type primary, GCC vector_size fallback)
#if defined(__clang__)
#define QHK_VEC 1
typedef uint16_t qhk_v16u __attribute__((ext_vector_type(16)));
#define QHK_VSPL(T, x) ((T)(x))
#define QHK_UNPRED(x)  __builtin_unpredictable(!!(x))
#elif defined(__GNUC__) && __GNUC__ >= 12
#define QHK_VEC 1
typedef uint16_t qhk_v16u __attribute__((vector_size(32)));
template <class T, class E> static PLR_AINLINE T qhk_vspl_(E x) noexcept {
    using El = typename std::remove_cv<typename std::remove_reference<decltype(std::declval<T&>()[0])>::type>::type;
    const El e = El(x); return T{e, e, e, e, e, e, e, e, e, e, e, e, e, e, e, e};
}
#define QHK_VSPL(T, x) (qhk_vspl_<T>(x))
#define QHK_UNPRED(x)  (!!(x))
#else
#define QHK_UNPRED(x)  (!!(x))
#endif

namespace qhk {

static constexpr int RNS   = 255;
static constexpr int RNC   = 256;
static constexpr int RNT   = 512;
static constexpr int RLB   = 9;
static constexpr int R_EOB = 240;
static constexpr int R_LR0 = 241;
static constexpr int RG    = 16;

alignas(32) static constexpr uint16_t VC_BASE[16] = {
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 0
};
alignas(16) static constexpr uint8_t VC_REM[16] = {
    0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 6, 7, 8, 9, 0
};

struct Tables {
    // Specification table memory: 512 KB LUT (512 * 512 * 2 bytes) + 130 KB QHK_LEN in qhk_tables.h
    // For s < 240 and l <= 9: rlut entry has bit 15 = 1, bits 12..14 = lf_ctx, bits 8..11 = len, bits 4..7 = vc, bits 0..3 = run.
    // For s >= 240 and l <= 9: rlut entry has bit 15 = 0, bits 8..11 = len, bits 0..7 = s (so entry > 0).
    // For l > 9: rlut entry is 0.
    alignas(64) uint16_t rlut[RNT][1 << RLB];
    uint16_t renc[RNT][RNS];
    uint8_t  rsrt[RNT][RNS];
    uint16_t first14[RNT][16];
    int16_t  delta_off[RNT][16];

    // Small helper LUTs embedded in Tables so a single base pointer covers all lookups
    uint8_t  v_vc[1024];
    uint8_t  v_rem[1024];
    uint16_t v_mask[1024];
    uint8_t  k3_lut[1024];
    uint8_t  k7_lut[1024];
    uint8_t  s_lf4[1024];
    uint8_t  tb_k[1024];
    uint16_t tb_u[1024];
    uint8_t  sym_r[256];
    uint8_t  sym_vc[256];
    uint16_t vc_base[16];
    uint8_t  vc_rem[16];
    uint8_t  vc_lf_y[16];
    uint8_t  vc_lf_s[16];
    uint8_t  y_cls_up16_sh3[8];
    uint8_t  y_up_ur_sh3[16];
    uint8_t  s_cls_up8_sh3[8];

    void init() noexcept {
        static constexpr uint8_t c_y_cls[8] = {
            12 << 3, 12 << 3, 12 << 3, 13 << 3, 14 << 3, 14 << 3, 15 << 3, 15 << 3
        };
        static constexpr uint8_t c_y_up_ur[16] = {
            0,       9 << 3,  10 << 3, 11 << 3,
            0 << 3,  3 << 3,   6 << 3,  6 << 3,
            1 << 3,  4 << 3,   7 << 3,  7 << 3,
            2 << 3,  5 << 3,   8 << 3,  8 << 3
        };
        static constexpr uint8_t c_s_cls[8] = {
            3 << 3, 3 << 3, 3 << 3, 4 << 3, 5 << 3, 6 << 3, 7 << 3, 7 << 3
        };
        std::memcpy(y_cls_up16_sh3, c_y_cls, 8);
        std::memcpy(y_up_ur_sh3, c_y_up_ur, 16);
        std::memcpy(s_cls_up8_sh3, c_s_cls, 8);

        v_vc[0] = 0; v_rem[0] = 0; v_mask[0] = 0;
        k3_lut[0] = 0; k7_lut[0] = 0; s_lf4[0] = 0;
        tb_k[0] = 0; tb_u[0] = 1;
        for (int v = 1; v < 1024; ++v) {
            int k = 32 - __builtin_clz((uint32_t)v);
            int kc = k < 10 ? k : 10;
            int vc, rem;
            if (kc == 1) { vc = 0; rem = 0; }
            else if (kc <= 6) { vc = 1 + ((kc - 2) << 1) + ((v >> (k - 2)) & 1); rem = k - 2; }
            else { vc = 11 + (kc - 7); rem = k - 1; }
            v_vc[v] = (uint8_t)vc;
            v_rem[v] = (uint8_t)rem;
            v_mask[v] = (uint16_t)((1u << rem) - 1u);
            k3_lut[v] = (uint8_t)(k < 3 ? k : 3);
            k7_lut[v] = (uint8_t)(k < 7 ? k : 7);
            s_lf4[v]  = (uint8_t)(v == 1 ? 64 : (v <= 7 ? 128 : 192));

            uint32_t s1 = (uint32_t)(v + 1);
            int tk = 31 - __builtin_clz(s1);
            uint32_t tu = (2u << tk) - s1;
            tb_k[v] = (uint8_t)tk;
            tb_u[v] = (uint16_t)tu;
        }

        for (int i = 0; i < 16; ++i) {
            vc_base[i] = VC_BASE[i];
            vc_rem[i]  = VC_REM[i];
            vc_lf_y[i] = k7_lut[VC_BASE[i]];
            vc_lf_s[i] = s_lf4[VC_BASE[i]];
        }

        std::memset(sym_r, 0, sizeof(sym_r));
        std::memset(sym_vc, 0, sizeof(sym_vc));
        for (int s = 0; s < 240; ++s) {
            sym_r[s]  = (uint8_t)(s / 15);
            sym_vc[s] = (uint8_t)(s % 15);
        }

        std::memset(rlut, 0, sizeof(rlut));
        for (int t = 0; t < RNT; ++t) {
            uint16_t co[16] = {0};
            for (int s = 0; s < RNS; ++s) {
                co[QHK_LEN[t][s]]++;
            }
            uint16_t fo[16] = {0}, oo[16] = {0};
            int code = 0, off = 0;
            for (int l = 1; l <= 15; ++l) {
                fo[l] = (uint16_t)code;
                oo[l] = (uint16_t)off;
                first14[t][l] = (l <= 14) ? (uint16_t)(code << (14 - l)) : 16384u;
                delta_off[t][l] = (int16_t)(off - code);
                code = (code + co[l]) << 1;
                off += co[l];
            }
            for (int l = 1; l <= 15; ++l) {
                for (int s = 0; s < RNS; ++s) {
                    if (QHK_LEN[t][s] != l) continue;
                    int c = fo[l]++;
                    int o = oo[l]++;
                    renc[t][s] = (uint16_t)c;
                    rsrt[t][o] = (uint8_t)s;
                    if (l <= RLB) {
                        int sh = RLB - l;
                        int base = c << sh;
                        int cnt = 1 << sh;
                        uint16_t val;
                        if (s < 240) {
                            uint32_t run = sym_r[s];
                            uint32_t vc  = sym_vc[s];
                            uint32_t lf3 = (t < RNC) ? vc_lf_y[vc] : (uint32_t(vc_lf_s[vc]) >> 6);
                            val = (uint16_t)(0x8000u | (lf3 << 12) | (uint32_t(l) << 8) | (vc << 4) | run);
                        } else {
                            val = (uint16_t)(s | (l << 8));
                        }
                        for (int x = 0; x < cnt; ++x) {
                            rlut[t][base + x] = val;
                        }
                    }
                }
            }
        }
    }
};

static inline Tables g_tab;
static inline bool g_inited = false;
static inline void init() noexcept {
    if (!g_inited) {
        g_tab.init();
        g_inited = true;
    }
}

static PLR_AINLINE int bw32(uint32_t v) noexcept {
#if defined(__LZCNT__)
    return (int)(32u - _lzcnt_u32(v));
#else
    return v ? (32 - __builtin_clz(v)) : 0;
#endif
}

// Vectorized zero-run class scan in row above (or in ZB for S when ky == 0).
// Uses Clang/GCC vector extension type qhk_v16u for vector loads and comparison;
// _mm256_movemask_epi8 collapses the 16-lane comparison into a scalar bitmask for tzcnt.
static PLR_AINLINE int up_run_cls(const uint16_t* __restrict zp, int rem_x) noexcept {
    int lim = rem_x < 32 ? rem_x : 32;
#if defined(QHK_VEC) && defined(__AVX2__) && !defined(QHK_SCALAR_LOOPS)
    qhk_v16u v0, v1;
    std::memcpy(&v0, zp, 32);
    std::memcpy(&v1, zp + 16, 32);
    const qhk_v16u z0 = QHK_VSPL(qhk_v16u, 0);
    uint32_t m0 = ~(uint32_t)_mm256_movemask_epi8((__m256i)(v0 == z0));
    uint32_t m1 = ~(uint32_t)_mm256_movemask_epi8((__m256i)(v1 == z0));
    uint64_t m = (uint64_t)m0 | ((uint64_t)m1 << 32);
    int c = m ? (__builtin_ctzll(m) >> 1) : 32;
    if (c > lim) c = lim;
    return bw32((uint32_t)c);
#else
    int c = 0;
    while (c < lim && zp[c] == 0) ++c;
    return bw32((uint32_t)c);
#endif
}

// Encoder context functions
static PLR_AINLINE int ctx_y_enc(const Tables& T, const uint16_t* __restrict z, int P, int rem_x, int tw) noexcept {
    if (PLR_UNLIKELY(P < tw)) {
        return (P > 0 ? T.k7_lut[z[P - 1]] : 4) + 120;
    }
    const uint16_t* __restrict up_p = z + P - tw;
    uint16_t up_v = up_p[0];
    uint16_t ur_v = (rem_x > 1) ? up_p[1] : up_v;
    uint16_t ul_v = (rem_x < tw) ? up_p[-1] : up_v;
    uint16_t lf_v = (rem_x < tw) ? z[P - 1] : up_v;

    int base = T.k7_lut[lf_v] + (int(ul_v >= 2) << 7);
    if ((up_v | ur_v) != 0) {
        int up = T.k3_lut[up_v];
        int ur = T.k3_lut[ur_v];
        return base + T.y_up_ur_sh3[(up << 2) | ur];
    }
    return base + T.y_cls_up16_sh3[up_run_cls(up_p, rem_x)];
}

static PLR_AINLINE int ctx_s_enc(const Tables& T, const uint16_t* __restrict z, const uint16_t* __restrict zy,
                                 int P, int rem_x, int tw) noexcept {
    uint16_t ky_v = zy[P];
    int y8;
    if (ky_v != 0) {
        y8 = int(ky_v < 4);
    } else {
        int zry = up_run_cls(zy + P, rem_x);
        y8 = (zry < 6) ? (zry + 1) : 7;
    }

    if (PLR_UNLIKELY(P < tw)) {
        int lf4 = (P > 0) ? T.s_lf4[z[P - 1]] : 192;
        return y8 + (7 << 3) + lf4;
    }
    const uint16_t* __restrict up_p = z + P - tw;
    uint16_t up_v = up_p[0];
    uint16_t ur_v = (rem_x > 1) ? up_p[1] : up_v;
    uint16_t lf_v = (rem_x < tw) ? z[P - 1] : up_v;
    int base = y8 + T.s_lf4[lf_v];
    if ((up_v | ur_v) != 0) {
        int up2 = up_v < 2 ? int(up_v) : 2;
        return base + ((2 - up2) << 3);
    }
    return base + T.s_cls_up8_sh3[up_run_cls(up_p, rem_x)];
}

// Decoder context functions: use register-carried lf_reg (eliminating Store-to-Load Forwarding stall on z[P - 1]!)
static PLR_AINLINE int ctx_y_dec(const Tables& T, const uint16_t* __restrict z, int P, int rem_x, int tw, int lf_reg) noexcept {
    if (PLR_UNLIKELY(P < tw)) {
        return lf_reg + 120;
    }
    const uint16_t* __restrict up_p = z + P - tw;
    uint16_t up_v = up_p[0];
    uint16_t ur_v = (rem_x > 1) ? up_p[1] : up_v;
    uint16_t ul_v = (rem_x < tw) ? up_p[-1] : up_v;
    int lf = (rem_x < tw) ? lf_reg : int(T.k7_lut[up_v]);

    int base = lf + (int(ul_v >= 2) << 7);
    if ((up_v | ur_v) != 0) {
        int up = T.k3_lut[up_v];
        int ur = T.k3_lut[ur_v];
        return base + T.y_up_ur_sh3[(up << 2) | ur];
    }
    return base + T.y_cls_up16_sh3[up_run_cls(up_p, rem_x)];
}

static PLR_AINLINE int ctx_s_dec(const Tables& T, const uint16_t* __restrict z, const uint16_t* __restrict zy,
                                 int P, int rem_x, int tw, int lf4_reg) noexcept {
    uint16_t ky_v = zy[P];
    int y8;
    if (ky_v != 0) {
        y8 = int(ky_v < 4);
    } else {
        int zry = up_run_cls(zy + P, rem_x);
        y8 = (zry < 6) ? (zry + 1) : 7;
    }

    if (PLR_UNLIKELY(P < tw)) {
        return y8 + (7 << 3) + lf4_reg;
    }
    const uint16_t* __restrict up_p = z + P - tw;
    uint16_t up_v = up_p[0];
    uint16_t ur_v = (rem_x > 1) ? up_p[1] : up_v;
    int lf4 = (rem_x < tw) ? lf4_reg : int(T.s_lf4[up_v]);
    int base = y8 + lf4;
    if ((up_v | ur_v) != 0) {
        int up2 = up_v < 2 ? int(up_v) : 2;
        return base + ((2 - up2) << 3);
    }
    return base + T.s_cls_up8_sh3[up_run_cls(up_p, rem_x)];
}

static PLR_AINLINE int probe(const uint16_t* __restrict z, int n) noexcept {
    int nb = (n + 15) >> 4;
    int zq = 0, nz = 0, seen = 0;
    for (int i = 0; i < nb; i += 8) {
        seen++;
        const uint64_t* __restrict p = reinterpret_cast<const uint64_t*>(z + i * 16);
        uint64_t w0 = p[0], w1 = p[1], w2 = p[2], w3 = p[3];
        zq += (w0 == 0) + (w1 == 0) + (w2 == 0) + (w3 == 0);
        nz += ((w0 | w1 | w2 | w3) != 0);
    }
    int quads = seen * 4;
    if (zq * 10 >= quads * 9) return 1;
    if (zq * 20 <= quads || nz * 50 >= seen * 49) return 0;
    return 3;
}

static PLR_AINLINE void adv_row(int pos, int tw, int& row_end) noexcept {
    if (PLR_UNLIKELY(pos >= row_end)) {
        do { row_end += tw; } while (PLR_UNLIKELY(pos >= row_end));
    }
}

static PLR_AINLINE int count_zeros(const uint16_t* __restrict z, int pos, int n) noexcept {
    if (z[pos] != 0) return 0;
#if defined(QHK_VEC) && defined(__AVX2__) && !defined(QHK_SCALAR_LOOPS)
    const qhk_v16u z0 = QHK_VSPL(qhk_v16u, 0);
    int p = pos;
    for (;;) {
        qhk_v16u v;
        std::memcpy(&v, z + p, 32);
        uint32_t m = ~(uint32_t)_mm256_movemask_epi8((__m256i)(v == z0));
        if (m != 0) {
            p += (__builtin_ctz(m) >> 1);
            return p < n ? (p - pos) : (n - pos);
        }
        p += 16;
        if (p >= n) return n - pos;
    }
#else
    int zr = 0, rem = n - pos;
    while (zr < rem && z[pos + zr] == 0) ++zr;
    return zr;
#endif
}

// Direct RLK plane encoder into Lane.
template<int PL>
static PLR_NOINLINE void enc_core(const uint16_t* __restrict z,
                                  const uint16_t* __restrict zy,
                                  const uint16_t* __restrict zu,
                                  int n, int tw, rlgrl::Lane& lane) noexcept {
    const Tables& T = g_tab;
    const int tb = PL * RNC;

    size_t i0 = size_t(lane.head >> 6);
    size_t need = i0 + size_t(n >> 1) + 256;
    if (lane.w.size() < need) lane.w.resize(std::max(need, lane.w.size() * 2), 0);

    uint64_t* __restrict p = lane.w.data() + i0;
    unsigned fill = unsigned(lane.head & 63);
    uint64_t acc = fill ? (*p >> (64 - fill)) : 0;

    auto put_raw = [&](unsigned nb, uint64_t v) __attribute__((always_inline)) {
        if (PLR_LIKELY(fill + nb < 64)) {
            acc = (acc << nb) | v;
            fill += nb;
        } else {
            unsigned take = 64 - fill;
            unsigned rem  = nb - take;
            *p++ = (fill ? (acc << take) : 0ull) | (v >> rem);
            acc  = v;
            fill = rem;
        }
    };

    uint32_t byp[RG];
    int pos = 0, row_end = tw;

    while (pos < n) {
        int nv = 0;
        for (int g = 0; g < RG && pos < n; ++g) {
            int gap = count_zeros(z, pos, n);
            int q = pos + gap;
            int ctx = (PL == 0) ? ctx_y_enc(T, z, pos, row_end - pos, tw)
                                : ctx_s_enc(T, z, zy, pos, row_end - pos, tw);
            int t = tb + ctx;
            if (PLR_UNLIKELY(q >= n)) {
                put_raw(QHK_LEN[t][R_EOB], T.renc[t][R_EOB]);
                pos = n;
                break;
            }
            if (PLR_UNLIKELY(gap >= 16)) {
                int b = bw32((uint32_t)gap);
                int s = R_LR0 + b - 5;
                unsigned l = QHK_LEN[t][s];
                uint64_t c = T.renc[t][s];
                uint64_t low = (uint64_t)(gap & ((1 << (b - 1)) - 1));
                put_raw(l + unsigned(b - 1), (c << (b - 1)) | low);
                pos = q;
                adv_row(pos, tw, row_end);
                continue;
            }
            uint32_t v = z[q];
            int vc  = T.v_vc[v];
            unsigned rem = T.v_rem[v];
            int s   = gap * 15 + vc;
            put_raw(QHK_LEN[t][s], T.renc[t][s]);

            if constexpr (PL == 0) {
                byp[nv] = (rem << 24) | (v & T.v_mask[v]);
                nv += (rem != 0);
            } else {
                uint32_t a = zu[q];
                unsigned kk = T.tb_k[v];
                uint32_t u = T.tb_u[v];
                bool ge = (a >= u);
#if defined(__clang__)
                uint32_t tb_val = a + (__builtin_unpredictable(ge) ? u : 0u);
#else
                uint32_t ge_u = uint32_t(ge);
                uint32_t tb_val = a + (u & (0u - ge_u));
#endif
                unsigned tb_bits = kk + unsigned(ge);
                unsigned tot_byp = rem + tb_bits;
                uint32_t bval = ((v & T.v_mask[v]) << tb_bits) | tb_val;
                byp[nv++] = (tot_byp << 24) | bval;
            }
            pos = q + 1;
            adv_row(pos, tw, row_end);
        }

        for (int i = 0; i < nv; ++i) {
            put_raw(byp[i] >> 24, byp[i] & 0xFFFFFFu);
        }
    }

    *p = fill ? (acc << (64 - fill)) : 0ull;
    lane.head = uint64_t(p - lane.w.data()) * 64 + fill;
}

static inline void rewind_lane(rlgrl::Lane& l, uint64_t p0) noexcept {
    size_t i0 = size_t(p0 >> 6);
    int off = int(p0 & 63);
    size_t i1 = size_t((l.head + 63) >> 6);
    if (off) l.w[i0] &= ~(~0ull >> off);
    else     l.w[i0] = 0;
    for (size_t i = i0 + 1; i < i1; ++i) l.w[i] = 0;
    l.head = p0;
}

// Guaranteed single-instruction shldq on both Clang and GCC.
static PLR_AINLINE uint64_t fast_peek(const uint64_t* __restrict wp, uint64_t head) noexcept {
    const size_t i = size_t(head >> 6);
    uint64_t hi = wp[i], lo = wp[i + 1];
#if defined(__x86_64__) || defined(_M_X64)
    __asm__("shldq %b2, %1, %0" : "+r"(hi) : "r"(lo), "c"(head) : "cc");
    return hi;
#else
    const unsigned off = unsigned(head & 63);
    return off ? (hi << off) | (lo >> (64 - off)) : hi;
#endif
}

// High-speed templated RLK plane decoder:
// 1. Register-carried lf_reg eliminates Store-to-Load Forwarding stalls on z[P - 1].
// 2. 16-bit packed rlut entry (0x8000 | lf3<<12 | len<<8 | vc<<4 | run) decodes s < 240 in a single lookup without sym_info!
// 3. 4x4 window refill decodes 4 Huffman codes per fast_peek call (4 * 14 = 56 <= 64 bits).
template<int PL>
static PLR_NOINLINE void dec_core(uint16_t* __restrict z,
                                  const uint16_t* __restrict zy,
                                  uint16_t* __restrict zu,
                                  uint16_t* __restrict zv,
                                  int n, int tw, rlgrl::LaneR& r) {
    const Tables& T = g_tab;
    const int tb = PL * RNC;

    const uint64_t* __restrict wp = r.w;
    const size_t nw = r.nw;
    const uint64_t max_bits = uint64_t(nw > 2 ? (nw - 2) : nw) * 64ull;
    uint64_t head = r.head;

    uint32_t gpk[RG];
    int pos = 0, row_end = tw;
    int lf_reg = (PL == 0) ? 4 : 192;

    while (pos < n) {
        if (PLR_UNLIKELY(head > max_bits)) throw std::runtime_error("rlgr lane: read past the end");
        int nv = 0;
        for (int q = 0; q < 4 && pos < n; ++q) {
            uint64_t win = fast_peek(wp, head);
            for (int j = 0; j < 4 && pos < n; ++j) {
                int ctx = (PL == 0) ? ctx_y_dec(T, z, pos, row_end - pos, tw, lf_reg)
                                    : ctx_s_dec(T, z, zy, pos, row_end - pos, tw, lf_reg);
                int t = tb + ctx;
                int16_t se = (int16_t)T.rlut[t][win >> (64 - RLB)];
                if (PLR_LIKELY(se < 0)) {
                    uint32_t ue = (uint16_t)se;
                    int run = int(ue & 15u);
                    uint32_t vc = (ue >> 4) & 15u;
                    int len = int((ue >> 8) & 15u);
                    win <<= len;
                    head += (uint64_t)len;
                    pos += run;
                    if (PLR_UNLIKELY(pos >= n)) throw std::runtime_error("corrupt plk run");
                    z[pos] = T.vc_base[vc];
                    if constexpr (PL == 0) {
                        lf_reg = int((ue >> 12) & 7u);
                        uint32_t rem = T.vc_rem[vc];
                        gpk[nv] = (uint32_t)((pos << 4) | rem);
                        nv += (rem != 0);
                    } else {
                        lf_reg = int((ue >> 12) & 3u) << 6;
                        gpk[nv++] = (uint32_t)((pos << 4) | vc);
                    }
                    pos++;
                    adv_row(pos, tw, row_end);
                    continue;
                }

                // Slow path: s >= 240 or code length 10..14
                int s, len;
                if (se > 0) {
                    s = se & 255;
                    len = se >> 8;
                } else {
                    uint32_t c14 = (uint32_t)(win >> 50);
                    len = 10 + (c14 >= T.first14[t][11])
                             + (c14 >= T.first14[t][12])
                             + (c14 >= T.first14[t][13])
                             + (c14 >= T.first14[t][14]);
                    int idx = (int)(c14 >> (14 - len)) + (int)T.delta_off[t][len];
                    if (PLR_UNLIKELY((unsigned)idx >= (unsigned)RNS)) throw std::runtime_error("corrupt plk huffman code");
                    s = T.rsrt[t][idx];
                }
                win <<= len;
                head += (uint64_t)len;

                if (s < 240) {
                    int run = T.sym_r[s];
                    uint32_t vc = T.sym_vc[s];
                    pos += run;
                    if (PLR_UNLIKELY(pos >= n)) throw std::runtime_error("corrupt plk run");
                    z[pos] = T.vc_base[vc];
                    if constexpr (PL == 0) {
                        lf_reg = T.vc_lf_y[vc];
                        uint32_t rem = T.vc_rem[vc];
                        gpk[nv] = (uint32_t)((pos << 4) | rem);
                        nv += (rem != 0);
                    } else {
                        lf_reg = T.vc_lf_s[vc];
                        gpk[nv++] = (uint32_t)((pos << 4) | vc);
                    }
                    pos++;
                    adv_row(pos, tw, row_end);
                } else if (s >= R_LR0) {
                    int nb = s - R_LR0 + 4;
                    uint64_t rwin = fast_peek(wp, head);
                    int run = (1 << nb) | (int)(rwin >> (64 - nb));
                    head += (uint64_t)nb;
                    if (PLR_UNLIKELY(run > n - pos)) throw std::runtime_error("corrupt plk lrun");
                    pos += run;
                    lf_reg = 0;
                    adv_row(pos, tw, row_end);
                    if (j < 3 && pos < n) win = fast_peek(wp, head);
                } else { // s == R_EOB
                    pos = n;
                }
            }
        }

        // Batched group bypass loop
        if constexpr (PL == 0) {
            int i = 0;
            while (i < nv) {
                int step = std::min(nv - i, 7);
                uint64_t win = fast_peek(wp, head);
                int used = 0;
                for (int j = 0; j < step; ++j) {
                    uint32_t pr = gpk[i + j];
                    int p   = (int)(pr >> 4);
                    int rem = (int)(pr & 15u);
                    z[p] |= (uint16_t)(win >> (64 - rem));
                    win <<= rem;
                    used += rem;
                }
                head += (uint64_t)used;
                i += step;
            }
        } else {
            int i = 0;
            while (i < nv) {
                int step = std::min(nv - i, 3);
                uint64_t win = fast_peek(wp, head);
                int used = 0;
                for (int j = 0; j < step; ++j) {
                    uint32_t pk = gpk[i + j];
                    int p  = (int)(pk >> 4);
                    int vc = (int)(pk & 15u);
                    int rem = T.vc_rem[vc];
                    uint32_t extra = (uint32_t)((win >> 1) >> (63 - rem)) & ((1u << rem) - 1u);
                    uint32_t v = T.vc_base[vc] | extra;
                    win <<= rem;
                    used += rem;
                    int kk = T.tb_k[v];
                    uint32_t u = T.tb_u[v];
                    uint32_t top = (uint32_t)(win >> (63 - kk));
                    const bool ge = top >= (u << 1);
#if defined(__clang__)
                    const uint64_t w0 = win << kk, w1 = win << (kk + 1);
                    const uint32_t a = __builtin_unpredictable(ge) ? (top - u) : (top >> 1);
                    win = __builtin_unpredictable(ge) ? w1 : w0;
                    used += kk + int(ge);
#else
                    const uint32_t ge_u = uint32_t(ge), nb = uint32_t(kk) + ge_u;
                    const uint32_t a = (top >> (1u - ge_u)) - (u & (0u - ge_u));
                    win <<= nb;
                    used += int(nb);
#endif
                    if (PLR_UNLIKELY(a > v)) throw std::runtime_error("corrupt plk tb");
                    zu[p] = (uint16_t)a;
                    zv[p] = (uint16_t)(v - a);
                }
                head += (uint64_t)used;
                i += step;
            }
        }
    }

    if (PLR_UNLIKELY(head > max_bits)) throw std::runtime_error("corrupt plk eof");
    r.head = head;
}

} // namespace qhk
