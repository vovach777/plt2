// planes_bench.cpp - context-free coding of one gray plane (PALIC-1 style: MED -> zigzag z), encoder side:
//   A  PALIC-1 llrans as is: token bit_width(z) + raw mantissa inside 4-way rANS64 (windows 2^w, w = 12 / 15)
//   B  FSE4 over the tokens bit_width(z) + the mantissa bits raw in a separate branchless bit stream, chunk 4K / 64K
//   C  bit planes: the L low bits of z go raw as planes of 16-symbol blocks (movemask, 2 bytes per plane per block),
//      the high part z >> L whole through FSE4; L fixed per tile (0 = plain FSE over z)
//   D  as C with L per block of 16: L_b = max(0, bw(mean z of the block) - off); L_b through FSE4 (alphabet 8),
//      the high part in one FSE4 stream per L_b value (8 streams)
// Every variant is decoded back and compared (z), sizes are the full stream bytes; speed = entropy stage only
// (z is computed once), best of R runs, MB/s = Mpx/s (gray, 1 byte per pixel).
// build (from this folder):
//   clang++ -O3 -march=native -std=c++17 -I. -I../../../third_party/fse -I../../.. -I../../../palic-1 planes_bench.cpp
//           <fse objects: entropy_common fse_compress fse_decompress fseU16 hist> -o planes_bench
// usage: planes_bench tile.pgm ... [-r N]          (8-bit P5)
#define CTX4D_CODEC_NO_MAIN
#include "ctx4d_codec.cpp"
#define main llrans_main
#include "llrans.cpp"
#undef main
#include <smmintrin.h>
#include <chrono>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static bool read_pgm(const char* fn, std::vector<uint8_t>& px, int& W, int& H) {
    FILE* f = fopen(fn, "rb"); if (!f) return false; int mx;
    if (fscanf(f, "P5 %d %d %d", &W, &H, &mx) != 3 || mx != 255) { fclose(f); return false; }
    fgetc(f); px.resize((size_t)W * H); bool ok = fread(px.data(), 1, px.size(), f) == px.size(); fclose(f); return ok;
}
// PALIC-1 residual: y = 0: left (x = 0: 0); x = 0: up; else MED; 8-bit wrap, zigzag
static void residuals(const uint8_t* p, int W, int H, std::vector<uint16_t>& z) {
    z.resize((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        const uint8_t* r = p + (size_t)y * W; int pr;
        if (y == 0) pr = x ? r[x - 1] : 0; else if (x == 0) pr = r[x - W]; else pr = llrans::med(r[x - 1], r[x - W], r[x - W - 1]);
        z[(size_t)y * W + x] = llrans::zz_of<false>(r[x] - pr);
    }
}

// ---- raw bits, branchless (as llcomp fse -k 1)
// accumulator in a register, unconditional 8-byte store, advance by whole bytes (like zstd BIT_flushBitsFast):
// no branch and no load (an OR into memory re-reads the bytes just stored: store-forwarding stalls)
struct RawW { std::vector<uint8_t> b; uint8_t* p; uint64_t acc = 0; int fill = 0;
    explicit RawW(size_t maxbits) : b(maxbits / 8 + 16, 0) { p = b.data(); }
    inline void put(uint32_t v, int k) { acc |= uint64_t(v) << fill; fill += k; memcpy(p, &acc, 8); const int nb = fill >> 3; p += nb; acc = nb ? acc >> (nb * 8) : acc; fill &= 7; }
    size_t bytes() const { return size_t(p - b.data()) + (fill + 7) / 8; } };
struct RawR { const uint8_t* b; size_t bit = 0;
    inline uint32_t get(int k) { uint64_t w; memcpy(&w, b + (bit >> 3), 8); uint32_t v = uint32_t(w >> (bit & 7)) & ((1u << k) - 1); bit += size_t(k); return v; } };

// ---- B: tokens + raw mantissa
static size_t encB(const std::vector<uint16_t>& z, size_t chunk, std::vector<uint8_t>& out) {
    const size_t n = z.size(); Out o; std::vector<uint8_t> tmp; RawW rw(n * 8); std::vector<uint16_t> tok(chunk);
    for (size_t b = 0; b < n; b += chunk) {
        const size_t m = std::min(chunk, n - b);
        for (size_t i = 0; i < m; ++i) { uint32_t v = z[b + i]; int t = int(llrans::bw(v)), k = t - (t > 0); tok[i] = uint16_t(t); rw.put(v & ((1u << k) - 1), k); }
        put_stream(o, tok.data(), m, 9, tmp);
    }
    memcpy(rw.p, &rw.acc, 8); o.var(rw.bytes()); o.raw(rw.b.data(), rw.bytes()); out.swap(o.b); return out.size();
}
static bool decB(const std::vector<uint8_t>& d, size_t n, size_t chunk, const std::vector<uint16_t>& ref) {
    In in{d.data(), d.data() + d.size()}; UBuf<uint16_t> tb; std::vector<uint16_t> tok(n);
    for (size_t b = 0; b < n; b += chunk) { size_t m = get_stream(in, tb); memcpy(&tok[b], tb.data(), m * 2); }
    size_t rl = in.var(); std::vector<uint8_t> raw(rl + 16, 0); memcpy(raw.data(), in.take(rl), rl); RawR rr{raw.data()};
    for (size_t i = 0; i < n; ++i) { int t = tok[i]; uint32_t v = ((1u << t) >> 1) | rr.get(t - (t > 0)); if (v != ref[i]) return false; }
    return true;
}

// ---- C / D: bit planes of 16-symbol blocks (SSE movemask), high part through FSE
static inline void planes16(const uint16_t* s, int L, uint16_t*& pw, uint16_t* hi) {
    const __m128i a0 = _mm_loadu_si128((const __m128i*)s), a1 = _mm_loadu_si128((const __m128i*)(s + 8));
    for (int k = 0; k < L; ++k) { const __m128i c = _mm_cvtsi32_si128(15 - k);
        *pw++ = uint16_t(_mm_movemask_epi8(_mm_packs_epi16(_mm_sll_epi16(a0, c), _mm_sll_epi16(a1, c)))); }
    const __m128i c = _mm_cvtsi32_si128(L);
    _mm_storeu_si128((__m128i*)hi, _mm_srl_epi16(a0, c)); _mm_storeu_si128((__m128i*)(hi + 8), _mm_srl_epi16(a1, c));
}
static size_t encC(const std::vector<uint16_t>& z, int L, std::vector<uint8_t>& out) {   // n % 16 == 0 (tiles)
    const size_t n = z.size(); Out o; std::vector<uint8_t> tmp; std::vector<uint16_t> hi(n), pl(n / 16 * L + 1); uint16_t* pw = pl.data();
    for (size_t b = 0; b < n; b += 16) planes16(&z[b], L, pw, &hi[b]);
    put_stream(o, hi.data(), n, 256u >> L, tmp); o.raw(pl.data(), (pw - pl.data()) * 2); out.swap(o.b); return out.size();
}
static bool decC(const std::vector<uint8_t>& d, size_t n, int L, const std::vector<uint16_t>& ref) {
    In in{d.data(), d.data() + d.size()}; UBuf<uint16_t> hb; get_stream(in, hb); const uint8_t* pp = in.take(n / 16 * L * 2);
    for (size_t b = 0; b < n; b += 16) { uint16_t m[8]; memcpy(m, pp, L * 2); pp += L * 2;
        for (int j = 0; j < 16; ++j) { uint32_t v = uint32_t(hb.data()[b + j]) << L; for (int k = 0; k < L; ++k) v |= ((m[k] >> j) & 1u) << k; if (v != ref[b + j]) return false; } }
    return true;
}
static size_t encD(const std::vector<uint16_t>& z, int off, std::vector<uint8_t>& out) {
    const size_t n = z.size(), nb = n / 16; Out o; std::vector<uint8_t> tmp;
    std::vector<uint16_t> Lb(nb), hi[8], blk(16), pl(n / 2 + 1); uint16_t* pw = pl.data(); for (auto& h : hi) h.reserve(n);
    for (size_t i = 0; i < nb; ++i) {
        const uint16_t* s = &z[i * 16]; uint32_t sum = 0; for (int j = 0; j < 16; ++j) sum += s[j];
        int L = std::max(0, int(llrans::bw(sum >> 4)) - off); L = std::min(L, 7); Lb[i] = uint16_t(L);
        planes16(s, L, pw, blk.data()); hi[L].insert(hi[L].end(), blk.begin(), blk.end());
    }
    put_stream(o, Lb.data(), nb, 8, tmp);
    for (int L = 0; L < 8; ++L) put_stream(o, hi[L].data(), hi[L].size(), 256u >> L, tmp);
    o.raw(pl.data(), (pw - pl.data()) * 2); out.swap(o.b); return out.size();
}
static bool decD(const std::vector<uint8_t>& d, size_t n, const std::vector<uint16_t>& ref) {
    In in{d.data(), d.data() + d.size()}; const size_t nb = n / 16; UBuf<uint16_t> lb, hb[8]; get_stream(in, lb);
    for (int L = 0; L < 8; ++L) get_stream(in, hb[L]);
    size_t np = 0; for (size_t i = 0; i < nb; ++i) np += lb.data()[i];
    const uint8_t* pp = in.take(np * 2); size_t hp[8] = {0};
    for (size_t i = 0; i < nb; ++i) { int L = lb.data()[i]; uint16_t m[8]; memcpy(m, pp, L * 2); pp += L * 2;
        for (int j = 0; j < 16; ++j) { uint32_t v = uint32_t(hb[L].data()[hp[L]++]) << L; for (int k = 0; k < L; ++k) v |= ((m[k] >> j) & 1u) << k; if (v != ref[i * 16 + j]) return false; } }
    return true;
}

int main(int argc, char** argv) {
    int reps = 20; std::vector<const char*> files;
    for (int i = 1; i < argc; ++i) { if (!strcmp(argv[i], "-r") && i + 1 < argc) reps = atoi(argv[++i]); else files.push_back(argv[i]); }
    struct V { std::string name; size_t bytes = 0; double t = 0; bool ok = true; };
    std::vector<V> tot; size_t totpx = 0;
    auto add = [&](size_t k, const char* name, size_t b, double t, bool ok) { if (tot.size() <= k) tot.resize(k + 1); tot[k].name = name; tot[k].bytes += b; tot[k].t += t; tot[k].ok &= ok; };
    for (auto fn : files) {
        std::vector<uint8_t> px; int W, H; if (!read_pgm(fn, px, W, H)) { fprintf(stderr, "skip %s\n", fn); continue; }
        if ((size_t(W) * H) % 16) { fprintf(stderr, "skip %s (W*H %% 16)\n", fn); continue; }
        std::vector<uint16_t> z; double t0 = now_s(); for (int r = 0; r < reps; ++r) residuals(px.data(), W, H, z); double tz = (now_s() - t0) / reps;
        const size_t n = z.size(); totpx += n; size_t k = 0;
        printf("%s %dx%d  (residuals %.2f ns/px)\n", fn, W, H, tz * 1e9 / n);
        auto run = [&](const char* name, auto enc, auto dec) {
            std::vector<uint8_t> o; double best = 1e9; size_t b = 0;
            for (int r = 0; r < reps; ++r) { double a = now_s(); b = enc(o); best = std::min(best, now_s() - a); }
            bool ok = dec(o); add(k++, name, b, best, ok);
            printf("  %-26s %7.4f bpp  %7.1f Mpx/s  %s\n", name, 8.0 * b / n, n / best / 1e6, ok ? "ok" : "FAIL");
        };
        for (int sb : {12, 15}) {
            char nm[64]; snprintf(nm, sizeof nm, "A llrans rANS w=%d", sb);
            run(nm, [&](std::vector<uint8_t>& o) { auto w = llrans::enc_plane(z, sb); o.resize(w.size() * 4); memcpy(o.data(), w.data(), o.size()); return o.size(); },
                [&](const std::vector<uint8_t>& o) { std::vector<uint32_t> w(o.size() / 4 + 8, 0); memcpy(w.data(), o.data(), o.size()); std::vector<uint16_t> r(n);
                    llrans::dec_plane(w.data(), n, sb, r.data()); return r == z; });
        }
        for (size_t ch : {size_t(4096), size_t(65536)}) {
            char nm[64]; snprintf(nm, sizeof nm, "B FSE tok + raw, chunk %zuK", ch / 1024);
            run(nm, [&](std::vector<uint8_t>& o) { return encB(z, ch, o); }, [&](const std::vector<uint8_t>& o) { return decB(o, n, ch, z); });
        }
        for (int L = 0; L <= 4; ++L) {
            char nm[64]; snprintf(nm, sizeof nm, "C planes, L=%d per tile", L);
            run(nm, [&](std::vector<uint8_t>& o) { return encC(z, L, o); }, [&](const std::vector<uint8_t>& o) { return decC(o, n, L, z); });
        }
        for (int off : {1, 2, 3}) {
            char nm[64]; snprintf(nm, sizeof nm, "D planes, L/block off=%d", off);
            run(nm, [&](std::vector<uint8_t>& o) { return encD(z, off, o); }, [&](const std::vector<uint8_t>& o) { return decD(o, n, z); });
        }
    }
    printf("TOTAL %zu px\n", totpx);
    for (auto& v : tot) printf("  %-26s %7.4f bpp  %7.1f Mpx/s  %s\n", v.name.c_str(), 8.0 * v.bytes / totpx, totpx / v.t / 1e6, v.ok ? "ok" : "FAIL");
}
