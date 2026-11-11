// bench_entropy.cpp - entropy stage only, on the dumped PLR streams: adaptive RLGR (rlgr_lane.hpp, as in PLR) versus the
// static quad-Huffman prototype "QH" (tables from train_tables.py). Both write one bit lane per tile with the L raw bit
// planes of every block interleaved as in PLR, both are decoded and verified. Reports real bits and enc / dec speed.
// usage: bench_entropy TABLES.bin REPS dump1.dmp dump2.dmp ...
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include "qh.hpp"
using namespace qh;
#pragma pack(push, 1)
struct Rec { uint8_t pl, L; uint32_t bits, sa; uint16_t z[16]; };
#pragma pack(pop)
struct Blk { uint8_t pt, L; uint16_t sa; uint16_t h[16]; uint16_t m[8]; };          // high parts + L raw planes
struct Tile { std::vector<Blk> b; };
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s tables.bin reps dumps...\n", argv[0]); return 1; }
    load_tables(argv[1]); const int reps = atoi(argv[2]);
    std::vector<Tile> tiles; double npx = 0; uint64_t dump_rlgr_bits = 0, nblk = 0, rawbits = 0;
    for (int a = 3; a < argc; ++a) {
        FILE* f = fopen(argv[a], "rb"); if (!f) { perror(argv[a]); return 1; }
        int hd[4]; if (fread(hd, 4, 4, f) != 4) return 1; npx += double(hd[0]) * hd[1];
        Rec r; Tile cur;
        while (fread(&r, sizeof r, 1, f) == 1) {
            if (r.pl == 255) { tiles.push_back(std::move(cur)); cur = Tile(); continue; }
            Blk b; b.pt = r.pl ? 1 : 0; b.L = r.L; b.sa = uint16_t(r.sa == 0xffffffffu ? 64 : r.sa); dump_rlgr_bits += r.bits; ++nblk; rawbits += 16 * r.L;
            for (int j = 0; j < 16; ++j) b.h[j] = uint16_t(r.z[j] >> r.L);
            for (int p = 0; p < 8; ++p) { uint16_t mm = 0; if (p < r.L) for (int j = 0; j < 16; ++j) mm |= uint16_t(((r.z[j] >> p) & 1u) << j); b.m[p] = mm; }
            cur.b.push_back(b); }
        fclose(f);
    }
    const double nsym = double(nblk) * 16;
    printf("tables %s: K=%d maxlen=%d (decode tables %zu KB) | %zu tiles, %.0f blocks, %.2f Mpx\n", argv[1], K, ML, dect.size() * 2 / 1024, tiles.size(), double(nblk), npx / 1e6);
    // ---------- RLGR
    std::vector<std::vector<uint64_t>> la(tiles.size()), lb(tiles.size()); uint64_t bits_r = 0, bits_q = 0, words_r = 0, words_q = 0;
    double te_r = 1e9, td_r = 1e9, te_q = 1e9, td_q = 1e9; bool ok = true;
    std::vector<Blk> out;
    for (int rep = 0; rep < reps; ++rep) {
        double t0 = now_s(); bits_r = 0; words_r = 0;
        for (size_t i = 0; i < tiles.size(); ++i) { Lane lane; lane.w.assign(tiles[i].b.size() * 2 + 64, 0);
            std::vector<rlgrl::Encoder> e; e.reserve(16); for (int k = 0; k < 16; ++k) e.emplace_back(&lane);
            for (const Blk& b : tiles[i].b) { rlgrl::Encoder& c = e[b.pt * 8 + b.L]; for (int j = 0; j < 16; ++j) c.put(b.h[j]); for (int p = 0; p < b.L; ++p) lane.put(16, b.m[p]); }
            for (auto& x : e) x.flush(); bits_r += lane.head; words_r += lane.words(); lane.w.resize(lane.words() + 2); la[i] = std::move(lane.w); }
        double t1 = now_s(); te_r = std::min(te_r, t1 - t0);
        t0 = now_s();
        for (size_t i = 0; i < tiles.size(); ++i) { LaneR lr; lr.w = la[i].data(); lr.nw = la[i].size();
            std::vector<rlgrl::Decoder> d; d.reserve(16); for (int k = 0; k < 16; ++k) d.emplace_back(&lr);
            out.resize(tiles[i].b.size()); const Blk* src = tiles[i].b.data();
            for (size_t n = 0; n < out.size(); ++n) { Blk& o = out[n]; const int L = src[n].L; rlgrl::Decoder& c = d[src[n].pt * 8 + L];
                for (int j = 0; j < 16; ++j) o.h[j] = uint16_t(c.get()); for (int p = 0; p < L; ++p) o.m[p] = uint16_t(lr.get(16)); }
            if (rep == 0) for (size_t n = 0; n < out.size(); ++n) ok &= !memcmp(out[n].h, src[n].h, 32) && !memcmp(out[n].m, src[n].m, 2 * src[n].L);
        }
        t1 = now_s(); td_r = std::min(td_r, t1 - t0);
        // ---------- QH
        t0 = now_s(); bits_q = 0; words_q = 0;
        for (size_t i = 0; i < tiles.size(); ++i) { Lane lane; lane.w.assign(tiles[i].b.size() * 2 + 64, 0);
            for (const Blk& b : tiles[i].b) qh_enc_block(lane, b.pt, b.sa, b.L, b.h, b.m);
            bits_q += lane.head; words_q += lane.words(); lane.w.resize(lane.words() + 2); lb[i] = std::move(lane.w); }
        t1 = now_s(); te_q = std::min(te_q, t1 - t0);
        t0 = now_s();
        for (size_t i = 0; i < tiles.size(); ++i) { LaneR lr; lr.w = lb[i].data(); lr.nw = lb[i].size();
            out.resize(tiles[i].b.size()); const Blk* src = tiles[i].b.data();
            for (size_t n = 0; n < out.size(); ++n) qh_dec_block(lr, src[n].pt, src[n].sa, src[n].L, out[n].h, out[n].m);
            if (rep == 0) for (size_t n = 0; n < out.size(); ++n) ok &= !memcmp(out[n].h, src[n].h, 32) && !memcmp(out[n].m, src[n].m, 2 * src[n].L);
        }
        t1 = now_s(); td_q = std::min(td_q, t1 - t0);
    }
    printf("  RLGR lane : h-part %.4f bpp (dump said %.4f) + raw %.4f | enc %6.1f dec %6.1f Msym/s  (%.2f / %.2f ns per symbol)\n", (bits_r - rawbits) / npx, dump_rlgr_bits / npx, rawbits / npx, nsym / te_r / 1e6, nsym / td_r / 1e6, te_r / nsym * 1e9, td_r / nsym * 1e9);
    printf("  QH static : h-part %.4f bpp (%+.2f%%)            + raw %.4f | enc %6.1f dec %6.1f Msym/s  (%.2f / %.2f ns per symbol)  dec x%.2f enc x%.2f  %s\n", (bits_q - rawbits) / npx, 100.0 * (double(bits_q - rawbits) / double(bits_r - rawbits) - 1), rawbits / npx, nsym / te_q / 1e6, nsym / td_q / 1e6, te_q / nsym * 1e9, td_q / nsym * 1e9, td_r / td_q, te_r / te_q, ok ? "round trips ok" : "MISMATCH");
    printf("  word padding per tile: RLGR %.4f bpp, QH %.4f bpp\n", (words_r * 64.0 - bits_r) / npx, (words_q * 64.0 - bits_q) / npx);
    return ok ? 0 : 3;
}
