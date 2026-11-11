// pct_probe.cpp - does PPT2 (github vovach777/PCT, ppt_transform.hpp) help on what PLR codes?
// Per gray image, empirical order-0 entropy in bits per pixel (no side info, ideal coder):
//  MED      : MED residuals (PLR's predictor), one distribution per image
//  MED|L    : the same, one distribution per causal noise bin L (PLR's context: mean of the 16 residuals above, off 2)
//  PPT2 px  : PPT2 8x8 on the pixels (minus 128), one distribution per coefficient position (64)
//  PPT2 res : PPT2 1-D 8-point along x on the MED residuals (8 rows in the lanes), one distribution per position (8)
//  PPT2 res|L: the same, per position and per L of the first sample
// Build: clang++ -O2 -march=native -std=c++17 -I<PCT> pct_probe.cpp -o pct_probe ; run: pct_probe a.pgm b.pgm ...
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "ppt_transform.hpp"

static std::vector<uint8_t> read_pgm(const char* fn, int& W, int& H) {
    FILE* f = fopen(fn, "rb"); if (!f) { perror(fn); exit(1); }
    char m[3] = {0}; int mx; if (fscanf(f, "%2s %d %d %d", m, &W, &H, &mx) != 4 || m[1] != '5') { fprintf(stderr, "%s: need P5\n", fn); exit(1); }
    fgetc(f); std::vector<uint8_t> p(size_t(W) * H); if (fread(p.data(), 1, p.size(), f) != p.size()) exit(1); fclose(f); return p;
}
struct Hist { std::map<int, uint64_t> c; uint64_t n = 0; void add(int v) { ++c[v]; ++n; }
    double bits() const { double b = 0; for (auto& kv : c) b -= double(kv.second) * std::log2(double(kv.second) / double(n)); return b; } };
static int med(int a, int b, int c) { int mn = std::min(a, b), mx = std::max(a, b), p = a + b - c; return p < mn ? mn : p > mx ? mx : p; }
static int bw(unsigned v) { return v ? 32 - __builtin_clz(v) : 0; }
int main(int argc, char** argv) {
    double T[5] = {0}; double px = 0;
    for (int fi = 1; fi < argc; ++fi) {
        int W, H; auto p = read_pgm(argv[fi], W, H);
        auto P = [&](int x, int y) { return int(p[size_t(y) * W + x]); };
        std::vector<int> e(size_t(W) * H); std::vector<int> Lm(size_t(W) * H, 0);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            int pr = y == 0 ? (x ? P(x - 1, y) : 0) : x == 0 ? P(x, y - 1) : med(P(x - 1, y), P(x, y - 1), P(x - 1, y - 1));
            e[size_t(y) * W + x] = P(x, y) - pr; }
        // causal L like PLR: mean zigzag of the 16 residuals above (same x range, aligned to 16), off 2
        for (int y = 1; y < H; ++y) for (int x0 = 0; x0 < W; x0 += 16) { unsigned s = 0; int n = 0;
            for (int x = x0; x < std::min(W, x0 + 16); ++x) { int v = e[size_t(y - 1) * W + x]; s += unsigned(v < 0 ? -2 * v - 1 : 2 * v); ++n; }
            const int L = std::min(7, std::max(0, bw(s / unsigned(n)) - 2));
            for (int x = x0; x < std::min(W, x0 + 16); ++x) Lm[size_t(y) * W + x] = L; }
        Hist hm, hmL[8]; for (size_t i = 0; i < e.size(); ++i) { hm.add(e[i]); hmL[Lm[i]].add(e[i]); }
        // PPT2 8x8 on pixels (whole 8x8 blocks only)
        Hist hp[64]; const int W8 = W / 8 * 8, H8 = H / 8 * 8;
        for (int by = 0; by < H8; by += 8) for (int bx = 0; bx < W8; bx += 8) {
            v8hi d[8]; for (int r = 0; r < 8; ++r) for (int c = 0; c < 8; ++c) d[r][c] = int16_t(P(bx + c, by + r) - 128);
            ppt2_8x8_forward(d); for (int r = 0; r < 8; ++r) for (int c = 0; c < 8; ++c) hp[r * 8 + c].add(d[r][c]); }
        // PPT2 1-D along x on residuals: data[i] lane j = e(x0 + i, y0 + j)
        Hist hr[8], hrL[8][8];
        for (int y0 = 0; y0 + 8 <= H; y0 += 8) for (int x0 = 0; x0 + 8 <= W; x0 += 8) {
            v8hi d[8]; for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) d[i][j] = int16_t(e[size_t(y0 + j) * W + x0 + i]);
            ppt2_packet_transform_8(d);
            for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) { hr[i].add(d[i][j]); hrL[i][Lm[size_t(y0 + j) * W + x0]].add(d[i][j]); } }
        double b[5] = {hm.bits(), 0, 0, 0, 0};
        for (auto& h : hmL) b[1] += h.bits();
        for (auto& h : hp) b[2] += h.bits();
        for (auto& h : hr) b[3] += h.bits();
        for (auto& a : hrL) for (auto& h : a) b[4] += h.bits();
        const double n = double(W) * H, n8 = double(W8) * H8, nr = double((W / 8) * 8) * ((H / 8) * 8);
        printf("%-40s MED %.4f  MED|L %.4f  PPT2px %.4f  PPT2res %.4f  PPT2res|L %.4f\n", argv[fi], b[0] / n, b[1] / n, b[2] / n8, b[3] / nr, b[4] / nr);
        T[0] += b[0]; T[1] += b[1]; T[2] += b[2] * n / n8; T[3] += b[3] * n / nr; T[4] += b[4] * n / nr; px += n;
    }
    printf("TOTAL bpp: MED %.4f  MED|L %.4f  PPT2 8x8 pixels %.4f  PPT2 1-D on residuals %.4f  (|L %.4f)\n", T[0] / px, T[1] / px, T[2] / px, T[3] / px, T[4] / px);
}
