// letters (4 px x 2 bit, PLR block L) -> FSE / huff0 / H0, no context
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
extern "C" {
#include "fse.h"
#include "huf.h"
}
static int SKIP = 0; static double skipped[2];
static int bw(uint32_t v){ return v ? 32 - __builtin_clz(v) : 0; }
static int chooseL(int s){ uint32_t q = (uint32_t)s >> 4; int L = bw(q) - 2; return L < 0 ? 0 : (L > 7 ? 7 : L); }
static size_t pack(std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int fse) {
  if (src.empty()) return 0;
  dst.resize(FSE_compressBound(src.size()) + HUF_compressBound(src.size()) + 64);
  size_t r = fse ? FSE_compress(dst.data(), dst.size(), src.data(), src.size())
                 : HUF_compress(dst.data(), dst.size(), src.data(), src.size());
  if (FSE_isError(r) || HUF_isError(r)) { fprintf(stderr, "err\n"); exit(1); }
  if (r == 0) return src.size();     // incompressible: stored
  return r;                           // r == 1: RLE byte
}
static double H0(std::vector<uint8_t>& s) {
  if (s.empty()) return 0; double c[256] = {0}; for (auto b : s) c[b]++;
  double h = 0, n = s.size(); for (int i = 0; i < 256; i++) if (c[i]) h -= c[i] * log2(c[i] / n); return h;
}
int main(int argc, char** argv) {
  SKIP = argc > 2; FILE* f = fopen(argv[1], "rb"); if (!f) return 1;
  double px = 0, plrbits = 0;
  double tH[2] = {0}, tF[2] = {0}, tU[2] = {0}, nl[2] = {0}, zl[2] = {0};
  std::vector<uint8_t> glob[2], cur, out;
  std::vector<uint16_t> z(3 * 288 * 288 * 4);
  int32_t hd[4];
  while (fread(hd, 4, 4, f) == 4) {
    int tw = hd[0], th = hd[1], n = tw * th; plrbits += hd[2] + hd[3]; px += n;
    z.resize(3 * (size_t)n); if (fread(z.data(), 2, 3 * (size_t)n, f) != 3 * (size_t)n) return 2;
    for (int pl = 0; pl < 2; pl++) {
      const uint16_t* p = z.data() + (size_t)pl * n; cur.clear();
      int nb = (n + 15) / 16;
      for (int i = 0; i < nb; i++) {
        int b0 = i * 16, s = 0, L;
        if (tw >= 16 && b0 >= tw) { for (int j = 0; j < 16; j++) s += p[b0 - tw + j]; L = chooseL(s & 0xffff); }
        else if (i == 0) L = 2;
        else { for (int j = 0; j < 16; j++) s += p[b0 - 16 + j]; L = chooseL(s & 0xffff); }
        bool causal0 = (i > 0 || (tw >= 16 && b0 >= tw)) && s == 0;
        if (SKIP && causal0) { bool allz = true; for (int j = 0; j < 16 && b0 + j < n; j++) if (p[b0 + j] >> L) allz = false; if (allz) { skipped[pl]++; continue; } }
        for (int q = 0; q < 4; q++) {
          int let = 0;
          for (int k = 0; k < 4; k++) { int idx = b0 + q * 4 + k; int v = idx < n ? p[idx] >> L : 0; if (v > 3) v = 3; let |= v << (2 * k); }
          cur.push_back((uint8_t)let); zl[pl] += let == 0;
        }
      }
      nl[pl] += cur.size();
      tH[pl] += H0(cur); tF[pl] += 8.0 * pack(cur, out, 1); tU[pl] += 8.0 * pack(cur, out, 0);
      glob[pl].insert(glob[pl].end(), cur.begin(), cur.end());
    }
  }
  printf("skip=%d skipped blocks Y %.0f S %.0f\n", SKIP, skipped[0], skipped[1]);
  printf("%s  px %.0f  PLR total %.4f bpp\n", argv[1], px, plrbits / px);
  const char* nm[2] = {"Y", "S"};
  for (int pl = 0; pl < 2; pl++) {
    double gH = H0(glob[pl]);
    // whole-file stream, cut in 128 KiB blocks (FSE/huff0 block limit)
    double gF = 0, gU = 0; std::vector<uint8_t> blk;
    for (size_t o = 0; o < glob[pl].size(); o += 131072) {
      blk.assign(glob[pl].begin() + o, glob[pl].begin() + std::min(glob[pl].size(), o + 131072));
      gF += 8.0 * pack(blk, out, 1); gU += 8.0 * pack(blk, out, 0);
    }
    printf("  %s letters %.0f (zero %.1f%%)  per tile-plane: H0 %.4f  FSE %.4f  huff0 %.4f | whole file: H0 %.4f  FSE(128K) %.4f  huff0(128K) %.4f  bpp\n",
           nm[pl], nl[pl], 100 * zl[pl] / nl[pl], tH[pl] / px, tF[pl] / px, tU[pl] / px, gH / px, gF / px, gU / px);
  }
}
