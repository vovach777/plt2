// blkgeo.cpp - block geometry for the causal L (Vladimir, 2026-11-15): 16x1 (PLR) vs 2-D blocks, L always causal.
// Input: RLK dumps (per tile: int32 tw, th, PLR Y bits, PLR S bits; u16 zY[n], S[n], zU[n]).
// Per plane (Y, S) of every tile, blocks of BW x BH = 16 pixels; 4 letters of 4 pixels min(z >> L, 3) (16x1: 4 in a
// row, 2-D: 2x2 squares); L = clamp(bw(s >> 4) - 2, 0, 7) from a causal sum s of 16 decoded pixels:
//   up  : the block above (same shape); first block row: the block to the left; first block: s = 64 (L = 2)
//   max / min / avg : of up and left where both exist, else the one that exists
// Cost (the same model for every variant, ideal entropies over the whole set, no table cost):
//   blocks with s == 0: a flag "all zero" (ideal entropy, ctx plane); all-zero ones cost only the flag
//   letters: ideal entropy with ctx (plane, min(bw(s), 15)); tails h >= 3: Exp-Golomb 0 of h - 3; L raw bits per pixel
// Padding of edge blocks (tile side not a multiple of the block side) is coded as zeros, raw bits only for real pixels.
// usage: blkgeo dump.bin
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>
static int bw(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
struct Var { const char* name; int BW, BH, mode; };
static const Var V[] = {{"16x1 up (PLR)", 16, 1, 0}, {"16x1 max(up,left)", 16, 1, 1}, {"16x1 min(up,left)", 16, 1, 2},
                        {"16x1 avg(up,left)", 16, 1, 3}, {"8x2 up", 8, 2, 0}, {"8x2 max", 8, 2, 1}, {"8x2 min", 8, 2, 2},
                        {"8x2 avg", 8, 2, 3}, {"4x4 up", 4, 4, 0}, {"4x4 max", 4, 4, 1}, {"4x4 min", 4, 4, 2}, {"4x4 avg", 4, 4, 3}};
static const int NV = 12;
struct Acc { double let[2][16][256]; double flag[2][2]; double tails, raw; };
static const int NO = 8; static Acc A[NV][NO];
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb"); if (!f) return 1;
  memset(A, 0, sizeof A); double px = 0;  // A[variant][off]
  int32_t hd[4]; std::vector<uint16_t> z;
  while (fread(hd, 4, 4, f) == 4) {
    int tw = hd[0], th = hd[1], n = tw * th; px += n;
    z.resize(3 * (size_t)n); if (fread(z.data(), 2, 3 * (size_t)n, f) != 3 * (size_t)n) return 2;
    for (int v = 0; v < NV; v++) {
      const int BWd = V[v].BW, BH = V[v].BH, nbx = (tw + BWd - 1) / BWd, nby = (th + BH - 1) / BH;
      for (int off = 0; off < NO; off++) {
      Acc& a = A[v][off];
      for (int pl = 0; pl < 2; pl++) {
        const uint16_t* p = z.data() + (size_t)pl * n;
        auto at = [&](int x, int y) -> int { return (x < tw && y < th) ? p[y * tw + x] : 0; };
        std::vector<int> bs(nbx * nby);
        for (int by = 0; by < nby; by++) for (int bx = 0; bx < nbx; bx++) {
          int s = 0; for (int j = 0; j < BH; j++) for (int i = 0; i < BWd; i++) s += at(bx * BWd + i, by * BH + j);
          bs[by * nbx + bx] = s;
        }
        for (int by = 0; by < nby; by++) for (int bx = 0; bx < nbx; bx++) {
          int up = by ? bs[(by - 1) * nbx + bx] : -1, lf = bx ? bs[by * nbx + bx - 1] : -1, s;
          if (V[v].mode == 0) s = up >= 0 ? up : (lf >= 0 ? lf : 64);
          else if (up < 0 && lf < 0) s = 64;
          else if (up < 0 || lf < 0) s = up < 0 ? lf : up;
          else s = V[v].mode == 1 ? (up > lf ? up : lf) : V[v].mode == 2 ? (up < lf ? up : lf) : (up + lf + 1) >> 1;
          s &= 0xffff;
          int L = bw((uint32_t)s >> 4) - off; L = L < 0 ? 0 : (L > 7 ? 7 : L);
          int x0 = bx * BWd, y0 = by * BH;
          int real = 0; for (int j = 0; j < BH; j++) for (int i = 0; i < BWd; i++) real += (x0 + i < tw && y0 + j < th);
          bool zero = bs[by * nbx + bx] == 0;
          if (s == 0) { a.flag[pl][zero]++; if (zero) continue; }
          int c = bw(s); if (c > 15) c = 15;
          for (int q = 0; q < 4; q++) {
            int let = 0;
            for (int k = 0; k < 4; k++) {
              int xx, yy;
              if (BH == 1) { xx = x0 + q * 4 + k; yy = y0; }
              else { int qx = q % (BWd / 2), qy = q / (BWd / 2); xx = x0 + qx * 2 + (k & 1); yy = y0 + qy * 2 + (k >> 1); }
              int h = at(xx, yy) >> L;
              if (h >= 3) a.tails += 2 * bw((uint32_t)(h - 3 + 1)) - 1;
              let |= (h > 3 ? 3 : h) << (2 * k);
            }
            a.let[pl][c][let]++;
          }
          a.raw += (double)L * real;
        }
      }
      }
    }
  }
  printf("%s  %.0f px   (bpp: letters + flags + tails + raw; zU split not counted, same for all)\n", argv[1], px);
  double base = 0;
  for (int v = 0; v < NV; v++) {
   double bestt = 1e300; int bo = 0; double r4[4];
   for (int off = 0; off < NO; off++) {
    Acc& a = A[v][off]; double hl = 0, hf = 0;
    for (int pl = 0; pl < 2; pl++) {
      for (int c = 0; c < 16; c++) { double r = 0; for (int s = 0; s < 256; s++) r += a.let[pl][c][s];
        for (int s = 0; s < 256; s++) if (a.let[pl][c][s]) hl -= a.let[pl][c][s] * log2(a.let[pl][c][s] / r); }
      double r = a.flag[pl][0] + a.flag[pl][1]; for (int k = 0; k < 2; k++) if (a.flag[pl][k]) hf -= a.flag[pl][k] * log2(a.flag[pl][k] / r);
    }
    double tot = (hl + hf + a.tails + a.raw) / px;
    if (!v && off == 2) base = tot;
    if (tot < bestt) { bestt = tot; bo = off; r4[0] = hl / px; r4[1] = hf / px; r4[2] = a.tails / px; r4[3] = a.raw / px; }
    if (getenv("DETAIL") && strstr(V[v].name, getenv("DETAIL")))
      printf("    %-20s off %d: total %.4f  coded (no raw L planes) %.4f  raw %.4f  [letters %.4f flags %.4f tails %.4f]\n",
             V[v].name, off, tot, (hl + hf + a.tails) / px, a.raw / px, hl / px, hf / px, a.tails / px);
    if (off == 2) printf("  %-20s off 2: %.4f (%+.1f%%)", V[v].name, tot, 100 * (tot / base - 1));
   }
   printf("   best off %d: %.4f (%+.1f%%)   letters %.4f flags %.4f tails %.4f raw %.4f\n", bo, bestt, 100 * (bestt / base - 1), r4[0], r4[1], r4[2], r4[3]);
  }
}
