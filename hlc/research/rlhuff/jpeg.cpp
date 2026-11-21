// jpeg.cpp - "pure JPEG" tokens on the llrice coder streams: symbol = (run 0..15, size = bw(v)), ZRL = 16 zeros,
// raw size bits; Huffman per (stream, context of the previous symbol). Two table policies:
//  semi = tables per image (+ header estimate 8 + 14 bits per used symbol), pool = one table set per data set
//  (all images pooled, no header: an optimistic static table trained on the test itself).
// Contexts: 0 = one table; 1 = prev size group {0(ZRL),1,2,3-4,5+} x prev run {0, 1+}.  Compare: concept.cpp.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <queue>
#include <fstream>
#include <sstream>
#include <algorithm>
static int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
static double huff(const std::vector<uint64_t>& f, int& used) {
    std::vector<uint64_t> v; for (auto x : f) if (x) v.push_back(x); used = v.size();
    if (used <= 1) return used ? double(v[0]) : 0;
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>> h(v.begin(), v.end());
    double bits = 0; while (h.size() > 1) { auto a = h.top(); h.pop(); auto b = h.top(); h.pop(); bits += a + b; h.push(a + b); }
    return bits;   // sum of internal node weights = total code bits (no length limit)
}
int main(int argc, char** argv) {
    std::string dir = argv[1]; int ns = atoi(argv[2]);
    std::ifstream sz(dir + "/sizes.txt"); std::string line; std::vector<uint32_t> s;
    double px = 0, llr = 0, man = 0, semi[2] = {}, hdr[2] = {}; uint64_t ntok = 0;
    std::map<int, std::vector<uint64_t>> pool[2];
    while (std::getline(sz, line)) {
        std::istringstream is(line); std::string nm; uint64_t bytes; uint32_t w, h; is >> nm >> bytes >> w >> h;
        px += double(w) * h; llr += double(bytes - 16) * 8;
        for (int k = 0; k < ns; ++k) {
            std::ifstream f(dir + "/" + nm + ".s" + std::to_string(k), std::ios::binary | std::ios::ate);
            size_t L = f.tellg() / 4; f.seekg(0); s.resize(L); f.read((char*)s.data(), L * 4);
            std::vector<int> sym; uint32_t rl = 0;
            for (uint32_t x : s) {
                if (!x) { if (++rl == 16) { sym.push_back(0xF0); rl = 0; } continue; }   // ZRL
                const int b = bw(x); sym.push_back((int(rl) << 4) | b); man += b - 1; rl = 0;
            }
            if (rl) sym.push_back(0x00);                                                  // EOB-like end
            ntok += sym.size();
            for (int m = 0; m < 2; ++m) {
                std::map<int, std::vector<uint64_t>> fr; int prev = -1;
                for (int y : sym) {
                    int c = 0;
                    if (m == 1 && prev >= 0) { const int ps = prev & 15, pr = prev >> 4;
                        const int sg = prev == 0xF0 ? 0 : ps <= 1 ? 1 : ps == 2 ? 2 : ps <= 4 ? 3 : 4; c = sg * 2 + (pr > 0) + 1; }
                    auto& v = fr[c]; if (v.empty()) v.assign(256, 0); v[y]++;
                    auto& pv = pool[m][k * 100 + c]; if (pv.empty()) pv.assign(256, 0); pv[y]++;
                    prev = y;
                }
                for (auto& [c, f] : fr) { int u; semi[m] += huff(f, u); hdr[m] += 8 + 14.0 * u; }
            }
        }
    }
    double pl[2] = {}; for (int m = 0; m < 2; ++m) for (auto& [c, f] : pool[m]) { int u; pl[m] += huff(f, u); }
    printf("%-12s llrice %.4f | tokens/px %.3f\n", dir.c_str(), llr / px, ntok / px);
    const char* nmz[2] = {"one table", "prev symbol (11 ctx)"};
    for (int m = 0; m < 2; ++m)
        printf("  JPEG %-20s semi %.4f (%+.1f%%) | pooled static %.4f (%+.1f%%)\n", nmz[m], (semi[m] + man + hdr[m]) / px,
               100 * ((semi[m] + man + hdr[m]) / llr - 1), (pl[m] + man) / px, 100 * ((pl[m] + man) / llr - 1));
}
