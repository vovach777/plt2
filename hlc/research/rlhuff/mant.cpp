// mant.cpp - how much the raw mantissas could still give: the empirical entropy of the value mantissa given the
// symbol class (HALF: half octaves), of its first bit only, and of the run mantissa given the run class.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
static int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
static double H(const std::map<uint32_t, uint64_t>& m) { uint64_t t = 0; for (auto& [k, v] : m) t += v; double b = 0; for (auto& [k, v] : m) b -= double(v) * std::log2(double(v) / t); return b; }
int main(int argc, char** argv) {
    std::string dir = argv[1]; int ns = atoi(argv[2]);
    std::ifstream sz(dir + "/sizes.txt"); std::string line; double px = 0;
    // per image (semi-static view) and pooled over the set (static view)
    double raw_v = 0, ent_v_img = 0, ent_v1_img = 0, raw_r = 0, ent_r_img = 0;
    std::map<int, std::map<uint32_t, uint64_t>> pool_v, pool_r; std::vector<uint32_t> s;
    while (std::getline(sz, line)) {
        std::istringstream is(line); std::string nm; uint64_t bytes; uint32_t w, h; int n; is >> nm >> bytes >> w >> h >> n; px += double(w) * h;
        for (int k = 0; k < ns; ++k) {
            std::ifstream f(dir + "/" + nm + ".s" + std::to_string(k), std::ios::binary | std::ios::ate);
            size_t L = f.tellg() / 4; f.seekg(0); s.resize(L); f.read((char*)s.data(), L * 4);
            std::map<int, std::map<uint32_t, uint64_t>> mv, mv1, mr; uint32_t rl = 0;
            for (uint32_t x : s) {
                if (!x) { ++rl; continue; }
                if (rl > 1) { const int b = bw(rl); raw_r += b - 1; mr[k * 64 + b][rl & ((1u << (b - 1)) - 1)]++; pool_r[k * 64 + b][rl & ((1u << (b - 1)) - 1)]++; }
                rl = 0;
                const int b = bw(x);
                if (b > 2) {                                   // half octave: bit b-2 is in the symbol, b-2 raw bits left
                    const int cls = k * 64 + 2 * b + int((x >> (b - 2)) & 1);
                    const uint32_t m = x & ((1u << (b - 2)) - 1); raw_v += b - 2;
                    mv[cls][m]++; pool_v[cls][m]++; mv1[cls][m >> (b - 3)]++;
                }
            }
            for (auto& [c, m] : mv) ent_v_img += H(m);
            for (auto& [c, m] : mv1) ent_v1_img += H(m) + 0;   // first bit only
            for (auto& [c, m] : mr) ent_r_img += H(m);
        }
    }
    double ent_v_pool = 0, ent_r_pool = 0; for (auto& [c, m] : pool_v) ent_v_pool += H(m); for (auto& [c, m] : pool_r) ent_r_pool += H(m);
    // first-bit entropy saving: raw first bit costs 1 per value with b > 2
    printf("%-12s value mantissa: raw %.4f bpp | entropy per image %.4f (pooled %.4f) | first bit: entropy %.4f of its raw bits\n",
           dir.c_str(), raw_v / px, ent_v_img / px, ent_v_pool / px, ent_v1_img / px);
    printf("%-12s run mantissa:   raw %.4f bpp | entropy per image %.4f (pooled %.4f)\n", "", raw_r / px, ent_r_img / px, ent_r_pool / px);
}
