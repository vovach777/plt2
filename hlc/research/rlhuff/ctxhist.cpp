// ctxhist.cpp - how the contexts "previous symbol" are used: share of tokens per (prev value bit width, prev run 0/1/2+),
// and the pooled entropy of the next symbol (half octaves) in each context against the one-table entropy.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
static int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
static int vcls(uint32_t x) { const int b = bw(x); return b <= 1 ? b : 2 * b - 2 + int((x >> (b - 2)) & 1); }
int main(int argc, char** argv) {
    std::string dir = argv[1]; int ns = atoi(argv[2]);
    std::ifstream sz(dir + "/sizes.txt"); std::string line; std::vector<uint32_t> s;
    std::map<int, std::map<int, uint64_t>> f; std::map<int, uint64_t> all; uint64_t tot = 0;
    while (std::getline(sz, line)) {
        std::istringstream is(line); std::string nm; is >> nm;
        for (int k = 0; k < ns; ++k) {
            std::ifstream in(dir + "/" + nm + ".s" + std::to_string(k), std::ios::binary | std::ios::ate);
            size_t L = in.tellg() / 4; in.seekg(0); s.resize(L); in.read((char*)s.data(), L * 4);
            uint32_t rl = 0; int pv = -1, pr = 0;
            for (uint32_t x : s) {
                if (!x) { ++rl; continue; }
                const int sym = std::min(bw(rl), 27) * 40 + vcls(x);
                const int c = pv < 0 ? 999 : std::min(pv, 15) * 3 + std::min(pr, 2);
                f[c][sym]++; all[sym]++; ++tot;
                pv = bw(x); pr = bw(rl); rl = 0;
            }
        }
    }
    auto H = [](const std::map<int, uint64_t>& m, uint64_t& t) { t = 0; for (auto& [a, b] : m) t += b; double h = 0; for (auto& [a, b] : m) h -= double(b) * std::log2(double(b) / t); return h; };
    uint64_t ta; const double h1 = H(all, ta) / ta;
    printf("%s: %llu tokens, one table %.3f bits/token\n", dir.c_str(), (unsigned long long)tot, h1);
    printf("prev k | run0: share  bits | run1: share  bits | run2+: share  bits\n");
    for (int pv = 1; pv <= 15; ++pv) {
        bool any = false; for (int r = 0; r < 3; ++r) any |= f.count(pv * 3 + r) > 0;
        if (!any) continue;
        printf("%6d |", pv);
        for (int r = 0; r < 3; ++r) {
            auto it = f.find(pv * 3 + r);
            if (it == f.end()) { printf("      -     -   |"); continue; }
            uint64_t t; const double h = H(it->second, t) / t;
            printf(" %6.2f%% %5.2f  |", 100.0 * t / tot, h);
        }
        printf("\n");
    }
}
