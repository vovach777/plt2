// tok.cpp - export RLH tokens of llrice CODER 3 value dumps: per image, per stream (g, r, b): uint32 n, then n x uint16
// sym = cr << 6 | cv (cr = bw(zero run) 0..27, cv = half-octave value class 1..39, 0 = trailing zeros at the stream end).
// Also exports the raw run length and value? No: mantissas do not depend on the context. max_tok limits tokens/stream.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
static int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
static int vcls(uint32_t x) { const int b = bw(x); return b <= 1 ? b : 2 * b - 2 + int((x >> (b - 2)) & 1); }
int main(int argc, char** argv) {   // dir out first_img n_img max_tok
    std::string dir = argv[1]; FILE* o = fopen(argv[2], "wb"); int first = atoi(argv[3]), nimg = atoi(argv[4]); size_t maxt = atol(argv[5]);
    std::ifstream sz(dir + "/sizes.txt"); std::string line; std::vector<uint32_t> s; int i = 0, n = 0;
    while (std::getline(sz, line)) {
        if (i++ < first || n >= nimg) continue; ++n;
        std::istringstream is(line); std::string nm; is >> nm;
        for (int k = 0; k < 3; ++k) {
            std::ifstream f(dir + "/" + nm + ".s" + std::to_string(k), std::ios::binary | std::ios::ate);
            size_t L = f.tellg() / 4; f.seekg(0); s.resize(L); f.read((char*)s.data(), L * 4);
            std::vector<uint16_t> t; uint32_t rl = 0;
            for (uint32_t x : s) { if (t.size() >= maxt) break; if (!x) { ++rl; continue; } t.push_back(uint16_t(std::min(bw(rl), 27) << 6 | vcls(x))); rl = 0; }
            if (rl && t.size() < maxt) t.push_back(uint16_t(std::min(bw(rl), 27) << 6));
            uint32_t c = t.size(); fwrite(&c, 4, 1, o); fwrite(t.data(), 2, c, o);
        }
        fprintf(stderr, "%s\n", nm.c_str());
    }
    fclose(o);
}
