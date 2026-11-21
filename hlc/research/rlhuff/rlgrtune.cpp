// rlgrtune.cpp - tune the llrice RLGR adaptation constants on RLGR itself (exact bit count of llrice engine 1 without
// page tails), per image type: screens / photos / both; holdout = crops. Then the found rules are used as the
// Huffman context state (tune.cpp eval).
// Bits: run mode (k_ = k >> 3 > 0): full window 1 bit; run end 1 + k_ bits + Rice(v - 1, kr_); direct: Rice(v, kr_);
// Rice(x, m) = q + 1 + m bits, q = x >> m; q >= 32 -> 32 + 32 bits (llrice escape).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <thread>
struct Set { std::vector<std::vector<uint16_t>> st[3]; double px = 0, llr = 0; std::string name; };
static Set load(const std::string& dir) {
    Set S; S.name = dir; std::ifstream sz(dir + "/sizes.txt"); std::string line; std::vector<uint32_t> s;
    while (std::getline(sz, line)) {
        std::istringstream is(line); std::string nm; uint64_t bytes; uint32_t w, h; is >> nm >> bytes >> w >> h;
        S.px += double(w) * h; S.llr += double(bytes - 16) * 8;
        for (int k = 0; k < 3; ++k) {
            std::ifstream f(dir + "/" + nm + ".s" + std::to_string(k), std::ios::binary | std::ios::ate);
            size_t L = f.tellg() / 4; f.seekg(0); s.resize(L); f.read((char*)s.data(), L * 4);
            std::vector<uint16_t> v(L); for (size_t i = 0; i < L; ++i) v[i] = uint16_t(std::min<uint32_t>(s[i], 65535)); S.st[k].push_back(std::move(v));
        }
    }
    return S;
}
struct P { int UP = 5, DOWN = 9, UPD = 2, DOWND = 3, DR = 2, KG = 1; };
static inline uint64_t rice(uint32_t x, uint32_t m) { const uint32_t q = x >> m; return q >= 32 ? 64 : q + 1 + m; }
static double cost(const Set& S, const P& p) {
    double bits = 0;
    for (int k = 0; k < 3; ++k)
        for (auto& s : S.st[k]) {
            uint32_t k8 = 8, kr = 8, rl = 0; uint64_t b = 0; bool inrun = false;
            for (uint16_t x : s) {
                const uint32_t k_ = k8 >> 3, kr_ = kr >> 3;
                auto akr = [&](uint32_t v) { const uint32_t vk = v >> kr_; if (vk == 0) kr -= kr > uint32_t(p.DR) ? p.DR : kr; else if (vk != 1) { kr += vk * p.KG; if (kr > 240) kr = 240; } };
                if (k_ > 0) {
                    inrun = true;
                    if (x == 0) { if (++rl == (1u << k_)) { b += 1; rl = 0; k8 += p.UP; if (k8 > 240) k8 = 240; } }
                    else { b += 1 + k_ + rice(x - 1, kr_); k8 -= k8 > uint32_t(p.DOWN) ? p.DOWN : k8; akr(x - 1); rl = 0; inrun = false; }
                } else { b += rice(x, kr_); akr(x); rl = 0; if (x == 0) { k8 += p.UPD; if (k8 > 240) k8 = 240; } else k8 -= k8 > uint32_t(p.DOWND) ? p.DOWND : k8; }
            }
            if (inrun && rl) b += 1 + (k8 >> 3);
            bits += double(b);
        }
    return bits / S.px;
}
int main(int argc, char** argv) {
    Set A = load(argv[1]), B = load(argv[2]), C = load(argv[3]);
    P d; printf("check (llrice rule) model vs real file: %s %.4f / %.4f  %s %.4f / %.4f  %s %.4f / %.4f\n",
                A.name.c_str(), cost(A, d), A.llr / A.px, B.name.c_str(), cost(B, d), B.llr / B.px, C.name.c_str(), cost(C, d), C.llr / C.px);
    int P::* fld[] = {&P::UP, &P::DOWN, &P::UPD, &P::DOWND, &P::DR, &P::KG};
    const char* nm[] = {"UP", "DOWN", "UPD", "DOWND", "DR", "KG"};
    std::vector<std::vector<int>> grid = {{2,3,4,5,6,7,8,10,12,16}, {2,3,4,5,6,7,9,12,16,20,24}, {1,2,3,4,6,8}, {1,2,3,4,6,8}, {1,2,3,4,6}, {1,2,3}};
    for (int obj = 0; obj < 3; ++obj) {           // 0 screens, 1 photos, 2 both (relative)
        auto score = [&](const P& p) {
            if (obj == 0) return cost(A, p) / (A.llr / A.px);
            if (obj == 1) return cost(B, p) / (B.llr / B.px);
            double a, b; std::thread t([&] { a = cost(A, p); }); b = cost(B, p); t.join(); return a / (A.llr / A.px) + b / (B.llr / B.px); };
        P p; double best = score(p);
        for (int round = 0; round < 3; ++round)
            for (int i = 0; i < 6; ++i) { int keep = p.*fld[i];
                for (int v : grid[i]) { if (v == keep) continue; P q = p; q.*fld[i] = v; double sc = score(q); if (sc < best - 1e-7) { best = sc; p = q; } } }
        printf("%-8s UP=%d DOWN=%d UPD=%d DOWND=%d DR=%d KG=%d | RLGR bpp: %s %.4f  %s %.4f  %s %.4f\n",
               obj == 0 ? "screens" : obj == 1 ? "photos" : "both", p.UP, p.DOWN, p.UPD, p.DOWND, p.DR, p.KG,
               A.name.c_str(), cost(A, p), B.name.c_str(), cost(B, p), C.name.c_str(), cost(C, p));
        fflush(stdout);
    }
}
