// ema.cpp - context from LINEAR filters instead of the RLGR state: per token, a = EMA of bw(value), r = EMA of bw(run),
// fixed point x16, weight 2^-s (s = "inertia": 1 -> 1/2 ... 5 -> 1/32). ctx = bucket(a) x bucket(r) x prev k-group (5).
// Buckets: 1 bit (a/16), NB x NB. Cost = pooled entropy per set (static view) + raw mantissas, bpp.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
static int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
static int vcls(uint32_t x) { const int b = bw(x); return b <= 1 ? b : 2 * b - 2 + int((x >> (b - 2)) & 1); }
struct Set { std::vector<std::vector<uint16_t>> st[3]; double px = 0; };
static Set load(const std::string& dir) {
    Set S; std::ifstream sz(dir + "/sizes.txt"); std::string line; std::vector<uint32_t> s;
    while (std::getline(sz, line)) {
        std::istringstream is(line); std::string nm; uint64_t bytes; uint32_t w, h; is >> nm >> bytes >> w >> h; S.px += double(w) * h;
        for (int k = 0; k < 3; ++k) {
            std::ifstream f(dir + "/" + nm + ".s" + std::to_string(k), std::ios::binary | std::ios::ate);
            size_t L = f.tellg() / 4; f.seekg(0); s.resize(L); f.read((char*)s.data(), L * 4);
            std::vector<uint16_t> v(L); for (size_t i = 0; i < L; ++i) v[i] = uint16_t(std::min<uint32_t>(s[i], 65535)); S.st[k].push_back(std::move(v));
        }
    }
    return S;
}
constexpr int NV = 40, NR_ = 28, NS = NV * NR_;
static int g_P = 16;   // internal precision: units per bit (FSM state count = buckets x P)
static double cost(const Set& S, int sa, int sr, int NB, int QA, int QR) {
    double bits = 0, man = 0; const int NC = NB * NB * 5; std::vector<uint32_t> cnt;
    for (int k = 0; k < 3; ++k) {
        cnt.assign(size_t(NC) * NS, 0);
        for (auto& s : S.st[k]) {
            int a = 0, r = 0, pg = 4; uint32_t rl = 0;
            for (uint16_t x : s) {
                if (!x) { ++rl; continue; }
                const int c = (std::min(a / QA, NB - 1) * NB + std::min(r / QR, NB - 1)) * 5 + pg;
                const int cr = bw(rl), vb = bw(x);
                cnt[size_t(c) * NS + std::min(cr, NR_ - 1) * NV + vcls(x)]++;
                man += (cr > 1 ? cr - 1 : 0) + (vb > 2 ? vb - 2 : 0);
                a += (vb * g_P - a + (1 << (sa - 1))) >> sa; r += (cr * g_P - r + (1 << (sr - 1))) >> sr;   // rounded EMA   // EMA update after the token (causal)
                pg = vb <= 1 ? 0 : vb == 2 ? 1 : vb <= 4 ? 2 : 3; rl = 0;
            }
        }
        for (int c = 0; c < NC; ++c) { const uint32_t* f = &cnt[size_t(c) * NS]; double t = 0; for (int i = 0; i < NS; ++i) t += f[i];
            for (int i = 0; i < NS; ++i) if (f[i]) bits -= f[i] * std::log2(f[i] / t); }
    }
    return (bits + man) / S.px;
}
int main(int argc, char** argv) {
    Set A = load(argv[1]), B = load(argv[2]), C = load(argv[3]);
    for (int P : {16, 8, 4, 2, 1}) { g_P = P;
        const int states_a = 6 * P, states_r = 6 * P;
        printf("precision %2d/bit: FSM states %d x %d x 5 = %d | full %.4f  clic16 %.4f  crops %.4f\n", P, states_a, states_r, states_a * states_r * 5,
               cost(A, 3, 2, 6, P, P), cost(B, 3, 2, 6, P, P), cost(C, 3, 2, 6, P, P)); fflush(stdout); }
}
