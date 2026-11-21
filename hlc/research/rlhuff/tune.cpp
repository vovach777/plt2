// tune.cpp - tune the RLGR state used as the RLH context (it codes nothing, so any rule is free):
// ctx = bucket(k) x bucket(kr) x prev value-width group (5), symbol = (run octave, half-octave value class).
// Cost = pooled entropy over a set per stream (static-table view, no header) + raw mantissas, bpp.
// Params: UP DOWN (run window k, 1/8 bit), UPD DOWND (k when no runs), DR (kr down), KG (kr up = vk*KG),
//         QK QR (bucket step in 1/8 bit), NK NR (number of buckets).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <thread>
static int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
static int vcls(uint32_t x) { const int b = bw(x); return b <= 1 ? b : 2 * b - 2 + int((x >> (b - 2)) & 1); }
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
static int g_cutk[64], g_cutr[64]; static int g_ncut = 0;   // quantile cuts (QK==0 / QR==0)
static inline int bucket(uint32_t v, int q, int n, const int* cut) {
    if (q > 0) return std::min<int>(v / q, n - 1);
    if (q < 0) { const uint32_t x = v / uint32_t(-q); return std::min<int>(x ? 32 - __builtin_clz(x) : 0, n - 1); }
    int b = 0; while (b < n - 1 && int(v) >= cut[b]) ++b; return b;
}
struct P { int UP = 5, DOWN = 9, UPD = 2, DOWND = 3, DR = 2, KG = 1, QK = 8, QR = 8, NK = 8, NR = 8; };
constexpr int NV = 40, NR_ = 28, NS = NV * NR_;
static double cost(const Set& S, const P& p) {
    double bits = 0, man = 0;
    const int NC = p.NK * p.NR * 5 + 1;
    std::vector<uint32_t> cnt;
    for (int k = 0; k < 3; ++k) {
        cnt.assign(size_t(NC) * NS, 0);
        for (auto& s : S.st[k]) {
            uint32_t k8 = 8, kr = 8, wrl = 0, rl = 0; int c = NC - 1; bool start = true; int pg = 4;
            for (uint16_t x : s) {
                if (start) { c = (bucket(k8, p.QK, p.NK, g_cutk) * p.NR + bucket(kr, p.QR, p.NR, g_cutr)) * 5 + pg; start = false; }
                const uint32_t k_ = k8 >> 3, kr_ = kr >> 3;
                auto akr = [&](uint32_t v) { const uint32_t vk = v >> kr_; if (vk == 0) kr -= kr > uint32_t(p.DR) ? p.DR : kr; else if (vk != 1) { kr += vk * p.KG; if (kr > 240) kr = 240; } };
                if (k_ > 0) {
                    if (x == 0) { if (++wrl == (1u << k_)) { wrl = 0; k8 += p.UP; if (k8 > 240) k8 = 240; } }
                    else { k8 -= k8 > uint32_t(p.DOWN) ? p.DOWN : k8; akr(x - 1); wrl = 0; }
                } else { akr(x); wrl = 0; if (x == 0) { k8 += p.UPD; if (k8 > 240) k8 = 240; } else k8 -= k8 > uint32_t(p.DOWND) ? p.DOWND : k8; }
                if (!x) { ++rl; continue; }
                const int cr = bw(rl), vb = bw(x);
                cnt[size_t(c) * NS + std::min(cr, NR_ - 1) * NV + vcls(x)]++;
                man += (cr > 1 ? cr - 1 : 0) + (vb > 2 ? vb - 2 : 0);
                pg = vb <= 1 ? 0 : vb == 2 ? 1 : vb <= 4 ? 2 : 3; rl = 0; start = true;
            }
            if (rl) cnt[size_t(c) * NS + std::min(bw(rl), NR_ - 1) * NV]++;
        }
        for (int c = 0; c < NC; ++c) {
            const uint32_t* f = &cnt[size_t(c) * NS]; double t = 0; for (int i = 0; i < NS; ++i) t += f[i];
            for (int i = 0; i < NS; ++i) if (f[i]) bits -= f[i] * std::log2(f[i] / t);
        }
    }
    return (bits + man) / S.px;
}
int main(int argc, char** argv) {
    Set A = load(argv[1]), B = load(argv[2]), C = load(argv[3]);
    if (argc > 4) {   // eval list: "UP,DOWN,UPD,DOWND,DR,KG,QK,QR,NK,NR" ...
        auto quant = [&](const P& p) {     // equal-token-count cuts of k8 and kr at token starts over A and B
            std::vector<uint64_t> hk(241, 0), hr(241, 0); uint64_t n = 0;
            for (const Set* S : {&A, &B}) for (int k = 0; k < 3; ++k) for (auto& s : S->st[k]) {
                uint32_t k8 = 8, kr = 8, wrl = 0; bool start = true;
                for (uint16_t x : s) {
                    if (start) { hk[k8]++; hr[kr]++; ++n; start = false; }
                    const uint32_t k_ = k8 >> 3, kr_ = kr >> 3;
                    auto akr = [&](uint32_t v) { const uint32_t vk = v >> kr_; if (vk == 0) kr -= kr > uint32_t(p.DR) ? p.DR : kr; else if (vk != 1) { kr += vk * p.KG; if (kr > 240) kr = 240; } };
                    if (k_ > 0) { if (x == 0) { if (++wrl == (1u << k_)) { wrl = 0; k8 += p.UP; if (k8 > 240) k8 = 240; } } else { k8 -= k8 > uint32_t(p.DOWN) ? p.DOWN : k8; akr(x - 1); wrl = 0; } }
                    else { akr(x); wrl = 0; if (x == 0) { k8 += p.UPD; if (k8 > 240) k8 = 240; } else k8 -= k8 > uint32_t(p.DOWND) ? p.DOWND : k8; }
                    if (x) start = true;
                }
            }
            auto cuts = [&](const std::vector<uint64_t>& h, int nb, int* cut) { uint64_t acc = 0; int b = 0;
                for (int v = 0; v <= 240 && b < nb - 1; ++v) { acc += h[v]; if (acc * nb >= n * uint64_t(b + 1)) cut[b++] = v + 1; } while (b < nb - 1) { cut[b] = 241 + b; ++b; } };
            cuts(hk, p.NK, g_cutk); cuts(hr, p.NR, g_cutr);
        };
        for (int a = 4; a < argc; ++a) { P p; sscanf(argv[a], "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", &p.UP, &p.DOWN, &p.UPD, &p.DOWND, &p.DR, &p.KG, &p.QK, &p.QR, &p.NK, &p.NR);
            if (p.QK == 0 || p.QR == 0) quant(p);
            printf("%-28s ctx %4d | full %.4f  clic16 %.4f  crops %.4f\n", argv[a], p.NK * p.NR * 5 + 1, cost(A, p), cost(B, p), cost(C, p)); fflush(stdout); }
        return 0;
    }   // tune on A+B (screens + photos), C = holdout
    auto both = [&](const P& p) { double a, b; std::thread t([&] { a = cost(A, p); }); b = cost(B, p); t.join(); return std::make_pair(a, b); };
    auto score = [&](std::pair<double,double> r) { return r.first / (A.llr / A.px) + r.second / (B.llr / B.px); };   // relative to llrice
    P p; auto r0 = both(p); double best = score(r0);
    printf("start  %s %.4f  %s %.4f  holdout %.4f\n", A.name.c_str(), r0.first, B.name.c_str(), r0.second, cost(C, p));
    int P::* fld[] = {&P::UP, &P::DOWN, &P::UPD, &P::DOWND, &P::DR, &P::KG, &P::QK, &P::QR, &P::NK, &P::NR};
    const char* nm[] = {"UP", "DOWN", "UPD", "DOWND", "DR", "KG", "QK", "QR", "NK", "NR"};
    std::vector<std::vector<int>> grid = {{2,3,4,5,6,8,10,12}, {3,5,7,9,12,16,20,24}, {1,2,3,4,6}, {1,2,3,4,6,8}, {1,2,3,4,6}, {1,2,3},
                                          {4,8,12,16,24}, {4,8,12,16,24}, {4,6,8,10,12}, {4,6,8,10,12}};
    for (int round = 0; round < 2; ++round)
        for (int i = 0; i < 10; ++i) {
            int keep = p.*fld[i];
            for (int v : grid[i]) { if (v == keep) continue; P q = p; q.*fld[i] = v; double sc = score(both(q)); if (sc < best - 1e-6) { best = sc; p = q; } }
            auto r = both(p);
            printf("r%d %-6s -> %3d | %.4f %.4f\n", round, nm[i], p.*fld[i], r.first, r.second); fflush(stdout);
        }
    auto r = both(p);
    printf("final UP=%d DOWN=%d UPD=%d DOWND=%d DR=%d KG=%d QK=%d QR=%d NK=%d NR=%d | %s %.4f  %s %.4f  holdout %s %.4f\n",
           p.UP, p.DOWN, p.UPD, p.DOWND, p.DR, p.KG, p.QK, p.QR, p.NK, p.NR, A.name.c_str(), r.first, B.name.c_str(), r.second, C.name.c_str(), cost(C, p));
}
