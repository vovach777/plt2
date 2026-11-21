// concept.cpp - the "(run class, value class) symbol + raw mantissas + Huffman, tables by the group of the previous
// symbol" coder as a model on the exact value streams the llrice RLGR coders get. Sizes only (bits), per image
// semi-static: Huffman code lengths per (stream, context) + a header estimate. Compared with the llrice file.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <queue>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include "qwen_fsm_v5.h"
static int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
constexpr int NV = 80, NR = 28, NS = NV * NR;        // symbol = cr * NV + cv
// Huffman code lengths (limit 16 by a simple rebalance: Kraft-fix), returns the payload bits
static double huff_bits(const std::vector<uint64_t>& f, int& used) {
    used = 0; std::vector<std::pair<uint64_t,int>> v;
    for (int i = 0; i < (int)f.size(); ++i) if (f[i]) { v.push_back({f[i], i}); ++used; }
    if (used <= 1) return used ? (double)v[0].first : 0.0;   // one symbol: 1 bit each (conservative)
    // standard Huffman by a heap of (weight, node)
    std::vector<int> par(2 * used, -1); std::vector<uint64_t> w(2 * used);
    using P = std::pair<uint64_t,int>; std::priority_queue<P, std::vector<P>, std::greater<P>> h;
    for (int i = 0; i < used; ++i) { w[i] = v[i].first; h.push({w[i], i}); }
    int nx = used;
    while (h.size() > 1) { auto a = h.top(); h.pop(); auto b = h.top(); h.pop(); w[nx] = a.first + b.first; par[a.second] = nx; par[b.second] = nx; h.push({w[nx], nx}); ++nx; }
    std::vector<int> len(used);
    for (int i = 0; i < used; ++i) { int d = 0, j = i; while (par[j] >= 0) { j = par[j]; ++d; } len[i] = d; }
    // length limit 16: clamp and fix Kraft by lengthening the shortest codes (rarely needed)
    for (auto& l : len) if (l > 16) l = 16;
    auto kraft = [&]() { double k = 0; for (int l : len) k += std::ldexp(1.0, -l); return k; };
    while (kraft() > 1.0 + 1e-12) { int bi = -1; for (int i = 0; i < used; ++i) if (len[i] < 16 && (bi < 0 || v[i].first < v[bi].first)) bi = i; len[bi]++; }
    double bits = 0; for (int i = 0; i < used; ++i) bits += double(v[i].first) * len[i];
    return bits;
}
static double entropy_bits(const std::vector<uint64_t>& f) {
    uint64_t t = 0; for (auto x : f) t += x; double b = 0;
    for (auto x : f) if (x) b -= double(x) * std::log2(double(x) / double(t));
    return b;
}
struct Tok { int cr, cv; uint32_t rl, v; int vk; int k0, kr0; };   // k0/kr0: RLGR state (k>>3, kr>>3) at the token start   // vk = value bit width (for groups / mantissa)
#ifndef HALF
#define HALF 0
#endif
static int vcls(uint32_t x) { const int b = bw(x); if (!HALF || b <= 1) return b;
    if (HALF == 2 && b >= 3) return 4 * b - 4 + int((x >> (b - 3)) & 3);   // quarter octaves: 2 bits after the top one
    return (HALF == 2 ? 4 * b - 4 : 2 * b - 2) + int((x >> (b - 2)) & 1); }
static void tokenize(const std::vector<uint32_t>& s, std::vector<Tok>& t) {
    t.clear(); uint32_t rl = 0;
    uint32_t k = 1 << 3, kr = 1 << 3, wrl = 0; int k0 = 1, kr0 = 1; bool start = true;   // llrice RLGR adaptation
    auto akr = [&](uint32_t v, uint32_t kr_) { const uint32_t vk = v >> kr_; if (vk == 0) kr -= kr > 2 ? 2 : kr; else if (vk != 1) { kr += vk; if (kr > 240) kr = 240; } };
    for (uint32_t x : s) {
        if (start) { k0 = int(k >> 3); kr0 = int(kr >> 3); start = false; }
        const uint32_t k_ = k >> 3, kr_ = kr >> 3;
        if (k_ > 0) {
            if (x == 0) { if (++wrl == (1u << k_)) { wrl = 0; k += 5; if (k > 240) k = 240; } }
            else { k -= k > 9 ? 9 : k; akr(x - 1, kr_); wrl = 0; }
        } else { akr(x, kr_); wrl = 0; if (x == 0) { k += 2; if (k > 240) k = 240; } else k -= k > 3 ? 3 : k; }
        if (!x) { ++rl; continue; }
        t.push_back({bw(rl), vcls(x), rl, x, bw(x), k0, kr0}); rl = 0; start = true;
    }
    if (rl) t.push_back({bw(rl), 0, rl, 0, 0, k0, kr0});
}
// context of the next symbol from the previous one
static int ctx_of(int mode, const Tok* p, const Tok* p2 = nullptr, const Tok* cur = nullptr) {
    if (mode == 10) return std::min(cur->k0, 15) * 16 + std::min(cur->kr0, 15);                 // RLGR state k x kr (256)
    if (mode == 11) return (std::min(cur->k0, 7) * 8 + std::min(cur->kr0, 7)) * 5 + (!p ? 4 : p->vk <= 1 ? 0 : p->vk == 2 ? 1 : p->vk <= 4 ? 2 : 3);   // k x kr (8x8) x prev k-group
    if (mode == 12) return std::min(cur->k0, 15) * 16 + std::min(cur->kr0, 15) + (!p ? 0 : 256 * std::min(p->cr, 2));   // k x kr x prev run 0/1/2+
    if (!p) return 0;
    const int vb = p->vk <= 1 ? 0 : p->vk == 2 ? 1 : p->vk <= 4 ? 2 : 3;   // 4 groups of the value class
    switch (mode) {
        case 0: return 0;                                             // one table
        case 1: return vb;                                            // prev value group (4)
        case 2: return vb * 2 + (p->cr > 0);                          // x prev run? (8)
        case 3: return std::min(p->vk, 15);                           // prev value class exact (16)
        case 4: return std::min(p->vk, 15) * 3 + std::min(p->cr, 2);  // x prev run 0/1/2+ (48)
        case 6: return std::min(p->vk, 15) * 10 + std::min(p->cr, 9);         // prev k x prev run octave (full)
        case 7: return std::min(p->cv, 39) * 3 + std::min(p->cr, 2);          // prev FULL value class (with sub-bits) x run3
        case 8: { auto g = [](const Tok* t) { return !t ? 4 : t->vk <= 1 ? 0 : t->vk == 2 ? 1 : t->vk <= 4 ? 2 : 3; };
                  return (g(p) * 5 + g(p2)) * 2 + (p->cr > 0); }               // two previous symbols (k groups) x run
        case 9: return std::min(p->cr, 15);                                  // prev run octave ONLY (16)
        case 5: { const int kg = p->vk <= 3 ? p->vk : p->vk <= 5 ? 4 : p->vk <= 8 ? 5 : 6; return kg * 2 + (p->cr > 0); }   // merged (14)
    }
    return 0;
}
int main(int argc, char** argv) {
    // argv: dir nstreams
    std::string dir = argv[1]; int ns = atoi(argv[2]);
    std::ifstream sz(dir + "/sizes.txt"); std::string line;
    const int NM = 15; double tot_llr = 0, px = 0, man = 0, hb[NM] = {}, hd[NM] = {}, en[NM] = {}; uint64_t ntok = 0;
    std::vector<uint32_t> s; std::vector<Tok> t;
    while (std::getline(sz, line)) {
        std::istringstream is(line); std::string nm; uint64_t bytes; uint32_t w, h; int n; is >> nm >> bytes >> w >> h >> n;
        tot_llr += double(bytes - 16) * 8; px += double(w) * h;
        for (int k = 0; k < ns; ++k) {
            std::ifstream f(dir + "/" + nm + ".s" + std::to_string(k), std::ios::binary | std::ios::ate);
            size_t L = f.tellg() / 4; f.seekg(0); s.resize(L); f.read((char*)s.data(), L * 4);
            tokenize(s, t); ntok += t.size();
            for (auto& x : t) man += (x.cr > 1 ? x.cr - 1 : 0) + (x.vk > 1 ? x.vk - 1 - (HALF && x.vk >= 2 ? 1 : 0) - (HALF == 2 && x.vk >= 3 ? 1 : 0) : 0);
            std::vector<int> qctx, ectx(t.size());
            qwen_fsm_ctx(t, qctx);                                   // mode 13: Qwen's TASK v5 automaton (23 contexts)
            { int a = 0, r = 0, pg = 4;                              // mode 14: our EMA automata 48+48 (fsm_eval ctx_ema48)
              for (size_t i = 0; i < t.size(); ++i) { ectx[i] = (std::min(a / 8, 5) * 6 + std::min(r / 8, 5)) * 5 + pg;
                  a += (t[i].vk * 8 - a + 4) >> 3; r += (t[i].cr * 8 - r + 2) >> 2;
                  pg = t[i].vk <= 1 ? 0 : t[i].vk == 2 ? 1 : t[i].vk <= 4 ? 2 : 3; } }
            for (int m = 0; m < NM; ++m) {
                std::map<int, std::vector<uint64_t>> fr;
                for (size_t i = 0; i < t.size(); ++i) {
                    const int cx = m == 13 ? qctx[i] : m == 14 ? ectx[i] : ctx_of(m, i ? &t[i - 1] : nullptr, i > 1 ? &t[i - 2] : nullptr, &t[i]);
                    auto& v = fr[cx]; if (v.empty()) v.assign(NS, 0);
                    v[std::min(t[i].cr, NR - 1) * NV + t[i].cv]++;
                }
                for (auto& [c, f] : fr) { int used; hb[m] += huff_bits(f, used); hd[m] += 8 + used * 14.0; en[m] += entropy_bits(f); }
            }
        }
    }
    printf("%-14s llrice %.4f bpp | tokens/px %.3f, mantissa %.4f bpp\n", dir.c_str(), tot_llr / px, ntok / px, man / px);
    const char* nmz[NM] = {"1 table", "prev v-group(4)", "x prev run(8)", "prev v-class(16)", "x run 0/1/2+(48)", "k{..3,4-5,6-8,9+}xrun(14)", "k x run octave(150)", "full class x run3(120)", "2 prev k-groups x run(50)", "prev run octave only(16)", "RLGR k x kr(256)", "k x kr(64) x prev k-grp", "k x kr x prev run3(768)", "Qwen FSM v5 (23)", "EMA 48+48 (180)"};
    for (int m = 0; m < NM; ++m)
        printf("  %-18s huff %.4f + hdr %.4f = %.4f bpp (%+.1f%%)  | entropy %.4f\n", nmz[m], (hb[m] + man) / px, hd[m] / px,
               (hb[m] + man + hd[m]) / px, 100.0 * ((hb[m] + man + hd[m]) / tot_llr - 1), (en[m] + man) / px);
}
