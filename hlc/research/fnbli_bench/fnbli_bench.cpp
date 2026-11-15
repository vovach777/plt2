// fnbli_bench: in-memory benchmark for fNBLI (github WangXuan95/NBLI, src/fNBLI), same output as palic-2 bench.
// The image is loaded once; only fNBLIcompress / fNBLIdecompress are timed (best of -r runs), then verified.
// MB/s counts the raw image bytes (W * H * 3 for RGB, W * H for gray), as palic-2 and palic do.
//
// Build (fNBLI sources untouched, linked from its checkout):
//   N=~/NBLI/src; clang++ -O3 -march=native -std=c++17 -I$N/fNBLI -I$N/imageio fnbli_bench.cpp \
//     $N/fNBLI/fNBLI.cpp $N/imageio/imageio_pnm.c -o fnbli_bench
// Run: fnbli_bench dir|a.ppm ... [-r N]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "fNBLI.h"
extern "C" uint8_t* loadPNMImageFile(const char* p_filename, int* p_is_rgb, uint32_t* p_height, uint32_t* p_width);

int main(int argc, char** argv) {
    int runs = 3; std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-r" && i + 1 < argc) { runs = std::max(1, std::atoi(argv[++i])); continue; }
        if (std::filesystem::is_directory(a)) {
            std::vector<std::string> v;
            for (auto& e : std::filesystem::directory_iterator(a)) { const auto x = e.path().extension(); if (x == ".ppm" || x == ".pgm") v.push_back(e.path().string()); }
            std::sort(v.begin(), v.end()); files.insert(files.end(), v.begin(), v.end());
        } else files.push_back(a);
    }
    if (files.empty()) { std::fprintf(stderr, "usage: fnbli_bench dir|a.ppm ... [-r N]\n"); return 1; }
    using clk = std::chrono::steady_clock;
    double te = 0, td = 0, tpx = 0, tpix = 0, tb = 0; bool ok = true;
    for (const auto& fn : files) {
        int irgb = 0; uint32_t H = 0, W = 0;
        uint8_t* img = loadPNMImageFile(fn.c_str(), &irgb, &H, &W);
        if (!img) { std::fprintf(stderr, "cannot load %s\n", fn.c_str()); ok = false; continue; }
        const size_t n = size_t(W) * H * (irgb ? 3 : 1);
        std::vector<uint8_t> src(img, img + n), work(n); free(img);
        double be = 1e9, bd = 1e9; size_t sz = 0; bool g = true;
        for (int r = 0; r < runs; ++r) {
            std::memcpy(work.data(), src.data(), n);                        // in case the encoder writes to its input
            uint32_t crc = 0; size_t cs = 0;
            const auto t0 = clk::now();
            uint8_t* c = fNBLIcompress(cs, work.data(), irgb != 0, H, W, crc);
            const auto t1 = clk::now();
            if (!c) { g = false; break; }
            bool rgb2 = false; uint32_t h2 = 0, w2 = 0, crc2 = 0;
            const auto t2 = clk::now();
            uint8_t* d = fNBLIdecompress(c, rgb2, h2, w2, crc2);
            const auto t3 = clk::now();
            g = d && h2 == H && w2 == W && rgb2 == (irgb != 0) && std::memcmp(d, src.data(), n) == 0;
            delete[] d; delete[] c; sz = cs;
            be = std::min(be, std::chrono::duration<double>(t1 - t0).count());
            bd = std::min(bd, std::chrono::duration<double>(t3 - t2).count());
            if (!g) break;
        }
        ok &= g; tpx += double(n); tpix += double(W) * H; tb += double(sz); te += be; td += bd;
        std::printf("%-16s %5ux%-5u %9zu B %7.4f bpp  enc %6.1f dec %6.1f MB/s  %s\n", std::filesystem::path(fn).filename().string().c_str(),
                    W, H, sz, 8.0 * sz / (double(W) * H), n / be / 1e6, n / bd / 1e6, g ? "ok" : "FAIL");
    }
    std::printf("TOTAL %zu files %.4f bpp  enc %.1f dec %.1f MB/s  (fNBLI, in memory)  %s\n", files.size(), 8 * tb / tpix, tpx / te / 1e6, tpx / td / 1e6, ok ? "all ok" : "FAIL");
    return ok ? 0 : 3;
}
