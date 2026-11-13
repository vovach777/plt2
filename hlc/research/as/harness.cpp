// harness.cpp - runs an as2cpp.py translation like palic-2r-as/run.mjs does: c in.pgm|ppm out | d in out.pnm.
// Build: python3 as2cpp.py ../../../palic-2r-as/plr.ts > /tmp/plr_as.cpp &&
//        g++ -std=c++17 -O2 -fwrapv -I. -DAS_SRC='"/tmp/plr_as.cpp"' harness.cpp -o asb   [-fsanitize=address,undefined,shift]
#include AS_SRC
#include <vector>
#include <string>
static std::vector<uint8_t> rd(const char* f) { FILE* p = fopen(f, "rb"); if (!p) { perror(f); exit(2); } std::vector<uint8_t> d; uint8_t b[65536]; size_t n;
    while ((n = fread(b, 1, sizeof b, p)) > 0) d.insert(d.end(), b, b + n); fclose(p); return d; }
static void wr(const char* f, const uint8_t* p, size_t n) { FILE* o = fopen(f, "wb"); fwrite(p, 1, n, o); fclose(o); }
static void fit(uint32_t addr, uint64_t extra) { const uint64_t need = addr + extra, have = uint64_t(mem_size()) << 16; if (need > have) mem_grow(int32_t((need - have + 65535) >> 16)); }
int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: c in out | d in out\n"); return 1; }
    as_static_init(); const uint32_t arena = uint32_t(init()); std::vector<uint8_t> f = rd(argv[2]); std::string mode = argv[1];
    if (mode == "c") {
        size_t i = 0; auto tok = [&] { while (i < f.size() && (f[i] <= 32 || f[i] == '#')) { if (f[i] == '#') while (i < f.size() && f[i] != 10) ++i; else ++i; }
            size_t s = i; while (i < f.size() && f[i] > 32) ++i; return std::string((const char*)f.data() + s, i - s); };
        std::string mg = tok(); int W = std::stoi(tok()), H = std::stoi(tok()); tok(); ++i; const int ch = mg == "P6" ? 3 : 1;
        const size_t n = size_t(W) * H * ch; fit(arena, n + 65536); memcpy(MEM + arena, f.data() + i, n);
#if defined(AS_CHANNELS)
        const int32_t r = encode(arena, W, H, ch, getenv("AS_FAST") ? 1 : 0);
#else
        const int32_t r = encode(arena, W, H);
#endif
        if (r < 0) { fprintf(stderr, "encode failed\n"); return 3; }
        if (size_t(r) >= f.size()) wr(argv[3], f.data(), f.size()); else wr(argv[3], MEM + outPtr(), size_t(r));
    } else {
        if (f.size() > 2 && f[0] == 'P' && (f[1] == '5' || f[1] == '6')) { wr(argv[3], f.data(), f.size()); return 0; }
        fit(arena, f.size() + (uint64_t(1) << 28)); memcpy(MEM + arena, f.data(), f.size()); const uint32_t dst = arena + uint32_t(f.size());
        const int32_t rc = decode(arena, int32_t(f.size()), dst); if (rc < 0) { fprintf(stderr, "decode failed\n"); return 2; }
#if defined(AS_CHANNELS)
        const int ch = channels();
#else
        const int ch = 3;
#endif
        std::string h = std::string(ch == 3 ? "P6" : "P5") + "\n" + std::to_string(width()) + " " + std::to_string(height()) + "\n255\n";
        std::vector<uint8_t> o(h.begin(), h.end()); o.insert(o.end(), MEM + dst, MEM + dst + size_t(width()) * height() * ch); wr(argv[3], o.data(), o.size());
    }
    return 0;
}
