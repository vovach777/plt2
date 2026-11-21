// dump.cpp: for each image, the values every RLGR coder of llrice (engine 1) receives + the llrice file size.
// out: <out>/<name>.s<id> (uint32 LE), and a line "name bytes w h nstreams" to stdout.
#include <cstdio>
#include <string>
#include <filesystem>
#include "llrice.hpp"
int main(int argc, char** argv) {
    std::string out = argv[1];
    for (int i = 2; i < argc; ++i) {
        RawImage img; img.load(argv[i]); img.le();
        llrice::g_rec.clear(); llrice::g_rec_id = 0;
        auto c = llrice::compress(img);
        std::string nm = std::filesystem::path(argv[i]).stem().string();
        for (size_t s = 0; s < llrice::g_rec.size(); ++s) {
            FILE* f = fopen((out + "/" + nm + ".s" + std::to_string(s)).c_str(), "wb");
            fwrite(llrice::g_rec[s].data(), 4, llrice::g_rec[s].size(), f); fclose(f);
        }
        printf("%s %zu %u %u %zu\n", nm.c_str(), c.size() * 8, img.width, img.height, llrice::g_rec.size());
    }
}
