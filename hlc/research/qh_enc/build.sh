#!/bin/sh
# build.sh [extra flags] : builds /tmp/fable_enc/plr (clang) and plr_gcc from the patched copy
cd /tmp/fable_enc/palic-2 || exit 1
F="-O3 -march=native -static -DNDEBUG -std=c++17 -DPLN_SSE -Wno-psabi -I/home/claude/plt2_repo/third_party/fse"
clang++ $F "$@" palic2r.cpp /tmp/cg/p2_*.o -o ../plr 2>&1 | grep -v "avx10\|warnings generated"
g++ $F "$@" palic2r.cpp /tmp/cg/p2_*.o -o ../plr_gcc 2>&1 | grep -v avx10
exit 0
