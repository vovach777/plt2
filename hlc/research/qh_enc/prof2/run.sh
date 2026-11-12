#!/bin/bash
# run.sh [sets...] : stage profile of the current patched source (clang native)
cd /tmp/fable_enc2 && python3 prof/mkprof.py /tmp/fable_enc2/palic-2/palic2r.cpp prof/prof_new.cpp ${A:-prof/anchors_orig.py} && clang++ -O3 -march=native -static -DNDEBUG -std=c++17 -DPLN_SSE -Wno-psabi -I/home/claude/plt2_repo/third_party/fse prof/prof_new.cpp /tmp/cg/p2_*.o -o prof_new 2>&1 | grep -E "error"
S=/tmp/claude-0/-home-claude/739fa5c5-1bdc-534c-8b96-8baaa717ae9c/scratchpad
for d in "$@"; do d=${d/@S/$S}; echo "== $(basename $d)"; ./prof_new bench $d -r 3 | tail -17 | grep -v "ok$"; done
