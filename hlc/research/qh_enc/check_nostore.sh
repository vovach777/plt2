#!/bin/bash
# check_nostore.sh : the same cmp with the "store the input if it does not shrink" rule switched off in both builds
# (random / binary noise images: escapes, long tails, lane growth), clang and gcc, PLR and PLF
F="-O3 -march=native -static -DNDEBUG -std=c++17 -DPLN_SSE -Wno-psabi -I/home/claude/plt2_repo/third_party/fse"
T=/tmp/fable_enc/ns; mkdir -p $T/o $T/n
for v in o n; do src=/tmp/fable_enc/orig/palic-2; [ $v = n ] && src=/tmp/fable_enc/palic-2
  cp $src/*.hpp $src/*.h $T/$v/ 2>/dev/null; cp -r $src/llrice $T/$v/ 2>/dev/null
  sed 's/o.size() >= src.size()/false/; s#\.\./palic-1/netpbm.hpp#/tmp/fable_enc/palic-1/netpbm.hpp#' $src/palic2r.cpp > $T/$v/palic2r.cpp; done
(cd $T/o && clang++ $F palic2r.cpp /tmp/cg/p2_*.o -o ../o_bin 2>/dev/null)
G="-O3 -static -DNDEBUG -std=c++17 -DPLN_SSE -Wno-psabi -I/home/claude/plt2_repo/third_party/fse"
(cd $T/n && clang++ $F palic2r.cpp /tmp/cg/p2_*.o -o ../n_clang 2>/dev/null; g++ $F palic2r.cpp /tmp/cg/p2_*.o -o ../n_gcc 2>/dev/null
 clang++ $G -march=x86-64-v2 palic2r.cpp /tmp/cg/p2_*.o -o ../n_clang_v2 2>/dev/null; g++ $G -march=x86-64-v2 palic2r.cpp /tmp/cg/p2_*.o -o ../n_gcc_v2 2>/dev/null
 clang++ $G -march=x86-64-v2 -UPLN_SSE palic2r.cpp /tmp/cg/p2_*.o -o ../n_v2_scalar 2>/dev/null; clang++ $G -march=x86-64-v3 -UPLN_SSE palic2r.cpp /tmp/cg/p2_*.o -o ../n_v3_nosse 2>/dev/null)
for B in n_clang n_gcc n_clang_v2 n_gcc_v2 n_v2_scalar n_v3_nosse; do bad=0; n=0
for f in /tmp/fable_enc/edge/*.p?m /tmp/fable_rlgr/synth/*.pgm /tmp/qhw/sets/clic_test_color/*.ppm; do for fl in "" "-f"; do
  $T/o_bin c $f $T/a.plr $fl; $T/$B c $f $T/b.plr $fl; n=$((n+1)); cmp -s $T/a.plr $T/b.plr || { bad=$((bad+1)); echo "DIFF $B $f $fl"; }
  $T/$B d $T/b.plr $T/b.out && cmp -s $T/b.out $f || { bad=$((bad+1)); echo "RT-FAIL $B $f $fl"; }
done; done; echo "$B: $n files, $bad problems"; done
