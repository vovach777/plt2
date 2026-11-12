#!/bin/bash
# final_bench.sh : in-process bench (best of 5 / 3), original vs patched, clang and gcc; then process wall clock of "c" and "d"
S=/tmp/claude-0/-home-claude/739fa5c5-1bdc-534c-8b96-8baaa717ae9c/scratchpad
cd /tmp/fable_enc
for d in $S/kodak768 /tmp/qhw/sets/clic_t117_color /tmp/qhw/sets/clic_t117_gray; do r=3; [ $(basename $d) = kodak768 ] && r=5
  for b in plr_orig plr plr_orig_gcc plr_gcc plr_sse; do echo "$(basename $d) $b: $(./$b bench $d -r $r | tail -1)"; done; done
mkdir -p wc; for d in /tmp/qhw/sets/clic_t117_color /tmp/qhw/sets/clic_t117_gray; do mb=$(cat $d/*.p?m | wc -c)
  for b in plr_orig plr; do best=999999; bestd=999999
    for rep in 1 2 3; do t0=$(date +%s%N); for f in $d/*.p?m; do ./$b c $f wc/$(basename $f).plr; done; t1=$(date +%s%N)
      for f in wc/*.plr; do ./$b d $f wc/out.pnm; done; t2=$(date +%s%N)
      e=$(( (t1-t0)/1000000 )); dd=$(( (t2-t1)/1000000 )); [ $e -lt $best ] && best=$e; [ $dd -lt $bestd ] && bestd=$dd; rm -f wc/*.plr; done
    echo "wall $(basename $d) $b: enc $(( mb/1000*1000/best/1000 )) MB/s  dec $(( mb/1000*1000/bestd/1000 )) MB/s (ms $best / $bestd)"; done; done
rm -rf wc
