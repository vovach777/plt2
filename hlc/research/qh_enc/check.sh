#!/bin/bash
# check.sh BIN [BIN...] : .plr files of BIN byte-identical to plr_orig on every set (also -f = PLF) + decode round trip
S=/tmp/claude-0/-home-claude/739fa5c5-1bdc-534c-8b96-8baaa717ae9c/scratchpad
T=/tmp/fable_enc/chk; mkdir -p $T
for B in "$@"; do bad=0; n=0
for d in $S/kodak768 $S/gray $S/col19 /tmp/qhw/sets/clic_test_color /tmp/qhw/sets/clic_test_gray /tmp/qhw/sets/clic_t117_color /tmp/qhw/sets/clic_t117_gray /tmp/fable_rlgr/synth /tmp/fable_enc/edge; do
  for f in $d/*.p?m; do [ -f "$f" ] || continue
    for fl in "" "-f"; do
      /tmp/fable_enc/plr_orig c $f $T/a.plr $fl; /tmp/fable_enc/$B c $f $T/b.plr $fl
      n=$((n+1)); cmp -s $T/a.plr $T/b.plr || { bad=$((bad+1)); echo "DIFF $B $(basename $d) file#$n $fl"; }
      /tmp/fable_enc/$B d $T/b.plr $T/b.out && cmp -s $T/b.out $f || { bad=$((bad+1)); echo "RT-FAIL $B $(basename $d) file#$n $fl"; }
    done
  done
done; echo "$B: $n files compared, $bad problems"; done
rm -rf $T
