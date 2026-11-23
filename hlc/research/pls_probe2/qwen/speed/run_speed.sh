#!/bin/sh
# Cost of the PLS probe: 4 modes of ONE build (tables 45/50), -r 7, 3 rounds, alternating order.
# A=-s (old detector) B=-s=zthr (probe-1) C=-s=probe2 (probe-2 45/50) D=-s=autocnt (count, old decision)
BIN=/tmp/r8f5/speed/b/palic-2r
B=/tmp/r8f5/bench
O=/home/devin/train/bench-independent/speed
mkdir -p "$O/logs"
SUM="$O/raw.txt"; : > "$SUM"
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "CPU: $CPU" | tee -a "$SUM"
mode_flag() { case $1 in A) echo "-s";; B) echo "-s=zthr";; C) echo "-s=probe2";; D) echo "-s=autocnt";; esac }
ORDERS="ABCD DCBA BADC"
for set in ios_test desktop_test websight_test docs_test scans_test full12 web8 valid41 kodak24; do
  r=0
  for ord in $ORDERS; do
    r=$((r+1))
    for m in $(echo $ord | fold -w1); do
      f=$(mode_flag $m)
      log="$O/logs/${set}__${m}__r${r}.txt"
      nice -n 19 taskset -c 3 "$BIN" bench "$B/$set" -r 7 "$f" > "$log" 2>&1
      printf '%-14s %s r%d %s ' "$set" "$m" "$r" "$f" | tee -a "$SUM"
      tail -1 "$log" | tee -a "$SUM"
    done
  done
done
echo SPEED-DONE | tee -a "$SUM"