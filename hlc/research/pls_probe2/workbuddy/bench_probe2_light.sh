#!/bin/sh
cd /tmp/r8f5
log=/tmp/r8f5/bench_probe2_light.txt
: > "$log"
run() { printf '%-16s ' "$1" | tee -a "$log"; shift; "$@" 2>&1 | tail -1 | tee -a "$log"; }
for s in ios_test desktop_test websight_test docs_test scans_test full12 web8 valid kodak; do
  case "$s" in full12|web8|valid|kodak) d=light_$s ;; *) d=light_$s ;; esac
  echo "=== $s ===" | tee -a "$log"
  run "probe1-zthr-old" ./pls_probe      bench "$d" -r 1 -s=zthr
  run "probe2-65/65"    ./pls_probe2_new bench "$d" -r 1 -s=probe2
  run "probe2-50/50"    ./thr_Y50_S50    bench "$d" -r 1 -s=probe2
  run "probe2-55/55"    ./thr_Y55_S55    bench "$d" -r 1 -s=probe2
  run "r8f5-e9"         ./bench_e9       "$d" -r 1
done
echo BENCH_DONE | tee -a "$log"
