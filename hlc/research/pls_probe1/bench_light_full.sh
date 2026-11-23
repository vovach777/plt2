#!/bin/sh
cd /tmp/r8f5
log=/tmp/r8f5/bench_light_full.txt
: > "$log"
run() { printf '%-12s ' "$1" | tee -a "$log"; shift; "$@" 2>&1 | tail -1 | tee -a "$log"; }
for s in ios_test desktop_test websight_test docs_test scans_test full12 web8 valid kodak; do
  case "$s" in full12|web8|valid|kodak) d=/tmp/r8f5/scr/$s ;; *) d=/tmp/r8f5/sets/$s ;; esac
  n=$(ls "$d"/*.ppm 2>/dev/null | wc -l)
  sd=/tmp/r8f5/light_$s; rm -rf "$sd"; mkdir -p "$sd"
  ls "$d"/*.ppm | sort | awk 'NR%int('$n'/8+1)==1' | head -8 | xargs -I{} cp {} "$sd/" 2>/dev/null
  c=$(ls "$sd"/*.ppm 2>/dev/null | wc -l)
  echo "=== $s (sample $c/$n) ===" | tee -a "$log"
  run "ctx5-auto"  ./pls_probe      bench "$sd" -r 1 -s
  run "zthr-old"   ./pls_probe      bench "$sd" -r 1 -s=zthr
  run "zthr-100"   ./pls_probe_full bench "$sd" -r 1 -s=zthr
  run "zthrc-100"  ./pls_probe_full bench "$sd" -r 1 -s=zthrc
  run "r8f5-e9"    ./bench_e9       "$sd" -r 1
done
echo LIGHT_DONE | tee -a "$log"
