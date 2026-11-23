#!/bin/sh
cd /tmp/r8f5
log=/tmp/r8f5/holdout_full.txt
: > "$log"
for s in ios_hold desktop_hold websight_hold docs_hold scans_hold web_hold; do
  d=/tmp/r8f5/sets/$s; n=$(ls "$d"/*.ppm | wc -l)
  sd=/tmp/r8f5/hlf_$s; rm -rf "$sd"; mkdir -p "$sd"
  [ "$n" -gt 8 ] && ls "$d"/*.ppm | sort | awk 'NR%int('$n'/8+1)==1' | head -8 | xargs -I{} cp {} "$sd/" || cp "$d"/*.ppm "$sd/"
  c=$(ls "$sd"/*.ppm | wc -l)
  echo "=== $s (sample $c/$n) ===" | tee -a "$log"
  printf '%-12s ' old-ctx5tbl | tee -a "$log"; ./pls_probe      bench "$sd" -r 1 -s=zthr 2>&1 | tail -1 | tee -a "$log"
  printf '%-12s ' new20pct     | tee -a "$log"; ./pls_probe_new  bench "$sd" -r 1 -s=zthr 2>&1 | tail -1 | tee -a "$log"
  printf '%-12s ' new100pct    | tee -a "$log"; ./pls_probe_full bench "$sd" -r 1 -s=zthr 2>&1 | tail -1 | tee -a "$log"
done
echo HOLDOUT_DONE | tee -a "$log"
