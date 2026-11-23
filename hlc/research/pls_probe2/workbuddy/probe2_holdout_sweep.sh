#!/bin/sh
# usage: probe2_holdout_sweep.sh <binary-with-new-tables-65/65>  (thr binaries must be rebuilt with new tables too)
cd /tmp/r8f5
log=/tmp/r8f5/probe2_holdout.txt
: > "$log"
B65=$1
# make sure thr binaries carry the new tables as well
for Y in 55 75; do for S in 55 65 75; do
  [ "$Y" = "65" ] && continue
  rm -rf bt; cmake -S plt2probe2new/palic-2 -B bt -DCMAKE_BUILD_TYPE=Release -DPLS_ZTHR=$Y -DPLS_ZTHR_S=$S > /dev/null 2>&1
  cmake --build bt --target palic-2r -j4 > /dev/null 2>&1 && cp bt/palic-2r "thr_Y${Y}_S${S}"
done; done
for S in 55 75; do
  rm -rf bt; cmake -S plt2probe2new/palic-2 -B bt -DCMAKE_BUILD_TYPE=Release -DPLS_ZTHR=65 -DPLS_ZTHR_S=$S > /dev/null 2>&1
  cmake --build bt --target palic-2r -j4 > /dev/null 2>&1 && cp bt/palic-2r "thr_Y65_S${S}"
done
# holdout samples
for s in ios_hold desktop_hold websight_hold docs_hold scans_hold web_hold; do
  d=/tmp/r8f5/sets/$s; n=$(ls "$d"/*.ppm | wc -l)
  sd=/tmp/r8f5/h2_$s; rm -rf "$sd"; mkdir -p "$sd"
  [ "$n" -gt 8 ] && ls "$d"/*.ppm | sort | awk 'NR%int('$n'/8+1)==1' | head -8 | xargs -I{} cp {} "$sd/" || cp "$d"/*.ppm "$sd/"
done
echo "=== holdout: probe-1 (ae3dbaf tables, -s=zthr) vs probe-2 (new tables, -s=probe2) ===" | tee -a "$log"
for s in ios_hold desktop_hold websight_hold docs_hold scans_hold web_hold; do
  sd=/tmp/r8f5/h2_$s; c=$(ls "$sd"/*.ppm | wc -l)
  echo "=== $s (sample $c) ===" | tee -a "$log"
  printf '%-16s ' probe1-zthr-old | tee -a "$log"; ./pls_probe2 bench "$sd" -r 1 -s=zthr  2>&1 | tail -1 | tee -a "$log"
  printf '%-16s ' probe2-65/65    | tee -a "$log"; "$B65"        bench "$sd" -r 1 -s=probe2 2>&1 | tail -1 | tee -a "$log"
done
echo "=== threshold sweep on holdout (new tables, -s=probe2): Y x S ===" | tee -a "$log"
printf '%-16s ' "thr Y\\S" | tee -a "$log"
for S in 55 65 75; do printf '%10s' "S=$S" | tee -a "$log"; done; echo "" | tee -a "$log"
for Y in 55 65 75; do
  printf '%-16s ' "Y=$Y" | tee -a "$log"
  for S in 55 65 75; do
    if [ "$Y" = "65" ] && [ "$S" = "65" ]; then B="$B65"; else B="thr_Y${Y}_S${S}"; fi
    r=$("$B" bench /tmp/r8f5/h2_ios_hold /tmp/r8f5/h2_desktop_hold /tmp/r8f5/h2_websight_hold /tmp/r8f5/h2_docs_hold /tmp/r8f5/h2_scans_hold /tmp/r8f5/h2_web_hold -r 1 -s=probe2 2>&1 | grep TOTAL | awk '{print $5}')
    printf '%10s' "$r" | tee -a "$log"
  done
  echo "" | tee -a "$log"
done
echo SWEEP_DONE | tee -a "$log"
