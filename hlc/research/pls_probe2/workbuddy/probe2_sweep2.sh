#!/bin/sh
cd /tmp/r8f5
log=/tmp/r8f5/probe2_sweep.txt
: > "$log"
B65=./pls_probe2_new
D="/tmp/r8f5/h2_ios_hold /tmp/r8f5/h2_desktop_hold /tmp/r8f5/h2_websight_hold /tmp/r8f5/h2_docs_hold /tmp/r8f5/h2_scans_hold /tmp/r8f5/h2_web_hold"
printf 'thr Y\\S' | tee -a "$log"
for S in 55 65 75; do printf '%10s' "S=$S" | tee -a "$log"; done; echo "" | tee -a "$log"
for Y in 55 65 75; do
  printf '%-8s' "Y=$Y" | tee -a "$log"
  for S in 55 65 75; do
    if [ "$Y" = "65" ] && [ "$S" = "65" ]; then B="$B65"; else B="./thr_Y${Y}_S${S}"; fi
    r=$("$B" bench $D -r 1 -s=probe2 2>&1 | grep TOTAL | awk '{print $4}')
    printf '%10s' "$r" | tee -a "$log"
  done
  echo "" | tee -a "$log"
done
echo SWEEP_DONE | tee -a "$log"
