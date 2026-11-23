#!/bin/sh
# Full-set bench, PLS-probe-2 task (2026-11-23): ctx5 / probe-1 / probe-2 (archive tables, 50-50) /
# probe-2 (retrained tables, 45-50) / R8F5 0448cfb. -r 3, one pinned core, nice -n 19.
B=/tmp/r8f5/bench
O=/home/devin/train/bench-independent/p2/bench
mkdir -p "$O/logs"
SUM="$O/summary.txt"; : > "$SUM"
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "CPU: $CPU" | tee -a "$SUM"
run() { set=$1; cfg=$2; bin=$3; shift 3
  log="$O/logs/${set}__${cfg}.txt"
  case "$bin" in
    *palic-2r|*pls_probe2*) sub=bench ;; *) sub= ;;
  esac
  nice -n 19 taskset -c 3 "$bin" $sub "$B/$set" -r 3 "$@" > "$log" 2>&1
  printf '%-14s %-14s ' "$set" "$cfg" | tee -a "$SUM"
  tail -1 "$log" | tee -a "$SUM"
}
for set in ios_test desktop_test websight_test docs_test scans_test full12 web8 valid41 kodak24; do
  echo "=== $set ===" | tee -a "$SUM"
  run "$set" ctx5-auto      /tmp/opencode/gh/plt2-ctx5/b/palic-2r            -s
  run "$set" probe1-zthr    /tmp/opencode/gh/plt2-pls-probe-1/b/palic-2r     -s=zthr
  run "$set" p2-arc-50-50   /tmp/r8f5/pls_probe2                             -s=probe2
  run "$set" p2-new-45-50   /tmp/r8f5/pls_probe2_new                         -s=probe2
  run "$set" r8f5-0448cfb   /tmp/opencode/gh/bench_r8f5
done
echo BENCH-DONE | tee -a "$SUM"