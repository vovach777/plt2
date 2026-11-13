#!/bin/sh
# test.sh PALIC2R img... - the wasm port against the C++ codec: encode in both modes must give the same bytes, and
# the wasm decoder must restore the C++ files (pixels compared with the C++ decoder's output).
P=$1; shift; D=$(dirname "$0"); T=$(mktemp -d); ok=0; bad=0
for f in "$@"; do for m in "" "-f"; do
  node "$D/run.mjs" c "$f" "$T/a" $m && "$P" c "$f" "$T/b" $m || { echo "FAIL encode $f $m"; bad=$((bad+1)); continue; }
  if cmp -s "$T/a" "$T/b" && node "$D/run.mjs" d "$T/b" "$T/o" && "$P" d "$T/b" "$T/o2" >/dev/null && cmp -s "$T/o" "$T/o2"; then ok=$((ok+1)); else echo "MISMATCH $f $m"; bad=$((bad+1)); fi
done; done; rm -rf "$T"; echo "ok $ok, bad $bad"; [ $bad -eq 0 ]
