#!/bin/bash
# fuzz.sh : corrupted .plr files (byte flips, truncation) through the ASan+UBSan decoders: no crash, no sanitizer report
cd /tmp/fable_enc2; T=fz; mkdir -p $T; n=0; bad=0; rej=0
srcs="$(ls /tmp/qhw/sets/clic_test_color/*.ppm | head -1) $(ls /tmp/qhw/sets/clic_test_gray/*.pgm | head -1) edge/low_300x290.ppm edge/spk_300x290.pgm /tmp/fable_rlgr/synth/gradient.pgm"
i=0; for f in $srcs; do for fl in "" "-f"; do i=$((i+1)); ./plr c $f $T/s$i.plr $fl; done; done
for s in $T/s*.plr; do for k in $(seq 1 12); do
  python3 -I - $s $T/c.plr $k <<'P'
import sys, random
s, o, k = sys.argv[1], sys.argv[2], int(sys.argv[3]); random.seed(hash((s, k)) & 0xffffff); b = bytearray(open(s, 'rb').read()); n = len(b)
if k % 4 == 0: b = b[:random.randrange(16, n)]
else:
    for _ in range(1 if k % 4 == 1 else 8): p = random.randrange(12, len(b)); b[p] ^= 1 << random.randrange(8)
open(o, 'wb').write(b)
P
  for B in plr_asan_native plr_asan_x86-64-v2; do n=$((n+1)); ./$B d $T/c.plr $T/o.pnm > $T/log 2>&1; rc=$?
    [ $rc = 2 ] && rej=$((rej+1)); if [ $rc != 0 ] && [ $rc != 2 ] || grep -q "ERROR: AddressSanitizer\|runtime error" $T/log; then bad=$((bad+1)); echo "BAD $B $s $k rc=$rc"; head -3 $T/log; fi; done
done; done; echo "fuzz: $n decodes, $rej rejected with an error, $bad crashes / sanitizer reports"; rm -rf $T
