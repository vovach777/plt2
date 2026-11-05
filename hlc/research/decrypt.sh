#!/usr/bin/env bash
# usage: ./decrypt.sh <password> [archive]   -> extracts ./hlc-samples/ (needs openssl, tar, xz; Git Bash has them)
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <password> [archive]" >&2; exit 1; }
A="${2:-$(dirname "$0")/hlc-samples.tar.xz.enc}"
openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass "pass:$1" -in "$A" | tar xJf -
echo "ok: $(ls hlc-samples | wc -l) files in ./hlc-samples"
