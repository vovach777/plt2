# Encoder review of PLR (QH) — Claude Fable subagent, 2026-11-12, verified in the main session

Profile (cycles / px before): tails h>=3 (branch per symbol) and the per-pixel zU split were most of the time, plus
the word-crossing branch in Lane::put64 per code; qz_enc is linear (not O(n^2)); allocations < 1%.
Byte-identical speedups (now in palic-2): `rlgrl::LaneW` (branchless register writer, one room check per block),
`qh::qh_enc_block_w` (SSE letters, branchless tails while codes <= 15 bits, AVX2 joining of 16 codes), `tb_put16`
(branchless zU split), `qh_plane_enc` (one loop per plane), vectorisable `rgb_split`. Old path: `-DPLR_OLD_ENC`.
Encoder in-process 146..188 -> 541..551 MB/s (AVX2), ~2x without AVX2; decoder unchanged; files identical
(main session: 474 cmp on clang / gcc / x86-64-v2 builds, PLF too, ASan + UBSan).
Gradient review: "83% blocks equal to the block above" were zeros over zeros; non-zero blocks equal to the one above:
0%. Ideas evaluated on dumps (`variants.py`, `runs_var.py`, `tails_gap.py`, `grad_look.py`, `qhsim.py`): repeat-above
flag 0; EOB letter / run code at sa <= 8 hurt CLIC; "zero run" as a 257th symbol of the first letter: gradient -24%,
CLIC -0.05% (needs a second decoder table set). Paths in the scripts point to /tmp/fable_enc.
