# QH — table-driven replacement of RLGR in PLR (research, not in the format yet)

Research run by a Claude Fable subagent on 2026-11-11/12 at Vladimir's request ("approximate RLGR with tables"),
verified in the main session. Paths in the scripts point to the sandbox (`/tmp/fable_rlgr`, the scratchpad sets).

QH: per 16-block, table = map[plane Y/S][Q(sa)] (sa = the causal sum that gives L; 68 levels merged into 32 tables);
letter = 4 neighbours, each min(h, 3) in 2 bits (256 letters), canonical Huffman <= 11 bits; tail h-3 for symbols
equal to 3 as Rice(k of the table), escape 24 ones + 10 raw bits; then the L raw planes as before. Everything is
written at the lane head at once: no holes, no state between blocks. Decoder: 11-bit lookup -> letter | len << 8.

Held-out (tables of the other set), real codec, round trips ok, verified:
| | PLR (RLGR lane) | PLR + QH |
|---|---|---|
| full Kodak (col19 tables) | 10.1422, 106 / 83 MB/s | 10.0093 (−1.31%), ~180 / 190 |
| gray 17 (Kodak tables) | 4.5666, 99 / 77 | 4.5275 (−0.86%), ~245 / 300 |
| 19 colour (Kodak tables) | 8.2334, 124 / 99 | 8.1411 (−1.12%), ~210 / 230 |
| Kodak 500 crops (col19 tables) | 10.3554, 106 / 83 | 10.2193 (−1.31%), 167 / 176 |
Weak point: flat / synthetic content (flat 0.0047 → 0.2548 bpp, gradient 0.15 → 0.44, stripes 0.011 → 0.26):
no runs below 1 bit per 4 symbols. Proposed fix (simulated only): runs of zero blocks when sa == 0.

Files: `qh.hpp` (coder/decoder of a block), `palic2r_proto.diff` (-DPLR_DUMP, -DPLR_QH against palic2r.cpp of
dec7993; tables via env QH_TABLES), `train_tables.py`, `exp1..7.py`, `lib.py`, `load.py`, `bench_entropy.cpp`,
`tables/*.bin` (statistics only, trained on Kodak / col19).

**Update:** QH is now the PLR format (`palic-2/qh.hpp`, `palic-2/qh_tables.h`), with off 2 and zero-block runs;
see PLN1.md "PLR format: static QH tables". `lib.py` / `load.py` / `train_tables.py` here are the updated versions:
several sets joined with `+`, `QH_DUMP` (dump dir), `QH_OFF`, and the zero-block runs simulated exactly as the codec
does them (the estimate matches the codec to 4 decimals). Dumps: build palic2r with `-DPLR_DUMP -DPLR_NO_QH`, run
`PLR_DUMP_FILE=x.dmp palic-2r c img x.plr`. Format tables: `tables/qh_pub_off2_K32_L11.bin` (Kodak + 12 CLIC gray).
