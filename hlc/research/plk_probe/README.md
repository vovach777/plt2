# plk_probe — how to choose RLK vs PLR per plane / per image (PLK, `palic2r -k`)

Base: Argon's RLK v9 v2 (`palic-2/palic2r.cpp`, `qhk.hpp`, `qhk_tables.h` from Drive `from_argon/2026-11-15_rlk9_clang_v2_code.zip`, not in this repo yet).

- `pk_log_v2.patch` — instrumented encoder: EVERY plane is coded both ways (oracle choice, the stream stays valid), prints
  `PL <chroma> <n> <probe pr> <rb> <plr> <zero-quad share> <all-zero 8x8 share> <nonzero share> <mean bit width>` to stderr.
- `pk_zq_v2.patch` — one-pass rule: RLK if >= 32.5% of the 4-px quads of the plane are zero, else PLR (format unchanged).
- `plk_rule.py` — fits single-feature thresholds on logs, compares with the oracle and with the current `-k`.
- `plk_center.py` — image-level decisions: central tile (Y, or Y and S), two tiles, luma-only centre.

Results (2026-11-15), loss vs the per-plane oracle: `zq` per plane +0.15..0.31%; centre decides the image +0.05..0.38%
on homogeneous sets, but on web screenshots with a photo banner in the centre (`data_screens_train`) the centre rule
(Argon v3) is +4.3% vs v2 while `zq` is +0.8%.
