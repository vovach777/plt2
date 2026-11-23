# PLS-probe-2: per-plane screen/photo probe, oracle, speed (experiment, 2026-11-23)

Branch `pls-probe-2` (from `pls-probe-1` ae3dbaf). Experiment only, not merged into ctx5: the new
`qhs_tables.h` changes the PLS2 bitstream under the same signature.

## Encoder modes (palic2r.cpp, decoder unchanged)

| flag | probe |
|---|---|
| `-s` | old `qhs::screen_plane_mode`: 1/8 of the blocks (zeros vs small 1..7) -> sure photo / unsure / sure screen; unsure planes: PLS vs PLR bit costing inside the PLS variant pass |
| `-s=zthr` | probe-1: zeros of Y+zU+zV counted in the MED vector loop, one choice per tile (65%) |
| `-s=probe2` (= `-s=zthrc`) | probe-2: zeros of Y and S counted in the MED loop, per plane, `PLS_ZTHR_Y2` 45 / `PLS_ZTHR_S2` 50 |
| `-s=hybrid` | probe-2 zeros decide clear planes, bit costing within +-`PLS_HYB_W` (15) |
| `-s=autocnt` | speed control: probe-2 counting, old `-s` decision; output byte-identical to `-s` |

`qhs_tables.h`: Qwen, 100% of the six R8F5 screen train groups (iOS-1K, ScreenSpot-Pro, WebSight, DocLayNet,
DIBCO scans, web+gb82; Google Drive: datasets/), planes selected by probe-2 45/50, equal pixel weight per group.

## Result (full test parts, Qwen, Ryzen 9 9900X, 1 core, median of 3 rounds of best-of-7)

| set (files) | `-s`: bpp / enc MB/s | `autocnt` enc (cost of counting) | `probe2`: bpp / enc | probe2 vs `-s`: bpp, enc |
|---|---|---|---|---|
| iOS-1K test (200) | **1.2943** / 2697 | 2629 (-2.5%) | 1.2978 / 2699 | +0.27%, +0.1% |
| ScreenSpot-Pro test (317) | **2.2471** / 2290 | 2239 (-2.2%) | 2.2499 / 2279 | +0.12%, -0.5% |
| WebSight test (160) | 0.2145 / **6390** | 5974 (-6.5%) | 0.2144 / 6109 | -0.05%, -4.4% |
| DocLayNet test (160) | **1.7078** / 2512 | 2451 (-2.4%) | 1.7109 / 2559 | +0.18%, +1.9% |
| DIBCO scans test (12) | **5.7454** / 1216 | 1201 (-1.3%) | 5.7540 / 1238 | +0.15%, +1.8% |
| full12 | **1.2818** / 2264 | 2209 (-2.4%) | 1.2819 / 2267 | +0.01%, +0.1% |
| web8 | **1.9511** / 2287 | 2233 (-2.4%) | 1.9587 / 2331 | +0.39%, +1.9% |
| valid41 | **8.7227** / 1187 | 1173 (-1.2%) | 8.7329 / 1188 | +0.12%, +0.1% |
| Kodak | **9.9586** / 1220 | 1204 (-1.3%) | 9.9607 / 1210 | +0.02%, -0.9% |

perf (full12): the PLS coder takes almost all encode time; the old probe ~1.4% of it, counting on the fly ~2.5%,
bit costing of unsure planes < 0.05% of total time.

Conclusion: in PLS counting zeros on the fly does not pay. `-s` with the new tables is the best bpp on all sets;
all the gain over ctx5 (iOS -0.50%, desktop -0.40%, docs -0.64%, full12 -0.75%, scans -0.68%) comes from the tables.

## Oracle (Claude, diag/)

`diag/pls_diag_oracle.patch` (on top of probe-2): with `PLS_DIAG=1` every tile is coded 4 times (Y, S) x (PLS, PLR)
and logged with zero counts, residual histograms and the old probe's decision. Scripts: `oracle_rules.py`,
`oracle_planes.py`, `oracle_scans.py`, `hybrid_band.py`, `features.py`, `old_probe_bias.py`.
- Y and S disagree in 19% of screen tiles but it is worth 0.17% (photos 0.01%); on DIBCO scans in 68% of tiles
  (Y photo, S screen) and the per-plane oracle beats the per-tile one by 1.3%.
- Loss vs the per-plane oracle (same tables): old `-s` 0.00-0.21%, probe-2 0.03-0.49%.
- Old `-s` errors are its "sure photo" planes (0.1-0.2% on scans, iPhone, web8); unsure planes are almost never wrong.
- Extra on-the-fly counters (small 1..7, mid 8..31, outliers >= 32) add nothing over the zero share (0.247% vs 0.251%).
- Hybrid +-15: scans -0.15% vs `-s`, but worse on web8 / iPhone / photo and up to 10% slower on photo.

## Workers

- `workbuddy/`: first per-plane training (65/65) and sweep (8-file samples).
- `qwen/retrain/`: holdout oracle, 45/50 choice, 100% training, full bench.
- `qwen/speed/`: speed of the probes (REPORT.md, raw.txt).
Full logs: Google Drive claude-phone/2026-11-23_qwen_pls-probe-2_logs.tar.gz, 2026-11-23_qwen_pls-probe_speed_logs.tar.gz.
