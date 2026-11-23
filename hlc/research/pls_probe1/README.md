# PLS-probe-1: qhs tables retrained for the R8F5 zero probe (WorkBuddy, 2026-11-23)

Encoder -s=zthr (palic2r.cpp, commit 8934c99): zeros of Y + zU + zV >= 65% of a 288x288 tile -> both planes PLS.
palic-2/qhs_tables.h on this branch is retrained (exp_hybrid.py H0 -> gen_qhs_tables.py) on the screen data of llrice
R8F5 0448cfb (Google Drive: datasets/ screens_ios1k, screens_screenspot_pro, screens_websight_v02_png800,
docs_doclaynet_small, scans_dibco_2016_2019, screenshot_web + train_gb82-sc_p6), split by sorted name (i%5==4 test,
i%10==3 held out), equal pixel weight per group, 20% of the train images per group (stride 5), only tiles that
-s=zthr sends to PLS (train_qhs_probe1.py). qh_tables.h (photo / PLR) unchanged.
NOTE: new qhs tables = a different PLS2 bitstream under the same signature; files of ctx5 -s and of this branch do
not decode across. Experiment only.

Claude's check (bench, 1 thread, bpp):
| set | ctx5 -s auto | zthr, old qhs | zthr, new qhs | auto, new qhs | R8F5 0448cfb |
|---|---|---|---|---|---|
| 12 screens | 1.2914 | 1.2925 | 1.2845 | 1.2835 | 1.2510 |
| 6 private iPhone | 1.9903 | 1.9942 | 1.9835 | 1.9815 | 1.9248 |
| web8 (old qhs were trained on it) | 1.9455 | 1.9557 | 1.9648 | 1.9534 | 1.9012 |
| valid41 | 8.7226 | 8.7302 | 8.7277 | 8.7228 | 8.7447 |
WorkBuddy's light bench (8 files per set) and plane shares: bench_light.txt, holdout_light.txt, plane_frac.py.
