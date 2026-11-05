# hlc/research — numpy experiments behind HLC5

Small, self-contained scripts used to decide HLC5's contexts (zero-order entropy of MED residuals, tables per file,
FSE headers ignored). Most take image paths/globs on the command line; the older ones read `photos16h/*.ppm` and
`cat_h.ppm` from the current directory and `exec` helpers from this folder (`ctxest.py`, `axesfit.py`, `act2.py`,
`xresctx.py`, `buckets.py`), so run them from a directory that holds those images and these scripts.

Test images (16 photos + cat, JPEG 4:2:0 downscaled 2x, PPM) are not in git (private photos); they were handed over
out of band (hlc-samples.zip / the encrypted
`hlc-samples.tar.xz.enc`, unpack with `./decrypt.sh <password>` (`decrypt.cmd` on Windows); files are
`hlc-samples/p01..p16.ppm`, `cat.ppm` (rename / symlink to `photos16h/` and `cat_h.ppm` for the older scripts).

| script | question |
|---|---|
| ctxest.py | activity variants without the up-right d (|a-c|+|c-b|+|a-aa| chosen) |
| ctxdist.py, ctxper.py | share / bits per HLC5 bucket, corpus and per photo |
| thsearch.py, thtile.py, hlc5_thresholds.py | threshold fit: common, per file, per tile; corpus script for CLIC |
| buckets.py | FSE-quantised bits per bucket with tables from 1/1, 1/2, 1/4 of the bucket; top codes bottom |
| signest.py, signcell.py, signtiny.py | sign from the context: flip only, sign cells, llcomp tiny rule and dead zones |
| act2.py, bwprod.py, bwsum.py, bwk.py, bw8.py | two-term activity, threshold-free bit-width classes, scale factor |
| axes2d.py, axesfit.py, axesfit2.py, axesplot.py | full two-axis map (signed bit width), oracle cell merging, plots |
| rawaxes.py, wnorm.py | raw-axis mean |e| contours; max / L1 / L2 / weighted norms |
| tri.py, triq.py | triangle sectors + ridge c=b x radius; radius quantisation |
| qcmp.py, q7cells.py, tinycmp.py | llcomp q3/q5/q7/q11 two-axis contexts, q7 cell table |
| predctx.py | direction x 2-bit disagreement MED vs HALIC predictors (/3, avg, none) and planar |
| sign_cmp.py | HLC5 sign rule vs llcomp_rc / llcomp Standard mirror signs, plus sign of neighbour errors |
| sign_err.py | sign rule when the neighbour error is a context axis (geometric wins with mirrored q cells) |
| q7q7e.py | q7 x q7 (mirrored) x lg2(|e_a|+|e_b|), caps |
| q7share_plot.py | per-photo q7 x q7 cell shares and mean |E1|+|E2|, small-multiple heatmaps + CSV |
| q7full.py | full unmirrored q7 x q7 grid: shares, mean signed residual, cost of folding mirror pairs |
| symmetry_rank.py | entropy cost of folding q7 sectors by symmetry groups (central mirror, sign of one axis, axis swap) |
| x_bias_fold.py | bias on the second axis after the mirror, then fold its sign (constant, per-row median, alpha*y) |
| uv_axes.py | diagonal axes u = 2c-a-b, v = b-a: (|u|,|v|) with sign(u) flip vs q7 sectors / mirror / mirror + swap |
| axis_scale.py | axis multipliers from the CLIC sector table: c-b x k (q axes), v x k (u,v); result: all within 0.1% |
| asym_q.py | asymmetric quantisation after mirror (tiny sign flip): different c-b edges on extremum vs ramp side; result: within 0.1% |
| oct_scale.py | per-octant axis scale by least squares on |e| (8 octants / 4 after mirror); result: worse (+0.9…+2.3%) |
| oct_scale_ent.py | per-octant radial scale fitted directly on entropy (4 octants after tiny mirror) |
| givens_rot.py | Givens rotation of (c-a, c-b) by a small angle: worse symmetry and entropy at any angle |
| third_axis.py | third axis instead of d: neighbour errors, |a-aa|, |b-bb| vs true |d-b| |
| xresctx.py, xresgrain.py, xresfit.py | cross-chroma on top of contexts; granularity; threshold refit |
