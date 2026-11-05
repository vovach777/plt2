# hlc/research — numpy experiments behind HLC5

Small, self-contained scripts used to decide HLC5's contexts (zero-order entropy of MED residuals, tables per file,
FSE headers ignored). Most take image paths/globs on the command line; the older ones read `photos16h/*.ppm` and
`cat_h.ppm` from the current directory and `exec` helpers from this folder (`ctxest.py`, `axesfit.py`, `act2.py`,
`xresctx.py`, `buckets.py`), so run them from a directory that holds those images and these scripts.

Test images (16 photos + cat, JPEG 4:2:0 downscaled 2x, PPM) are not in git (private photos): encrypted archive
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
| third_axis.py | third axis instead of d: neighbour errors, |a-aa|, |b-bb| vs true |d-b| |
| xresctx.py, xresgrain.py, xresfit.py | cross-chroma on top of contexts; granularity; threshold refit |
