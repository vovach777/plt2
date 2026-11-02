# PLT2 in the browser: one Web Worker per strip, canvas

`index.html` — pick a .plt (file input) and the number of workers; or call `window.run(pltUrl, ppmUrl, K)` (decode
best of 3, then compare the canvas with the reference PPM pixel by pixel). Serve the folder over http
(`python3 -m http.server`), wasm is compiled once and the Module is posted to the workers.

Each worker: own wasm instance, the whole .plt in its memory, takes strips i = k, k+K, ... ;
`plt_decode_strip_rgba` writes the strip straight to RGBA (w×H×4), the buffer is transferred back and the main
thread does `putImageData(strip, x0, 0)`. Strips are independent: no shared memory, no COOP/COEP needed.

Headless Chromium (Playwright, `node pw.cjs plt2_dec_simd128.wasm`), 2-core VM, MB/s of RGB, incl. transfer + putImageData:
| | strips | 1 worker | 2 workers |
|---|---|---|---|
| 1920×1080, A31, SIMD128 | 3 | 35.2 | 51.6 (×1.47, 2+1 strips) |
| 4000×3000, A31, SIMD128 | 7 | 40.0 | 65.5 (×1.64) |
| 4000×3000, A15, SIMD128 | 7 | 50.9 | **80.5** (×1.58) |
| 4000×3000, A31, scalar | 7 | 18.5 | 30.0 |
All canvases pixel-exact vs the source.

Strip count is chosen by the encoder (`plt2 c ... -s N`, stored in the header). Equal halves for 2 workers:
| | strips | 1 worker | 2 workers |
|---|---|---|---|
| 1920×1080, default | 3 | 176 ms | 119 ms (×1.48) |
| **1920×1080, `-s 2`** | 2 | 177 ms | **90.5 ms (×1.95)** |
| 4000×3000, default | 7 | 874 ms | 551 ms (×1.59) |
| **4000×3000, `-s 2`** | 2 | 881 ms | **452 ms (×1.95)** |
