# PLT2 decoder for WebAssembly

Freestanding (no libc / libc++, no WASI): system `clang --target=wasm32` + `wasm-ld`. Decodes the default layout
(one buffer per strip, 4 streams), A31/A15, both models. Single thread; bit-exact with plt2.cpp.

```
F="--target=wasm32 -O3 -std=c++17 -nostdlib -fno-exceptions -fno-rtti -Iinc -Wl,--no-entry -Wl,--export-dynamic -Wl,-z,stack-size=1048576"
clang++ $F -msimd128 -DPLT_WASM_SIMD plt2_dec_wasm.cpp -o plt2_dec_simd128.wasm   # wasm SIMD128 (all modern browsers)
clang++ $F plt2_dec_wasm.cpp -o plt2_dec_scalar.wasm
node run.mjs plt2_dec_simd128.wasm file.plt:reference.ppm ...                        # decode + compare + MB/s
```
Exports: `plt_alloc(n)`, `plt_reset()`, `plt_width(in)`, `plt_height(in)`, `plt_decode(in, size, out_rgb)` → 0 ok.
Memory: bump arena in linear memory, grows on demand. Size ~320 KB (almost all of it: tables).

Node 22 (V8), 1 thread, MB/s of RGB output:
| | 13 images | 4000×3000 |
|---|---|---|
| A31 scalar | 17.4 | 18.6 |
| **A31 SIMD128** | **41.7** | **44.0** |
| A15 scalar | 23.0 | 25.0 |
| **A15 SIMD128** | **48.5** | **53.3** |
| native AVX-512, A31 (reference) | 75.8 | 88 |

In a browser: one Worker per strip (strips are independent; N = W/512), each with its own instance.
