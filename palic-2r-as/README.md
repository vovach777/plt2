# palic-2r-as — PLR / PLF in AssemblyScript (WebAssembly)

`plr.ts` is the AssemblyScript port of `palic-2/palic2r.cpp`. It produces and reads the same files byte for byte:

| format | what | encode |
|---|---|---|
| PLR2 / PLR1 | RGB / gray: RCT (RGB only) + MED + QH (static Huffman letters, Rice tails, raw low bits, zero-block runs) | `encode(src, W, H, 3 \| 1, 0)` |
| PLF2 / PLF1 | fast mode (`palic2r -f`): RLGR pairs (llrice adaptation), truncated-binary split, one bit lane per tile with run-header holes and the give-back of full windows | `encode(src, W, H, 3 \| 1, 1)` |

`decode(src, n, dst)` reads all four. Not ported: PLS (`-s`, screen mode). It is still moving: the per-plane detector and the screen tables.

Why AssemblyScript and not plain JS: the codec needs exact 64-bit words (`u64`) and 32-bit integer arithmetic. JS `Number` is a double, and `BigInt` is slow.

## Build and run

```bash
npx asc plr.ts -o palic2r.wasm -O3 --runtime stub          # AssemblyScript 0.28
node run.mjs c in.ppm|pgm out.plr [-f]                      # -f: PLF
node run.mjs d in.plr out.ppm|pgm
./test.sh /path/to/palic-2r img...                         # byte identity with the C++ codec, both modes
```

**The committed `palic2r.wasm` is from the previous version** (RGB PLR only, `encode(src, W, H)`): rebuild it after pulling.

## Testing without the AS compiler

`../hlc/research/as/as2cpp.py` translates `plr.ts` mechanically into C++ with wasm32 semantics (`as_rt.hpp`: one linear memory, `usize` = u32, little-endian `load`/`store`). `harness.cpp` then runs it like `run.mjs`. This is how the gray and PLF parts were written and checked: byte-identical to `palic-2r` on 75 + 74 + 72 images (CLIC colour and gray, screenshots, edge sizes 1×1…577×300), part of the runs under ASan/UBSan with shift checks. The real wasm build still has to be checked with `test.sh`.
