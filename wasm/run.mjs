// node run.mjs dec.wasm file.plt:ref.ppm ...  -> decode in wasm, compare with reference PPM, report MB/s (RGB)
import { readFileSync } from 'node:fs';
const [wasmPath, ...pairs] = process.argv.slice(2);
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const ex = instance.exports;
let totPx = 0, totT = 0, allOk = true;
for (const pr of pairs) {
  const [pltPath, ppmPath] = pr.split(':');
  const plt = readFileSync(pltPath), ppm = readFileSync(ppmPath);
  ex.plt_reset();
  const inPtr = ex.plt_alloc(plt.length + 16);
  new Uint8Array(ex.memory.buffer, inPtr, plt.length).set(plt);
  const W = ex.plt_width(inPtr), H = ex.plt_height(inPtr), n = W * H * 3;
  const outPtr = ex.plt_alloc(n);
  let best = 1e30, rc = 0;
  for (let r = 0; r < 3; ++r) {
    const t0 = process.hrtime.bigint(); rc = ex.plt_decode(inPtr, plt.length, outPtr); const t1 = process.hrtime.bigint();
    best = Math.min(best, Number(t1 - t0) / 1e9);
  }
  const out = new Uint8Array(ex.memory.buffer, outPtr, n), ref = ppm.subarray(ppm.length - n);
  const ok = rc === 0 && Buffer.compare(Buffer.from(out), ref) === 0; allOk &&= ok;
  totPx += W * H; totT += best;
  console.log(`${pltPath.split('/').pop().padEnd(28)} ${W}x${H}  rc=${rc}  dec ${(3 * W * H / 1e6 / best).toFixed(1)} MB/s  ${ok ? 'ok' : 'FAIL'}`);
}
console.log(`TOTAL dec ${(3 * totPx / 1e6 / totT).toFixed(1)} MB/s (RGB)  ${allOk ? 'all ok' : 'FAIL'}`);
