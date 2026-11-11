// Либа одного тайла. Три плоскости в out, заголовок файла сюда не входит.
// 82944 = 288*288. 663552 = 82944*8. 36288 = 5184*7. 663560 байт пакета RLGR.
// 3145728 байт на весь тайл: три плоскости, худший RLGR.

let yPlane: Uint8Array = new Uint8Array(82944);
let uPlane: Uint16Array = new Uint16Array(82944);
let vPlane: Uint16Array = new Uint16Array(82944);
let out: Uint8Array = new Uint8Array(3145728);
let outCount = 0;

function writeVarint(value: number): void {
  let n = value >>> 0;
  while (n >= 128) {
    out[outCount] = (n & 127) | 128;
    outCount = outCount + 1;
    n = n >>> 7;
  }
  out[outCount] = n & 127;
  outCount = outCount + 1;
}

function frame(count: number, bits: number): void {
  let packed = encodeLoaded(count);
  out[outCount] = bits;
  outCount = outCount + 1;
  writeVarint(count);
  writeVarint(packed);
  let i = 0;
  while (i < packed) {
    out[outCount] = pack[i];
    outCount = outCount + 1;
    i = i + 1;
  }
}

function emitPlane(chroma: number): void {
  let L = 0;
  while (L < 8) {
    let count = coeffPos[L];
    let bits = 8;
    let max = maxSymbol[L];
    if (max >= 32768) max = max - 65536;
    if (chroma !== 0 && L === 0 && count !== 0 && max > 255) bits = 16;
    if (chroma === 0) load8(coeff8, L * stride, count);
    else load16(coeff16, L * stride, count);
    frame(count, bits);
    L = L + 1;
  }
  load16(raw, 0, rawCount);
  frame(rawCount, 16);
}

function compressTile(rgb: Uint8Array, width: number, height: number): number {
  if (width > 288 || height > 288 || width <= 0 || height <= 0) return 0;
  let n = width * height;
  splitRgb(rgb, yPlane, uPlane, vPlane, n);
  outCount = 0;
  widen(yPlane, n);
  transform(wide, width, height, 0);
  emitPlane(0);
  transform(uPlane, width, height, 1);
  emitPlane(1);
  transform(vPlane, width, height, 1);
  emitPlane(1);
  return outCount;
}

let demo: Uint8Array = new Uint8Array(12);
let digest = 0;

function main(): void {
  let i = 0;
  while (i < 12) {
    demo[i] = (i * 17) & 255;
    i = i + 1;
  }
  let n = compressTile(demo, 2, 2);
  digest = n;
  i = 0;
  while (i < n) {
    digest = digest + out[i];
    i = i + 1;
  }
}
