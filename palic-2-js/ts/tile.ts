// Либа одного тайла. Сжимает RGB и дописывает три плоскости в out.
// На плоскость девять потоков: корзины L0..L7, затем сырые слова.
// Заголовок файла сюда не входит.

let yPlane: number[] = [];
let uPlane: number[] = [];
let vPlane: number[] = [];
let out: number[] = [];
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

function frame(symbols: number[], base: number, count: number, bits: number): void {
  let packed = encodeSymbols(symbols, base, count);
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
    if (chroma !== 0 && L === 0 && count !== 0 && maxSymbol[L] > 255) bits = 16;
    frame(coeff, L * stride, count, bits);
    L = L + 1;
  }
  frame(raw, 0, rawCount, 16);
}

function compressTile(rgb: number[], width: number, height: number): number {
  if (width > 288 || height > 288) return 0;
  let n = width * height;
  splitRgb(rgb, yPlane, uPlane, vPlane, n);
  outCount = 0;
  transformPlane(yPlane, width, height, 0);
  emitPlane(0);
  transformPlane(uPlane, width, height, 1);
  emitPlane(1);
  transformPlane(vPlane, width, height, 1);
  emitPlane(1);
  return outCount;
}

let demo: number[] = [];

function main(): number {
  let i = 0;
  while (i < 12) {
    demo[i] = (i * 17) & 255;
    i = i + 1;
  }
  let n = compressTile(demo, 2, 2);
  let sum = n;
  i = 0;
  while (i < n) {
    sum = sum + out[i];
    i = i + 1;
  }
  return sum;
}
