// Тайл для TypeScript Native 0.0.pre35, GC.
// Буферы: Uint8 и Uint16, выделены на 288x288. Знак хромы — младшие 16 бит.
// Склейка ts/rgb.ts ts/plane.ts ts/rlgr.ts ts/tile.ts. main(): void.

// Слой 1. RGB -> плоскости. Y байты, U и V знаковые 16 бит.

function splitRgb(rgb: Uint8Array, y: Uint8Array, u: Uint16Array, v: Uint16Array, n: number): void {
  let i = 0;
  while (i < n) {
    let p = i * 3;
    let r = rgb[p];
    let g = rgb[p + 1];
    let b = rgb[p + 2];
    let du = r - g;
    let v0 = b - g;
    y[i] = (g + ((du + v0 + 2) >> 2)) & 255;
    u[i] = du & 65535;
    v[i] = (v0 - (du >> 2)) & 65535;
    i = i + 1;
  }
}
// Слой 2. Плоскость -> корзины L0..L7 и сырые слова.
// Буферы на максимальный тайл 288x288 выделяются один раз.
// Яркость: коэффициенты Uint8. Хрома: Uint16, потому что L0 доходит до 511.

let hist: Uint16Array = new Uint16Array(288);
let blockZ: Uint16Array = new Uint16Array(16);
let wide: Uint16Array = new Uint16Array(82944);
let coeff8: Uint8Array = new Uint8Array(663552);
let coeff16: Uint16Array = new Uint16Array(663552);
let coeffPos: Uint32Array = new Uint32Array(8);
let maxSymbol: Uint16Array = new Uint16Array(8);
let raw: Uint16Array = new Uint16Array(36288);
let rawCount = 0;
let stride = 0;
let planeChroma = 0;

function bitWidth(value: number): number {
  let n = value >>> 0;
  let w = 0;
  while (n !== 0) {
    n = n >>> 1;
    w = w + 1;
  }
  return w;
}

function wrap8(value: number): number {
  return ((value + 128) & 255) - 128;
}

function wrap9(value: number): number {
  return ((value + 256) & 511) - 256;
}

function zigzag(value: number): number {
  if (value >= 0) return value * 2;
  return -value * 2 - 1;
}

let pred = 0;

function sx(bits: number): number {
  if (bits >= 32768) return bits - 65536;
  return bits;
}

function predict(samples: Uint16Array, width: number, x: number, y: number): void {
  if (y === 0) {
    pred = x === 0 ? 0 : sx(samples[x - 1]);
    return;
  }
  if (x === 0) {
    pred = sx(samples[(y - 1) * width + x]);
    return;
  }
  let i = y * width + x;
  let a = sx(samples[i - 1]);
  let b = sx(samples[i - width]);
  let c = sx(samples[i - width - 1]);
  let lo = a < b ? a : b;
  let hi = a > b ? a : b;
  if (c >= hi) { pred = lo; return; }
  if (c <= lo) { pred = hi; return; }
  pred = a + b - c;
}

function chooseL(blockIndex: number, width: number, previousSum: number): number {
  if (blockIndex === 0) return 2;
  let p = blockIndex * 16;
  let reference = previousSum;
  if (width >= 16 && p >= width) {
    reference = 0;
    let cursor = p % width;
    let i = 0;
    while (i < 16) {
      reference = reference + hist[cursor];
      cursor = cursor + 1;
      if (cursor === width) cursor = 0;
      i = i + 1;
    }
  }
  let w = bitWidth(reference >> 4) - 2;
  if (w < 0) return 0;
  if (w > 7) return 7;
  return w;
}

function bindPlane(padded: number): void {
  stride = padded;
  let i = 0;
  while (i < 8) {
    coeffPos[i] = 0;
    maxSymbol[i] = 65535;
    i = i + 1;
  }
  rawCount = 0;
}

function appendBlock(L: number): void {
  let bit = 0;
  while (bit < L) {
    let word = 0;
    let symbol = 0;
    while (symbol < 16) {
      word = word | (((blockZ[symbol] >>> bit) & 1) << symbol);
      symbol = symbol + 1;
    }
    raw[rawCount] = word;
    rawCount = rawCount + 1;
    bit = bit + 1;
  }
  let pos = coeffPos[L];
  let max = maxSymbol[L];
  let symbol = 0;
  let base = L * stride + pos;
  while (symbol < 16) {
    let top = blockZ[symbol] >>> L;
    if (planeChroma === 0) coeff8[base + symbol] = top;
    else coeff16[base + symbol] = top;
    if (top > sx(max)) max = top;
    symbol = symbol + 1;
  }
  coeffPos[L] = pos + 16;
  maxSymbol[L] = max & 65535;
}

function transform(samples: Uint16Array, width: number, height: number, chroma: number): void {
  planeChroma = chroma;
  let size = width * height;
  let blockCount = (size + 15) >> 4;
  let i = 0;
  while (i < width) {
    hist[i] = 0;
    i = i + 1;
  }
  bindPlane(blockCount * 16);
  let previousSum = 0;
  let histPos = 0;
  let rasterX = 0;
  let rasterY = 0;
  let blockIndex = 0;
  while (blockIndex < blockCount) {
    let L = chooseL(blockIndex, width, previousSum);
    let blockSum = 0;
    let slot = 0;
    while (slot < 16) {
      let pixel = blockIndex * 16 + slot;
      if (pixel >= size) {
        blockZ[slot] = 0;
      } else {
        let x = rasterX;
        let y = rasterY;
        predict(samples, width, x, y);
        let original = sx(samples[pixel]) - pred;
        let delta = chroma === 0 ? wrap8(original) : wrap9(original);
        let z = zigzag(delta);
        blockZ[slot] = z;
        blockSum = blockSum + z;
        hist[histPos] = z;
        histPos = histPos + 1;
        if (histPos === width) histPos = 0;
        rasterX = rasterX + 1;
        if (rasterX === width) {
          rasterX = 0;
          rasterY = rasterY + 1;
        }
      }
      slot = slot + 1;
    }
    appendBlock(L);
    previousSum = blockSum;
    blockIndex = blockIndex + 1;
  }
}

function widen(src: Uint8Array, n: number): void {
  let i = 0;
  while (i < n) {
    wide[i] = src[i];
    i = i + 1;
  }
}
// Слой 3. RLGR. Вход копируется в Uint16, выход пишется в заранее выделенный Uint8.
// k и kr хранятся умноженными на 8. Частное 32 и больше — 32 единицы и сырое слово.

let acc = 0;
let nbits = 0;
let kr = 8;
let sym: Uint16Array = new Uint16Array(82944);
let pack: Uint8Array = new Uint8Array(663560);
let packCount = 0;

function writeBits(value: number, width: number): void {
  let i = width - 1;
  while (i >= 0) {
    acc = (acc << 1) | ((value >>> i) & 1);
    nbits = nbits + 1;
    if (nbits === 8) {
      pack[packCount] = acc & 255;
      packCount = packCount + 1;
      acc = 0;
      nbits = 0;
    }
    i = i - 1;
  }
}

function ones(q: number): void {
  while (q >= 8) {
    writeBits(255, 8);
    q = q - 8;
  }
  if (q !== 0) writeBits((1 << q) - 1, q);
}

function writeRice(x: number, m: number): void {
  let q = 0;
  if (m < 31) q = x >>> m;
  if (q >= 32) {
    writeBits(0xffffffff, 32);
    writeBits(x, 32);
  } else {
    ones(q);
    writeBits(0, 1);
    if (m !== 0) writeBits(x & ((1 << m) - 1), m);
  }
  let vk = 0;
  if (m < 31) vk = x >>> m;
  if (vk === 0) {
    if (kr > 2) kr = kr - 2;
    else kr = 0;
  } else if (vk !== 1) {
    kr = kr + vk;
    if (kr > 240) kr = 240;
  }
}

function encodeLoaded(count: number): number {
  acc = 0;
  nbits = 0;
  kr = 8;
  packCount = 0;
  let k = 8;
  let rl = 0;
  let s = 0;
  while (s < count) {
    let value = sym[s];
    let kShift = k >> 3;
    let krShift = kr >> 3;
    if (kShift > 0) {
      if (value === 0) {
        rl = rl + 1;
        if (rl === (1 << kShift)) {
          writeBits(0, 1);
          rl = 0;
          k = k + 5;
          if (k > 240) k = 240;
        }
      } else {
        writeBits(1, 1);
        if (kShift !== 0) writeBits(rl, kShift);
        if (k > 9) k = k - 9;
        else k = 0;
        rl = 0;
        writeRice(value - 1, krShift);
      }
    } else {
      rl = 0;
      writeRice(value, krShift);
      if (value === 0) {
        k = k + 2;
        if (k > 240) k = 240;
      } else if (k > 3) k = k - 3;
      else k = 0;
    }
    s = s + 1;
  }
  if (rl > 0) {
    let kShift = k >> 3;
    writeBits(1, 1);
    if (kShift !== 0) writeBits(rl, kShift);
  }
  if (nbits !== 0) {
    pack[packCount] = (acc << (8 - nbits)) & 255;
    packCount = packCount + 1;
  }
  return packCount;
}

function load8(src: Uint8Array, base: number, count: number): void {
  let i = 0;
  while (i < count) {
    sym[i] = src[base + i];
    i = i + 1;
  }
}

function load16(src: Uint16Array, base: number, count: number): void {
  let i = 0;
  while (i < count) {
    sym[i] = src[base + i];
    i = i + 1;
  }
}
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
