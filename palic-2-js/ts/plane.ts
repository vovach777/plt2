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
