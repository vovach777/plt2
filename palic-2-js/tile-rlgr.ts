// One tile, RLGR, port of entropy_rlgr.js.
// Godbolt: TypeScript Native Compiler 0.0.pre35 JIT (No GC).
// Look at encodeTile. Buffers below live for the tile; the loop does not allocate.
// k and kr are stored scaled by 8. A Rice quotient of 32 or more is 32 ones and the raw word.

let acc = 0;
let nbits = 0;
let outCount = 0;
let kr = 8;

let symbols: number[] = [
  0, 0, 1, 0, 4, 0, 0, 2,
  0, 0, 0, 0, 7, 1, 0, 3
];

let out: number[] = [];

function clearOut(n: number): void {
  let i = 0;
  while (i < n) {
    out[i] = 0;
    i = i + 1;
  }
}

function writeBits(value: number, width: number): void {
  let i = width - 1;
  while (i >= 0) {
    acc = (acc << 1) | ((value >>> i) & 1);
    nbits = nbits + 1;
    if (nbits === 8) {
      out[outCount] = acc & 255;
      outCount = outCount + 1;
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

function encodeTile(count: number): number {
  acc = 0;
  nbits = 0;
  outCount = 0;
  kr = 8;
  let k = 8;
  let rl = 0;
  let s = 0;
  while (s < count) {
    let value = symbols[s];
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
      } else {
        if (k > 3) k = k - 3;
        else k = 0;
      }
    }
    s = s + 1;
  }
  if (rl > 0) {
    let kShift = k >> 3;
    writeBits(1, 1);
    if (kShift !== 0) writeBits(rl, kShift);
  }
  if (nbits !== 0) {
    out[outCount] = (acc << (8 - nbits)) & 255;
    outCount = outCount + 1;
  }
  return outCount;
}

function main(): number {
  clearOut(64);
  let n = encodeTile(16);
  let sum = n;
  let i = 0;
  while (i < n) {
    sum = sum + out[i];
    i = i + 1;
  }
  return sum;
}
