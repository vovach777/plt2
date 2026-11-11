'use strict';

// Rice of one typed array. k is floor(log2) of the mean of the last three symbols.
// The history is three slots. The byte buffer grows by doubling, not by pushing boxed numbers.

function riceK(hist, hn) {
  if (hn === 0) return 0;
  const used = hn < 3 ? hn : 3;
  let sum = 0;
  for (let i = hn - used; i < hn; i++) sum += hist[i];
  const pred = sum / used;
  if (pred <= 1) return 0;
  return Math.min(16, 31 - Math.clz32(Math.round(pred)));
}

function remember(hist, hn, sym) {
  if (hn < 3) { hist[hn] = sym; return hn + 1; }
  hist[0] = hist[1];
  hist[1] = hist[2];
  hist[2] = sym;
  return 3;
}

function encode(values) {
  if (!(values instanceof Uint8Array) && !(values instanceof Uint16Array)) {
    throw new TypeError('expected Uint8Array or Uint16Array');
  }
  let bytes = new Uint8Array(values.length + 1);
  let length = 0;
  let acc = 0, nbits = 0;
  function byte(value) {
    if (length === bytes.length) {
      const next = new Uint8Array(bytes.length * 2);
      next.set(bytes);
      bytes = next;
    }
    bytes[length++] = value;
  }
  function writeBits(value, bits) {
    for (let i = bits - 1; i >= 0; i--) {
      acc = (acc << 1) | ((value >>> i) & 1);
      nbits++;
      if (nbits === 8) { byte(acc & 255); acc = 0; nbits = 0; }
    }
  }
  const hist = new Uint16Array(3);
  let hn = 0;
  for (let s = 0; s < values.length; s++) {
    const sym = values[s];
    const k = riceK(hist, hn);
    let q = sym >>> k;
    while (q >= 24) { writeBits(0xffffff, 24); q -= 24; }
    if (q) writeBits((1 << q) - 1, q);
    writeBits(0, 1);
    if (k) writeBits(sym & ((1 << k) - 1), k);
    hn = remember(hist, hn, sym);
  }
  if (nbits) byte((acc << (8 - nbits)) & 255);
  return bytes.subarray(0, length);
}

function decode(bytes, length, bits) {
  const values = new (bits === 16 ? Uint16Array : Uint8Array)(length);
  let acc = 0, nbits = 0, pos = 0;
  function readBit() {
    if (nbits === 0) {
      if (pos >= bytes.length) throw new RangeError('rice truncated');
      acc = bytes[pos++];
      nbits = 8;
    }
    nbits--;
    return (acc >> nbits) & 1;
  }
  const hist = new Uint16Array(3);
  let hn = 0;
  for (let s = 0; s < length; s++) {
    const k = riceK(hist, hn);
    let q = 0;
    while (readBit() === 1) q++;
    let r = 0;
    for (let b = 0; b < k; b++) r = (r << 1) | readBit();
    const sym = (q << k) | r;
    values[s] = sym;
    hn = remember(hist, hn, sym);
  }
  return values;
}

module.exports = { encode, decode };
