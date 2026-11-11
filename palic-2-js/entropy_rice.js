'use strict';

// Rice of one typed array. k is floor(log2) of the mean of the last three symbols.

function riceK(hist) {
  const n = hist.length;
  if (n === 0) return 0;
  const used = Math.min(n, 3);
  let sum = 0;
  for (let i = n - used; i < n; i++) sum += hist[i];
  const pred = sum / used;
  if (pred <= 1) return 0;
  return Math.min(16, 31 - Math.clz32(Math.round(pred)));
}

function encode(values) {
  if (!(values instanceof Uint8Array) && !(values instanceof Uint16Array)) {
    throw new TypeError('expected Uint8Array or Uint16Array');
  }
  const bytes = [];
  let acc = 0, nbits = 0;
  function writeBits(value, bits) {
    for (let i = bits - 1; i >= 0; i--) {
      acc = (acc << 1) | ((value >>> i) & 1);
      nbits++;
      if (nbits === 8) { bytes.push(acc & 255); acc = 0; nbits = 0; }
    }
  }
  const hist = [];
  for (let s = 0; s < values.length; s++) {
    const sym = values[s];
    const k = riceK(hist);
    let q = sym >>> k;
    while (q >= 24) { writeBits(0xffffff, 24); q -= 24; }
    if (q) writeBits((1 << q) - 1, q);
    writeBits(0, 1);
    if (k) writeBits(sym & ((1 << k) - 1), k);
    hist.push(sym);
    if (hist.length > 3) hist.shift();
  }
  if (nbits) bytes.push((acc << (8 - nbits)) & 255);
  return Uint8Array.from(bytes);
}

function decode(bytes, length, bits) {
  const Type = bits === 16 ? Uint16Array : Uint8Array;
  const values = new Type(length);
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
  const hist = [];
  for (let s = 0; s < length; s++) {
    const k = riceK(hist);
    let q = 0;
    while (readBit() === 1) q++;
    let r = 0;
    for (let b = 0; b < k; b++) r = (r << 1) | readBit();
    const sym = (q << k) | r;
    values[s] = sym;
    hist.push(sym);
    if (hist.length > 3) hist.shift();
  }
  return values;
}

module.exports = { encode, decode };
