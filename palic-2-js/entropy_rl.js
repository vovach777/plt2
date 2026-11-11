'use strict';

// Nibble run-length of one typed array.
// 0..14 is one symbol, two per byte. F0 and two nibbles: a zero run from length 4.
// F1 and four nibbles: one symbol above 14.

function encode(values) {
  if (!(values instanceof Uint8Array) && !(values instanceof Uint16Array)) {
    throw new TypeError('expected Uint8Array or Uint16Array');
  }
  const bytes = [];
  let acc = 0, half = false;
  function nib(n) {
    if (!half) { acc = n << 4; half = true; }
    else { bytes.push(acc | n); half = false; }
  }
  let i = 0;
  while (i < values.length) {
    const value = values[i];
    if (value === 0) {
      let run = 1;
      while (i + run < values.length && values[i + run] === 0) run++;
      let left = run;
      while (left >= 4) {
        const n = Math.min(255, left);
        nib(0xf); nib(0); nib(n >> 4); nib(n & 15);
        left -= n;
      }
      while (left-- > 0) nib(0);
      i += run;
      continue;
    }
    if (value <= 14) { nib(value); i++; continue; }
    nib(0xf); nib(1);
    nib((value >> 12) & 15); nib((value >> 8) & 15);
    nib((value >> 4) & 15); nib(value & 15);
    i++;
  }
  if (half) nib(0);
  return Uint8Array.from(bytes);
}

function decode(bytes, length, bits) {
  const Type = bits === 16 ? Uint16Array : Uint8Array;
  const values = new Type(length);
  let acc = 0, half = false, pos = 0, n = 0;
  function nib() {
    if (!half) {
      if (pos >= bytes.length) throw new RangeError('nibble truncated');
      acc = bytes[pos++];
      half = true;
      return acc >> 4;
    }
    half = false;
    return acc & 15;
  }
  while (n < length) {
    const token = nib();
    if (token < 15) { values[n++] = token; continue; }
    const tag = nib();
    if (tag === 0) {
      const run = (nib() << 4) | nib();
      for (let k = 0; k < run; k++) values[n++] = 0;
    } else if (tag === 1) {
      values[n++] = (nib() << 12) | (nib() << 8) | (nib() << 4) | nib();
    } else throw new RangeError(`bad run tag ${tag}`);
  }
  if (n !== length) throw new RangeError('run count');
  return values;
}

module.exports = { encode, decode };
