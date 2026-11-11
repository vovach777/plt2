'use strict';

// Run-length on PLN-TXT. Two symbols per byte. Not a CLI output.

const { parseTxt, renderParsed } = require('./pln-text');

function pushVarint(out, value) {
  let n = value;
  while (n >= 128) { out.push((n & 127) | 128); n = Math.floor(n / 128); }
  out.push(n);
}

function readVarint(bytes, cursor) {
  let value = 0, shift = 0;
  while (cursor.i < bytes.length) {
    const byte = bytes[cursor.i++];
    value += (byte & 127) * (2 ** shift);
    if ((byte & 128) === 0) return value;
    shift += 7;
    if (shift > 28) throw new RangeError('varint overflow');
  }
  throw new RangeError('truncated varint');
}

function encode(values) {
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
    if (value > 0 && value <= 14) { nib(value); i++; continue; }
    nib(0xf); nib(1);
    nib((value >> 12) & 15); nib((value >> 8) & 15);
    nib((value >> 4) & 15); nib(value & 15);
    i++;
  }
  if (half) nib(0);
  const out = [];
  pushVarint(out, values.length);
  for (const byte of bytes) out.push(byte);
  return Uint8Array.from(out);
}

function decode(bytes) {
  const cursor = { i: 0 };
  const count = readVarint(bytes, cursor);
  let acc = 0, half = false, pos = cursor.i;
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
  const values = [];
  while (values.length < count) {
    const n = nib();
    if (n < 15) { values.push(n); continue; }
    const tag = nib();
    if (tag === 0) {
      const len = (nib() << 4) | nib();
      for (let k = 0; k < len; k++) values.push(0);
    } else if (tag === 1) {
      values.push((nib() << 12) | (nib() << 8) | (nib() << 4) | nib());
    } else throw new RangeError(`bad run tag ${tag}`);
  }
  if (values.length !== count) throw new RangeError('run count');
  return values;
}

function compress(text) {
  const doc = parseTxt(text);
  if (!doc.coded) throw new RangeError('entropy input is PLN-TXT');
  const out = Array.from(Buffer.from(doc.gray ? 'RL01' : 'RL02'));
  pushVarint(out, doc.width);
  pushVarint(out, doc.height);
  for (const planes of doc.tiles) for (const rows of planes) for (const values of rows) {
    const packed = encode(values);
    pushVarint(out, packed.length);
    for (const byte of packed) out.push(byte);
  }
  return Uint8Array.from(out);
}

function decompress(bytes) {
  const magic = Buffer.from(bytes.subarray(0, 4)).toString('ascii');
  if (magic !== 'RL01' && magic !== 'RL02') throw new RangeError(`bad magic ${magic}`);
  const gray = magic === 'RL01';
  const cursor = { i: 4 };
  const width = readVarint(bytes, cursor);
  const height = readVarint(bytes, cursor);
  const nx = Math.ceil(width / 288);
  const ny = Math.ceil(height / 288);
  const planes = gray ? 1 : 3;
  const tiles = [];
  for (let tile = 0; tile < nx * ny; tile++) {
    const tilePlanes = [];
    for (let plane = 0; plane < planes; plane++) {
      const rows = [];
      for (let s = 0; s < 9; s++) {
        const length = readVarint(bytes, cursor);
        const packed = bytes.subarray(cursor.i, cursor.i + length);
        cursor.i += length;
        rows.push(decode(packed));
      }
      tilePlanes.push(rows);
    }
    tiles.push(tilePlanes);
  }
  if (cursor.i !== bytes.length) throw new RangeError('trailing bytes');
  return renderParsed({ magic: gray ? 'PLN1-TXT' : 'PLN2-TXT', width, height, gray, coded: true, tiles });
}

module.exports = { compress, decompress };
