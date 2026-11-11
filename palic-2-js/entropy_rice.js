'use strict';

// Rice on PLN-TXT. k is floor(log2) of the mean of the last three symbols.
// Not a CLI output.

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
  for (const sym of values) {
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
  const out = [];
  pushVarint(out, values.length);
  for (const byte of bytes) out.push(byte);
  return Uint8Array.from(out);
}

function decode(bytes) {
  const cursor = { i: 0 };
  const count = readVarint(bytes, cursor);
  let acc = 0, nbits = 0, pos = cursor.i;
  function readBit() {
    if (nbits === 0) {
      if (pos >= bytes.length) throw new RangeError('rice truncated');
      acc = bytes[pos++];
      nbits = 8;
    }
    nbits--;
    return (acc >> nbits) & 1;
  }
  const values = [];
  const hist = [];
  for (let s = 0; s < count; s++) {
    const k = riceK(hist);
    let q = 0;
    while (readBit() === 1) q++;
    let r = 0;
    for (let b = 0; b < k; b++) r = (r << 1) | readBit();
    const sym = (q << k) | r;
    values.push(sym);
    hist.push(sym);
    if (hist.length > 3) hist.shift();
  }
  return values;
}

function compress(text) {
  const doc = parseTxt(text);
  if (!doc.coded) throw new RangeError('entropy input is PLN-TXT');
  const out = Array.from(Buffer.from(doc.gray ? 'RC01' : 'RC02'));
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
  if (magic !== 'RC01' && magic !== 'RC02') throw new RangeError(`bad magic ${magic}`);
  const gray = magic === 'RC01';
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
