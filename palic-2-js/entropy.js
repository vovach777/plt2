'use strict';

// Entropy compressors for the ranking. Input is PLN-TXT from the text codec,
// output is bytes. decompress() returns that same text, rendered again by
// the text codec.

const { parseTxt, renderParsed } = require('./pln-text');

function pushVarint(out, value) {
  let n = value;
  while (n >= 128) {
    out.push((n & 127) | 128);
    n = Math.floor(n / 128);
  }
  out.push(n);
}

function readVarint(bytes, cursor) {
  let value = 0;
  let shift = 0;
  while (cursor.i < bytes.length) {
    const byte = bytes[cursor.i++];
    value += (byte & 127) * (2 ** shift);
    if ((byte & 128) === 0) return value;
    shift += 7;
    if (shift > 28) throw new RangeError('varint overflow');
  }
  throw new RangeError('truncated varint');
}

function encodeRunsBin(values) {
  const bytes = [];
  let acc = 0;
  let half = false;
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

function decodeRunsBin(bytes) {
  const cursor = { i: 0 };
  const count = readVarint(bytes, cursor);
  let acc = 0;
  let half = false;
  let pos = cursor.i;
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
    } else {
      throw new RangeError(`bad run tag ${tag}`);
    }
  }
  if (values.length !== count) throw new RangeError('run count');
  return values;
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

function encodeRice(values) {
  const bytes = [];
  let acc = 0;
  let nbits = 0;
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

function decodeRice(bytes) {
  const cursor = { i: 0 };
  const count = readVarint(bytes, cursor);
  let acc = 0;
  let nbits = 0;
  let pos = cursor.i;
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

const CODERS = {
  RL02: { gray: false, encode: encodeRunsBin, decode: decodeRunsBin },
  RL01: { gray: true, encode: encodeRunsBin, decode: decodeRunsBin },
  RC02: { gray: false, encode: encodeRice, decode: decodeRice },
  RC01: { gray: true, encode: encodeRice, decode: decodeRice }
};

function compress(text, kind) {
  const doc = parseTxt(text);
  if (!doc.coded) throw new RangeError('entropy input is PLN-TXT');
  const magic = (kind === 'rice' ? 'RC' : 'RL') + (doc.gray ? '01' : '02');
  const out = Array.from(Buffer.from(magic));
  pushVarint(out, doc.width);
  pushVarint(out, doc.height);
  for (const planes of doc.tiles) {
    for (const rows of planes) {
      for (const values of rows) {
        const packed = CODERS[magic].encode(values);
        pushVarint(out, packed.length);
        for (const byte of packed) out.push(byte);
      }
    }
  }
  return Uint8Array.from(out);
}

function decompress(bytes) {
  const magic = Buffer.from(bytes.subarray(0, 4)).toString('ascii');
  const coder = CODERS[magic];
  if (!coder) throw new RangeError(`bad magic ${magic}`);
  const cursor = { i: 4 };
  const width = readVarint(bytes, cursor);
  const height = readVarint(bytes, cursor);
  const nx = Math.ceil(width / 288);
  const ny = Math.ceil(height / 288);
  const planes = coder.gray ? 1 : 3;
  const tiles = [];
  for (let tile = 0; tile < nx * ny; tile++) {
    const tilePlanes = [];
    for (let plane = 0; plane < planes; plane++) {
      const rows = [];
      for (let s = 0; s < 9; s++) {
        const length = readVarint(bytes, cursor);
        const packed = bytes.subarray(cursor.i, cursor.i + length);
        cursor.i += length;
        rows.push(coder.decode(packed));
      }
      tilePlanes.push(rows);
    }
    tiles.push(tilePlanes);
  }
  if (cursor.i !== bytes.length) throw new RangeError('trailing bytes');
  return renderParsed({
    magic: coder.gray ? 'PLN1-TXT' : 'PLN2-TXT',
    width,
    height,
    gray: coder.gray,
    coded: true,
    tiles
  });
}

module.exports = { compress, decompress };
