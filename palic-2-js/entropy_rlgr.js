'use strict';

// llrice RLGR::Encoder. Not the JPEG-XR constants and not RLGR::Simple.
// k and kr are stored scaled by 8. UP_K = 5, DOWN_K = 9, direct +2/−3.
// A Rice quotient of 32 or more is 32 ones and the raw word.
// Input is PLN-TXT. compress() is bytes, renderText() is the same decisions
// written so they can be read.

const { parseTxt, renderParsed } = require('./pln-text');

const MAX_K = 240;
const TILE = 288;

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

function encodeValues(values) {
  const events = [];
  const bytes = [];
  let acc = 0, nbits = 0;
  function writeBits(value, width) {
    for (let i = width - 1; i >= 0; i--) {
      acc = (acc << 1) | ((value >>> i) & 1);
      if (++nbits === 8) { bytes.push(acc & 255); acc = 0; nbits = 0; }
    }
  }
  function ones(q) {
    while (q >= 8) { writeBits(0xff, 8); q -= 8; }
    if (q) writeBits((1 << q) - 1, q);
  }
  let k = 8, kr = 8, rl = 0;
  function writeRice(x, m) {
    const q = m >= 31 ? 0 : x >>> m;
    if (q >= 32) {
      writeBits(0xffffffff, 32);
      writeBits(x, 32);
    } else {
      ones(q);
      writeBits(0, 1);
      if (m) writeBits(x & ((1 << m) - 1), m);
    }
    const vk = m >= 31 ? 0 : x >>> m;
    if (vk === 0) kr = kr > 2 ? kr - 2 : 0;
    else if (vk !== 1) { kr += vk; if (kr > MAX_K) kr = MAX_K; }
  }
  for (const value of values) {
    const k_ = k >> 3;
    const kr_ = kr >> 3;
    if (k_ > 0) {
      if (value === 0) {
        rl++;
        if (rl === (1 << k_)) {
          writeBits(0, 1);
          events.push('W');
          rl = 0;
          k += 5;
          if (k > MAX_K) k = MAX_K;
        }
      } else {
        writeBits(1, 1);
        if (k_) writeBits(rl, k_);
        events.push('P', String(rl), String(value));
        k = k > 9 ? k - 9 : 0;
        rl = 0;
        writeRice(value - 1, kr_);
      }
    } else {
      rl = 0;
      events.push('D', String(value));
      writeRice(value, kr_);
      if (value === 0) { k += 2; if (k > MAX_K) k = MAX_K; }
      else k = k > 3 ? k - 3 : 0;
    }
  }
  if (rl > 0) {
    const k_ = k >> 3;
    writeBits(1, 1);
    if (k_) writeBits(rl, k_);
    events.push('T', String(rl));
  }
  if (nbits) bytes.push((acc << (8 - nbits)) & 255);
  return { events, bytes };
}

function decodeValues(bytes, count) {
  let pos = 0, acc = 0, nbits = 0;
  function bit() {
    if (nbits === 0) {
      if (pos >= bytes.length) throw new RangeError('rlgr truncated');
      acc = bytes[pos++];
      nbits = 8;
    }
    nbits--;
    return (acc >> nbits) & 1;
  }
  function bits(width) {
    let v = 0;
    for (let i = 0; i < width; i++) v = (v << 1) | bit();
    return v >>> 0;
  }
  let k = 8, kr = 8, rl = 0, runMode = false, full = false;
  function readRice(addOne) {
    const m = kr >> 3;
    let ones = 0;
    while (ones < 32 && bit() === 1) ones++;
    const value = ones === 32 ? bits(32) : ((ones << m) | (m ? bits(m) : 0)) >>> 0;
    const vk = m >= 31 ? 0 : value >>> m;
    if (vk === 0) kr = kr > 2 ? kr - 2 : 0;
    else if (vk !== 1) { kr += vk; if (kr > MAX_K) kr = MAX_K; }
    return value + (addOne ? 1 : 0);
  }
  function finishPartial() {
    k = k > 9 ? k - 9 : 0;
    runMode = false;
    return readRice(true);
  }
  function fromRun() {
    if (full) {
      rl--;
      if (rl === 0) {
        k += 5;
        if (k > MAX_K) k = MAX_K;
        runMode = false;
      }
      return 0;
    }
    if (rl === 0) return finishPartial();
    rl--;
    return 0;
  }
  const values = [];
  while (values.length < count) {
    if (runMode) { values.push(fromRun()); continue; }
    const k_ = k >> 3;
    if (k_ > 0) {
      full = bit() === 0;
      runMode = true;
      if (full) rl = 1 << k_;
      else {
        rl = bits(k_);
        if (rl === 0) { values.push(finishPartial()); continue; }
      }
      values.push(fromRun());
    } else {
      const v = readRice(false);
      if (v === 0) { k += 2; if (k > MAX_K) k = MAX_K; }
      else k = k > 3 ? k - 3 : 0;
      values.push(v);
    }
  }
  return values;
}

function valuesFromTokens(tokens) {
  const values = [];
  let k = 8;
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token === 'W') {
      const n = 1 << (k >> 3);
      for (let z = 0; z < n; z++) values.push(0);
      k += 5;
      if (k > MAX_K) k = MAX_K;
    } else if (token === 'P') {
      const run = Number(tokens[++i]);
      const value = Number(tokens[++i]);
      for (let z = 0; z < run; z++) values.push(0);
      values.push(value);
      k = k > 9 ? k - 9 : 0;
    } else if (token === 'D') {
      const value = Number(tokens[++i]);
      values.push(value);
      if (value === 0) { k += 2; if (k > MAX_K) k = MAX_K; }
      else k = k > 3 ? k - 3 : 0;
    } else if (token === 'T') {
      const run = Number(tokens[++i]);
      for (let z = 0; z < run; z++) values.push(0);
    } else {
      throw new RangeError(`bad rlgr token ${token}`);
    }
  }
  return values;
}

function pack(text, take) {
  const doc = parseTxt(text);
  if (!doc.coded) throw new RangeError('entropy input is PLN-TXT');
  const streams = [];
  for (const planes of doc.tiles) for (const rows of planes) for (const values of rows) {
    streams.push(take(values));
  }
  return { doc, streams };
}

function compress(text) {
  const { doc, streams } = pack(text, values => {
    const { bytes } = encodeValues(values);
    const packed = [];
    pushVarint(packed, values.length);
    for (const byte of bytes) packed.push(byte);
    return Uint8Array.from(packed);
  });
  const out = Array.from(Buffer.from(doc.gray ? 'RG01' : 'RG02'));
  pushVarint(out, doc.width);
  pushVarint(out, doc.height);
  for (const packed of streams) {
    pushVarint(out, packed.length);
    for (const byte of packed) out.push(byte);
  }
  return Uint8Array.from(out);
}

function decompress(bytes) {
  const magic = Buffer.from(bytes.subarray(0, 4)).toString('ascii');
  if (magic !== 'RG01' && magic !== 'RG02') throw new RangeError(`bad magic ${magic}`);
  const gray = magic === 'RG01';
  const cursor = { i: 4 };
  const width = readVarint(bytes, cursor);
  const height = readVarint(bytes, cursor);
  const nx = Math.ceil(width / TILE);
  const ny = Math.ceil(height / TILE);
  const planeCount = gray ? 1 : 3;
  const tiles = [];
  for (let tile = 0; tile < nx * ny; tile++) {
    const tilePlanes = [];
    for (let plane = 0; plane < planeCount; plane++) {
      const rows = [];
      for (let s = 0; s < 9; s++) {
        const length = readVarint(bytes, cursor);
        const packed = bytes.subarray(cursor.i, cursor.i + length);
        cursor.i += length;
        const inner = { i: 0 };
        const count = readVarint(packed, inner);
        rows.push(decodeValues(packed.subarray(inner.i), count));
      }
      tilePlanes.push(rows);
    }
    tiles.push(tilePlanes);
  }
  if (cursor.i !== bytes.length) throw new RangeError('trailing bytes');
  return renderParsed({ magic: gray ? 'PLN1-TXT' : 'PLN2-TXT', width, height, gray, coded: true, tiles });
}

function renderText(text) {
  const doc = parseTxt(text);
  const nx = Math.ceil(doc.width / TILE);
  const ny = Math.ceil(doc.height / TILE);
  const groups = doc.tiles.map(planes => planes.map(rows => rows.map((values, index) => {
    const tag = index < 8 ? `L${index}` : 'M';
    const { events } = encodeValues(values);
    return `${tag} ${events.length ? events.join(' ') : '-'}`;
  })));
  let dataLine = 3 + ny;
  const cells = groups.map((group, tile) => {
    const starts = [];
    for (const plane of group) { starts.push(dataLine); dataLine += plane.length; }
    return { starts, row: Math.floor(tile / nx) };
  });
  const index = [];
  for (let row = 0; row < ny; row++) {
    index.push(cells.filter(cell => cell.row === row).map(cell => cell.starts.join(' ')).join('\t'));
  }
  return ['RLGR-TXT', `${doc.width} ${doc.height}`, ...index, ...groups.flat(2)].join('\n') + '\n';
}

function parseText(text) {
  const lines = text.trimEnd().split('\n');
  if (lines[0] !== 'RLGR-TXT') throw new RangeError(`bad magic ${lines[0]}`);
  const [width, height] = lines[1].split(' ').map(Number);
  const ny = Math.ceil(height / TILE);
  const nx = Math.ceil(width / TILE);
  const data = lines.slice(2 + ny);
  const gray = data.length === nx * ny * 9;
  const planes = gray ? 1 : 3;
  if (data.length !== nx * ny * planes * 9) throw new RangeError('rlgr text size');
  const tiles = [];
  let cursor = 0;
  for (let tile = 0; tile < nx * ny; tile++) {
    const tilePlanes = [];
    for (let plane = 0; plane < planes; plane++) {
      const rows = [];
      for (let s = 0; s < 9; s++) {
        const line = data[cursor++];
        const body = line.slice(line.indexOf(' ') + 1);
        rows.push(body === '-' ? [] : valuesFromTokens(body.split(' ')));
      }
      tilePlanes.push(rows);
    }
    tiles.push(tilePlanes);
  }
  return renderParsed({
    magic: gray ? 'PLN1-TXT' : 'PLN2-TXT',
    width, height, gray, coded: true, tiles
  });
}

module.exports = { compress, decompress, renderText, parseText };
