'use strict';

// llrice RLGR::Encoder on one typed array. Not JPEG-XR, not RLGR::Simple.
// k and kr are stored scaled by 8. UP_K = 5, DOWN_K = 9, direct +2/−3.
// A Rice quotient of 32 or more is 32 ones and the raw word.

const MAX_K = 240;

function encodeValues(values) {
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
          rl = 0;
          k += 5;
          if (k > MAX_K) k = MAX_K;
        }
      } else {
        writeBits(1, 1);
        if (k_) writeBits(rl, k_);
        k = k > 9 ? k - 9 : 0;
        rl = 0;
        writeRice(value - 1, kr_);
      }
    } else {
      rl = 0;
      writeRice(value, kr_);
      if (value === 0) { k += 2; if (k > MAX_K) k = MAX_K; }
      else k = k > 3 ? k - 3 : 0;
    }
  }
  if (rl > 0) {
    const k_ = k >> 3;
    writeBits(1, 1);
    if (k_) writeBits(rl, k_);
  }
  if (nbits) bytes.push((acc << (8 - nbits)) & 255);
  return Uint8Array.from(bytes);
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

function typed(values, length, bits) {
  const Type = bits === 16 ? Uint16Array : Uint8Array;
  const out = new Type(length);
  for (let i = 0; i < length; i++) out[i] = values[i];
  return out;
}

function encode(values) {
  if (!(values instanceof Uint8Array) && !(values instanceof Uint16Array)) {
    throw new TypeError('expected Uint8Array or Uint16Array');
  }
  return Uint8Array.from(encodeValues(values));
}

function decode(bytes, length, bits) {
  return typed(length ? decodeValues(bytes, length) : [], length, bits);
}

module.exports = { encode, decode };
