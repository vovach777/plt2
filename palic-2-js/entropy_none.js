'use strict';

// Raw symbols. Uint8Array is copied. Uint16Array is little-endian bytes.
// bzip2 of a PLN1-NONE or PLN2-NONE file is the entropy reference.

function encode(values) {
  if (values instanceof Uint8Array) return new Uint8Array(values);
  if (!(values instanceof Uint16Array)) throw new TypeError('expected Uint8Array or Uint16Array');
  const out = new Uint8Array(values.length * 2);
  for (let i = 0; i < values.length; i++) {
    out[i * 2] = values[i] & 255;
    out[i * 2 + 1] = values[i] >> 8;
  }
  return out;
}

function decode(bytes, length, bits) {
  if (bits === 8) {
    if (bytes.length < length) throw new RangeError('none truncated');
    return bytes.slice(0, length);
  }
  if (bits !== 16) throw new RangeError(`bad width ${bits}`);
  if (bytes.length < length * 2) throw new RangeError('none truncated');
  const out = new Uint16Array(length);
  for (let i = 0; i < length; i++) out[i] = bytes[i * 2] | (bytes[i * 2 + 1] << 8);
  return out;
}

module.exports = { encode, decode };
