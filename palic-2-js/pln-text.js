'use strict';

// Text view of a palic-2 image, and two binary coders on the same baskets.
//
// Text, for the eye:
//   PLN2 / PLN2-TXT
//   W H
//   startY startU startV     (tab between tiles of one row)
//   L0..L7 and M, Y then U then V
//
// Binary run-length (RL02 colour, RL01 gray): nibbles, two per byte.
//   0..E                 one symbol
//   F0, then two nibbles a zero run of that length (only used from length 4)
//   F1, then four nibbles one symbol above 14
//   a varint count is in front, so a trailing pad nibble is ignored
//
// Rice (RC02 / RC01): k is the bit width of the mean of the last three
// symbols. Fewer than three: the mean of those we have. k = 0 at the start.

const { decomposeImage } = require('./tile-fse-layout');

const TILE = 288;

function encodeVarint(value) {
  if (!Number.isInteger(value) || value < 0) throw new RangeError(`bad varint ${value}`);
  const bytes = [];
  let n = value;
  while (n >= 128) {
    bytes.push((n & 127) | 128);
    n = Math.floor(n / 128);
  }
  bytes.push(n);
  return bytes.map(byte => byte.toString(16).toUpperCase().padStart(2, '0')).join('');
}

function decodeVarint(hex) {
  if (!/^[0-9A-F]+$/i.test(hex) || hex.length % 2 !== 0) throw new RangeError(`bad varint ${hex}`);
  let value = 0;
  let shift = 0;
  for (let i = 0; i < hex.length; i += 2) {
    const byte = Number.parseInt(hex.slice(i, i + 2), 16);
    value += (byte & 127) * (2 ** shift);
    if ((byte & 128) === 0) {
      if (i + 2 !== hex.length) throw new RangeError(`varint tail ${hex}`);
      return value;
    }
    shift += 7;
  }
  throw new RangeError(`truncated varint ${hex}`);
}

function encodeRuns(values) {
  const tokens = [];
  let i = 0;
  while (i < values.length) {
    const value = values[i];
    let run = 1;
    while (i + run < values.length && values[i + run] === value) run++;
    if (value >= 0 && value <= 14) {
      let left = run;
      while (left > 0) {
        const n = Math.min(14, left);
        tokens.push(n.toString(16).toUpperCase() + value.toString(16).toUpperCase());
        left -= n;
      }
    } else {
      tokens.push('F', encodeVarint(run), encodeVarint(value));
    }
    i += run;
  }
  return tokens.join(' ');
}

function decodeRuns(text) {
  if (text === '') return [];
  const tokens = text.split(' ');
  const values = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token === 'F') {
      const run = decodeVarint(tokens[++i]);
      const value = decodeVarint(tokens[++i]);
      for (let k = 0; k < run; k++) values.push(value);
    } else if (/^[1-9A-E][0-9A-E]$/.test(token)) {
      const run = Number.parseInt(token[0], 16);
      const value = Number.parseInt(token[1], 16);
      for (let k = 0; k < run; k++) values.push(value);
    } else {
      throw new RangeError(`bad token ${token}`);
    }
  }
  return values;
}

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
    if (!half) {
      acc = n << 4;
      half = true;
    } else {
      bytes.push(acc | n);
      half = false;
    }
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
    if (value > 0 && value <= 14) {
      nib(value);
      i++;
      continue;
    }
    nib(0xf); nib(1);
    nib((value >> 12) & 15);
    nib((value >> 8) & 15);
    nib((value >> 4) & 15);
    nib(value & 15);
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
    if (n < 15) {
      values.push(n);
      continue;
    }
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

function planeStreams(plane) {
  const streams = [];
  for (let L = 0; L < 8; L++) streams.push(Array.from(plane.buckets[L].coefficients));
  streams.push(Array.from(plane.rawWords));
  return streams;
}

function packRuns(image) {
  const gray = image.format === 'PLN1';
  const out = Array.from(Buffer.from(gray ? 'RL01' : 'RL02'));
  pushVarint(out, image.width);
  pushVarint(out, image.height);
  for (const tile of image.tiles) {
    const planes = gray ? [tile.data.planes[0]] : tile.data.planes;
    for (const plane of planes) {
      for (const values of planeStreams(plane)) {
        const packed = encodeRunsBin(values);
        pushVarint(out, packed.length);
        for (const byte of packed) out.push(byte);
      }
    }
  }
  return Uint8Array.from(out);
}

function unpackRuns(bytes, image) {
  const gray = image.format === 'PLN1';
  const magic = Buffer.from(bytes.subarray(0, 4)).toString('ascii');
  if (magic !== (gray ? 'RL01' : 'RL02')) throw new RangeError(`bad magic ${magic}`);
  const cursor = { i: 4 };
  const width = readVarint(bytes, cursor);
  const height = readVarint(bytes, cursor);
  if (width !== image.width || height !== image.height) throw new RangeError('size mismatch');
  for (const tile of image.tiles) {
    const planes = gray ? [tile.data.planes[0]] : tile.data.planes;
    for (const plane of planes) {
      const streams = planeStreams(plane);
      for (let s = 0; s < streams.length; s++) {
        const length = readVarint(bytes, cursor);
        const packed = bytes.subarray(cursor.i, cursor.i + length);
        cursor.i += length;
        const values = decodeRunsBin(packed);
        if (values.length !== streams[s].length || values.some((v, k) => v !== streams[s][k])) {
          throw new RangeError(`stream mismatch ${plane.name} ${s}`);
        }
        if (s < 8) plane.buckets[s].coefficients = values;
        else plane.rawWords = values;
      }
    }
  }
  if (cursor.i !== bytes.length) throw new RangeError('trailing bytes');
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
      if (nbits === 8) {
        bytes.push(acc & 255);
        acc = 0;
        nbits = 0;
      }
    }
  }
  const hist = [];
  for (const sym of values) {
    const k = riceK(hist);
    let q = sym >>> k;
    while (q >= 24) {
      writeBits(0xffffff, 24);
      q -= 24;
    }
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

function packCoded(image, magicGray, magicColor, encode) {
  const gray = image.format === 'PLN1';
  const out = Array.from(Buffer.from(gray ? magicGray : magicColor));
  pushVarint(out, image.width);
  pushVarint(out, image.height);
  for (const tile of image.tiles) {
    const planes = gray ? [tile.data.planes[0]] : tile.data.planes;
    for (const plane of planes) {
      for (const values of planeStreams(plane)) {
        const packed = encode(values);
        pushVarint(out, packed.length);
        for (const byte of packed) out.push(byte);
      }
    }
  }
  return Uint8Array.from(out);
}

function unpackCoded(bytes, image, magicGray, magicColor, decode) {
  const gray = image.format === 'PLN1';
  const magic = Buffer.from(bytes.subarray(0, 4)).toString('ascii');
  if (magic !== (gray ? magicGray : magicColor)) throw new RangeError(`bad magic ${magic}`);
  const cursor = { i: 4 };
  if (readVarint(bytes, cursor) !== image.width || readVarint(bytes, cursor) !== image.height) {
    throw new RangeError('size mismatch');
  }
  for (const tile of image.tiles) {
    const planes = gray ? [tile.data.planes[0]] : tile.data.planes;
    for (const plane of planes) {
      const streams = planeStreams(plane);
      for (let s = 0; s < streams.length; s++) {
        const length = readVarint(bytes, cursor);
        const packed = bytes.subarray(cursor.i, cursor.i + length);
        cursor.i += length;
        const values = decode(packed);
        if (values.length !== streams[s].length || values.some((v, k) => v !== streams[s][k])) {
          throw new RangeError(`stream mismatch ${plane.name} ${s}`);
        }
        if (s < 8) plane.buckets[s].coefficients = values;
        else plane.rawWords = values;
      }
    }
  }
  if (cursor.i !== bytes.length) throw new RangeError('trailing bytes');
}

function packRice(image) { return packCoded(image, 'RC01', 'RC02', encodeRice); }
function unpackRice(bytes, image) { unpackCoded(bytes, image, 'RC01', 'RC02', decodeRice); }

function payload(values, coder) {
  return values.length ? coder(values) : '-';
}

function planeLines(plane, coder) {
  const lines = [];
  for (let L = 0; L < 8; L++) lines.push(`L${L} ${payload(Array.from(plane.buckets[L].coefficients), coder)}`);
  lines.push(`M ${payload(Array.from(plane.rawWords), coder)}`);
  return lines;
}

function renderCoded(rgb, width, height, magic, coder) {
  const image = decomposeImage(rgb, width, height);
  const nx = Math.ceil(width / TILE);
  const ny = Math.ceil(height / TILE);
  const groups = image.tiles.map(tile => tile.data.planes.map(plane => planeLines(plane, coder)));
  let dataLine = 3 + ny;
  const cells = groups.map((group, tile) => {
    const startY = dataLine;
    const startU = startY + group[0].length;
    const startV = startU + group[1].length;
    dataLine = startV + group[2].length;
    return { startY, startU, startV, row: Math.floor(tile / nx) };
  });
  const index = [];
  for (let row = 0; row < ny; row++) {
    index.push(cells.filter(cell => cell.row === row).map(cell =>
      `${cell.startY} ${cell.startU} ${cell.startV}`
    ).join('\t'));
  }
  return {
    image,
    text: [`${magic}`, `${width} ${height}`, ...index, ...groups.flat(2)].join('\n') + '\n'
  };
}

function render(rgb, width, height) {
  return renderCoded(rgb, width, height, 'PLN2', values => values.join(' ')).text;
}

function renderTxt(rgb, width, height) {
  return renderCoded(rgb, width, height, 'PLN2-TXT', encodeRuns).text;
}

function synthetic8() {
  const width = 8;
  const height = 8;
  const rgb = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 3;
      rgb[i] = x * 32;
      rgb[i + 1] = y * 32;
      rgb[i + 2] = 128;
    }
  }
  return rgb;
}

function roundtrip(rgb, width, height) {
  const plain = renderCoded(rgb, width, height, 'PLN2', values => values.join(' '));
  const coded = renderCoded(rgb, width, height, 'PLN2-TXT', encodeRuns);
  const plainLines = plain.text.trimEnd().split('\n').slice(3);
  const codedLines = coded.text.trimEnd().split('\n').slice(3);
  if (plainLines.length !== codedLines.length) throw new Error('line count');
  for (let i = 0; i < plainLines.length; i++) {
    const [tag, ...rest] = plainLines[i].split(' ');
    const body = rest.join(' ');
    const codedBody = codedLines[i].slice(tag.length + 1);
    const back = codedBody === '-' ? [] : decodeRuns(codedBody);
    const original = body === '-' ? [] : body.split(' ').map(Number);
    if (tag.startsWith('L')) {
      if (back.join(' ') !== original.join(' ')) throw new Error(`L mismatch ${codedLines[i]}`);
    } else {
      const masks = Array.from(plain.image.tiles[0].data.planes[Math.floor(i / 9)].rawWords);
      if (back.join(' ') !== masks.join(' ')) throw new Error(`M mismatch ${codedLines[i]}`);
    }
  }
}

if (require.main === module) {
  const rgb = synthetic8();
  roundtrip(rgb, 8, 8);
  process.stdout.write(renderTxt(rgb, 8, 8));
}

module.exports = {
  render,
  renderTxt,
  encodeRuns,
  decodeRuns,
  encodeRunsBin,
  decodeRunsBin,
  packRuns,
  unpackRuns,
  encodeRice,
  decodeRice,
  packRice,
  unpackRice,
  encodeVarint,
  decodeVarint
};
