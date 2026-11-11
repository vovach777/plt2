'use strict';

// The codec is the library. This program only chooses the last step.
// -e name loads ./entropy_<name>.js. A new file is a new method.
//
//   node palicjs-cli.js -e rlgr image.ppm -o image.pjs
//   node palicjs-cli.js -d image.pjs -o image.ppm

const fs = require('fs');
const layout = require('./tile-fse-layout');

const C = layout.constants;

function loadEntropy(name) {
  if (!/^[a-z][a-z0-9]*$/.test(name)) throw new Error(`bad module ${name}`);
  return require(`./entropy_${name}.js`);
}

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

function token(buf, cursor) {
  let i = cursor.i;
  while (i < buf.length) {
    if (buf[i] === 35) { while (i < buf.length && buf[i] !== 10) i++; continue; }
    if (buf[i] <= 32) { i++; continue; }
    break;
  }
  const start = i;
  while (i < buf.length && buf[i] > 32) i++;
  cursor.i = i;
  return buf.toString('latin1', start, i);
}

function loadNetpbm(path) {
  const buf = fs.readFileSync(path);
  const cursor = { i: 0 };
  const magic = token(buf, cursor);
  const width = Number(token(buf, cursor));
  const height = Number(token(buf, cursor));
  const max = Number(token(buf, cursor));
  if ((magic !== 'P5' && magic !== 'P6') || max !== 255) throw new Error(`unsupported ${magic}`);
  if (buf[cursor.i] <= 32) cursor.i++;
  const need = width * height * (magic === 'P6' ? 3 : 1);
  const data = Uint8Array.from(buf.subarray(cursor.i, cursor.i + need));
  if (data.length !== need) throw new Error('truncated netpbm');
  return { magic, width, height, data };
}

function saveNetpbm(image) {
  const magic = image.magic;
  const head = Buffer.from(`${magic}\n${image.width} ${image.height}\n255\n`);
  return Buffer.concat([head, image.data]);
}

function rowsOf(plane) {
  const rows = [];
  for (let L = 0; L < 8; L++) rows.push(plane.buckets[L].coefficients);
  rows.push(plane.rawWords);
  return rows;
}

function encodePjs(image, name) {
  const entropy = loadEntropy(name);
  const planeCount = image.format === C.FORMAT_PLN1 ? 1 : 3;
  const bin = [];
  for (const tile of image.tiles) {
    const planes = planeCount === 1 ? [tile.data.planes[0]] : tile.data.planes;
    for (const plane of planes) {
      for (const values of rowsOf(plane)) {
        const bits = values instanceof Uint16Array ? 16 : 8;
        const packed = entropy.encode(values);
        bin.push(bits);
        pushVarint(bin, values.length);
        pushVarint(bin, packed.length);
        for (let i = 0; i < packed.length; i++) bin.push(packed[i]);
      }
    }
  }
  const formatName = image.format === C.FORMAT_PLN1 ? 'PLN1' : 'PLN2';
  const head = Buffer.from(`${formatName}-${name.toUpperCase()}\n${image.width} ${image.height}\n`);
  return Buffer.concat([head, Buffer.from(bin)]);
}

function bucket(coefficients, plane, L) {
  const alphabetSize = plane === 0 ? 256 >>> L : 512 >>> L;
  let maxSymbol = -1;
  const frequencies = new Uint32Array(alphabetSize);
  for (let i = 0; i < coefficients.length; i++) {
    const value = coefficients[i];
    if (value > maxSymbol) maxSymbol = value;
    frequencies[value]++;
  }
  return {
    plane, L, coefficients,
    symbolCount: coefficients.length,
    alphabetSize,
    mode: 'fse',
    frequencies,
    maxSymbol
  };
}

function tileData(width, height, format, planes) {
  return {
    version: C.VERSION,
    format,
    scope: 'tile',
    width,
    height,
    off: C.DEFAULT_OFF,
    blockSize: C.BLOCK_SIZE,
    blockLayout: C.BLOCK_LAYOUT,
    predictor: C.PREDICTOR,
    layout: 'planar-row-major',
    colorTransform: format === C.FORMAT_PLN2 ? C.COLOR_TRANSFORM : null,
    planes
  };
}

function decodePjs(buf) {
  let headerEnd = 0, newlines = 0;
  while (headerEnd < buf.length && newlines < 2) {
    if (buf[headerEnd++] === 10) newlines++;
  }
  const [magic, sizeLine] = buf.toString('latin1', 0, headerEnd).trimEnd().split('\n');
  const method = /^PLN([12])-([A-Z0-9]+)$/.exec(magic);
  if (!method) throw new Error(`bad magic ${magic}`);
  const planeCount = method[1] === '1' ? 1 : 3;
  const name = method[2].toLowerCase();
  const [width, height] = sizeLine.split(' ').map(Number);
  const entropy = loadEntropy(name);
  const format = planeCount === 1 ? C.FORMAT_PLN1 : C.FORMAT_PLN2;
  const cursor = { i: headerEnd };
  const tiles = [];
  for (let y = 0; y < height; y += C.TILE_SIZE) {
    const tileHeight = Math.min(C.TILE_SIZE, height - y);
    for (let x = 0; x < width; x += C.TILE_SIZE) {
      const tileWidth = Math.min(C.TILE_SIZE, width - x);
      const planes = [];
      for (let p = 0; p < planeCount; p++) {
        const rows = [];
        for (let s = 0; s < 9; s++) {
          const bits = buf[cursor.i++];
          const count = readVarint(buf, cursor);
          const length = readVarint(buf, cursor);
          const packed = buf.subarray(cursor.i, cursor.i + length);
          cursor.i += length;
          rows.push(entropy.decode(packed, count, bits));
        }
        const built = [];
        for (let L = 0; L < 8; L++) built.push(bucket(rows[L], p, L));
        planes.push({ buckets: built, rawWords: rows[8] });
      }
      tiles.push({
        x, y, width: tileWidth, height: tileHeight,
        data: tileData(tileWidth, tileHeight, format, planes)
      });
    }
  }
  if (cursor.i !== buf.length) throw new Error('trailing bytes');
  const image = {
    version: C.VERSION,
    format,
    scope: 'image',
    width, height,
    off: C.DEFAULT_OFF,
    tileSize: C.TILE_SIZE,
    tileOrder: 'row-major-independent',
    tiles
  };
  const data = format === C.FORMAT_PLN1
    ? layout.recomposeGrayImage(image)
    : layout.recomposeImage(image);
  return { magic: planeCount === 1 ? 'P5' : 'P6', width, height, data };
}

function decomposeInput(image) {
  return image.magic === 'P5'
    ? layout.decomposeGrayImage(image.data, image.width, image.height)
    : layout.decomposeImage(image.data, image.width, image.height);
}

const args = process.argv.slice(2);
let method = null, decode = false, out = null, file = null;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '-e') method = args[++i];
  else if (args[i] === '-d') decode = true;
  else if (args[i] === '-o') out = args[++i];
  else if (args[i] === '--help') method = 'help';
  else file = args[i];
}

if (!file || method === 'help' || (!decode && !method)) {
  process.stderr.write('node palicjs-cli.js -e none|rlgr|rl|rice image.ppm -o out.pjs\n');
  process.stderr.write('node palicjs-cli.js -d out.pjs -o image.ppm\n');
  process.exit(method === 'help' && file == null ? 0 : 1);
}

const result = decode
  ? saveNetpbm(decodePjs(fs.readFileSync(file)))
  : encodePjs(decomposeInput(loadNetpbm(file)), method);

if (out) fs.writeFileSync(out, result);
else process.stdout.write(result);
