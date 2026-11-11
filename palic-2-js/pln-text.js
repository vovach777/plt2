'use strict';

// Text codec only. It turns a picture into PLN-TXT and reads that text back.
// Entropy compressors live in entropy.js: they take this text and return bytes.

const { decomposeImage, decomposeGrayImage } = require('./tile-fse-layout');

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

function basketsOf(image) {
  const gray = image.format === 'PLN1';
  return image.tiles.map(tile => {
    const planes = gray ? [tile.data.planes[0]] : tile.data.planes;
    return planes.map(plane => {
      const rows = [];
      for (let L = 0; L < 8; L++) rows.push(Array.from(plane.buckets[L].coefficients));
      rows.push(Array.from(plane.rawWords));
      return rows;
    });
  });
}

function payload(values, coder) {
  return values.length ? coder(values) : '-';
}

function streamLine(index, values, coder) {
  const tag = index < 8 ? `L${index}` : 'M';
  return `${tag} ${payload(values, coder)}`;
}

// tiles: per tile, per plane, nine number rows (L0..L7, then masks).
function renderDocument(width, height, tiles, magic, coder) {
  const gray = tiles[0].length === 1;
  const nx = Math.ceil(width / TILE);
  const ny = Math.ceil(height / TILE);
  const groups = tiles.map(planes => planes.map(rows => rows.map((values, i) => streamLine(i, values, coder))));
  let dataLine = 3 + ny;
  const cells = groups.map((group, tile) => {
    const starts = [];
    for (const plane of group) {
      starts.push(dataLine);
      dataLine += plane.length;
    }
    return { starts, row: Math.floor(tile / nx) };
  });
  const index = [];
  for (let row = 0; row < ny; row++) {
    index.push(cells.filter(cell => cell.row === row).map(cell => cell.starts.join(' ')).join('\t'));
  }
  if (gray && magic.startsWith('PLN2')) throw new RangeError('gray document');
  return [`${magic}`, `${width} ${height}`, ...index, ...groups.flat(2)].join('\n') + '\n';
}

function renderImage(image, magic, coder) {
  return renderDocument(image.width, image.height, basketsOf(image), magic, coder);
}

function render(rgb, width, height) {
  return renderImage(decomposeImage(rgb, width, height), 'PLN2', values => values.join(' '));
}

function renderTxt(rgb, width, height) {
  return renderImage(decomposeImage(rgb, width, height), 'PLN5-TXT', encodeRuns);
}

function renderGrayTxt(gray, width, height) {
  return renderImage(decomposeGrayImage(gray, width, height), 'PLN5-TXT', encodeRuns);
}

function parseTxt(text) {
  const lines = text.trimEnd().split('\n');
  const magic = lines[0];
  const coded = magic.endsWith('-TXT');
  const [width, height] = lines[1].split(' ').map(Number);
  const ny = Math.ceil(height / TILE);
  const nx = Math.ceil(width / TILE);
  const gray = magic.startsWith('PLN1')
    || (magic.startsWith('PLN5') && lines[2].split('\t')[0].split(' ').length === 1);
  const planes = gray ? 1 : 3;
  const data = lines.slice(2 + ny);
  const expected = nx * ny * planes * 9;
  if (data.length !== expected) throw new RangeError(`lines ${data.length}, expected ${expected}`);

  const tiles = [];
  let cursor = 0;
  for (let tile = 0; tile < nx * ny; tile++) {
    const tilePlanes = [];
    for (let plane = 0; plane < planes; plane++) {
      const rows = [];
      for (let s = 0; s < 9; s++) {
        const line = data[cursor++];
        const space = line.indexOf(' ');
        const body = line.slice(space + 1);
        rows.push(body === '-' ? [] : (coded ? decodeRuns(body) : body.split(' ').map(Number)));
      }
      tilePlanes.push(rows);
    }
    tiles.push(tilePlanes);
  }
  return { magic, width, height, gray, coded, tiles };
}

function renderParsed(doc) {
  const coder = doc.coded ? encodeRuns : values => values.join(' ');
  return renderDocument(doc.width, doc.height, doc.tiles, doc.magic, coder);
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

if (require.main === module) {
  const text = renderTxt(synthetic8(), 8, 8);
  if (renderParsed(parseTxt(text)) !== text) throw new Error('text roundtrip');
  process.stdout.write(text);
}

module.exports = {
  render,
  renderTxt,
  renderGrayTxt,
  parseTxt,
  renderParsed,
  encodeRuns,
  decodeRuns,
  encodeVarint,
  decodeVarint
};
