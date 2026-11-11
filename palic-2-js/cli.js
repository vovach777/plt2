'use strict';

// node cli.js [image.ppm|pgm] [--out text|rlgr|rlgr-txt]
// text is the default. rlgr is loaded only when asked.
// Rice and runlen stay on the text; they are not outputs of this CLI.

const fs = require('fs');
const { renderTxt, renderGrayTxt } = require('./pln-text');

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
  if ((magic !== 'P5' && magic !== 'P6') || max !== 255) throw new RangeError(`unsupported ${magic} ${max}`);
  if (buf[cursor.i] <= 32) cursor.i++;
  const need = width * height * (magic === 'P6' ? 3 : 1);
  const data = Uint8Array.from(buf.subarray(cursor.i, cursor.i + need));
  if (data.length !== need) throw new RangeError('truncated netpbm');
  return { magic, width, height, data };
}

function synthetic() {
  const rgb = new Uint8Array(8 * 8 * 3);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
    const i = (y * 8 + x) * 3;
    rgb[i] = x * 32;
    rgb[i + 1] = y * 32;
    rgb[i + 2] = 128;
  }
  return rgb;
}

function textOf(file) {
  if (!file) return renderTxt(synthetic(), 8, 8);
  const image = loadNetpbm(file);
  return image.magic === 'P5'
    ? renderGrayTxt(image.data, image.width, image.height)
    : renderTxt(image.data, image.width, image.height);
}

const args = process.argv.slice(2);
let out = 'text';
let file = null;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--out') out = args[++i];
  else if (args[i] === '--help') out = 'help';
  else file = args[i];
}

if (out === 'help' || (out !== 'text' && out !== 'rlgr' && out !== 'rlgr-txt')) {
  process.stderr.write('node cli.js [image.ppm|pgm] [--out text|rlgr|rlgr-txt]\n');
  process.exit(out === 'help' ? 0 : 1);
}

const text = textOf(file);
if (out === 'text') {
  process.stdout.write(text);
} else {
  const rlgr = require('./entropy_rlgr');
  if (out === 'rlgr') process.stdout.write(rlgr.compress(text));
  else process.stdout.write(rlgr.renderText(text));
}
