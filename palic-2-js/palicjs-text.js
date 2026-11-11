'use strict';

// Debug text only. No entropy.
//   node palicjs-text.js image.ppm

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

const file = process.argv[2];
if (!file || file === '--help') {
  process.stderr.write('node palicjs-text.js image.ppm\n');
  process.exit(file === '--help' ? 0 : 1);
}

const buf = fs.readFileSync(file);
const cursor = { i: 0 };
const magic = token(buf, cursor);
const width = Number(token(buf, cursor));
const height = Number(token(buf, cursor));
const max = Number(token(buf, cursor));
if ((magic !== 'P5' && magic !== 'P6') || max !== 255) throw new Error(`unsupported ${magic}`);
if (buf[cursor.i] <= 32) cursor.i++;
const data = Uint8Array.from(buf.subarray(cursor.i));
process.stdout.write(magic === 'P5'
  ? renderGrayTxt(data, width, height)
  : renderTxt(data, width, height));
