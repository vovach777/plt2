import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const dir = dirname(fileURLToPath(import.meta.url));
const wasm = readFileSync(join(dir, "palic2r.wasm"));
const instance = new WebAssembly.Instance(new WebAssembly.Module(wasm), {
  env: { abort() { throw new Error("palic2r abort"); } }
});
const api = instance.exports;
const memory = api.memory;
const arena = api.init();

function fit(addr, extra) {
  const need = Number(addr) + extra;
  const have = memory.buffer.byteLength;
  if (need <= have) return;
  memory.grow(Math.ceil((need - have) / 65536));
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
  return buf.toString("latin1", start, i);
}

export function loadNetpbm(path) {
  const buf = readFileSync(path);
  const cursor = { i: 0 };
  const magic = token(buf, cursor);
  const width = Number(token(buf, cursor));
  const height = Number(token(buf, cursor));
  const max = Number(token(buf, cursor));
  if ((magic !== "P5" && magic !== "P6") || max !== 255) throw new Error(`unsupported ${magic} ${path}`);
  if (buf[cursor.i] <= 32) cursor.i++;
  const channels = magic === "P6" ? 3 : 1;
  const data = new Uint8Array(buf.subarray(cursor.i, cursor.i + width * height * channels));
  if (data.length !== width * height * channels) throw new Error("truncated " + path);
  return { magic, width, height, channels, data, file: buf };
}

export function encodeImage(image) {
  fit(arena, image.data.length + 65536);
  new Uint8Array(memory.buffer).set(image.data, Number(arena));
  const n = api.encode(arena, image.width, image.height);
  if (n < 0) throw new Error("encode failed");
  const out = Buffer.from(new Uint8Array(memory.buffer, Number(api.outPtr()), n));
  if (image.file && out.length >= image.file.length) return Buffer.from(image.file);
  return out;
}

export function encodeFile(path) { return encodeImage(loadNetpbm(path)); }

export function decodeFile(file) {
  const text = file.toString("latin1", 0, Math.min(file.length, 64));
  if (text.startsWith("P6") || text.startsWith("P5")) return Buffer.from(file);
  const nl = text.indexOf("\n");
  const nl2 = text.indexOf("\n", nl + 1);
  const [width, height] = text.slice(nl + 1, nl2).split(" ").map(Number);
  const pixels = width * height * 3;
  fit(arena, file.length + pixels + 65536);
  const bytes = new Uint8Array(memory.buffer);
  const src = Number(arena);
  bytes.set(file, src);
  const dst = src + file.length;
  const rc = api.decode(src, file.length, dst);
  if (rc < 0) throw new Error("decode failed");
  const data = Buffer.from(new Uint8Array(memory.buffer, dst, pixels));
  return Buffer.concat([Buffer.from(`P6\n${width} ${height}\n255\n`), data]);
}

const args = process.argv.slice(2);
if (args[0] === "c" && args.length === 3) {
  writeFileSync(args[2], encodeFile(args[1]));
} else if (args[0] === "d" && args.length === 3) {
  writeFileSync(args[2], decodeFile(readFileSync(args[1])));
}
