// Слой 1. RGB -> плоскости Y, U, V.
// Часть tile-lib.ts. Склейка: rgb, plane, rlgr, tile. import этому компилятору не нужен.

function splitRgb(rgb: number[], y: number[], u: number[], v: number[], n: number): void {
  let i = 0;
  while (i < n) {
    let p = i * 3;
    let r = rgb[p];
    let g = rgb[p + 1];
    let b = rgb[p + 2];
    let du = r - g;
    let v0 = b - g;
    y[i] = (g + ((du + v0 + 2) >> 2)) & 255;
    u[i] = du;
    v[i] = v0 - (du >> 2);
    i = i + 1;
  }
}
