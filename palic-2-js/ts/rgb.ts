// Слой 1. RGB -> плоскости. Y байты, U и V знаковые 16 бит.

function splitRgb(rgb: Uint8Array, y: Uint8Array, u: Uint16Array, v: Uint16Array, n: number): void {
  let i = 0;
  while (i < n) {
    let p = i * 3;
    let r = rgb[p];
    let g = rgb[p + 1];
    let b = rgb[p + 2];
    let du = r - g;
    let v0 = b - g;
    y[i] = (g + ((du + v0 + 2) >> 2)) & 255;
    u[i] = du & 65535;
    v[i] = (v0 - (du >> 2)) & 65535;
    i = i + 1;
  }
}
