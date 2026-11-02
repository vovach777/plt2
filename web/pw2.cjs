const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch(); const page = await browser.newPage();
  page.on('pageerror', e => console.log('page error:', e.message));
  await page.goto('http://127.0.0.1:8765/index.html');
  const cases = [['fhd.plt', 'fhd.ppm'], ['fhd_s2.plt', 'fhd.ppm'], ['big_a31.plt', 'big.ppm'], ['big_s2.plt', 'big.ppm'], ['big_s4.plt', 'big.ppm']];
  for (const [plt, ppm] of cases) for (const K of [1, 2]) {
    let best = null;
    for (let t = 0; t < 2; ++t) { const r = await page.evaluate(([a, b, k]) => window.run(a, b, k, 3), [plt, ppm, K]); if (!best || r.ms < best.ms) best = r; }
    console.log(`${plt.padEnd(12)} ${best.W}x${best.H} strips=${best.strips} workers=${best.workers}  ${best.ms} ms  ${best.mbs} MB/s  badPixels=${best.badPixels}`);
  }
  await browser.close();
})();
