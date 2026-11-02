const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on('console', m => { if (m.type() === 'error') console.log('console error:', m.text()); });
  page.on('pageerror', e => console.log('page error:', e.message));
  await page.goto('http://127.0.0.1:8765/index.html' + (process.argv[2] ? '?wasm=' + process.argv[2] : ''));
  console.log('hardwareConcurrency =', await page.evaluate(() => navigator.hardwareConcurrency));
  const cases = [['fhd.plt', 'fhd.ppm'], ['big_a31.plt', 'big.ppm'], ['big_a15.plt', 'big.ppm'], ['astro.plt', 'astro.ppm']];
  for (const [plt, ppm] of cases) for (const K of [1, 2]) {
    const r = await page.evaluate(([a, b, k]) => window.run(a, b, k, 3), [plt, ppm, K]);
    console.log(`${plt.padEnd(12)} ${r.W}x${r.H} strips=${r.strips} workers=${r.workers}  ${r.ms} ms  ${r.mbs} MB/s  badPixels=${r.badPixels}`);
  }
  await browser.close();
})();
