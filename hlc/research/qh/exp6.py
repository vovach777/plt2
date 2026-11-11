# exp6.py - (1) domain mismatch of static tables: held-out vs self-trained QH; (2) synthetic stress images for plr vs plr_qh
import numpy as np, sys, os, subprocess
from lib import *
from train_tables import *
if sys.argv[1] == 'mismatch':
    for te, tr in [('kodak768', 'col19'), ('kodak768', 'kodak768'), ('col19', 'kodak768'), ('col19', 'col19'), ('gray', 'kodak768'), ('gray', 'gray'), ('uv_col19', 'uv_kodak768'), ('uv_col19', 'uv_col19')]:
        T = train(tr, 32, 11); lb, xb, d = cost(T, te); n = d['npx']; r = d['bits'].sum()
        print(f'test {te:12s} tables from {tr:12s}: {(lb + xb) / n:.4f} bpp ({100 * ((lb + xb) / r - 1):+.2f}% of the RLGR part), file {(d["fbits"] + lb + xb - r) / n:.4f}' + ('   <- self-trained, not held-out' if te == tr else ''))
else:
    rng = np.random.default_rng(1); W, H = 1024, 768; out = '/tmp/fable_rlgr/synth'; os.makedirs(out, exist_ok=True)
    yy, xx = np.mgrid[0:H, 0:W]
    imgs = {
      'flat': np.full((H, W), 128),
      'gradient': (xx * 255 // W + yy * 64 // H) % 256,
      'noise_uniform': rng.integers(0, 256, (H, W)),
      'noise_sigma2': np.clip(128 + rng.normal(0, 2, (H, W)), 0, 255),
      'noise_sigma20': np.clip(128 + rng.normal(0, 20, (H, W)), 0, 255),
      'flat_rects': None, 'sparse_spikes': np.where(rng.random((H, W)) < 0.002, 255, 0),
      'stripes_1px': (xx % 2) * 255,
    }
    r = np.full((H, W), 240)
    for _ in range(300):
        x0, y0 = rng.integers(0, W - 40), rng.integers(0, H - 20); r[y0:y0 + rng.integers(2, 60), x0:x0 + rng.integers(2, 200)] = rng.integers(0, 256)
    imgs['flat_rects'] = r
    print('%-16s %10s %10s   (bpp; QH tables trained on Kodak)' % ('synthetic gray', 'PLR', 'PLR+QH'))
    for nm, a in imgs.items():
        fn = f'{out}/{nm}.pgm'; open(fn, 'wb').write(b'P5\n%d %d\n255\n' % (W, H) + np.asarray(a, np.uint8).tobytes())
        res = []
        for exe, env in (('plr', {}), ('plr_qh', {'QH_TABLES': '/tmp/fable_rlgr/tables/qh_kodak_K32_L11.bin'})):
            o = subprocess.run([f'/tmp/fable_rlgr/palic-2/{exe}', 'bench', fn, '-r', '1'], capture_output=True, text=True, env={**os.environ, **env}).stdout.strip().split('\n')[-1]
            res.append(o.split('bpp')[0].split()[-1] + (' ok' if 'all ok' in o else ' FAIL'))
        print('%-16s %10s %10s' % (nm, res[0], res[1]))
