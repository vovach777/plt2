# ffv1_sweep.py - FFV1 sizes (raw packets, no container) for gray P5 files over context model (0 small, 1 large),
# coder (range_def, range_tab) and slices (1 = level 1, else level 3). Prints bpp per config; checks the best
# config of each context for round trip.   usage: python ffv1_sweep.py dir [dir ...]
import os, subprocess, sys, tempfile, itertools
tmp = tempfile.mkdtemp(); out = os.path.join(tmp, 'o.bin')
def enc(f, ctx, coder, sl, dst=out, fmt='rawvideo'):
    lv = ['-level', '1'] if sl == 1 else ['-level', '3', '-slices', str(sl), '-slicecrc', '0']
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f, '-c:v', 'ffv1', '-coder', coder, '-context', str(ctx), '-g', '1'] + lv + ['-f', fmt, dst], check=True)
    return os.path.getsize(dst)
def px(f): t = open(f, 'rb').read(40).split(); return int(t[1]) * int(t[2])
for d in sys.argv[1:]:
    files = sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith('.pgm')); N = sum(px(f) for f in files); res = {}
    for ctx, coder, sl in itertools.product((0, 1), ('range_def', 'range_tab'), (1, 4, 6, 9, 12, 16)):
        res[(ctx, coder, sl)] = 8 * sum(enc(f, ctx, coder, sl) for f in files) / N
    print('== %s (%d files, %.1f Mpx)' % (d, len(files), N / 1e6))
    for k, v in sorted(res.items(), key=lambda kv: kv[1]): print('  context %d %-9s slices %2d: %.4f bpp' % (k + (v,)))
    for ctx in (0, 1):
        k = min((k for k in res if k[0] == ctx), key=res.get); ok = True
        for f in files:
            nut = os.path.join(tmp, 'o.nut'); enc(f, *k, dst=nut, fmt='nut'); dec = os.path.join(tmp, 'd.pgm')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', nut, '-f', 'image2', '-c:v', 'pgm', '-pix_fmt', 'gray', dec], check=True)
            a = open(f, 'rb').read(); b = open(dec, 'rb').read(); ok &= a[-px(f):] == b[-px(f):]
        print('  best context %d: %s -> round trip %s' % (ctx, k, 'ok' if ok else 'FAIL'))
