# gray_vs_halic.py - whole-file comparison on 8-bit P5 files: planes_codec variants vs HALIC binaries (local only,
# HALIC is not distributed with plt2). Times include process start and file I/O for every codec (best of R).
# usage: python gray_vs_halic.py <planes_codec> <halic dir> dir|a.pgm ... [-r N]
import os, subprocess, sys, tempfile, time
args = sys.argv[1:]; pln, hdir = args.pop(0), args.pop(0); reps = 3; files = []
while args:
    a = args.pop(0)
    if a == '-r': reps = int(args.pop(0))
    elif os.path.isdir(a): files += sorted(os.path.join(a, f) for f in os.listdir(a) if f.endswith('.pgm'))
    else: files.append(a)
H = lambda n: os.path.join(hdir, n)
codecs = {
    'HALIC': ([H('HALIC_ENCODE_0.7.2_ST_x64_avx2_Linux_Static')], [H('HALIC_DECODE_0.7.2_ST_x64_avx2_Linux_Static')]),
    'HALIC FAST': ([H('HALIC_ENCODE_FAST_0.7.2_ST_x64_avx2_Linux_Static')], [H('HALIC_DECODE_FAST_0.7.2_ST_x64_avx2_Linux_Static')]),
    'PLN m1 off1': ([pln, 'c'], [pln, 'd'], ['-m', '1', '-off', '1']),
    'PLN m1 off2': ([pln, 'c'], [pln, 'd'], ['-m', '1', '-off', '2']),
    'PLN m0 (FSE z)': ([pln, 'c'], [pln, 'd'], ['-m', '0']),
}
tmp = tempfile.mkdtemp(); c, o = os.path.join(tmp, 'c.bin'), os.path.join(tmp, 'o.pgm')
def body(p): b = open(p, 'rb').read(); t = b.split(maxsplit=3); return t[1], t[2], t[3][1:] if t[3][:1].isspace() else t[3]
tot = {k: [0, 0.0, 0.0, True] for k in codecs}; npx = 0
print('%-14s' % 'file' + ''.join('%22s' % k for k in codecs))
for f in files:
    w, h, _ = body(f); px = int(w) * int(h); npx += px; line = '%-14s' % os.path.basename(f)[:14]
    for k, v in codecs.items():
        ce = v[0] + [f, c] + (v[2] if len(v) > 2 else []); de = v[1] + [c, o]; te = td = 1e9
        for _ in range(reps):
            t0 = time.perf_counter(); subprocess.run(ce, check=True, stdout=subprocess.DEVNULL); t1 = time.perf_counter()
            subprocess.run(de, check=True, stdout=subprocess.DEVNULL); t2 = time.perf_counter(); te = min(te, t1 - t0); td = min(td, t2 - t1)
        sz = os.path.getsize(c); ok = body(o)[2] == body(f)[2]
        T = tot[k]; T[0] += sz; T[1] += te; T[2] += td; T[3] &= ok; line += '%15.4f %s' % (8 * sz / px, 'ok    ' if ok else 'BAD   ')
    print(line)
print('TOTAL %d files %.1f Mpx (bpp; MB/s incl. process start + I/O, 1 thread)' % (len(files), npx / 1e6))
b0 = 8 * tot['HALIC'][0] / npx
for k, (s, te, td, ok) in tot.items():
    print('  %-16s %7.4f bpp (%+6.2f%% vs HALIC)  enc %6.1f  dec %6.1f MB/s  %s' % (k, 8 * s / npx, 100 * (8 * s / npx / b0 - 1), npx / te / 1e6, npx / td / 1e6, 'all ok' if ok else 'MISMATCH'))
