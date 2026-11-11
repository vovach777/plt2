# time_rank.py - wall clock of whole processes (encode, decode) per codec over a folder, best of R runs per file,
# codecs from the same codecs.py as color_rank.py. MB/s = raw pixel bytes / summed seconds / 1e6.
# usage: python time_rank.py codecs.py dir [-r 3] [-only name1,name2]
import os, subprocess, sys, tempfile, time
cfg = {}; exec(open(sys.argv[1]).read(), cfg); d = sys.argv[2]; R = 3; only = None; a = sys.argv[3:]
while a:
    k = a.pop(0)
    if k == '-r': R = int(a.pop(0))
    elif k == '-only': only = a.pop(0).split(',')
files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.ppm', '.pgm')))
def raw(p): t = open(p, 'rb').read(40).split(); return int(t[1]) * int(t[2]) * (3 if t[0] == b'P6' else 1)
N = sum(raw(f) for f in files); tmp = tempfile.mkdtemp()
for name, enc, dec in cfg['codecs']:
    if only and not any(o in name for o in only): continue
    te = td = 0.0
    for f in files:
        c = os.path.join(tmp, 'c.bin'); o = os.path.join(tmp, 'o' + os.path.splitext(f)[1]); be = bd = 1e9
        for _ in range(R):
            t0 = time.perf_counter(); subprocess.run([x.format(i=f, c=c) for x in enc], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            t1 = time.perf_counter(); subprocess.run([x.format(c=c, o=o) for x in dec], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            t2 = time.perf_counter(); be = min(be, t1 - t0); bd = min(bd, t2 - t1)
        te += be; td += bd
    print('%-40s enc %6.3f s %6.1f MB/s   dec %6.3f s %6.1f MB/s' % (name, te, N / te / 1e6, td, N / td / 1e6), flush=True)
