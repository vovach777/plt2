# color_rank.py - compression ranking on P6 (or P5) files: each codec = (name, encode cmd, decode cmd, output ext);
# sizes of the encoded files, round trip checked by comparing the pixel bytes. Codec binaries are given in a JSON-ish
# python dict file (local paths; HALIC is not distributed).  usage: python color_rank.py codecs.py dir
import os, subprocess, sys, tempfile
cfg = {}; exec(open(sys.argv[1]).read(), cfg); codecs = cfg['codecs']; d = sys.argv[2]
files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.ppm', '.pgm')))
def body(p):
    b = open(p, 'rb').read(); t = b.split(maxsplit=4); w, h = int(t[1]), int(t[2]); n = w * h * (3 if t[0] == b'P6' else 1); return w * h, b[-n:]
tmp = tempfile.mkdtemp(); res = []
for name, enc, dec in codecs:
    tot = 0; npx = 0; bad = []
    for f in files:
        px, ref = body(f); c = os.path.join(tmp, 'c.bin'); o = os.path.join(tmp, 'o' + os.path.splitext(f)[1])
        subprocess.run([a.format(i=f, c=c) for a in enc], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tot += os.path.getsize(c); npx += px
        if dec:
            if os.path.exists(o): os.remove(o)
            subprocess.run([a.format(c=c, o=o) for a in dec], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if body(o)[1] != ref: bad.append(os.path.basename(f))
    res.append((8 * tot / npx, name, bad)); print('%-44s %.4f bpp  %s' % (name, 8 * tot / npx, ('BAD: ' + ' '.join(bad)) if bad else 'ok'), flush=True)
print('---'); [print('%-44s %.4f' % (n, b)) for b, n, _ in sorted(res)]
