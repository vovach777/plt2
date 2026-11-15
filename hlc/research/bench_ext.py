#!/usr/bin/env python3
"""A/B of external lossless codecs vs PLR / PLS by process wall time (single thread, includes file I/O).

usage: bench_ext.py OUTCSV file.ppm ...   (codec binaries from env: PLR, FNBLI, HALIC_DIR)
Every round trip is verified byte-exact against the input pixels.
"""
import csv, os, subprocess, sys, tempfile, time

PLR = os.environ.get('PLR', '/tmp/pfmt/p2/palic-2r')
FNBLI = os.environ.get('FNBLI', '/tmp/pfmt/fnbli')
HD = os.environ.get('HALIC_DIR', '/tmp/pfmt/halic')

CODECS = {
    'plr':   (lambda i, o: [PLR, 'c', i, o],              lambda i, o: [PLR, 'd', i, o]),
    'pls':   (lambda i, o: [PLR, 'c', i, o, '-s'],        lambda i, o: [PLR, 'd', i, o]),
    'fnbli': (lambda i, o: [FNBLI, '-f', i, '-o', o],     lambda i, o: [FNBLI, '-f', i, '-o', o]),
    'halic': (lambda i, o: [f'{HD}/HALIC_ENCODE_0.7.2_ST_x64_avx2_Linux_Static', i, o],
              lambda i, o: [f'{HD}/HALIC_DECODE_0.7.2_ST_x64_avx2_Linux_Static', i, o]),
}
EXT = {'plr': '.plr', 'pls': '.pls', 'fnbli': '.fnbli', 'halic': '.halic'}


def pixels(path):
    d = open(path, 'rb').read()
    # P6/P5 header: magic, w, h, maxval, then one whitespace
    tok, i = [], 0
    while len(tok) < 4:
        while d[i:i + 1].isspace(): i += 1
        if d[i:i + 1] == b'#':
            while d[i:i + 1] != b'\n': i += 1
            continue
        j = i
        while not d[j:j + 1].isspace(): j += 1
        tok.append(d[i:j]); i = j
    return tok[0], int(tok[1]), int(tok[2]), d[i + 1:]


def run(cmd):
    t = time.perf_counter()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode: raise RuntimeError(f'{cmd}: {r.stderr[-300:]}')
    return time.perf_counter() - t


def main():
    out, files = sys.argv[1], sys.argv[2:]
    codecs = os.environ.get('CODECS', 'plr,pls,fnbli,halic').split(',')
    tmp = tempfile.mkdtemp(dir='/tmp/pfmt')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'px', 'raw', 'codec', 'bytes', 'enc_s', 'dec_s', 'ok'])
        for fn in files:
            m, W, H, px = pixels(fn)
            for c in codecs:
                enc, dec = CODECS[c]
                z, back = f'{tmp}/x{EXT[c]}', f'{tmp}/y.ppm'
                for p in (z, back):
                    if os.path.exists(p): os.remove(p)
                te = run(enc(fn, z)); td = run(dec(z, back))
                ok = pixels(back)[3] == px
                w.writerow([os.path.basename(fn), W * H, len(px), c, os.path.getsize(z), f'{te:.4f}', f'{td:.4f}', int(ok)])
                f.flush()


if __name__ == '__main__':
    main()
