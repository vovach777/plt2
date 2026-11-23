# plane_frac2.py - PLS-probe-2 per-plane share (Y by zY zeros, S by zS zeros, 65% each), sidecar-based.
import sys, os, subprocess, tempfile, struct, numpy as np

def frac(d):
    fs = sorted(f for f in (os.path.join(d, x) for x in os.listdir(d)) if f.endswith('.ppm'))
    tiles = 0; my = 0; ms = 0
    tmp = tempfile.mkdtemp()
    for img in fs:
        log = os.path.join(tmp, 'a.probe')
        subprocess.run(['/tmp/r8f5/pls_probe2', 'c', img, os.path.join(tmp, 'a.pls'), '-s=probe2'],
                       env=dict(os.environ, PLS_PROBE_LOG=log), check=True, stdout=subprocess.DEVNULL)
        raw = open(log, 'rb').read(); nt = struct.unpack('<I', raw[:4])[0]
        modes = np.frombuffer(raw, np.uint8, nt * 2, 4).reshape(nt, 2)
        tiles += nt; my += int((modes[:, 0] == 2).sum()); ms += int((modes[:, 1] == 2).sum())
    name = os.path.basename(os.path.normpath(d))
    print('%-14s imgs=%3d tiles=%5d | Y PLS %6d/%6d = %6.2f%% | S PLS %6d/%6d = %6.2f%%' %
          (name, len(fs), tiles, my, tiles, 100.0 * my / max(1, tiles),
           ms, tiles, 100.0 * ms / max(1, tiles)), flush=True)

for d in sys.argv[1:]:
    frac(d)
