# plane_frac.py - PLS plane share under PLS-probe-1 rules, from the instrumented encoder sidecar.
import sys, os, subprocess, tempfile, struct, numpy as np

def frac(d):
    fs = sorted(f for f in (os.path.join(d, x) for x in os.listdir(d)) if f.endswith('.ppm'))
    tiles = 0; planes = 0; z = np.zeros((4,), np.int64)
    tmp = tempfile.mkdtemp()
    for img in fs:
        log = os.path.join(tmp, 'a.probe')
        subprocess.run(['/tmp/r8f5/pls_probe', 'c', img, os.path.join(tmp, 'a.pls'), '-s=zthr'],
                       env=dict(os.environ, PLS_PROBE_LOG=log), check=True, stdout=subprocess.DEVNULL)
        raw = open(log, 'rb').read(); nt = struct.unpack('<I', raw[:4])[0]
        modes = np.frombuffer(raw, np.uint8, nt * 2, 4).reshape(nt, 2)
        zz = np.frombuffer(raw, '<u4', nt * 5, 4 + nt * 2).reshape(nt, 5)
        tiles += nt; planes += 2 * nt
        z[0] += (zz[:, 0] * 100 >= 65 * zz[:, 4]).sum()   # zthrc: Y by zY zeros
        z[1] += (zz[:, 3] * 100 >= 65 * zz[:, 4]).sum()   # zthrc: S by zS zeros
        z[2] += (modes[:, 0] == 2).sum()                  # zthr: both planes of tile
        z[3] += (modes[:, 1] == 2).sum()
    name = os.path.basename(os.path.normpath(d))
    print('%-14s imgs=%3d tiles=%5d | zthr planes %6d/%6d = %6.2f%% | zthrc Y %6.2f%% S %6.2f%%' %
          (name, len(fs), tiles, 2 * z[2], planes, 100.0 * 2 * z[2] / max(1, planes),
           100.0 * z[0] / max(1, tiles), 100.0 * z[1] / max(1, tiles)), flush=True)

for d in sys.argv[1:]:
    frac(d)
