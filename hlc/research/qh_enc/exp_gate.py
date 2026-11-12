# exp_gate.py - Qwen's "free" zero-run gate for PLR: try a run of zero blocks when sa == 0 (format) OR the previous
# block is empty (decoder knows both). Variants: A = format (sa == 0), B = sa == 0 || left empty, C = B but not right
# after a run (Argon's after_run in PLS: the block after a maximal run is never empty). Coded blocks cost as in the
# format (letters + tails + 16 L at the causal L, format tables); runs EG0(n); everything else unchanged.
# Check: A must reproduce the dump's run bits. usage: exp_gate.py DUMPER NAME img...
import sys, os, numpy as np
sys.argv, _a = sys.argv[:1] + ['x', 'y'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_pred.py')).read().split("dm, dw = sys.argv[1], sys.argv[2]")[0])
dumper, name, imgs = _a[1], _a[2], _a[3:]
eg0 = lambda n: 2 * int(np.floor(np.log2(n + 1))) + 1
S = dict(tot=0.0, A=0.0, B=0.0, C=0.0, npx=0, chkA=0.0, chkR=0.0)
for img in imgs:
    d, c, parts = costs(dumper, img); z = d['z']; lb, tb, pb = qhsim.block_bits(d, T); bc = lb + tb + pb
    empty = (z == 0).all(1); sa = d['sa']; pk = d['tile'].astype(np.int64) * 4 + d['pl']; first = np.r_[True, pk[1:] != pk[:-1]]
    n = len(z); res = {}
    for v in 'ABC':
        bits = 0.0; i = 0; after = False
        while i < n:
            if first[i]: after = False
            trig = (not first[i]) and (sa[i] == 0 or (v != 'A' and empty[i - 1] and not (v == 'C' and after)))
            if trig:
                m = 0
                while i + m < n and empty[i + m] and (m == 0 or not first[i + m]): m += 1
                bits += eg0(m)
                if m: i += m; after = True; continue
            bits += bc[i]; after = False; i += 1
        res[v] = bits
    rest = c.sum() - (parts['runs'] + (bc * ~d['skip']).sum())
    S['chkA'] += res['A']; S['chkR'] += parts['runs'] + (bc * ~d['skip']).sum()
    for v in 'ABC': S[v] += res[v] + rest
    S['tot'] += c.sum(); S['npx'] += d['npx']
n = S['npx']
print(f"{name}: format {S['A'] / n:.4f} bpp (check vs dump {100 * (S['chkA'] / S['chkR'] - 1):+.3f}%) | B sa==0||left empty {S['B'] / n:.4f} ({100 * (S['B'] / S['A'] - 1):+.2f}%) | C = B, not after a run {S['C'] / n:.4f} ({100 * (S['C'] / S['A'] - 1):+.2f}%)", flush=True)
