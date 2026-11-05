import numpy as np, pickle, sys
Hc = pickle.load(open(sys.argv[1], 'rb'))            # [type][l1 0..9][l2+9][512]
def ent(h):
    h = h[h > 0].astype(np.float64); return float(-(h * np.log2(h / h.sum()))).sum() if h.size else 0.0
def ent(h):
    h = h[h > 0].astype(np.float64); return float(-(h * np.log2(h / h.sum())).sum()) if h.size else 0.0
L1, L2 = np.meshgrid(np.arange(10), np.arange(-9, 10), indexing='ij')
def cost_map(t, cls):                                   # cls: [10,19] -> class id
    return sum(ent(Hc[t][cls == k].sum(0)) for k in np.unique(cls[Hc[t].sum(-1) > 0]))
def oracle(t, N):                                       # greedy agglomerative merge of cells
    cells = [(i, j) for i in range(10) for j in range(19) if Hc[t][i, j].sum() > 0]
    H = [Hc[t][i, j].astype(np.int64) for i, j in cells]; E = [ent(h) for h in H]; mem = [[c] for c in cells]
    while len(H) > N:
        best = None
        for a in range(len(H)):
            for b in range(a + 1, len(H)):
                d = ent(H[a] + H[b]) - E[a] - E[b]
                if best is None or d < best[0]: best = (d, a, b)
        _, a, b = best; H[a] = H[a] + H[b]; E[a] = ent(H[a]); mem[a] += mem[b]; del H[b], E[b], mem[b]
    return sum(E), mem
N0 = {t: Hc[t].sum() for t in (0, 1)}
full = {t: cost_map(t, L1 * 19 + L2 + 9) for t in (0, 1)}
M = np.maximum(L1, np.abs(L2)); SM = L1 + np.abs(L2)
forms = {
  'max(l1,|l2|)': M,
  'max, cap 6': np.minimum(M, 6),
  'l1+|l2|': SM,
  'l1+|l2| cap 8': np.minimum(SM, 8),
  '(l1+|l2|+max+1)>>1': (SM + M + 1) >> 1,
  'max x shape(l2<0)': M * 2 + (L2 < 0),
  'max cap6 x shape': np.minimum(M, 6) * 2 + (L2 < 0),
}
for t, nm in ((0, 'Y'), (1, 'UV')):
    n = N0[t]; print(f'== {nm}: all 190 cells {full[t] / n:.4f} bits/sample')
    for N in (6, 8, 12):
        c, mem = oracle(t, N); print(f'  oracle {N:2d} buckets: {c / n:.4f} ({100 * c / full[t] - 100:+.2f}% vs all cells)')
        if N == 6:
            for g in sorted(mem, key=lambda m: min(max(i, abs(j - 9)) for i, j in m)):
                print('     bucket: ' + ' '.join(f'({i},{j - 9:+d})' for i, j in sorted(g))[:300])
    for k, f in forms.items():
        c = cost_map(t, f); print(f'  formula {k:22s} {len(np.unique(f[Hc[t].sum(-1) > 0])):2d} buckets: {c / n:.4f} ({100 * c / full[t] - 100:+.2f}%)')
