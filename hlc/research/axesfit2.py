import numpy as np, pickle, sys
exec(open('axesfit.py').read().split('N0 = {')[0])
L1, L2 = np.meshgrid(np.arange(10), np.arange(-9, 10), indexing='ij')
M = np.maximum(L1, np.abs(L2)); SM = L1 + np.abs(L2)
px = Hc[0].sum()                                        # Y samples = pixels
def bpp(cy, cu): return (cost_map(0, cy) + cost_map(1, cu)) / px
y6 = np.where(M < 4, M, np.where(M < 6, 4, 5))         # oracle-6 rings of Y: 0,1,2,3,{4,5},{6+}
cands = {
  'oracle 6 + oracle 6': None,
  'Y max rings(6)  | UV l1+|l2| cap 5 (6)': (y6, np.minimum(SM, 5)),
  'Y max rings(6)  | UV l1+|l2| cap 7 (8)': (y6, np.minimum(SM, 7)),
  'Y max cap 8 (9) | UV l1+|l2| cap 8 (9)': (np.minimum(M, 8), np.minimum(SM, 8)),
  'Y max rings(6)  | UV sum cap5 + same-dir diag': (y6, np.minimum(SM, 5) + 6 * ((L2 > 0) & (L1 > 0) & (SM <= 4))),
}
oy, _ = oracle(0, 6); ou, _ = oracle(1, 6); print(f'{"oracle 6 + oracle 6":46s} {(oy + ou) / px:.4f} bpp (12 buckets)')
for k, v in list(cands.items())[1:]:
    cy, cu = v; nb = len(np.unique(cy[Hc[0].sum(-1) > 0])) + len(np.unique(cu[Hc[1].sum(-1) > 0]))
    print(f'{k:46s} {bpp(cy, cu):.4f} bpp ({nb} buckets)')
