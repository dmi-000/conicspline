"""
tests/check_find_transitions_stability.py
------------------------------------------
Verifies that find_transitions correctly identifies STABLE all-conic n-values
(two consecutive n both all-conic) rather than transient ones.

The bump curve (circle + narrow Gaussian) is the key regression case:
  n=10,11,12 are all-conic, n=13 has splines → stable all-conic = 10 (not 12).

Run from repo root:
    python3 tests/check_find_transitions_stability.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import demo_n_transitions as dn

# ── Bump circle: the regression case ─────────────────────────────────────────
xy_bump = lambda t: (np.cos(t) + 0.3 * np.exp(-((t - 1.5)**2) / 0.05), np.sin(t))
panels = dn.find_transitions(xy_bump, (0.0, 2 * np.pi), n_min=8, n_max=40)
print('Bump curve:', panels)
print()

# ── All DEMO_CURVES ───────────────────────────────────────────────────────────
for entry in dn.DEMO_CURVES:
    name, xy_func, t_range = entry[0], entry[1], entry[2]
    panels = dn.find_transitions(xy_func, t_range, n_min=8, n_max=120)
    print(f'{name:35s}: {[(n, l) for n, l in panels]}')
