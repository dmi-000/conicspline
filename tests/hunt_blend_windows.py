"""
tests/hunt_blend_windows.py
-----------------------------
Scans all DEMO_CURVES (plus the bump-circle extra) over n=8..80, looking for
any window whose method is 'blend' (conic+spline disagreement weighted mix).

'blend' windows are listed in the architecture as "rarely used" and have never
been visually confirmed.  This script either finds one or conclusively rules
them out for the standard curve set.

Run from repo root:
    python3 tests/hunt_blend_windows.py

Exit 0 in both cases; prints a summary either way.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
spec  = importlib.util.spec_from_file_location('bd', os.path.join(_here, '..', 'blend_demo.py'))
bd    = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

import demo_n_transitions as dn

# ── Curve list ────────────────────────────────────────────────────────────────
extra = [
    ('Bump circle',
     lambda t: (np.cos(t) + 0.3 * np.exp(-((t - 1.5)**2) / 0.05), np.sin(t)),
     (0.0, 2 * np.pi)),
]
all_curves = [(e[0], e[1], e[2]) for e in dn.DEMO_CURVES] + extra

# ── Scan ──────────────────────────────────────────────────────────────────────
found_blend = []

for name, xy_func, t_range in all_curves:
    for n in range(8, 81):
        pts, times = bd.sample_curve(xy_func, t_range, n)
        _, _, _, wins, ms, me, methods = bd._run_blend(pts, times, False, 2)
        blend_idx = [i for i, m in enumerate(methods) if m == 'blend']
        if blend_idx:
            found_blend.append((name, n, blend_idx, methods))

# ── Report ────────────────────────────────────────────────────────────────────
METHOD_CHAR = {'conic': 'C', 'spline': 's', 'blend': 'B'}

if found_blend:
    print(f'Found blend windows in {len(found_blend)} (curve, n) cases:\n')
    for name, n, idxs, methods in found_blend[:40]:
        abbrev = ''.join(METHOD_CHAR.get(m, '?') for m in methods)
        print(f'  {name} n={n}: [{abbrev}]  blend@{idxs}')
    if len(found_blend) > 40:
        print(f'  … and {len(found_blend) - 40} more')
else:
    print('No blend windows found in n=8..80 for any curve.')
    print('The blend path in _blended_conic_spline is not triggered by standard inputs.')
