"""
tests/inspect_blend_windows.py
--------------------------------
Prints the methods arrays for Kepler+drift and 5-petal flower at n-values
where 'blend' windows appear, to help select good demo panels.

Run from repo root:
    python3 tests/inspect_blend_windows.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import importlib.util
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
spec  = importlib.util.spec_from_file_location('bd', os.path.join(_here, '..', 'blend_demo.py'))
bd    = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

import demo_n_transitions as dn

METHOD_CHAR = {'conic': 'C', 'spline': 's', 'blend': 'B'}


def scan_blend(name, xy_func, t_range, n_values):
    print(f'{name}:')
    for n in n_values:
        pts, times = bd.sample_curve(xy_func, t_range, n)
        _, _, _, wins, ms, me, methods = bd._run_blend(pts, times, False, 2)
        abbrev = ''.join(METHOD_CHAR.get(m, '?') for m in methods)
        nb = sum(m == 'blend' for m in methods)
        if nb:
            print(f'  n={n}: [{abbrev}]')
    print()


# ── Kepler + drift ────────────────────────────────────────────────────────────
kepler = next(e for e in dn.DEMO_CURVES if e[0] == 'Kepler + drift')
scan_blend(kepler[0], kepler[1], kepler[2], range(8, 30))

# ── 5-petal flower ────────────────────────────────────────────────────────────
petal = next(e for e in dn.DEMO_CURVES if e[0] == '5-petal flower')
scan_blend(petal[0], petal[1], petal[2], [35, 38, 40, 45, 50])
