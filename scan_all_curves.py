"""
Summary scan: for each of the 8 test curves plus extra analytic conics, find
budget n and report max_dev plus method-window counts.  Asserts n matches
known baselines.

Usage:
    python3 scan_all_curves.py
"""

import numpy as np
import importlib.util, os
from collections import Counter

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('bd', os.path.join(_here, 'blend_demo.py'))
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)

# Hyperbola (1 period): x=sec(t), y=tan(t) on one period ±0.5 margin around
# the asymptote.  The ground truth IS the conic → direct regression for the
# projective-arc (phi-branch) path with genuine cross-branch windows.
_ONE_PERIOD = (-np.pi / 2 + 0.5, 3 * np.pi / 2 - 0.5)
_EXTRA_CURVES = [
    ('Hyperbola (1 period)',
     lambda t: (1.0 / np.cos(t), np.tan(t)),
     _ONE_PERIOD),
]

# Baseline n values (adaptive_n_budget at max_dev_target=0.018, N_order=2).
# Metric: max(||blended − truth||) * Menger_kappa (curvature-scaled, ≈ radians angular error).
# Threshold 0.018 ≈ δ·κ at old Euclidean-0.01 baseline for Random spline (n=28, δ·κ=0.01777).
# Update after any intentional change to conicspline or the CURVES list.
#
# Van der Pol uses speed-adaptive sampling (sample_curve_speed, alpha=1.0):
#   density ∝ phase-space speed + 1 baseline → 57 points vs 128 uniform (55% reduction).
#   Uniform baseline is n=128; speed-adaptive achieves the same max_dev target with n=57.
BASELINES = {
    'Logarithmic spiral':  7,
    'Lissajous (3:2)':    38,
    'Rose curve (k=3)':   18,
    'Damped oscillation':  7,
    '5-petal flower':     28,
    'Kepler + drift':      7,
    'Random spline':      21,
    'Van der Pol (mu=3)': 57,
    'Hyperbola (1 period)':  8,
}

# Per-curve sampler overrides: use speed-adaptive for Van der Pol.
_CURVE_SAMPLERS = {
    'Van der Pol (mu=3)': bd.sample_curve_speed,
}

print(f"{'Curve':26s}  {'n':>4}  {'max_dev':>10}  {'baseline':>8}  methods")
print("-" * 80)
failures = []
for title, xy_func, t_range in list(bd.CURVES) + _EXTRA_CURVES:
    sampler = _CURVE_SAMPLERS.get(title)
    n = bd.adaptive_n_budget(xy_func, t_range, sampler=sampler)
    pts, times = (sampler or bd.sample_curve)(xy_func, t_range, n)
    _, _, _, wins, ms, me, methods = bd._run_blend(pts, times, False, 2)
    wj, md, _ = bd._find_worst_interval(
        pts, times, wins, ms, me, xy_func, t_range, 2)
    mc = Counter(methods)
    base = BASELINES.get(title, '?')
    flag = '' if n == base else '  *** REGRESSION ***' if isinstance(base, int) and n > base else '  *** IMPROVED ***'
    print(f"{title[:26]:26s}  {n:4d}  {md:.4e}  {str(base):>8}{flag}  {dict(mc)}")
    if isinstance(base, int) and n != base:
        failures.append(f"{title}: expected n={base}, got n={n}")

if failures:
    print()
    print("ASSERTION FAILURES:")
    for f in failures:
        print(f"  {f}")
    raise AssertionError(f"{len(failures)} baseline(s) changed")
