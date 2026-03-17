"""
tests/hunt_edge_cases_67.py
-----------------------------
Investigates two remaining edge cases:

  #6  phi diff = 0 exactly (two consecutive control points with identical
      stereographic slope from canonical P₀).  Strict monotonicity check
      (all > 0 or all < 0) classifies this as non-monotone → theta/spline
      fallback even though the arc may be geometrically fine.

  #7  Window spanning a full conic period (arc ≥ 2π on an ellipse).
      For the ellipse arc t∈[0,10] (~1.6 turns) at low n, windows are wide.
      If _is_conic_monotone still returns True, the orbit is geometrically
      valid.  If False → spline fallback regardless.

Run from repo root:
    python3 tests/hunt_edge_cases_67.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
spec  = importlib.util.spec_from_file_location('bd', os.path.join(_here, '..', 'blend_demo.py'))
bd    = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

import demo_n_transitions as dn

_label = ['?']

# ── #6: phi diff == 0 exactly ─────────────────────────────────────────────────
zero_phi_diff = []
_orig_icm = bd._is_conic_monotone

def _patched_icm(pts5_xy):
    pts5_xy = np.asarray(pts5_xy, dtype=float)
    try:
        A, B, C, D, E, F = bd.fit_conic(pts5_xy)
    except Exception:
        return _orig_icm(pts5_xy)
    xs, ys = pts5_xy[:, 0], pts5_xy[:, 1]
    L = 2*A*xs + B*ys + D;  M = B*xs + 2*C*ys + E
    Ms = np.where(np.abs(M) >= 1e-15, M, np.sign(M + 1e-300) * 1e-15)
    k0 = int(np.argmin(np.abs(L / Ms)))
    L_e, M_e = L[k0], M[k0]
    s0 = -L_e / M_e if abs(M_e) > 1e-15 else 0.0
    x0, y0 = pts5_xy[k0]
    dxi = pts5_xy[:, 0] - x0;  dyi = pts5_xy[:, 1] - y0
    mask = np.abs(dxi) >= 1e-9 * (np.abs(dyi) + 1.)
    s = np.where(mask, dyi / np.where(mask, dxi, 1.), np.sign(dyi + 1e-300) * 1e15)
    s[k0] = s0
    phi = np.unwrap(2.0 * np.arctan(s - s0))
    d = np.diff(phi)
    if np.any(d == 0.0):
        zero_phi_diff.append((_label[0], list(np.round(d, 6))))
    return _orig_icm(pts5_xy)

bd._is_conic_monotone = _patched_icm

# ── #7: wide-arc windows on ellipse ──────────────────────────────────────────
# Report the arc span (in radians on the ellipse) for each window at low n.
wide_arc = []

_orig_build = bd._build_projective_arc_window

def _patched_build(pts_2d, pts, ts, center3d, e1, e2, coeffs):
    A, B, C, D, E, F = coeffs
    det = 4*A*C - B*B
    if det > 0:   # ellipse
        # Estimate arc span from ts range relative to curve period.
        # The ellipse arc t∈[0,10] has period 2π ≈ 6.28; window ts span / period.
        span = ts[-1] - ts[0]
        # Rough period: assume uniform speed, full t_range passed via label
        if span > 1.0:   # wider than 1 radian of parameter space
            wide_arc.append((_label[0], round(span, 3)))
    return _orig_build(pts_2d, pts, ts, center3d, e1, e2, coeffs)

bd._build_projective_arc_window = _patched_build

# ── Scan ──────────────────────────────────────────────────────────────────────
for entry in dn.DEMO_CURVES:
    name, xy_func, t_range = entry[0], entry[1], entry[2]
    for n in range(8, 51):
        _label[0] = f'{name} n={n}'
        pts, times = bd.sample_curve(xy_func, t_range, n)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bd._run_blend(pts, times, False, 2)

bd._is_conic_monotone           = _orig_icm
bd._build_projective_arc_window = _orig_build

# ── Report ────────────────────────────────────────────────────────────────────
print('── #6: phi diff == 0 exactly ────────────────────────────────────────────')
if zero_phi_diff:
    print(f'  Found {len(zero_phi_diff)} case(s):')
    for label, d in zero_phi_diff[:10]:
        print(f'    {label}  diffs={d}')
else:
    print('  None found across n=8..50.')

print()
print('── #7: wide-arc ellipse windows (ts span > 1.0) ─────────────────────────')
if wide_arc:
    # Group by curve, show max span
    from collections import defaultdict
    by_curve = defaultdict(list)
    for label, span in wide_arc:
        by_curve[label].append(span)
    for label, spans in sorted(by_curve.items())[:15]:
        print(f'  {label}: max span={max(spans):.3f}')
    print(f'  ({len(wide_arc)} total wide-arc windows)')
else:
    print('  No wide-arc ellipse windows found (ts span > 1.0).')
