"""
test_cross_guard.py — three cases verifying the pure-geometric window selection
architecture (all quality heuristics removed from _try_conic / _build_projective_arc_window).

Architecture (2026-03-07): the only gates for using a projective arc are:
  1. _is_conic_monotone — arc must trace control points in correct parameter order
  2. NaN in orbit_func output drives adaptive_n_budget to increase n until the
     asymptote falls outside the blend region
No cross.sum guard, no NaN-in-blend early exit, no position-disagreement check.

Window extraction mirrors diag_xbranch.py logic so results are reproducible.
"""
import importlib.util, os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('bd', os.path.join(_here, 'blend_demo.py'))
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

CURVE_MAP = {n: (f, r) for n, f, r in bd.CURVES}

# ─────────────────────────────────────────────────────────────────────────────
def extract_window(curve_name, n, win_idx):
    """Return (pts_2d, p5, t5, center, e1, e2, coeffs) for window win_idx."""
    xy_func, t_range = CURVE_MAP[curve_name]
    pts, times = bd.sample_curve(xy_func, t_range, n)
    p5 = pts[win_idx : win_idx + 5]
    t5 = times[win_idx : win_idx + 5]
    center = p5.mean(axis=0)
    centered = p5 - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    if (centered[-1] - centered[0]) @ e1 < 0:
        e1 = -e1
    pts_2d = np.column_stack([centered @ e1, centered @ e2])
    coeffs = bd.fit_conic(pts_2d)
    return pts_2d, p5, t5, center, e1, e2, coeffs


def cross_sum_of(pts_2d, coeffs):
    """Recompute cross.sum directly from the conic, matching _build_projective_arc_window."""
    A, B, C, D, E, F = coeffs
    det = 4.0 * A * C - B * B
    if abs(det) < 1e-10:
        return None  # parabola
    cx = (-2.0 * C * D + B * E) / det
    cy = (-2.0 * A * E + B * D) / det
    f0 = A*cx**2 + B*cx*cy + C*cy**2 + D*cx + E*cy + F
    if abs(f0) < 1e-15:
        return None  # degenerate
    dx = pts_2d[:, 0] - cx
    dy = pts_2d[:, 1] - cy
    M = np.array([[A, B / 2.0], [B / 2.0, C]])
    eigvals, eigvecs = np.linalg.eigh(M)
    if not (eigvals[0] * eigvals[1] < 0):
        return 0  # ellipse/parabola: no branches
    pos_idx = 1 if eigvals[1] > 0 else 0
    e_t = eigvecs[:, pos_idx]
    proj = dx * e_t[0] + dy * e_t[1]
    if np.any(proj == 0):
        return None
    branch = np.sign(proj)
    cross = (branch[:4] * branch[1:]) < 0
    return int(np.sum(cross))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Normal case
# Damped oscillation win[1], n=40: cross.sum=1, rel≈0.002, NaN-free.
# Expectation (BEFORE and AFTER changes): _try_conic returns 'conic'.
# This verifies the guard does not break genuine single-crossing windows.
#
# Note (2026-03-07): with the stereographic parameterisation, Q(s) has no
# sign change for this window (both Q roots lie at positive s, all control-
# point slopes are negative).  Case 3 detects this as "no actual asymptote
# crossing" and builds phi = arctan(s − s₀) directly — orbit_func is NaN-free
# and accurate.  _try_conic uses the projective arc and returns 'conic'.
# ─────────────────────────────────────────────────────────────────────────────
def test_normal_single_crossing():
    pts_2d, p5, t5, center, e1, e2, coeffs = extract_window('Damped oscillation', 40, 1)
    cs = cross_sum_of(pts_2d, coeffs)
    assert cs == 1, f"Expected cross.sum=1, got {cs}"

    proj = bd._build_projective_arc_window(pts_2d, p5, t5, center, e1, e2, coeffs)
    assert proj is not None, "_build_projective_arc_window returned None for cross.sum=1 window"
    _, _, is_cross = proj
    assert is_cross, "Expected is_cross_branch=True"

    # NaN check — Case 3 (no asymptote) must be NaN-free in the blend region
    t_check = np.linspace(t5[1], t5[3], 50)
    check = proj[0](t_check)
    assert np.all(np.isfinite(check)), "NaN in blend region for clean single-crossing window"

    result = bd._try_conic(pts_2d, p5, t5, center, e1, e2, use_spline=True)
    assert result is not None, "_try_conic returned None"
    _, _, method = result
    assert method == 'conic', f"Expected method='conic', got {method!r}"
    print(f"  PASS  test_normal_single_crossing  (cross.sum={cs}, method={method!r})")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Multi-alternation cross-branch window
# Damped oscillation win[11], n=40: cross.sum=3, NaN-free.
# Under the pure-geometric architecture: cross.sum=3 is a valid conic
# (passes through ∞ three times) — _try_conic returns 'conic'.
# adaptive_n_budget handles quality: Damped still converges at n=43.
# ─────────────────────────────────────────────────────────────────────────────
def test_edge_multi_alternation():
    pts_2d, p5, t5, center, e1, e2, coeffs = extract_window('Damped oscillation', 40, 11)
    cs = cross_sum_of(pts_2d, coeffs)
    assert cs is not None and cs != 1, \
        f"Expected cross.sum != 1 (multi-alternation), got {cs}"

    proj = bd._build_projective_arc_window(pts_2d, p5, t5, center, e1, e2, coeffs)
    assert proj is not None, \
        f"_build_projective_arc_window returned None for cross.sum={cs} window (no guard expected)"

    result = bd._try_conic(pts_2d, p5, t5, center, e1, e2, use_spline=True)
    assert result is not None, "_try_conic returned None for valid cross-branch conic"
    _, _, method = result
    assert method == 'conic', f"Expected method='conic', got {method!r}"
    print(f"  PASS  test_edge_multi_alternation  "
          f"(cross.sum={cs}, method={method!r} — valid conic, no guard)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Cross-branch conic window accepted with phi_c = −B/(2C)
# VdP win[1], n=131 (converged n): cross.sum=1, phi-monotone with the
# intrinsic phi_c = −B/(2C_e), NaN-free.
#
# Why n=131 and not n=40?
#   VdP near-spike windows at low n have very wide stereographic slopes
#   (e.g. s_vals spanning [−1.5, 1.35]).  With phi_c = −B/(2C) the arc
#   maps these to non-monotone φ — they correctly fall back to spline.
#   At converged n (n=131) the windows are narrower, φ is monotone, and
#   the projective arc is geometrically correct.
#
# Core assertion: the ONLY gate is phi-monotone (geometric order check).
# No quality filter (position disagreement, cross.sum limit) applies.
# ─────────────────────────────────────────────────────────────────────────────
def test_vdp_high_disagreement():
    pts_2d, p5, t5, center, e1, e2, coeffs = extract_window('Van der Pol (mu=3)', 131, 1)
    cs = cross_sum_of(pts_2d, coeffs)
    assert cs == 1, f"Expected cross.sum=1 for VdP win[1] n=131, got {cs}"

    proj = bd._build_projective_arc_window(pts_2d, p5, t5, center, e1, e2, coeffs)
    assert proj is not None, "Expected proj arc to succeed for phi-monotone cross.sum=1"

    t_check = np.linspace(t5[1], t5[3], 50)
    check = proj[0](t_check)
    assert np.all(np.isfinite(check)), "Unexpected NaN in VdP win[1] n=131 blend region"
    spline_f = bd._make_spline_window(p5, t5)
    diff = float(np.max(np.linalg.norm(check[:, :2] - spline_f(t_check)[:, :2], axis=1)))
    span = float(np.sum(np.linalg.norm(np.diff(p5[:, :2], axis=0), axis=1)))
    rel = diff / span

    result = bd._try_conic(pts_2d, p5, t5, center, e1, e2, use_spline=True, coeffs=coeffs)
    assert result is not None, "_try_conic returned None for valid cross-branch conic"
    _, _, method = result
    assert method == 'conic', f"Expected method='conic' (no quality gate), got {method!r}"
    print(f"  PASS  test_vdp_high_disagreement  "
          f"(cross.sum={cs}, rel={rel:.3f}, method={method!r} — "
          f"conic accepted; quality delegated to adaptive_n_budget)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\nRunning test_cross_guard.py")
    print("=" * 60)
    test_normal_single_crossing()
    test_edge_multi_alternation()
    test_vdp_high_disagreement()
    print("=" * 60)
    print("Done.\n")
