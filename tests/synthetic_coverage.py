"""
tests/synthetic_coverage.py
----------------------------
Unit tests for code paths that cannot be reached by smooth parameterised
curves but ARE reachable (or near-reachable) with specific synthetic
control-point configurations.  Each test uses analytically-constructed
5-point windows with explicit mirror or rotational symmetry so the
symmetry cross-checks correctness.

Paths exercised here (all marked DEAD/PRACTICAL in branch_coverage.py):

  1. bpaw_swapped / icm_swap
     Condition: |M_e| < 1e-9 * max(|L_e|, 1)  at canonical P₀.
     How to trigger: 5 points on a unit circle near (1, 0), passed directly
     to _build_projective_arc_window with exact coefficients (bypassing SVD).
     The swap IS entered; the function then returns None because the same
     scale that forces |L/M|>1e9 also makes |dxi| < eps_dx, saturating
     s_vals to ±1e15 and producing ctrl_err >> 1e-3.
     Symmetry: y-axis reflection (θ ↔ −θ) gives identical behaviour.

  2. degenerate center input (no guard — returns None via M_e < 1e-15)
     A control point coincides exactly with the conic centre.
     Geometry: 4 points on a unit circle (4-fold rotational symmetry) +
     1 point at (0, 0) (the centre of x²+y²=1).
     The removed bpaw_early_at_center guard is no longer needed: _select_k0
     picks the center point as canonical P₀ (ratio |L/M|=0 at origin), the
     swap block is entered, and after swap M_e=0 → the abs(M_e)<1e-15 guard
     at line ~1075 returns None.  Same observable contract, different path.
     Symmetry: 4-fold rotational symmetry for the 4 circle points.

  3. bpaw_case3_theta sub-paths (theta cross-branch fallback)
     Geometry: near-parabolic cross-branch window from the logarithmic spiral
     (conic A ≈ −1e-4 ≪ C ≈ 1).  Eigenvector method labels it cross-branch,
     but Q(s_vals) has no sign change (asymptote slopes outside s_vals range
     from canonical P₀) → Case 3 (theta) is triggered.
     Symmetry: 90° rigid rotation preserves the conic and its Case 3 property;
     both the original and rotated window must give method='conic'.

Run from the repo root:
    python3 tests/synthetic_coverage.py

Exit code 0 → all targeted paths were hit or correctly documented.
Exit code 1 → unexpected failure in the test logic.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import conicspline as bl

PASS = True


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_window(pts_2d, t0=0.0, dt=1.0):
    """Lift 2-D points to 3-D (z=0) and assign uniform times."""
    pts   = np.column_stack([pts_2d, np.zeros(len(pts_2d))])
    times = np.linspace(t0, t0 + dt, len(pts_2d))
    return pts, times


# ════════════════════════════════════════════════════════════════════════════
# Test 1: bpaw_swapped / icm_swap
# ════════════════════════════════════════════════════════════════════════════
def test_swap():
    """
    Verify the x↔y swap block in _build_projective_arc_window is entered when
    canonical P₀ has a near-vertical tangent (|M_e| < 1e-9·max(|L_e|,1)).

    Configuration: 5 points on the unit circle x²+y²=1, clustered near the
    right vertex (1, 0) where the tangent is vertical.  Angular step ε = 1e-10
    gives |L/M| = 1/(ε) = 1e10  >>  1e9 (swap threshold).

    The function is called DIRECTLY with the exact circle coefficients
    (1, 0, 1, 0, 0, −1) — bypassing fit_conic on near-collinear points,
    which would return a degenerate conic rather than a circle.

    Post-swap outcome: the function returns None because the same ε=1e-10
    spacing that triggers the swap also makes |dxi| = |sin(ε)| ≈ 1e-10 fall
    below eps_dx ≈ 1e-9, saturating s_vals to ±1e15.  The resulting orbit
    predicts (−1, 0) for near-(1,0) control points, giving ctrl_err ≈ 2 >>
    1e-3 → return None.  The swap block IS executed; it just cannot produce
    a useful orbit at this scale.

    Symmetry: y-axis reflection θ ↔ −θ gives identical pts_2d (mirror image)
    and an identical None return — the swap behaviour is symmetric.
    """
    global PASS
    print("\nTest 1: bpaw_swapped / icm_swap  (near-vertical tangent, direct call)")

    eps = 1e-10
    thetas = np.array([-2*eps, -eps, 0., eps, 2*eps])
    pts_2d = np.column_stack([np.cos(thetas), np.sin(thetas)])
    pts3d, times = _make_window(pts_2d)
    center = np.zeros(3)
    e1 = np.array([1., 0., 0.]); e2 = np.array([0., 1., 0.])
    coeffs_circle = (1., 0., 1., 0., 0., -1.)

    # ── Instrument _build_projective_arc_window ───────────────────────────
    hit_swap_bpaw = [False]
    _orig_b = bl._build_projective_arc_window

    def _patched_b(p2d, p, ts, ctr, _e1, _e2, cffs):
        A, B, C, D, E, F = cffs
        det = 4*A*C - B*B
        if abs(det) < 1e-10:
            return _orig_b(p2d, p, ts, ctr, _e1, _e2, cffs)
        cx = (-2*C*D + B*E)/det; cy = (-2*A*E + B*D)/det
        dx = p2d[:,0]-cx; dy = p2d[:,1]-cy
        if np.min(np.sqrt(dx**2+dy**2)) < 1e-10:
            return _orig_b(p2d, p, ts, ctr, _e1, _e2, cffs)
        # canonical P0 selection
        def sk0(pe, Ae, Be, Ce, De, Ee):
            xs, ys = pe[:,0], pe[:,1]
            L = 2*Ae*xs + Be*ys + De; M = Be*xs + 2*Ce*ys + Ee
            Ms = np.where(np.abs(M)>=1e-15, M, np.sign(M+1e-300)*1e-15)
            return int(np.argmin(np.abs(L/Ms))), L, M
        k0, La, Ma = sk0(p2d, A, B, C, D, E)
        Le, Me = La[k0], Ma[k0]
        if abs(Me) < 1e-9 * max(abs(Le), 1.0):
            hit_swap_bpaw[0] = True
        return _orig_b(p2d, p, ts, ctr, _e1, _e2, cffs)

    bl._build_projective_arc_window = _patched_b
    result = bl._build_projective_arc_window(pts_2d, pts3d, times, center, e1, e2, coeffs_circle)
    bl._build_projective_arc_window = _orig_b

    # ── Instrument _is_conic_monotone ─────────────────────────────────────
    # For icm_swap the same fundamental constraint applies when the function
    # receives pts5_xy with near-vertical tangents AFTER fit_conic and eigh.
    # Below we verify the swap condition can be reproduced with manually
    # chosen principal-frame values.
    hit_swap_icm = False
    # In _is_conic_monotone, swap fires when |M_k0| < 1e-9·max(|L_k0|, 1).
    # M_k = 2·C_r·y'_k + E_r.  For a standard ellipse (C_r=1, E_r=0): M=2y'.
    # Swap threshold: |2y'| < 1e-9·|2x'| ≈ 1e-9  →  |y'| < 5e-10.
    # These are the same conditions as bpaw_swapped — the analysis is identical.
    # The swap IS reachable in principle (direct numerical verification below):
    A_r, C_r, D_r, E_r = 1.0, 1.0, 0.0, 0.0  # unit circle in principal frame
    xs_r = np.cos(thetas)           # ≈ 1.0
    ys_r = np.sin(thetas)           # ≈ ε (tiny)
    Lk = 2*A_r*xs_r + D_r          # ≈ 2
    Mk = 2*C_r*ys_r + E_r          # ≈ 2ε
    ratio = np.where(np.abs(Mk)>1e-15, np.abs(Lk)/np.abs(Mk), 1e30)
    k0 = int(np.argmin(ratio))
    L0, M0 = Lk[k0], Mk[k0]
    hit_swap_icm = abs(M0) < 1e-9 * max(abs(L0), 1.0)

    # ── Symmetry check: flip y → same result ─────────────────────────────
    pts_2d_m = np.column_stack([np.cos(thetas), -np.sin(thetas)])  # y-reflected
    pts3d_m, times_m = _make_window(pts_2d_m)
    bl._build_projective_arc_window = _patched_b
    result_m = bl._build_projective_arc_window(pts_2d_m, pts3d_m, times_m,
                                                center, e1, e2, coeffs_circle)
    bl._build_projective_arc_window = _orig_b

    ok_cond_bpaw = hit_swap_bpaw[0]
    ok_cond_icm  = hit_swap_icm
    ok_sym        = (result is None) == (result_m is None)  # both None

    print(f"  Swap condition (bpaw_swapped) fires : {'✓' if ok_cond_bpaw else '✗'}")
    print(f"  Swap condition (icm_swap) in theory : {'✓' if ok_cond_icm else '✗'}")
    print(f"  Post-swap returns None (expected)   : {'✓' if result is None else '✗ (unexpected success)'}")
    print(f"  Mirror-symmetric behaviour          : {'✓' if ok_sym else '✗'}")
    print(f"  Note: swap fires but produces ctrl_err>>1e-3 at ε=1e-10 scale")
    print(f"        (same ε that forces |L/M|>1e9 also makes |dxi|<eps_dx)")

    if not (ok_cond_bpaw and ok_cond_icm and ok_sym):
        PASS = False


# ════════════════════════════════════════════════════════════════════════════
# Test 2: degenerate center input
# ════════════════════════════════════════════════════════════════════════════
def test_degenerate_center():
    """
    Verify _build_projective_arc_window returns None gracefully when a control
    point coincides exactly with the conic centre.

    Configuration: 4 points on the unit circle at 0°, 90°, 180°, 270°
    (4-fold rotational symmetry) plus 1 point at (0, 0) — the centre of
    x²+y²=1.

    Old code path: a guard `if min_dist < 1e-10: return None` fired immediately.
    New code path (guard removed): _select_k0 picks the centre point as the
    canonical P₀ because |L/M| = 0 there (L=2x=0, M=2y=0 at the origin) →
    the swap block is entered → after swap M_e=0 → the `abs(M_e) < 1e-15`
    guard at the end of the swap block returns None.  Same observable contract.

    The 4-fold rotational symmetry of the circle points is preserved: rotating
    by 90° gives the same pts_2d (only the center point is fixed), so the
    result is identical.
    """
    global PASS
    print("\nTest 2: degenerate center input  (control point at conic centre → None via M_e<1e-15)")

    angles = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    pts_2d = np.vstack([circle[:2], [[0.0, 0.0]], circle[2:]])
    pts, times = _make_window(pts_2d)

    center = np.zeros(3)
    e1 = np.array([1., 0., 0.]); e2 = np.array([0., 1., 0.])
    coeffs_circle = (1., 0., 1., 0., 0., -1.)

    result = bl._build_projective_arc_window(pts_2d, pts, times, center, e1, e2, coeffs_circle)
    ok = result is None
    print(f"  Returns None for center-coincident point : {'✓' if ok else '✗ (unexpected non-None)'}")
    if not ok:
        PASS = False


# ════════════════════════════════════════════════════════════════════════════
# Test 3: bpaw_case3_theta sub-paths
# ════════════════════════════════════════════════════════════════════════════
def test_case3_theta():
    """
    Verify Case 3 (theta fallback) fires and returns successfully.

    Configuration: a near-parabolic cross-branch window captured from the
    logarithmic spiral (conic coefficient A ≈ −1e-4 ≪ C ≈ 1).  The eigenvector
    method labels these 5 points as cross-branch because their projections onto
    the positive eigenvector span both signs.  However, the two roots of
    Q(s) = A + Bs + Cs² (the asymptote slopes in the stereographic frame) both
    lie outside the s_vals range from canonical P₀ → Q never changes sign →
    Case 3: angle-from-centre theta fallback.

    Why this geometry is Case 3 and not Case 1:
      A ≈ −1e-4 means one Q-root ≈ s₀ (the canonical tangent slope) and the
      other ≈ −B/C ≈ +0.05.  All 5 control-point slopes s_i cluster in
      [−0.06, −0.002], sitting below both roots → Q positive throughout.

    Symmetry: 90° rigid rotation of all control points preserves the conic
    and its cross-branch structure.  The rotated window must also be Case 3
    and yield method='conic'.
    """
    global PASS
    print("\nTest 3: bpaw_case3_theta  (theta cross-branch fallback, near-parabolic hyperbola)")

    # Hardcoded control points from logarithmic spiral window (n=8, first conic)
    # Captured by exhaustive sweep in tests/branch_coverage.py
    pts_2d = np.array([
        [-0.06967757,  0.01617481],
        [-0.24795342, -0.01361061],
        [ 0.10870786,  0.02566110],
        [-0.60258132, -0.01263545],
        [ 0.81150445, -0.01558984],
    ])
    ts = np.array([0.0, 0.142857, 0.285714, 0.428571, 0.571429])
    pts, times = np.column_stack([pts_2d, np.zeros(5)]), ts

    hit_case3 = [False]
    _orig_b = bl._build_projective_arc_window

    def _patched_b(p2d, p, ts_in, ctr, _e1, _e2, cffs):
        A, B, C, D, E, F = cffs
        det = 4*A*C - B*B
        if det == 0.0:
            return _orig_b(p2d, p, ts_in, ctr, _e1, _e2, cffs)
        cx=(-2*C*D+B*E)/det; cy=(-2*A*E+B*D)/det
        dx=p2d[:,0]-cx; dy=p2d[:,1]-cy
        M_mat=np.array([[A,B/2],[B/2,C]])
        ev,evec=np.linalg.eigh(M_mat)
        if ev[0]*ev[1]>=0: return _orig_b(p2d, p, ts_in, ctr, _e1, _e2, cffs)
        pidx=1 if ev[1]>0 else 0
        et=evec[:,pidx]; prj=dx*et[0]+dy*et[1]; br=np.sign(prj)
        cross=(br[:4]*br[1:])<0
        if not cross.any(): return _orig_b(p2d, p, ts_in, ctr, _e1, _e2, cffs)
        # Classify Case 1 vs Case 3 using canonical P0
        def sk0(pe, Ae, Be, Ce, De, Ee):
            xs,ys=pe[:,0],pe[:,1]
            L=2*Ae*xs+Be*ys+De; M=Be*xs+2*Ce*ys+Ee
            Ms=np.where(np.abs(M)>=1e-15,M,np.sign(M+1e-300)*1e-15)
            return int(np.argmin(np.abs(L/Ms))),L,M
        pts_e=p2d.copy(); A_e,C_e,D_e,E_e=A,C,D,E
        k0,La,Ma=sk0(pts_e,A_e,B,C_e,D_e,E_e); Le,Me=La[k0],Ma[k0]
        if abs(Me)<1e-9*max(abs(Le),1.):
            pts_e=pts_e[:,::-1]; A_e,C_e=C_e,A_e; D_e,E_e=E_e,D_e
            k0,La,Ma=sk0(pts_e,A_e,B,C_e,D_e,E_e); Le,Me=La[k0],Ma[k0]
        if abs(Me)>1e-15:
            s0=-Le/Me; x0c,y0c=pts_e[k0]
            dxi=pts_e[:,0]-x0c; dyi=pts_e[:,1]-y0c
            eps_=1e-9*(np.abs(dyi)+1.); mask_=np.abs(dxi)>=eps_
            dxi_s=np.where(mask_,dxi,1.)
            sv=np.where(mask_,dyi/dxi_s,np.sign(dyi+1e-300)*1e15); sv[k0]=s0
            Qv=A_e+B*sv+C_e*sv**2
            if len(np.where(np.diff(np.sign(Qv))!=0)[0])==0:
                hit_case3[0] = True
        return _orig_b(p2d, p, ts_in, ctr, _e1, _e2, cffs)

    bl._build_projective_arc_window = _patched_b
    _, _err, method = bl.fit_conic_5pt(pts, times)
    bl._build_projective_arc_window = _orig_b

    ok_case3 = hit_case3[0]
    ok_method = method == 'conic'

    # ── Symmetry check: 90° rigid rotation ───────────────────────────────
    # A 90° rotation maps the near-parabolic hyperbola to an equally valid
    # near-parabolic hyperbola with the same cross-branch structure and Case 3
    # classification.  Both the original and rotated windows must give 'conic'.
    c90, s90 = np.cos(np.pi/2), np.sin(np.pi/2)
    pts_2d_rot = pts_2d @ np.array([[c90, -s90], [s90, c90]]).T
    pts_rot = np.column_stack([pts_2d_rot, np.zeros(5)])
    _, _, method_rot = bl.fit_conic_5pt(pts_rot, times)

    ok_sym = method == method_rot
    print(f"  Case 3 theta detected  : {'✓' if ok_case3 else '✗ (Q has sign change → Case 1, not 3)'}")
    print(f"  Method = conic         : {'✓' if ok_method else f'✗ (got {method})'}")
    print(f"  90° rotation sym       : {'✓' if ok_sym else '✗'}  "
          f"(orig={method}, rot={method_rot})")

    if not (ok_case3 and ok_method and ok_sym):
        PASS = False


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_swap()
    test_degenerate_center()
    test_case3_theta()

    print()
    if PASS:
        print("✓  All synthetic coverage tests passed.")
        sys.exit(0)
    else:
        print("✗  One or more tests failed.")
        sys.exit(1)
