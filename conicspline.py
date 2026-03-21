"""
conicspline.py — C^N parametric curve interpolation.

Fits conic arc sections wherever possible (spline fallback otherwise),
using overlapping 5-point windows and a quintic smoothstep blend.

When the input data lies exactly on a conic, the blend is near
machine-epsilon accurate everywhere between control points.  Each conic
type uses its natural intrinsic parametrization: ellipses and circles use
the eccentric-anomaly orbit anchored at the conic center (an algebraic
invariant); hyperbolas use the stereographic orbit anchored at the conic
vertex (also an algebraic invariant); parabolas use the vertex path with
O(h²) accuracy.  Both overlapping windows on any segment share the same
anchor → identical orbit functions → exact blend on the conic.

Demo curves (CURVES list) and visualization live in blend_demo.py.
"""

import numpy as np
from scipy.special import betainc
from scipy.interpolate import CubicSpline, PchipInterpolator, CubicHermiteSpline, BPoly
# ── Numerical thresholds ──────────────────────────────────────────────────────
# Each constant serves a distinct semantic purpose; identical numeric values
# are NOT interchangeable — do not consolidate.

_EPS_GRAD      = 1e-30  # Gradient/normal norm below which the tangent direction
                         # is undefined; guards divisions before normalisation.

_EPS_CHORD     = 1e-15  # Minimum meaningful 1-D length: chord or arc-length gap
                         # below which two samples are coincident or a window is
                         # degenerate.  Used as a denominator floor for ratios.

_EPS_COEFF     = 1e-15  # Conic coefficient (Ae, Ce, Ee, Me, Mk) treated as zero:
                         # triggers coordinate swap, parabola branch, or
                         # degenerate-conic short-circuit.  Same numeric value as
                         # _EPS_CHORD but semantically distinct.

_EPS_DET       = 1e-12  # |det(M)| = |4AC−B²| below which the conic is
                         # near-parabolic and center computation is ill-conditioned;
                         # also used for Q(s) asymptote proximity checks.

_EPS_SWAP      = 1e-9   # Relative threshold for x↔y coordinate swap: swap when
                         # |A_e| < _EPS_SWAP·max(|C_e|,1) or slope denominator
                         # |dx| < _EPS_SWAP·(|dy|+1).

_EPS_TSPAN     = 1e-9   # Relative parameter proximity: a sample at t is
                         # "at control point tᵢ" when |t−tᵢ| < _EPS_TSPAN·span.
                         # Same numeric value as _EPS_SWAP but semantically distinct.

_EPS_CTRL      = 1e-3   # Orbit quality gate: maximum allowed distance from the
                         # orbit to a control point.  Orbits exceeding this are
                         # rejected and replaced by the arc-length / spline path.

_EVAL_SEP_NEAR = 0.2    # Eigenvalue-separation threshold for "near-circle":
                         # (|λ_max|−|λ_min|)/(|λ_max|+|λ_min|) < this value.
                         # Load-bearing gate in _try_conic; the matching guard
                         # inside _build_projective_arc_window is redundant
                         # (computation optimisation only).

_ALPHA_FALLBACK = 0.3   # Conic/spline disagreement ratio above which the
                         # pure-spline window wins (≈ 22% of mean chord).

_SIGN_NUG      = 1e-300 # Infinitesimal nudge before np.sign() so that
                         # np.sign(0) = +1 rather than 0; also stabilises
                         # eval-sep denominators against exact zero.

# ── Conic fitting ─────────────────────────────────────────────────────────────
# Inlined from orbitkit/conic_fit.py (local package, not on PyPI).

def fit_conic(points):
    """Fit Ax²+Bxy+Cy²+Dx+Ey+F=0 to N≥5 2-D points via SVD null space."""
    points = np.asarray(points, dtype=float)
    M = np.column_stack([
        points[:, 0]**2, points[:, 0] * points[:, 1], points[:, 1]**2,
        points[:, 0], points[:, 1], np.ones(len(points)),
    ])
    _, _, Vt = np.linalg.svd(M)
    return Vt[-1]


# ── Generalized polar curve: r = r0 / (1 + a*sin(k*θ) + b*cos(k*θ)) ─────

def _fit_gen_polar(pts_2d):
    """Fit generalized polar curve with two harmonics to 5 2D points.

    Model: r = r0 / (1 + a1*sin(kθ) + b1*cos(kθ) + a2*sin(2kθ) + b2*cos(2kθ))

    With 5 params (r0, a1, b1, a2, b2) and 5 points, the fit is exact for
    each candidate k.  We pick the k that produces the smoothest, most
    well-behaved curve between the control points.

    Returns (coeffs, k, center, thetas) or None.
    coeffs = (r0, a1, b1, a2, b2)
    """
    n = len(pts_2d)
    centroid = pts_2d.mean(axis=0)

    span = np.ptp(pts_2d, axis=0)
    offsets = [np.zeros(2)]
    for frac in [-0.3, 0.3]:
        offsets.append(np.array([frac * span[0], 0]))
        offsets.append(np.array([0, frac * span[1]]))
        offsets.append(np.array([frac * span[0], frac * span[1]]))

    k_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    best = None
    best_score = np.inf  # lower = smoother

    for off in offsets:
        center = centroid + off
        dx = pts_2d[:, 0] - center[0]
        dy = pts_2d[:, 1] - center[1]
        r = np.sqrt(dx**2 + dy**2)

        # Need all points at reasonable distance from center
        if np.min(r) < 0.05 * np.max(r):
            continue

        theta = np.unwrap(np.arctan2(dy, dx))

        # Require monotonic angles
        if not np.all(np.diff(theta) > 0):
            for i in range(1, n):
                if theta[i] <= theta[i - 1]:
                    theta[i:] += 2 * np.pi
            if not np.all(np.diff(theta) > 0):
                continue

        for k in k_candidates:
            # Linear system (5 eq, 5 unknowns for exact fit):
            # r_i = r0 - r_i*[a1*sin(kθ) + b1*cos(kθ) + a2*sin(2kθ) + b2*cos(2kθ)]
            A_mat = np.column_stack([
                np.ones(n),
                -r * np.sin(k * theta),
                -r * np.cos(k * theta),
                -r * np.sin(2 * k * theta),
                -r * np.cos(2 * k * theta),
            ])
            cond = np.linalg.cond(A_mat)
            if cond > 1e10:
                continue

            try:
                coeffs = np.linalg.solve(A_mat, r)
            except np.linalg.LinAlgError:
                continue

            r0, a1, b1, a2, b2 = coeffs
            if r0 <= 0:
                continue

            # Check no singularity: denominator must stay positive
            th_dense = np.linspace(theta[0], theta[-1], 300)
            denom_dense = (1 + a1 * np.sin(k * th_dense)
                           + b1 * np.cos(k * th_dense)
                           + a2 * np.sin(2 * k * th_dense)
                           + b2 * np.cos(2 * k * th_dense))
            if np.min(denom_dense) <= 0.05:
                continue

            # Check r stays positive
            r_dense = r0 / denom_dense
            if np.min(r_dense) <= 0:
                continue

            # Smoothness score: total variation of r (less = smoother)
            dr = np.abs(np.diff(r_dense))
            score = np.sum(dr) / np.mean(r_dense)

            if score < best_score:
                best_score = score
                best = (coeffs, k, center.copy(), theta.copy())

    if best is None:
        return None
    return best


def _eval_gen_polar(theta, coeffs, k, center):
    """Evaluate 2-harmonic generalized polar at theta values."""
    r0, a1, b1, a2, b2 = coeffs
    theta = np.asarray(theta, dtype=float)
    denom = (1 + a1 * np.sin(k * theta)
             + b1 * np.cos(k * theta)
             + a2 * np.sin(2 * k * theta)
             + b2 * np.cos(2 * k * theta))
    r = r0 / denom
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y])


def _trace_gen_polar(coeffs, k, center, theta0, theta1, n_steps=100):
    """Evaluate generalized polar curve at dense theta values.

    Returns (n_steps+1, 2) array of 2D points.
    """
    thetas = np.linspace(theta0, theta1, n_steps + 1)
    return _eval_gen_polar(thetas, coeffs, k, center)


def smoothstep(s, N):
    """Generalized smoothstep of order N (C^N continuous)."""
    s = np.asarray(s, dtype=float)
    if N == 0:
        return s
    return betainc(N + 1, N + 1, s)


def _conic_gradient(coeffs, x, y):
    """Gradient of Ax²+Bxy+Cy²+Dx+Ey+F (normal to conic at (x,y))."""
    A, B, C, D, E, F = coeffs
    return np.array([2*A*x + B*y + D, B*x + 2*C*y + E])


def _project_onto_conic(coeffs, x, y, niter=8):
    """Newton-project (x,y) onto the nearest point on the conic."""
    A, B, C, D, E, F = coeffs
    for _ in range(niter):
        f = A*x**2 + B*x*y + C*y**2 + D*x + E*y + F
        g = _conic_gradient(coeffs, x, y)
        g2 = g[0]**2 + g[1]**2
        if g2 < _EPS_GRAD:
            break
        x -= f * g[0] / g2
        y -= f * g[1] / g2
    return x, y


def _trace_conic_arc(coeffs, p0, p1, n_steps=100, initial_tang=None):
    """Trace along the conic from p0 to p1 via predictor-corrector.

    initial_tang: if provided (unit vector), the traversal direction at p0 is
    matched to its sign rather than pointing toward p1.

    Returns (points, tang_end):
      points   – (m, 2) array of traced positions, ending exactly at p1
      tang_end – unit tangent at p1 consistent with the traversal direction
                 (exact algebraic gradient, suitable as initial_tang for the
                 next sub-arc)

    When the chosen direction takes the arc to p1 from the WRONG SIDE (the snap
    vector opposes tang_end and the snap distance is large), the arc is
    automatically re-traced with the reversed direction.  This avoids the
    "long-arc arrival" artifact where the tracer circles most of the conic
    and reaches p1 from an angle incompatible with the algebraic tangent,
    creating a large directional jump in the arc data fed to CubicSpline.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    chord = np.linalg.norm(p1 - p0)
    ds = chord * 1.5 / n_steps

    def _run_loop(tang0):
        """Single forward trace; returns (pts_list, final_loop_tang)."""
        pos = p0.copy()
        tang = tang0.copy()
        pts = [pos.copy()]
        for _ in range(n_steps * 8):
            new = pos + ds * tang
            new[0], new[1] = _project_onto_conic(coeffs, new[0], new[1])
            g = _conic_gradient(coeffs, new[0], new[1])
            t2 = np.array([-g[1], g[0]])
            tn = np.linalg.norm(t2)
            if tn > _EPS_GRAD:
                t2 /= tn
            if np.dot(t2, tang) < 0:
                t2 = -t2
            pos = new
            tang = t2
            pts.append(pos.copy())
            dist = np.linalg.norm(pos - p1)
            if dist < ds * 0.5:
                break
            if len(pts) > 3:
                prev = np.linalg.norm(np.asarray(pts[-2]) - p1)
                if dist > prev and dist < chord * 0.3:
                    break
        return pts, tang

    def _make_tang_end(tang_loop):
        """Algebraic unit tangent at p1, signed to match tang_loop direction."""
        g = _conic_gradient(coeffs, p1[0], p1[1])
        te = np.array([-g[1], g[0]])
        tn = np.linalg.norm(te)
        if tn > _EPS_GRAD:
            te /= tn
        if np.dot(te, tang_loop) < 0:
            te = -te
        return te

    # Choose initial direction
    g0 = _conic_gradient(coeffs, p0[0], p0[1])
    tang0 = np.array([-g0[1], g0[0]])
    tn = np.linalg.norm(tang0)
    if tn > _EPS_GRAD:
        tang0 /= tn
    if initial_tang is not None:
        if np.dot(tang0, initial_tang) < 0:
            tang0 = -tang0
    else:
        if np.dot(tang0, p1 - p0) < 0:
            tang0 = -tang0

    # Primary trace
    pts1, tang_loop1 = _run_loop(tang0)
    tang_end = _make_tang_end(tang_loop1)

    # Check for "wrong-side arrival": the snap vector (last tracked → p1) is
    # large AND points opposite to tang_end.  Re-trace with reversed direction.
    pts_final = pts1
    snap = np.asarray(p1) - np.asarray(pts1[-1])
    snap_len = np.linalg.norm(snap)
    if snap_len > ds * 1.0 and np.dot(snap, tang_end) < 0:
        pts2, tang_loop2 = _run_loop(-tang0)
        snap_len2 = np.linalg.norm(np.asarray(p1) - np.asarray(pts2[-1]))
        if snap_len2 < snap_len:
            pts_final = pts2
            tang_end = _make_tang_end(tang_loop2)

    # Truncate to closest approach, append exact p1.
    arr  = np.array(pts_final)
    best = int(np.argmin(np.linalg.norm(arr - p1, axis=1)))
    result = list(arr[:best + 1]) + [p1.copy()]
    return np.array(result), tang_end


def _build_spline_func(ts, center, e1, e2, pts_2d,
                       bc_left_2d=None, bc_right_2d=None):
    """Build cubic spline interpolant through 5 control points in SVD plane.

    bc_left_2d, bc_right_2d: optional (dx, dy) first-derivative constraints
    in the SVD (e1, e2) plane at ts[0] and ts[-1].  When provided the spline
    is clamped to match those tangent directions, reducing the velocity
    mismatch (vB'−vA') that the quintic smoothstep amplifies into curvature
    oscillations at conic↔spline window transitions.
    """
    # CubicSpline bc_type = (left_BC, right_BC); each side is either
    # (deriv_order, value) for a clamped condition or (2, 0.0) for natural.
    # Using the explicit (2, 0.0) form allows mixing clamped and natural
    # per-side without relying on string support in older scipy versions.
    left_x  = (1, bc_left_2d[0])  if bc_left_2d  is not None else (2, 0.0)
    right_x = (1, bc_right_2d[0]) if bc_right_2d is not None else (2, 0.0)
    left_y  = (1, bc_left_2d[1])  if bc_left_2d  is not None else (2, 0.0)
    right_y = (1, bc_right_2d[1]) if bc_right_2d is not None else (2, 0.0)
    bc_x = (left_x, right_x)
    bc_y = (left_y, right_y)
    x_spl = CubicSpline(ts, pts_2d[:, 0], bc_type=bc_x)
    y_spl = CubicSpline(ts, pts_2d[:, 1], bc_type=bc_y)

    def orbit_func(t):
        t = np.atleast_1d(t).astype(float)
        lx = x_spl(t)
        ly = y_spl(t)
        pos = (center
               + lx[:, None] * e1[None, :]
               + ly[:, None] * e2[None, :])
        return pos

    # Exact analytic derivatives (no finite differences).
    # d/dt pos = dx/dt * e1 + dy/dt * e2  (center is constant)
    _x_d1 = x_spl.derivative(1)
    _y_d1 = y_spl.derivative(1)
    _x_d2 = x_spl.derivative(2)
    _y_d2 = y_spl.derivative(2)

    def _d1(t):
        t = np.atleast_1d(t).astype(float)
        return _x_d1(t)[:, None] * e1[None, :] + _y_d1(t)[:, None] * e2[None, :]

    def _d2(t):
        t = np.atleast_1d(t).astype(float)
        return _x_d2(t)[:, None] * e1[None, :] + _y_d2(t)[:, None] * e2[None, :]

    orbit_func.d1 = _d1
    orbit_func.d2 = _d2
    return orbit_func


def _make_spline_window(pts5, ts5):
    """5-point natural cubic spline window — drop-in for non-monotone conic windows.

    Passes through all 5 control points exactly at ts5.  Has .d1 and .d2
    attributes, so it is compatible with every place that conic windows are used.
    Never produces zig-zag arcs; always well-behaved.
    """
    p = pts5.astype(float)
    center = p.mean(axis=0)
    _, _, Vt = np.linalg.svd(p - center, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    pts_2d = np.column_stack([(p - center) @ e1, (p - center) @ e2])
    return _build_spline_func(ts5, center, e1, e2, pts_2d)


def _build_conic_tangent_spline(pts, ts, center3d, e1, e2, pts_2d, coeffs, n_deriv=2):
    """Hermite spline: conic-tangent first derivatives + conic curvature (C^2).

    ── DEAD CODE ──────────────────────────────────────────────────────────────
    Previously called by _try_conic Phase 2 (zig-zag / cross-branch detection).
    Phase 2 has been removed: cross-branch windows are now handled by the
    projective monotonicity gate in _is_conic_monotone.  This function is
    retained for reference and potential future use (e.g. a projective arc
    window that correctly renders the ∞-crossing portion).
    ── END DEAD CODE ──────────────────────────────────────────────────────────

    Used when _trace_conic_arc produces a zig-zag (cross-branch hyperbola).
    First derivatives use the conic tangent [-Gy, Gx] at each control point,
    with sign chosen so it agrees with the centered finite-difference direction:
      • k=0 : one-sided chord  pts_2d[1]  – pts_2d[0]
      • k=1..3 : centered chord pts_2d[k+1] – pts_2d[k-1]
      • k=4 : one-sided chord  pts_2d[4]  – pts_2d[3]
    The centered chord spans both sides of any turning point, giving a net-
    forward reference even where the one-sided chord is nearly perpendicular.
    Second derivatives (curvature) still come from the analytical conic gradient.

    n_deriv=1: CubicHermiteSpline — matches position + first derivative (C^1).
    n_deriv=2: BPoly quintic Hermite — also matches second derivative (C^2).
    """
    A, B, C, D, E, F = coeffs
    dydx_2d = np.zeros((5, 2))
    for k in range(5):
        lx_k, ly_k = pts_2d[k]
        Gx = 2*A*lx_k + B*ly_k + D
        Gy = B*lx_k + 2*C*ly_k + E
        g  = np.sqrt(Gx**2 + Gy**2)
        if g < _EPS_GRAD:
            continue
        tau = np.array([-Gy, Gx]) / g        # unit conic tangent
        # Sign reference: centered chord for interior knots, one-sided for ends.
        # Centered chord pts[k+1]-pts[k-1] references the same original-space
        # vector for any window that shares p[k], guaranteeing consistent sign.
        if k == 0:
            ref = pts_2d[1] - pts_2d[0]
        elif k == 4:
            ref = pts_2d[4] - pts_2d[3]
        else:
            ref = pts_2d[k+1] - pts_2d[k-1]
        if tau @ ref < 0:
            tau = -tau
        # Scale to t-space: chord/dt estimate of arc speed
        if k == 0:
            chord, dt = np.linalg.norm(pts_2d[1]-pts_2d[0]), ts[1]-ts[0]
        elif k == 4:
            chord, dt = np.linalg.norm(pts_2d[4]-pts_2d[3]), ts[4]-ts[3]
        else:
            chord = np.linalg.norm(pts_2d[k+1]-pts_2d[k-1])
            dt    = ts[k+1] - ts[k-1]
        dydx_2d[k] = tau * (chord / dt if dt > _EPS_GRAD else 1.0)

    if n_deriv >= 2:
        # Analytical conic curvature: κ_vec = d²pos/ds² = -2·S2/g⁴ · (Gx,Gy)
        # where S2 = A·Gy²-B·Gx·Gy+C·Gx².  Scaled to t-space: v²·κ_vec
        # (centripetal acceleration, assuming constant arc speed per interval).
        d2ydx2_2d = np.zeros((5, 2))
        for k in range(5):
            lx_k, ly_k = pts_2d[k]
            Gx = 2*A*lx_k + B*ly_k + D
            Gy = B*lx_k + 2*C*ly_k + E
            g  = np.sqrt(Gx**2 + Gy**2)
            if g < _EPS_GRAD:
                continue
            S2 = A*Gy**2 - B*Gx*Gy + C*Gx**2
            kappa_vec = (-2*S2 / g**4) * np.array([Gx, Gy])  # d²pos/ds²
            v = np.linalg.norm(dydx_2d[k])
            d2ydx2_2d[k] = v**2 * kappa_vec                   # d²pos/dt²
        # ── Greedy sign refinement ────────────────────────────────────────────
        # Try flipping each knot's tangent sign independently; keep the flip
        # if it reduces the sum of squared midpoint deviations from the chord
        # polygon (piecewise-linear interpolant through the 5 control points).
        # d2ydx2_2d depends only on norm(dydx_2d[k]) — invariant to sign flips
        # — so we reuse it unchanged in every trial build.
        def _score(dydx):
            yx_ = [[pts_2d[k, 0], dydx[k, 0], d2ydx2_2d[k, 0]] for k in range(5)]
            yy_ = [[pts_2d[k, 1], dydx[k, 1], d2ydx2_2d[k, 1]] for k in range(5)]
            xs_ = BPoly.from_derivatives(ts, yx_)
            ys_ = BPoly.from_derivatives(ts, yy_)
            s = 0.0
            for k in range(4):
                t_mid = 0.5 * (ts[k] + ts[k + 1])
                p_mid = 0.5 * (pts_2d[k] + pts_2d[k + 1])
                q_mid = np.array([float(xs_(t_mid)), float(ys_(t_mid))])
                s += float(np.dot(q_mid - p_mid, q_mid - p_mid))
            return s
        score = _score(dydx_2d)
        for k in range(5):
            trial = dydx_2d.copy()
            trial[k] = -trial[k]
            s2 = _score(trial)
            if s2 < score:
                # Guard: don't flip if it reverses the chord-based direction.
                # The centered chord (interior knots) encodes the global forward
                # direction; overriding it breaks consistency with adjacent windows.
                if k == 0:
                    ref_k = pts_2d[1] - pts_2d[0]
                elif k == 4:
                    ref_k = pts_2d[4] - pts_2d[3]
                else:
                    ref_k = pts_2d[k + 1] - pts_2d[k - 1]
                if np.linalg.norm(ref_k) > _EPS_DET and float(trial[k] @ ref_k) < 0:
                    continue   # flip would reverse forward direction — skip
                dydx_2d[k] = -dydx_2d[k]
                score = s2
        y_x = [[pts_2d[k, 0], dydx_2d[k, 0], d2ydx2_2d[k, 0]] for k in range(5)]
        y_y = [[pts_2d[k, 1], dydx_2d[k, 1], d2ydx2_2d[k, 1]] for k in range(5)]
        x_spl = BPoly.from_derivatives(ts, y_x)
        y_spl = BPoly.from_derivatives(ts, y_y)
    else:
        x_spl = CubicHermiteSpline(ts, pts_2d[:, 0], dydx_2d[:, 0])
        y_spl = CubicHermiteSpline(ts, pts_2d[:, 1], dydx_2d[:, 1])
    _xd1, _yd1 = x_spl.derivative(1), y_spl.derivative(1)
    _xd2, _yd2 = x_spl.derivative(2), y_spl.derivative(2)

    def orbit_func(t):
        t = np.atleast_1d(t).astype(float)
        return (center3d
                + x_spl(t)[:, None] * e1[None, :]
                + y_spl(t)[:, None] * e2[None, :])

    def _d1(t):
        t = np.atleast_1d(t).astype(float)
        return _xd1(t)[:, None]*e1[None, :] + _yd1(t)[:, None]*e2[None, :]

    def _d2(t):
        t = np.atleast_1d(t).astype(float)
        return _xd2(t)[:, None]*e1[None, :] + _yd2(t)[:, None]*e2[None, :]

    orbit_func.d1, orbit_func.d2 = _d1, _d2
    pred     = orbit_func(ts)
    ctrl_err = float(np.max(np.linalg.norm(pred - pts, axis=1)))
    return orbit_func, ctrl_err, 'conic_hermite'


def _blended_conic_spline(pts, ts, center3d, e1, e2, pts_2d, conic_func):
    """Blend conic and spline interpolants based on local disagreement.

    Both interpolants pass through the 5 control points exactly, so any
    linear combination (1-α)*conic + α*spline also interpolates exactly.

    α(t) is computed from the pointwise disagreement between conic and
    spline, with a regularizer (smoothing) that favors slowly-varying α.
    The result is:
      - Pure conic where both agree (near-Keplerian segments)
      - Pure spline where the conic diverges (crossings, inflections)
      - Smooth transition in between
    """
    from scipy.ndimage import uniform_filter1d

    spline_func = _build_spline_func(ts, center3d, e1, e2, pts_2d)

    # Evaluate both on a dense grid
    t_dense = np.linspace(ts[0], ts[-1], 50)
    c_pts = conic_func(t_dense)
    s_pts = spline_func(t_dense)

    # Guard: projective arc may return NaN near asymptotes (forbidden zone).
    # Fall back to pure spline when this occurs.
    if not np.all(np.isfinite(c_pts)):
        pred = spline_func(ts)
        ctrl_err = float(np.max(np.linalg.norm(pred - pts, axis=1)))
        return spline_func, ctrl_err, 'spline'

    # Disagreement: 3D distance between conic and spline at each point
    disagreement = np.linalg.norm(c_pts - s_pts, axis=1)

    # Scale by mean chord length between consecutive control points
    chord_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mean_chord = np.mean(chord_lens)
    if mean_chord < _EPS_CHORD:
        mean_chord = 1.0

    # Sigmoid-like transition:
    #   disagreement < 10% of chord → α ≈ 0 (trust conic)
    #   disagreement > 50% of chord → α ≈ 1 (trust spline)
    alpha_raw = np.clip((disagreement / mean_chord - 0.1) / 0.4, 0, 1)
    max_alpha_raw = float(np.max(alpha_raw))

    # When the conic is significantly wrong anywhere in the window, the smoothed-α
    # approach leaves a residual conic contribution outside the α-peak region that
    # can corrupt the blend.  Guard: return pure spline immediately.
    # _ALPHA_FALLBACK ≈ conic disagrees by ≥ 22% of mean chord anywhere in window.
    # Good conic windows have max_alpha_raw ≈ 0; only problematic regions exceed it.
    if max_alpha_raw >= _ALPHA_FALLBACK:
        pred = spline_func(ts)
        ctrl_err = np.max(np.linalg.norm(pred - pts, axis=1))
        return spline_func, ctrl_err, 'spline'

    # Regularize: smooth α to prefer slowly-varying weights
    alpha_smooth = uniform_filter1d(alpha_raw, size=11, mode='nearest')
    alpha_smooth = np.clip(alpha_smooth, 0, 1)

    # Build smooth interpolant for α(t)
    alpha_interp = PchipInterpolator(t_dense, alpha_smooth)

    # Classify method by peak α
    max_alpha = float(np.max(alpha_smooth))

    _alpha_d1 = alpha_interp.derivative(1)
    _alpha_d2 = alpha_interp.derivative(2)

    def blended_func(t):
        t = np.atleast_1d(t).astype(float)
        a = np.clip(alpha_interp(t), 0, 1)
        c = conic_func(t)
        s = spline_func(t)
        return c * (1 - a[:, None]) + s * a[:, None]

    # Exact derivatives via product rule:
    #   d/dt blend = (1-α)·c' + α·s' + α'·(s - c)
    #   d²/dt² blend = (1-α)·c'' + α·s'' + 2α'·(s' - c') + α''·(s - c)
    def _d1(t):
        t  = np.atleast_1d(t).astype(float)
        a  = np.clip(alpha_interp(t), 0, 1)[:, None]
        da = _alpha_d1(t)[:, None]
        c, s   = conic_func(t), spline_func(t)
        dc, ds = conic_func.d1(t), spline_func.d1(t)
        return dc * (1 - a) + ds * a + da * (s - c)

    def _d2(t):
        t   = np.atleast_1d(t).astype(float)
        a   = np.clip(alpha_interp(t), 0, 1)[:, None]
        da  = _alpha_d1(t)[:, None]
        d2a = _alpha_d2(t)[:, None]
        c,  s   = conic_func(t),    spline_func(t)
        dc, ds  = conic_func.d1(t), spline_func.d1(t)
        d2c,d2s = conic_func.d2(t), spline_func.d2(t)
        return (d2c * (1 - a) + d2s * a
                + 2 * da * (ds - dc)
                + d2a * (s - c))

    blended_func.d1 = _d1
    blended_func.d2 = _d2

    pred = blended_func(ts)
    ctrl_err = np.max(np.linalg.norm(pred - pts, axis=1))

    method = 'blend' if max_alpha >= 0.1 else 'conic'
    return blended_func, ctrl_err, method


def _arc_sagitta(arc, p0, p1):
    """Max perpendicular distance of arc points from chord p0→p1."""
    chord = p1 - p0
    cl = np.linalg.norm(chord)
    if cl < _EPS_CHORD:
        return 0.0
    perp = np.array([-chord[1], chord[0]]) / cl
    return float(np.max(np.abs((arc - p0) @ perp)))


def _build_arc_interpolant(pts, ts, center3d, e1, e2, pts_2d,
                           all_arc_pts, all_arc_s, s_ctrl, use_spline,
                           conic_coeffs=None):
    """Build conic orbit_func from arc-traced 2D points with s(t) mapping.

    Uses PchipInterpolator for s(t) — guarantees monotonicity since s_ctrl
    values are cumulative arc lengths (always increasing).

    conic_coeffs: optional (A,B,C,D,E,F) for f(lx,ly)=0 in the (e1,e2) frame.
    When provided, derivatives at control-point times use the algebraic
    conic gradient instead of the CubicSpline chain rule, avoiding the
    sub-arc junction artifact.

    Returns (orbit_func, ctrl_err) or None if arc tracing failed.
    """
    all_arc_s = np.array(all_arc_s)
    all_arc_pts = np.array(all_arc_pts)
    s_ctrl = np.array(s_ctrl)

    # PchipInterpolator preserves monotonicity of the strictly-increasing
    # s_ctrl data, eliminating the old binary monotonicity gate.
    s_of_t = PchipInterpolator(ts, s_ctrl)

    unique_mask = np.diff(all_arc_s, prepend=-1) > _EPS_CHORD
    arc_s_unique = all_arc_s[unique_mask]
    arc_pts_unique = all_arc_pts[unique_mask]

    # CubicSpline (replaces interp1d) gives exact .derivative() support.
    x_of_s = CubicSpline(arc_s_unique, arc_pts_unique[:, 0])
    y_of_s = CubicSpline(arc_s_unique, arc_pts_unique[:, 1])

    _s_lo, _s_hi = arc_s_unique[0], arc_s_unique[-1]

    def orbit_func(t):
        t = np.atleast_1d(t).astype(float)
        s = np.clip(s_of_t(t), _s_lo, _s_hi)
        lx = x_of_s(s)
        ly = y_of_s(s)
        pos = (center3d
               + lx[:, None] * e1[None, :]
               + ly[:, None] * e2[None, :])
        return pos

    # ── Pre-compute algebraic tangent & curvature at control-point times ──
    # The CubicSpline fitted to concatenated sub-arc data has wrong tangents
    # at junction times (where sub-arc k arrives ≠ sub-arc k+1 departs).
    # The algebraic conic gradient gives the exact tangent independently of
    # how the arc was sampled.
    #
    # For f(lx,ly) = A·lx²+B·lx·ly+C·ly²+D·lx+E·ly+F = 0:
    #   ∇f = (Gx, Gy) = (2A·lx+B·ly+D,  B·lx+2C·ly+E)
    #   Unit tangent:  τ = ±(-Gy, Gx)/|∇f|   (sign chosen for forward arc direction)
    #   Curvature vec: κ_vec = -2(A·Gy²-B·Gx·Gy+C·Gx²)/|∇f|⁴ · (Gx, Gy)
    #
    # Derivation of κ_vec: from d²f/ds²=0 along the arc,
    #   Gx·x''+Gy·y'' = -(2A·(x')²+2B·x'·y'+2C·(y')²)
    # Combined with (x'',y'') ⊥ τ, solving gives the formula above.
    _ctrl_tang = None   # shape (5, 2) unit tangent in (e1, e2) frame
    _ctrl_curv = None   # shape (5, 2) curvature vector d²(lx,ly)/ds²

    if conic_coeffs is not None:
        A, B, C, D, E, F = conic_coeffs
        _ctrl_tang = np.zeros((5, 2))
        _ctrl_curv = np.zeros((5, 2))
        for k in range(5):
            lx_k, ly_k = pts_2d[k]
            Gx = 2*A*lx_k + B*ly_k + D
            Gy = B*lx_k + 2*C*ly_k + E
            g  = np.sqrt(Gx**2 + Gy**2)
            if g < _EPS_GRAD:
                continue  # degenerate gradient — keep zeros
            tau = np.array([-Gy, Gx]) / g
            # Sign: forward direction (toward next control pt, or from prev)
            if k < 4:
                fwd = pts_2d[k + 1] - pts_2d[k]
            else:
                fwd = pts_2d[k] - pts_2d[k - 1]
            if tau @ fwd < 0:
                tau = -tau
            _ctrl_tang[k] = tau
            # Curvature vector = -2(A·Gy²-B·Gx·Gy+C·Gx²)/g⁴ · (Gx,Gy)
            S2 = A*Gy**2 - B*Gx*Gy + C*Gx**2
            _ctrl_curv[k] = (-2*S2 / g**4) * np.array([Gx, Gy])

    # Time-span for control-point proximity test
    _t_span = float(ts[-1] - ts[0])

    _x_ds1 = x_of_s.derivative(1)
    _y_ds1 = y_of_s.derivative(1)
    _x_ds2 = x_of_s.derivative(2)
    _y_ds2 = y_of_s.derivative(2)
    _s_dt1 = s_of_t.derivative(1)
    _s_dt2 = s_of_t.derivative(2)

    def _d1(t):
        t   = np.atleast_1d(t).astype(float)
        s   = np.clip(s_of_t(t), _s_lo, _s_hi)
        dsd = _s_dt1(t)
        result = (_x_ds1(s)[:, None] * e1[None, :] +
                  _y_ds1(s)[:, None] * e2[None, :]) * dsd[:, None]
        if _ctrl_tang is not None:
            for i, ti in enumerate(t):
                k_arr = np.where(np.abs(ts - ti) < _t_span * _EPS_TSPAN)[0]
                if len(k_arr):
                    tau = _ctrl_tang[k_arr[0]]
                    result[i] = (tau[0]*e1 + tau[1]*e2) * dsd[i]
        return result

    def _d2(t):
        t     = np.atleast_1d(t).astype(float)
        s     = np.clip(s_of_t(t), _s_lo, _s_hi)
        dsd   = _s_dt1(t)
        d2sd  = _s_dt2(t)
        tang  = (_x_ds1(s)[:, None] * e1[None, :] +
                 _y_ds1(s)[:, None] * e2[None, :])
        curv  = (_x_ds2(s)[:, None] * e1[None, :] +
                 _y_ds2(s)[:, None] * e2[None, :])
        result = curv * (dsd**2)[:, None] + tang * d2sd[:, None]
        if _ctrl_curv is not None:
            for i, ti in enumerate(t):
                k_arr = np.where(np.abs(ts - ti) < _t_span * _EPS_TSPAN)[0]
                if len(k_arr):
                    k    = k_arr[0]
                    tau  = _ctrl_tang[k]
                    cv   = _ctrl_curv[k]
                    result[i] = ((cv[0]*e1  + cv[1]*e2)  * dsd[i]**2
                                 + (tau[0]*e1 + tau[1]*e2) * d2sd[i])
        return result

    orbit_func.d1 = _d1
    orbit_func.d2 = _d2

    pred = orbit_func(ts)
    ctrl_err = np.max(np.linalg.norm(pred - pts, axis=1))
    return orbit_func, ctrl_err


def _ray_conic_intersect(coeffs, focus, theta, r_hint=None):
    """Intersect ray from focus at angle theta with the implicit conic.

    Ray: (x,y) = focus + r*(cos θ, sin θ)  for r > 0
    If r_hint is given, pick the positive root closest to r_hint
    (needed for hyperbolas where both branches may be hit).
    """
    A, B, C, D, E, F = coeffs
    fx, fy = focus
    ct, st = np.cos(theta), np.sin(theta)

    # Substitute x = fx + r*ct, y = fy + r*st into Ax²+Bxy+Cy²+Dx+Ey+F=0
    # Quadratic in r: a2*r² + a1*r + a0 = 0
    a2 = A * ct**2 + B * ct * st + C * st**2
    a1 = (2 * A * fx * ct + B * (fx * st + fy * ct) + 2 * C * fy * st
           + D * ct + E * st)
    a0 = A * fx**2 + B * fx * fy + C * fy**2 + D * fx + E * fy + F

    disc = a1**2 - 4 * a2 * a0
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    r1 = (-a1 + sq) / (2 * a2)
    r2 = (-a1 - sq) / (2 * a2)

    roots = [r for r in [r1, r2] if r > _EPS_CHORD]
    if not roots:
        return None
    if r_hint is not None and len(roots) > 1:
        # Pick the root closest to the hint distance
        return min(roots, key=lambda r: abs(r - r_hint))
    return min(roots)


def _try_kepler_time(pts_2d, pts, ts, center3d, e1, e2, conic_coeffs):
    """Try angle-from-focus parameterization on the exact conic.

    Uses the conic from tier 1 (exact through 5 points) with a focus as
    the angular origin.  At query time t, computes θ(t) via polynomial,
    then shoots a ray from the focus at angle θ and intersects with the
    exact implicit conic — giving an exact point on the curve.

    Avoids Kepler's equation entirely. Monotonicity comes from θ(t) being
    monotonic (checked explicitly).

    Returns (func, err, 'kepler') or None.
    """
    # Try conic_foci directly — skip classify_conic check so that
    # near-degenerate conics still get a chance.
    try:
        foci = conic_foci(conic_coeffs)
    except Exception:
        return None

    best = None
    best_err = np.inf

    for focus in foci:
        # Angles of control points from this focus
        dx = pts_2d[:, 0] - focus[0]
        dy = pts_2d[:, 1] - focus[1]
        r = np.sqrt(dx**2 + dy**2)

        if np.min(r) < 1e-10:
            continue

        theta = np.unwrap(np.arctan2(dy, dx))

        # Require monotonic angles
        dth = np.diff(theta)
        if not (np.all(dth > 0) or np.all(dth < 0)):
            continue

        # θ(t) via PCHIP: monotone-preserving cubic Hermite interpolation.
        # Unlike a degree-4 polynomial, PCHIP won't overshoot between knots
        # when the data is monotonic — exactly what we need here.
        th_of_t = PchipInterpolator(ts, theta)

        # Verify monotonicity (PCHIP preserves it for monotone data,
        # but check anyway in case of nearly-flat segments)
        t_check = np.linspace(ts[0], ts[-1], 50)
        th_check = th_of_t(t_check)
        dth_check = np.diff(th_check)
        if not (np.all(dth_check > 0) or np.all(dth_check < 0)):
            continue

        # Also interpolate r(t) for branch disambiguation
        r_of_t = PchipInterpolator(ts, r)

        # Build orbit_func via ray-conic intersection, with exact derivatives
        # attached via implicit differentiation of the conic polar equation.
        #
        # At angle θ, position is (fx + r cosθ, fy + r sinθ).
        # Differentiating the implicit conic f(x,y)=0 w.r.t. θ:
        #   Gx·(r'cosθ − r sinθ) + Gy·(r'sinθ + r cosθ) = 0
        #   where Gx = 2Ax+By+D, Gy = Bx+2Cy+E
        #   ⟹  r' = r·(Gx sinθ − Gy cosθ) / (Gx cosθ + Gy sinθ)
        #
        # Second derivative r'' follows from differentiating r' once more
        # using the same Gx,Gy and their θ-derivatives via dx/dθ, dy/dθ.
        def _polar_r_derivs(cc, fx, fy, th, r):
            """Return (r', r'') at a single (theta, r) point on the conic cc."""
            A, B, C, D, E, F = cc
            ct, st = np.cos(th), np.sin(th)
            x = fx + r * ct
            y = fy + r * st
            Gx = 2*A*x + B*y + D
            Gy = B*x + 2*C*y + E
            denom = Gx*ct + Gy*st
            if abs(denom) < _EPS_GRAD:
                return 0.0, 0.0          # tangent point – treat as straight
            N  = Gx*st - Gy*ct
            rp = r * N / denom           # r'(θ)
            # dx/dθ, dy/dθ
            dxdth = rp*ct - r*st
            dydth = rp*st + r*ct
            # dGx/dθ, dGy/dθ
            dGx = 2*A*dxdth + B*dydth
            dGy = B*dxdth + 2*C*dydth
            Np  = dGx*st + Gx*ct - dGy*ct + Gy*st   # dN/dθ
            Dp  = -Gx*st + dGx*ct + Gy*ct + dGy*st  # d(denom)/dθ
            rpp = (rp*N*denom + r*Np*denom - r*N*Dp) / denom**2
            return rp, rpp

        def _make_func(conic_coeffs, focus, th_of_t, r_of_t):
            _th_d1 = th_of_t.derivative(1)
            _th_d2 = th_of_t.derivative(2)

            def orbit_func(t):
                t = np.atleast_1d(t).astype(float)
                th = th_of_t(t)
                r_hint = np.abs(r_of_t(t))
                out = np.zeros((len(t), 3))
                for i in range(len(t)):
                    r_hit = _ray_conic_intersect(conic_coeffs, focus, th[i],
                                                  r_hint=r_hint[i])
                    if r_hit is not None:
                        lx = focus[0] + r_hit * np.cos(th[i])
                        ly = focus[1] + r_hit * np.sin(th[i])
                    else:
                        # Fallback: use interpolated r directly
                        lx = focus[0] + r_hint[i] * np.cos(th[i])
                        ly = focus[1] + r_hint[i] * np.sin(th[i])
                    out[i] = (center3d + lx * e1 + ly * e2)
                return out

            def _d1(t):
                # d/dt pos = (dx/dθ·e1 + dy/dθ·e2) · dθ/dt
                t   = np.atleast_1d(t).astype(float)
                th  = th_of_t(t)
                r_h = np.abs(r_of_t(t))
                dth = _th_d1(t)
                out = np.zeros((len(t), 3))
                for i in range(len(t)):
                    r_hit = _ray_conic_intersect(conic_coeffs, focus, th[i],
                                                  r_hint=r_h[i])
                    r_i = r_hit if r_hit is not None else r_h[i]
                    rp, _ = _polar_r_derivs(conic_coeffs, focus[0], focus[1],
                                            th[i], r_i)
                    ct, st = np.cos(th[i]), np.sin(th[i])
                    dxdth = rp*ct - r_i*st
                    dydth = rp*st + r_i*ct
                    out[i] = (dxdth*e1 + dydth*e2) * dth[i]
                return out

            def _d2(t):
                # d²/dt² pos = d²pos/dθ² · (dθ/dt)² + dpos/dθ · d²θ/dt²
                t    = np.atleast_1d(t).astype(float)
                th   = th_of_t(t)
                r_h  = np.abs(r_of_t(t))
                dth  = _th_d1(t)
                d2th = _th_d2(t)
                out  = np.zeros((len(t), 3))
                for i in range(len(t)):
                    r_hit = _ray_conic_intersect(conic_coeffs, focus, th[i],
                                                  r_hint=r_h[i])
                    r_i = r_hit if r_hit is not None else r_h[i]
                    rp, rpp = _polar_r_derivs(conic_coeffs, focus[0], focus[1],
                                              th[i], r_i)
                    ct, st   = np.cos(th[i]), np.sin(th[i])
                    dxdth    = rp*ct  - r_i*st
                    dydth    = rp*st  + r_i*ct
                    d2xdth2  = rpp*ct - 2*rp*st  - r_i*ct
                    d2ydth2  = rpp*st + 2*rp*ct  - r_i*st
                    out[i]   = ((d2xdth2*e1 + d2ydth2*e2) * dth[i]**2
                                + (dxdth *e1 + dydth *e2) * d2th[i])
                return out

            orbit_func.d1 = _d1
            orbit_func.d2 = _d2
            return orbit_func

        orbit_func = _make_func(conic_coeffs, focus, th_of_t, r_of_t)
        pred = orbit_func(ts)
        ctrl_err = np.max(np.linalg.norm(pred - pts, axis=1))

        if ctrl_err > 0.1:
            continue

        # Sanity: check dense interpolation doesn't wander far from chords.
        # Evaluate at midpoints between consecutive control points and
        # check that the interpolated point stays within a reasonable
        # distance of the chord midpoint.
        t_mids = (ts[:-1] + ts[1:]) / 2
        mid_pts = orbit_func(t_mids)
        chord_mids = (pts[:-1] + pts[1:]) / 2
        chord_lens = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        mid_devs = np.linalg.norm(mid_pts - chord_mids, axis=1)
        # Reject if midpoints wander more than 30% of chord length.
        # Near-collinear points (e.g. Lissajous crossing) produce extreme
        # conics that pass through the 5 control points but diverge wildly
        # in between — the chord-deviation ratio catches this.
        max_ratio = np.max(mid_devs / np.maximum(chord_lens, _EPS_CHORD))
        if max_ratio > _ALPHA_FALLBACK:
            continue

        if ctrl_err < best_err:
            best_err = ctrl_err
            best = (orbit_func, ctrl_err)

    if best is None or best_err > 0.1:
        return None
    return best[0], best[1], 'kepler'


def _try_gen_polar(pts_2d, pts, ts, center3d, e1, e2, use_spline):
    """Try fitting generalized polar curve.  Returns (func, err, 'polar') or None."""
    result = _fit_gen_polar(pts_2d)
    if result is None:
        return None

    coeffs, k_val, polar_center, thetas = result

    # Trace arc segments, snapping endpoints to actual control points
    all_arc_pts = []
    all_arc_s = []
    s_ctrl = [0.0]
    s_running = 0.0

    for seg_k in range(4):
        seg = _trace_gen_polar(coeffs, k_val, polar_center,
                               thetas[seg_k], thetas[seg_k + 1])
        seg[0] = pts_2d[seg_k]
        seg[-1] = pts_2d[seg_k + 1]

        diffs = np.diff(seg, axis=0)
        seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)

        for i in range(len(seg)):
            all_arc_pts.append(seg[i])
            all_arc_s.append(s_running)
            if i < len(seg_lens):
                s_running += seg_lens[i]

        s_ctrl.append(s_running)

    built = _build_arc_interpolant(pts, ts, center3d, e1, e2, pts_2d,
                                   all_arc_pts, all_arc_s, s_ctrl, use_spline)
    if built is None:
        return None
    return built[0], built[1], 'polar'


def _canonical_k0(xs, ys, A, B, C, D, E, candidates=None):
    """Return (k0, L_all, M_all) — canonical P₀ selection for stereographic map.

    k0 = argmin_k |L_k / M_k|  (control point with smallest tangent slope |L/M|
    from P₀, i.e. most horizontal tangent).  This minimises the chance that
    consecutive s-values straddle an asymptote direction (s → ±∞), which
    would cause a spurious φ ≈ ±π/2 jump in PCHIP between same-branch points.

    L_k = 2A·x_k + B·y_k + D   (x-component of conic gradient at each point)
    M_k = B·x_k + 2C·y_k + E   (y-component of conic gradient at each point)

    candidates: optional list of indices to restrict the argmin to.  When both
    overlapping windows for a blend segment restrict to their shared inner-3
    points (indices [1,2,3] of the 5-point window), they evaluate |L/M| at the
    same physical points and — for data on a single conic — choose the same P₀.
    Same P₀ + same phi_c → identical orbit functions in the shared blend region
    → blend stays exactly on the conic.

    Used by both `_build_projective_arc_window` and `_is_conic_monotone`
    (local-frame mode) so that both functions choose the same P₀.
    """
    L  = 2.0*A*xs + B*ys + D
    M  = B*xs + 2.0*C*ys + E
    Ms = np.where(np.abs(M) >= 1e-15, M, np.sign(M + _SIGN_NUG) * 1e-15)
    ratios = np.abs(L / Ms)
    if candidates is not None:
        k0 = candidates[int(np.argmin(ratios[candidates]))]
    else:
        k0 = int(np.argmin(ratios))
    return k0, L, M


def _build_projective_arc_window(pts_2d, pts, ts, center3d, e1, e2, coeffs):
    """Evaluate exact conic via rational stereographic parameterisation from P₀.

    For a point P₀ = (x₀, y₀) on the conic, every line through P₀ with slope s
    meets the conic in exactly one other point, giving the rational map:
        u(s) = −(L + s·M) / Q(s),   x = x₀ + u,   y = y₀ + s·u
    where L = 2A·x₀+B·y₀+D, M = B·x₀+2C·y₀+E (gradient at P₀),
    and Q(s) = A + B·s + C·s² (quadratic form in direction (1,s)).

    Q(s) = 0 at the two asymptote slopes from P₀; u → ∞ there.  For cross-branch
    windows the arc passes through ∞, so the PCHIP parameter is arctan(s − s*)
    (Case 1: finite asymptote s* in the s range of control points) or
    arctan(v − v*) with v = 1/s (Case 2: asymptote approached from s → ∞).
    Both give bounded parameters and produce NaN in the blend region, which
    adaptive_n_budget detects as inf deviation and resolves with more points.

    Cross-branch detection uses transverse-axis eigenvectors (unchanged from old
    code).  Same-branch windows return a dummy (orbit_func is unused in that
    code path of _try_conic).  All evaluation is vectorised.

    Returns (orbit_func, ctrl_err, is_cross_branch) or None if:
      - ctrl_err > _EPS_CTRL (bad fit), or
      - orbit_func produces NaN in the blend region (asymptote inside window).
    """
    A, B, C, D, E, F = coeffs

    # ── Conic center + cross-branch detection (eigenvector method, unchanged) ──
    det = 4.0 * A * C - B * B
    if abs(det) == 0.0:
        return None            # exactly singular — can't compute center
    # Near-parabola fits (det ≈ machine-epsilon) produce a large but finite
    # cx, cy.  This is harmless: eigvals[0]*eigvals[1] ≈ 0 → is_hyperbola=False,
    # so cx/cy are never used in the cross-branch path.  The stereographic map
    # proceeds normally (Q ≈ constant for a near-parabola → exact arc).
    cx = (-2.0 * C * D + B * E) / det
    cy = (-2.0 * A * E + B * D) / det

    dx = pts_2d[:, 0] - cx
    dy = pts_2d[:, 1] - cy

    M_mat = np.array([[A, B / 2.0], [B / 2.0, C]])
    eigvals, eigvecs = np.linalg.eigh(M_mat)
    is_hyperbola = eigvals[0] * eigvals[1] < 0

    if is_hyperbola:
        pos_idx = 1 if eigvals[1] > 0 else 0
        e_t = eigvecs[:, pos_idx]
        proj_eig = dx * e_t[0] + dy * e_t[1]
        branch = np.sign(proj_eig)
        # A control point exactly on the transverse axis gives branch=0.
        # We could guard here (`if np.any(branch == 0): return None`), but it
        # is unnecessary: Q(s) = 0 at the axis direction → u → ∞ → NaN in the
        # orbit, which propagates to ctrl_err = NaN → `not np.isfinite(ctrl_err)`
        # → return None a few lines below.  The NaN does the right thing.
        cross = (branch[:4] * branch[1:]) < 0
        is_cross_branch = bool(cross.any())
    else:
        is_cross_branch = False

    # ── Rotate to the conic's principal frame (B'=0) ─────────────────────────
    # In the principal frame every window fitting the same conic uses identical
    # coefficients (up to sign), the same phi_c = 0 (automatic since B'=0),
    # and the same vertex P₀ (where L=0, an intrinsic conic property independent
    # of which 5 points are sampled).  Same frame + same P₀ + same phi_c →
    # identical orbit functions for both overlapping windows → blend exactly on
    # the conic for locally-conic data.
    #
    # The SVD local frame is kept for the cross-branch eigenvector test above
    # (which already uses local-frame coefficients) and for the final
    # global-coordinate conversion in orbit_func.
    M_pf = np.array([[A, B / 2.0], [B / 2.0, C]])
    evals_pf, evecs_pf = np.linalg.eigh(M_pf)
    # Sort axes by |eigenvalue| — consistent assignment across sign-negated fits.
    idx_pf = np.argsort(np.abs(evals_pf))
    evals_pf  = evals_pf[idx_pf]
    evecs_pf  = evecs_pf[:, idx_pf]
    # Sign convention: physical first principal axis has positive x-component
    # in the 2D plane (dot product with e1).  This is globally consistent across
    # all windows on the same conic: e1 varies per window, but the physical
    # eigenvector direction is the same → same vertex → same phi at shared knots.
    pts_p_raw = pts_2d @ evecs_pf
    phys_x = float(evecs_pf[0, 0] * e1[0] + evecs_pf[1, 0] * e2[0])
    phys_y = float(evecs_pf[0, 0] * e1[1] + evecs_pf[1, 0] * e2[1])
    if phys_x < 0 or (abs(phys_x) < _EPS_DET and phys_y < 0):
        evecs_pf[:, 0] = -evecs_pf[:, 0]
        pts_p_raw[:, 0] = -pts_p_raw[:, 0]
    # Maintain right-hand orientation so the swap flag behaves consistently.
    if np.linalg.det(evecs_pf) < 0:
        evecs_pf[:, 1] = -evecs_pf[:, 1]
        pts_p_raw[:, 1] = -pts_p_raw[:, 1]
    # Principal-frame conic coefficients (B_p = 0 by construction).
    A_p   = float(evals_pf[0])
    C_p   = float(evals_pf[1])
    DE    = np.array([D, E])
    D_p   = float(DE @ evecs_pf[:, 0])
    E_p   = float(DE @ evecs_pf[:, 1])

    # ── Select orbit: eccentric-anomaly (ellipse) or principal-vertex (other) ──
    # evals_pf[0]*evals_pf[-1] > 0  ↔  same-sign eigenvalues  ↔  ellipse.
    # Hyperbolas (opposite-sign) and parabolas (one zero eigenvalue) fall
    # through to the vertex-P₀ stereographic path below.

    swapped = False

    # ── Ellipse path: eccentric-anomaly orbit — no P₀ needed ────────────────
    # Geometric context: the three conic types each have a natural intrinsic
    # parametrization in the principal frame:
    #   Ellipse  (det > 0):  E = arctan2((y−cy)/b, (x−cx)/a)   [eccentric anomaly]
    #                        x = cx + a·cos(E),  y = cy + b·sin(E)
    #   Parabola (det = 0):  D = t  (linear in arc-length near vertex)
    #                        x = x_v + t,  y = y_v + t²/(2p)
    #   Hyperbola(det < 0):  phi = 2·arctan(s − s₀)             [stereographic]
    #                        x = x₀+u(s), y = y₀+s·u(s)  (rational / projective)
    # All three are related: D = tan(E/2) at the parabola limit (e→1), and the
    # stereographic phi for hyperbolas is the imaginary-angle analogue of E.
    # The center (cx,cy) is intrinsic for ellipses: all windows on the same
    # ellipse compute the same value → same E at every shared knot → consistent
    # orbits → machine-epsilon blend for exact-conic input.
    # E "pre-scales" by semi-axes before arctan2, removing the angular
    # compression at the tips of elongated ellipses; for circles (a=b), E
    # equals the central angle θ exactly.
    #
    # Guard: eval_sep < _EVAL_SEP_NEAR restricts the E orbit to near-circles.  This is a
    # COMPUTATION guard, not a correctness gate — for elongated same-branch
    # ellipses the E orbit would be discarded by _try_conic anyway (it falls
    # through to the arc-length quality check; only _near_circle windows get
    # the early return that bypasses that check).  The guard just avoids
    # computing an orbit that will not be used.  The load-bearing threshold is
    # _near_circle in _try_conic, not this one.
    abs_evals = np.abs(evals_pf)
    if (evals_pf[0] * evals_pf[-1] > 0                          # ellipse
            and (abs_evals[-1] - abs_evals[0])
                / (abs_evals[-1] + abs_evals[0] + _SIGN_NUG) < _EVAL_SEP_NEAR):  # near-circular
        if abs(A_p) > _EPS_DET and abs(C_p) > _EPS_DET:
            cx_p  = -D_p / (2.0 * A_p)
            cy_p  = -E_p / (2.0 * C_p)
            Fc_p  = F - D_p**2 / (4.0 * A_p) - E_p**2 / (4.0 * C_p)
            # Fc_p and A_p have opposite signs iff center is inside the ellipse.
            if Fc_p * A_p < 0.0:
                a2 = -Fc_p / A_p;  b2 = -Fc_p / C_p   # semi-axes squared
                if a2 > 0 and b2 > 0:
                    a_e = np.sqrt(a2);  b_e = np.sqrt(b2)
                    ecc_raw  = np.arctan2((pts_p_raw[:, 1] - cy_p) / b_e,
                                         (pts_p_raw[:, 0] - cx_p) / a_e)
                    ecc_u    = np.unwrap(ecc_raw)
                    ecc_mono = (np.all(np.diff(ecc_u) > 0)
                                or np.all(np.diff(ecc_u) < 0))
                    if ecc_mono:
                        ecc_of_t  = PchipInterpolator(ts, ecc_u)
                        _decc     = ecc_of_t.derivative(1)
                        _d2ecc    = ecc_of_t.derivative(2)
                        _cx_p, _cy_p   = cx_p, cy_p
                        _a_e, _b_e     = a_e, b_e
                        _evpf          = evecs_pf

                        def orbit_func(t):
                            t   = np.atleast_1d(t).astype(float)
                            ec  = ecc_of_t(t)
                            xp  = _cx_p + _a_e * np.cos(ec)
                            yp  = _cy_p + _b_e * np.sin(ec)
                            loc = np.column_stack([xp, yp]) @ _evpf.T
                            return center3d + loc[:, 0:1] * e1 + loc[:, 1:2] * e2

                        def _d1(t):
                            t   = np.atleast_1d(t).astype(float)
                            ec  = ecc_of_t(t);  dec = _decc(t)
                            dxp = -_a_e * np.sin(ec) * dec
                            dyp =  _b_e * np.cos(ec) * dec
                            loc = np.column_stack([dxp, dyp]) @ _evpf.T
                            return loc[:, 0:1] * e1 + loc[:, 1:2] * e2

                        def _d2(t):
                            t    = np.atleast_1d(t).astype(float)
                            ec   = ecc_of_t(t)
                            dec  = _decc(t);  d2ec = _d2ecc(t)
                            d2xp = -_a_e * np.cos(ec) * dec**2 - _a_e * np.sin(ec) * d2ec
                            d2yp = -_b_e * np.sin(ec) * dec**2 + _b_e * np.cos(ec) * d2ec
                            loc  = np.column_stack([d2xp, d2yp]) @ _evpf.T
                            return loc[:, 0:1] * e1 + loc[:, 1:2] * e2

                        orbit_func.d1 = _d1;  orbit_func.d2 = _d2
                        pred     = orbit_func(ts)
                        ctrl_err = float(np.nanmax(np.linalg.norm(pred - pts, axis=1)))
                        if np.isfinite(ctrl_err) and ctrl_err <= _EPS_CTRL:
                            return orbit_func, ctrl_err, is_cross_branch

    # ── Principal-frame vertex P₀ path (general / fallback) ──────────────────
    # The vertex (L=0) is an intrinsic conic property → consistent P₀ across
    # all windows on the same physical conic → exact blend for non-circular
    # conic data.
    pts_e  = pts_p_raw.copy()
    A_e, C_e, D_e, E_e = A_p, C_p, D_p, E_p
    B_e    = 0.0           # B_p = 0 in principal frame

    def _try_vertex(Ae, Ce, De, Ee, pts_loc):
        """Return (x_v, y_v) where L=0, or None if degenerate."""
        if abs(Ae) < _EPS_COEFF:
            return None
        x_v = -De / (2.0 * Ae)
        const_v = F - De**2 / (4.0 * Ae)
        if abs(Ce) < _EPS_COEFF:
            if abs(Ee) < _EPS_COEFF:
                return None
            y_v = -const_v / Ee
        else:
            disc = Ee**2 - 4.0 * Ce * const_v
            if disc < 0:
                return None
            sqrt_d = np.sqrt(disc)
            y1 = (-Ee + sqrt_d) / (2.0 * Ce)
            y2 = (-Ee - sqrt_d) / (2.0 * Ce)
            cy = float(np.mean(pts_loc[:, 1]))
            y_v = y1 if abs(y1 - cy) <= abs(y2 - cy) else y2
        return x_v, y_v

    # Try vertex in current frame; swap x↔y if A_e too small (near-parabola).
    v_result = None
    _has_vertex = False
    if abs(A_e) >= _EPS_SWAP * max(abs(C_e), 1.0):
        v_result = _try_vertex(A_e, C_e, D_e, E_e, pts_e)
    if v_result is None:
        A_e, C_e = C_e, A_e
        D_e, E_e = E_e, D_e
        pts_e = pts_e[:, ::-1]
        swapped = True
        if abs(A_e) >= _EPS_SWAP * max(abs(C_e), 1.0):
            v_result = _try_vertex(A_e, C_e, D_e, E_e, pts_e)

    if v_result is not None:
        x0, y0 = v_result
        L_e = 0.0                        # vertex: L = 2A_e·x_v + D_e = 0
        M_e = 2.0 * C_e * y0 + E_e      # gradient M at vertex
        if abs(M_e) >= _EPS_COEFF:
            _has_vertex = True

    if not _has_vertex:
        # Fallback: inner-3 argmin canonical P₀
        _INNER3 = [1, 2, 3]
        k0, L_all, M_all = _canonical_k0(pts_e[:, 0], pts_e[:, 1], A_e, B_e, C_e, D_e, E_e,
                                          candidates=_INNER3)
        x0, y0 = pts_e[k0]
        L_e, M_e = L_all[k0], M_all[k0]
        if abs(M_e) < _EPS_COEFF:
            return None            # still degenerate after swap

    s0 = -L_e / M_e   # = 0 for vertex (L_e=0), or tangent slope for fallback
    dxi = pts_e[:, 0] - x0
    dyi = pts_e[:, 1] - y0
    eps_dx = _EPS_SWAP * (np.abs(dyi) + 1.0)
    mask = np.abs(dxi) >= eps_dx
    dxi_safe = np.where(mask, dxi, 1.0)
    s_vals = np.where(mask, dyi / dxi_safe, np.sign(dyi + _SIGN_NUG) * 1e15)
    if not _has_vertex:
        s_vals[k0] = s0   # exact tangent slope at argmin P₀ (vertex has no k0)

    # In the principal frame B_p = 0, so phi_c = −B_p/(2C_e) = 0 exactly.
    phi_c = 0.0

    # ── Phi-first: try φ = 2·arctan(s − phi_c) + np.unwrap ──────────────────
    # Works for same-branch windows (Case 0, np.unwrap is a no-op) and genuine
    # cross-branch windows where s_vals cross the asymptote cleanly (Case 1,
    # np.unwrap bridges the ±2π jump).
    # Falls back to angle-from-center θ when phi is non-monotone AND the window
    # is labelled cross-branch.  Non-monotone phi arises when:
    #   (a) Q(s_vals) has no sign change — false-positive cross-branch label
    #       (near-parabolic hyperbola; previously "Case 3"); or
    #   (b) s_vals have an outlier slope far from the rest — the ±2π unwrap
    #       jump does not restore monotonicity.
    # See tests/diag_unified_param.py (benchmark) and
    #     tests/diag_case1_between_knots.py (orbit quality analysis).
    phi_vals = np.unwrap(2.0 * np.arctan(s_vals - phi_c))
    phi_mono = np.all(np.diff(phi_vals) > 0) or np.all(np.diff(phi_vals) < 0)

    if not phi_mono and is_cross_branch:
        # ── DEAD CODE ──────────────────────────────────────────────────────────
        # _try_conic now calls _is_conic_monotone(pts_2d, coeffs=coeffs) as a
        # pre-filter before calling this function.  That pre-filter uses the
        # same P₀ selection and φ computation as below, so phi_mono is
        # guaranteed True on every call that reaches here.  This branch
        # (non-monotone φ + cross-branch → theta orbit) is therefore
        # unreachable in normal operation.
        # Kept for reference; remove once the pre-filter has been in production
        # for a full scan-baseline cycle.
        # φ non-monotone + cross-branch: circle-projection fails.
        # Use angle-from-center θ(t) — globally consistent across adjacent
        # windows, robust to all hyperbola geometries.
            f0 = A*cx**2 + B*cx*cy + C*cy**2 + D*cx + E*cy + F
            # f0 ≈ 0 means the conic center lies on the conic itself — geometrically
            # impossible for a proper ellipse/hyperbola but numerically reachable.
            # We could guard here (`if abs(f0) < _EPS_COEFF: return None`), but it is
            # unnecessary: ratio = −f0/a ≈ 0 → r = 0 → orbit collapses to the
            # center → ctrl_err = max distance from center to control points >>
            # _EPS_CTRL → return None below.  The downstream check does the right thing.
            theta_raw = np.arctan2(dy, dx)
            theta_u   = theta_raw.copy().astype(float)
            for i in range(4):
                step = theta_u[i + 1] - theta_u[i]
                step = (step + np.pi) % (2 * np.pi) - np.pi
                theta_u[i + 1] = theta_u[i] + step
            if cross.any():
                same_diffs = np.diff(theta_u)[~cross]
                # len(same_diffs) == 0 (all 4 pairs cross-branch) would give
                # np.sum([]) = 0 → overall_dir = 0, caught two lines below.
                overall_dir = float(np.sign(np.sum(same_diffs)))
                if overall_dir == 0:
                    # Zero net theta direction: cross-branch correction is undefined.
                    # We could let the orbit proceed, but ctrl_err >> _EPS_CTRL would
                    # catch the ambiguous result.  Returning early is cheaper.
                    return None
                theta_u = theta_raw.copy().astype(float)
                for i in range(4):
                    step = theta_u[i + 1] - theta_u[i]
                    step = (step + np.pi) % (2 * np.pi) - np.pi
                    if cross[i] and float(np.sign(step)) != overall_dir:
                        step += overall_dir * 2 * np.pi
                    theta_u[i + 1] = theta_u[i] + step
            th_of_t = PchipInterpolator(ts, theta_u)
            _th_d1  = th_of_t.derivative(1)
            _th_d2  = th_of_t.derivative(2)
            _f0 = f0;  _cx = cx;  _cy = cy;  _Ab = A;  _Bb = B;  _Cb = C

            def orbit_func(t):
                t   = np.atleast_1d(t).astype(float)
                th  = th_of_t(t)
                ct, st = np.cos(th), np.sin(th)
                a    = _Ab*ct**2 + _Bb*ct*st + _Cb*st**2
                valid = np.abs(a) >= _EPS_COEFF
                ratio = np.where(valid, -_f0 / a, np.nan)
                r    = np.where(ratio > 0, np.sqrt(ratio), np.nan)
                lx   = _cx + r * ct;  ly = _cy + r * st
                return center3d + lx[:, None] * e1 + ly[:, None] * e2

            def _d1(t):
                t   = np.atleast_1d(t).astype(float)
                th  = th_of_t(t);  dth = _th_d1(t)
                ct, st = np.cos(th), np.sin(th)
                a    = _Ab*ct**2 + _Bb*ct*st + _Cb*st**2
                valid = np.abs(a) >= _EPS_COEFF
                a_s  = np.where(valid, a, 1.0)
                ratio = np.where(valid, -_f0 / a_s, np.nan)
                r    = np.where(ratio > 0, np.sqrt(ratio), np.nan)
                ap   = (_Cb - _Ab)*np.sin(2*th) + _Bb*np.cos(2*th)
                rp   = np.where(valid, -r * ap / (2.0 * a_s), np.nan)
                dxdth = rp*ct - r*st
                dydth = rp*st + r*ct
                return (dxdth[:, None]*e1 + dydth[:, None]*e2) * dth[:, None]

            def _d2(t):
                t    = np.atleast_1d(t).astype(float)
                th   = th_of_t(t);  dth = _th_d1(t);  d2th = _th_d2(t)
                ct, st = np.cos(th), np.sin(th)
                a    = _Ab*ct**2 + _Bb*ct*st + _Cb*st**2
                valid = np.abs(a) >= _EPS_COEFF
                a_s  = np.where(valid, a, 1.0)
                ratio = np.where(valid, -_f0 / a_s, np.nan)
                r    = np.where(ratio > 0, np.sqrt(ratio), np.nan)
                ap   = (_Cb - _Ab)*np.sin(2*th) + _Bb*np.cos(2*th)
                app  = 2*(_Cb - _Ab)*np.cos(2*th) - 2*_Bb*np.sin(2*th)
                rp   = np.where(valid, -r * ap / (2.0 * a_s), np.nan)
                rpp  = np.where(valid,
                                r * (3.0*ap**2 / (4.0*a_s**2) - app / (2.0*a_s)),
                                np.nan)
                dxdth   = rp*ct - r*st
                dydth   = rp*st + r*ct
                d2xdth2 = rpp*ct - 2*rp*st - r*ct
                d2ydth2 = rpp*st + 2*rp*ct - r*st
                return ((d2xdth2[:, None]*e1 + d2ydth2[:, None]*e2) * dth[:, None]**2
                        + (dxdth[:, None]*e1 + dydth[:, None]*e2) * d2th[:, None])

            orbit_func.d1 = _d1;  orbit_func.d2 = _d2
            pred     = orbit_func(ts)
            # nanmax: NaN values (asymptote in blend region) → ctrl_err=inf → return None → spline fallback.
            ctrl_err = float(np.nanmax(np.linalg.norm(pred - pts, axis=1)))
            if not np.isfinite(ctrl_err) or ctrl_err > _EPS_CTRL:
                return None
            return orbit_func, ctrl_err, True

    # ── Phi is monotone: stereographic phi orbit ──────────────────────────────
    phi_of_t  = PchipInterpolator(ts, phi_vals)
    dphi_dt   = phi_of_t.derivative(1)
    d2phi_dt2 = phi_of_t.derivative(2)

    # ── Capture immutable closure variables ───────────────────────────────────
    _A_e, _B, _C_e, _L_e, _M_e = A_e, B_e, C_e, L_e, M_e   # B_e=0 in principal frame
    _x0, _y0 = x0, y0
    _swapped  = swapped
    _s0       = phi_c   # centre used in φ map; s = tan(φ/2) + phi_c
    _V        = evecs_pf  # principal frame → SVD local frame

    # ── s from phi: s = tan(φ/2) + phi_c  (inverse of φ = 2·arctan(s − phi_c)) ─
    def _s_from_phi(phi):
        return np.tan(phi / 2.0) + _s0

    # ── ds/dt = ½·sec²(φ/2)·dφ/dt ────────────────────────────────────────────
    def _dsdt(phi, dphi):
        tan_h = np.tan(phi / 2.0)
        return 0.5 * (1.0 + tan_h**2) * dphi

    # ── d²s/dt² = ½·sec²(φ/2)·[tan(φ/2)·(dφ/dt)² + d²φ/dt²] ────────────────
    def _d2sdt2(phi, dphi, d2phi):
        tan_h  = np.tan(phi / 2.0)
        sec2_h = 1.0 + tan_h**2
        return 0.5 * sec2_h * (tan_h * dphi**2 + d2phi)

    # ── Rational conic eval + derivatives w.r.t. s (vectorized) ──────────────
    def _xy_derivs(s_arr):
        """Return (x2d, y2d, dx/ds, dy/ds, d²x/ds², d²y/ds²) for array s_arr."""
        Q    = _A_e + _B * s_arr + _C_e * s_arr**2
        valid = np.abs(Q) >= _EPS_DET
        Q_s  = np.where(valid, Q, 1.0)
        u    = np.where(valid, -(_L_e + s_arr * _M_e) / Q_s, np.nan)
        x2d  = _x0 + u
        y2d  = _y0 + s_arr * u
        dQds = _B + 2.0 * _C_e * s_arr
        num  = _M_e * Q_s - (_L_e + s_arr * _M_e) * dQds
        duds = np.where(valid, -num / Q_s**2, np.nan)
        dxds = duds
        dyds = u + s_arr * duds
        dnumds  = -(_L_e + s_arr * _M_e) * (2.0 * _C_e)
        d2uds2  = np.where(valid,
                           -(dnumds * Q_s - num * 2.0 * dQds) / Q_s**3,
                           np.nan)
        d2xds2  = d2uds2
        d2yds2  = 2.0 * duds + s_arr * d2uds2
        return x2d, y2d, dxds, dyds, d2xds2, d2yds2

    # ── Vectorized orbit_func, _d1, _d2 ──────────────────────────────────────
    def orbit_func(t):
        t   = np.atleast_1d(t).astype(float)
        phi = phi_of_t(t)
        s_a = _s_from_phi(phi)
        x2d, y2d, _, _, _, _ = _xy_derivs(s_a)
        if _swapped:
            x2d, y2d = y2d, x2d
        # principal frame → SVD local frame:  [lx, ly] = [x_p, y_p] @ V.T
        lx = x2d * _V[0, 0] + y2d * _V[0, 1]
        ly = x2d * _V[1, 0] + y2d * _V[1, 1]
        return center3d + lx[:, None] * e1 + ly[:, None] * e2

    def _d1(t):
        t    = np.atleast_1d(t).astype(float)
        phi  = phi_of_t(t);  dphi = dphi_dt(t)
        s_a  = _s_from_phi(phi);  dsdt = _dsdt(phi, dphi)
        _, _, dxds, dyds, _, _ = _xy_derivs(s_a)
        dxdt = dxds * dsdt;  dydt = dyds * dsdt
        if _swapped:
            dxdt, dydt = dydt, dxdt
        dxl = dxdt * _V[0, 0] + dydt * _V[0, 1]
        dyl = dxdt * _V[1, 0] + dydt * _V[1, 1]
        return dxl[:, None] * e1 + dyl[:, None] * e2

    def _d2(t):
        t     = np.atleast_1d(t).astype(float)
        phi   = phi_of_t(t);  dphi = dphi_dt(t);  d2phi = d2phi_dt2(t)
        s_a   = _s_from_phi(phi);  dsdt = _dsdt(phi, dphi)
        d2s   = _d2sdt2(phi, dphi, d2phi)
        _, _, dxds, dyds, d2xds2, d2yds2 = _xy_derivs(s_a)
        d2xdt2 = d2xds2 * dsdt**2 + dxds * d2s
        d2ydt2 = d2yds2 * dsdt**2 + dyds * d2s
        if _swapped:
            d2xdt2, d2ydt2 = d2ydt2, d2xdt2
        d2xl = d2xdt2 * _V[0, 0] + d2ydt2 * _V[0, 1]
        d2yl = d2xdt2 * _V[1, 0] + d2ydt2 * _V[1, 1]
        return d2xl[:, None] * e1 + d2yl[:, None] * e2

    orbit_func.d1 = _d1
    orbit_func.d2 = _d2

    # ── Validate control-point reconstruction ─────────────────────────────────
    # The rational map is exact on the conic; ctrl_err should be ≈ machine ε.
    # NaN values (asymptote crossing) are excluded by nanmax.
    pred = orbit_func(ts)
    ctrl_err = float(np.nanmax(np.linalg.norm(pred - pts, axis=1)))
    if not np.isfinite(ctrl_err) or ctrl_err > _EPS_CTRL:
        return None

    return orbit_func, ctrl_err, is_cross_branch


def _try_conic(pts_2d, pts, ts, center3d, e1, e2, use_spline, N_order=2, coeffs=None):
    """Fit exact conic and blend with spline based on disagreement.

    Cross-branch path: phi-monotone windows → `_build_projective_arc_window`
    → rational stereographic arc (phi-orbit).  Returned directly because
    `_blended_conic_spline` would see NaN near the asymptote and fall back.

    Same-branch path: arc-length arc tracer (Phase 1) → `_blended_conic_spline`.
    Arc-length parameterisation agrees more closely with the natural spline,
    giving lower α → more conic contribution → better accuracy at low n.

    Fallback: pure arc-length path for parabolas (det≈0 → projective arc
    disabled), or when phi is non-monotone (_is_conic_monotone pre-filter
    returns False → proj = None).

    Returns (func, err, method) where method is 'conic'/'blend'/'spline'.
    Returns None only if the conic can't be fit at all.
    """
    if coeffs is None:
        try:
            coeffs = fit_conic(pts_2d)
        except Exception:
            return None

    # ── Cross-branch: projective arc ──────────────────────────────────────────
    # Pre-filter: skip projective arc if φ = 2·arctan(s−s₀) is non-monotone.
    # Exception: near-circles (eval_sep < _EVAL_SEP_NEAR, both eigenvalues same sign) use
    # an eccentric-anomaly orbit in _build_projective_arc_window rather than
    # stereographic phi, so stereographic monotonicity is irrelevant for them.
    # _near_circle is the load-bearing quality gate: it controls whether the
    # proj result gets an early return (bypassing the arc-length quality check)
    # at line ~1570.  The matching eval_sep < _EVAL_SEP_NEAR guard inside
    # _build_projective_arc_window is redundant for correctness — it just saves
    # computing an E orbit that would otherwise be discarded here.
    _ev_raw = np.linalg.eigvalsh(np.array([[coeffs[0], coeffs[1]/2.0],
                                           [coeffs[1]/2.0, coeffs[2]]]))
    _ev_abs = np.sort(np.abs(_ev_raw))
    _near_circle = (
        _ev_raw[0] * _ev_raw[-1] > 0
        and (_ev_abs[-1] - _ev_abs[0]) / (_ev_abs[-1] + _ev_abs[0] + _SIGN_NUG) < _EVAL_SEP_NEAR
    )
    if _near_circle or _is_conic_monotone(pts_2d, coeffs=coeffs):
        proj = _build_projective_arc_window(pts_2d, pts, ts, center3d, e1, e2, coeffs)
    else:
        proj = None
    if proj is not None:
        proj_func, proj_err, is_cross_branch = proj
        if is_cross_branch:
            # Cross-branch: projective arc is the geometrically correct arc
            # for these 5 control points.  NaN in the blend region means the
            # conic passes through ∞ there — this is correct for the given
            # control points.  adaptive_n_budget sees inf deviation and adds
            # more control points until the asymptote leaves the blend region.
            return proj_func, proj_err, 'conic'
        if _near_circle and np.isfinite(proj_err) and proj_err <= _EPS_CTRL:
            # Near-circle same-branch: eccentric-anomaly orbit is intrinsic
            # (same center and semi-axes for all windows on the same ellipse)
            # → machine-epsilon blend for exact-conic input.
            return proj_func, proj_err, 'conic'
        # Same-branch (non-near-circle): arc-length path preferred.
        # Fall through to arc-length code below.

    # ── Arc-length path (same-branch and parabola) ────────────────────────────
    # Phase 1: collect segments with initial_tang threading ──────────────────
    consistent_tang = None   # threaded across sub-arcs for direction consistency
    segs = []
    for k in range(4):
        try:
            seg, new_tang = _trace_conic_arc(
                coeffs, pts_2d[k], pts_2d[k + 1],
                initial_tang=consistent_tang)
        except Exception:
            # Arc tracing failed (e.g. extreme conic near collinear points).
            # Substitute a straight line segment — the disagreement-based α
            # will shift toward the spline in this region.
            seg = np.linspace(pts_2d[k], pts_2d[k + 1], 20)
            new_tang = None  # reset; next sub-arc picks direction freely
        segs.append(seg)
        consistent_tang = new_tang

    # ── Phase 2 (removed): zig-zag / cross-branch detection ──────────────────
    # Previously, when the arc tracer zig-zagged (snapped to the wrong branch),
    # this phase triggered _build_conic_tangent_spline ('conic_hermite' window).
    # The conic_hermite tangent approach is wrong for curves that reverse at a
    # finite point (e.g. Van der Pol spike): the asymptote-directed tangent bends
    # the Hermite toward infinity.
    #
    # Design decision: cross-branch arcs are projectively valid (the hyperbola
    # is a closed curve in RP²).  _is_conic_monotone now correctly identifies
    # projectively-monotone cross-branch windows (including the asymptote-gap fix
    # for steps > 180°).  Monotone windows proceed to Phase 3; genuinely
    # non-monotone windows are replaced by a 5-pt spline in _run_blend.
    # _build_conic_tangent_spline is retained as dead code.

    # ── Phase 3: build arc data from final segments ────────────────────────────
    all_arc_pts = []
    all_arc_s = []
    s_ctrl = [0.0]
    s_running = 0.0
    for k in range(4):
        seg = segs[k]
        diffs = np.diff(seg, axis=0)
        seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        for i in range(len(seg)):
            all_arc_pts.append(seg[i])
            all_arc_s.append(s_running)
            if i < len(seg_lens):
                s_running += seg_lens[i]
        s_ctrl.append(s_running)

    built = _build_arc_interpolant(pts, ts, center3d, e1, e2, pts_2d,
                                   all_arc_pts, all_arc_s, s_ctrl, use_spline,
                                   conic_coeffs=coeffs)
    if built is None:
        return None

    conic_func = built[0]
    return _blended_conic_spline(pts, ts, center3d, e1, e2, pts_2d, conic_func)


def fit_conic_5pt(points, times, use_spline=False, N_order=2):
    """Fit curve through 5 3D points.

    Primary approach: conic+spline blend with disagreement-based α(t).
    The conic contributes where it agrees with the spline; the spline
    takes over where the conic diverges. No binary fallback gate.

    Remaining tiers (kepler, polar, pure spline) handle edge cases
    where the conic can't even be fit (degenerate geometry).
    """
    pts = np.asarray(points, dtype=float)
    ts = np.asarray(times, dtype=float)
    center = pts.mean(axis=0)

    centered = pts - center
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    # Canonicalize: e1 must point in the forward traversal direction (pts[0]→pts[4]).
    # numpy.svd may orient e1 either way; if it points backward, centered-chord
    # sign references in _build_conic_tangent_spline choose the wrong tangent sign.
    if (centered[-1] - centered[0]) @ e1 < 0:
        e1 = -e1
    pts_2d = np.column_stack([centered @ e1, centered @ e2])

    # ── Exact conic (needed by multiple tiers) ─────────────────────
    try:
        conic_coeffs = fit_conic(pts_2d)
    except Exception:
        conic_coeffs = None

    # Tier 1: conic+spline blend (primary approach)
    if conic_coeffs is not None:
        result = _try_conic(pts_2d, pts, ts, center, e1, e2, use_spline,
                            N_order=N_order, coeffs=conic_coeffs)
        if result is not None:
            return result

    # ── DEAD CODE: Tier 2 (kepler angle-from-focus) and Tier 3 (gen polar) ──────
    # Neither tier has ever been triggered by any of the 8 test curves in CURVES
    # or by the analytic-conic extras (ellipse, hyperbola, parabola).
    # Tier 1 (_try_conic) always either succeeds or falls back to pure spline;
    # the quality guard / _blended_conic_spline cost function produces a spline
    # result (method='spline') before tiers 2–3 are reached.
    # Left here for reference; safe to remove in a future cleanup pass.
    # ─────────────────────────────────────────────────────────────────────────────
    if False:  # noqa: SIM210
        if conic_coeffs is not None:
            result = _try_kepler_time(pts_2d, pts, ts, center, e1, e2, conic_coeffs)
            if result is not None:
                return result
        result = _try_gen_polar(pts_2d, pts, ts, center, e1, e2, use_spline)
        if result is not None:
            return result

    # Tier 4: pure cubic spline (always works)
    func = _build_spline_func(ts, center, e1, e2, pts_2d)
    pred = func(ts)
    ctrl_err = np.max(np.linalg.norm(pred - pts, axis=1))
    return func, ctrl_err, 'spline'


def _is_conic_monotone(pts5_xy, coeffs=None):
    """Return True if the 5 points lie on their fitted conic in monotonic order.

    Two operating modes:

    **Local-frame mode** (``coeffs`` provided — called as pre-filter from
    ``_try_conic``):
      Uses the same canonical-P₀ selection and φ = 2·arctan(s − s₀)
      computation as ``_build_projective_arc_window``, so the verdict here
      matches ``phi_mono`` there exactly.  Returns True iff φ is monotone
      (after np.unwrap) — guaranteeing that ``_build_projective_arc_window``
      will take the phi path and never reach the theta fallback.

    **Principal-frame mode** (``coeffs=None`` — post-filter in ``_run_blend``):
      Works in the conic's own principal-axes frame (B'=0) so the result is
      invariant under any rigid motion (rotation, reflection, translation) of
      the 5 input points — including self-similar curves like the log spiral.

      Steps:
        1. Fit conic; diagonalise M = [[A, B/2],[B/2, C]] via eigh.
        2. Rotate all 5 points to the principal frame.
        3. Run the rational stereographic algorithm in that frame.

      Q(s) = A' + C'·s²  changes sign at each asymptote direction from P₀.
      Monotone iff n_cross ≤ 1 AND all consecutive s-diffs have the same sign.

    Both modes handle near-vertical tangent at P₀ via x↔y swap.
    Returns False if the conic fit or eigendecomposition is degenerate.
    """
    pts5_xy = np.asarray(pts5_xy, dtype=float)

    # ── Local-frame mode: mirror _build_projective_arc_window's phi_mono ──────
    # Rotate to the conic's principal frame (B'=0).  Use conic vertex as P₀ so
    # the phi_mono verdict matches _build_projective_arc_window exactly.
    # Sign convention: forward-traversal is OK here because with vertex P₀ and
    # B_p=0, Q(s)=A_p+C_p·s² is symmetric → phi_mono is sign-invariant.
    if coeffs is not None:
        A, B, C, D, E, F = coeffs
        M_pf = np.array([[A, B / 2.0], [B / 2.0, C]])
        try:
            evals_pf, evecs_pf = np.linalg.eigh(M_pf)
        except np.linalg.LinAlgError:
            return False
        idx_pf   = np.argsort(np.abs(evals_pf))
        evals_pf = evals_pf[idx_pf]
        evecs_pf = evecs_pf[:, idx_pf]
        pts_p    = pts5_xy @ evecs_pf
        if pts_p[-1, 0] - pts_p[0, 0] < 0:
            evecs_pf[:, 0] = -evecs_pf[:, 0];  pts_p[:, 0] = -pts_p[:, 0]
        if np.linalg.det(evecs_pf) < 0:
            evecs_pf[:, 1] = -evecs_pf[:, 1];  pts_p[:, 1] = -pts_p[:, 1]
        A_p = float(evals_pf[0]);  C_p = float(evals_pf[1])
        DE  = np.array([D, E])
        D_p = float(DE @ evecs_pf[:, 0]);  E_p = float(DE @ evecs_pf[:, 1])

        xs, ys = pts_p[:, 0].copy(), pts_p[:, 1].copy()
        A_e, C_e, D_e, E_e = A_p, C_p, D_p, E_p
        swapped = False

        def _try_v(Ae, Ce, De, Ee, ys_loc):
            if abs(Ae) < _EPS_COEFF:
                return None
            x_v = -De / (2.0 * Ae);  const_v = F - De**2 / (4.0 * Ae)
            if abs(Ce) < _EPS_COEFF:
                return None if abs(Ee) < _EPS_COEFF else (x_v, -const_v / Ee)
            disc = Ee**2 - 4.0 * Ce * const_v
            if disc < 0:
                return None
            sq = np.sqrt(disc)
            y1 = (-Ee + sq) / (2.0 * Ce);  y2 = (-Ee - sq) / (2.0 * Ce)
            cy = float(np.mean(ys_loc))
            return x_v, (y1 if abs(y1 - cy) <= abs(y2 - cy) else y2)

        v_result = None
        if abs(A_e) >= _EPS_SWAP * max(abs(C_e), 1.0):
            v_result = _try_v(A_e, C_e, D_e, E_e, ys)
        if v_result is None:
            A_e, C_e = C_e, A_e;  D_e, E_e = E_e, D_e;  xs, ys = ys, xs
            swapped = True
            if abs(A_e) >= _EPS_SWAP * max(abs(C_e), 1.0):
                v_result = _try_v(A_e, C_e, D_e, E_e, ys)

        has_vertex = False
        if v_result is not None:
            x0, y0 = v_result
            M_v = 2.0 * C_e * y0 + E_e
            if abs(M_v) >= _EPS_COEFF:
                has_vertex = True;  s0 = 0.0  # L=0 at vertex

        if not has_vertex:
            k0, L_all, M_all = _canonical_k0(xs, ys, A_e, 0.0, C_e, D_e, E_e,
                                              candidates=[1, 2, 3])
            L_e, M_e = L_all[k0], M_all[k0];  x0, y0 = xs[k0], ys[k0]
            if abs(M_e) < _EPS_COEFF:
                return False
            s0 = -L_e / M_e

        dxi = xs - x0;  dyi = ys - y0
        eps_dx   = _EPS_SWAP * (np.abs(dyi) + 1.0)
        mask     = np.abs(dxi) >= eps_dx
        dxi_safe = np.where(mask, dxi, 1.0)
        s        = np.where(mask, dyi / dxi_safe, np.sign(dyi + _SIGN_NUG) * 1e15)
        if not has_vertex:
            s[k0] = s0
        phi  = np.unwrap(2.0 * np.arctan(s))   # phi_c = 0 (B_p=0, vertex s0=0)
        d    = np.diff(phi)
        return bool(np.all(d > 0) or np.all(d < 0))

    # ── Principal-frame mode (post-filter in _run_blend) ──────────────────────
    try:
        A, B, C, D, E, F = fit_conic(pts5_xy)
    except Exception:
        return False

    # Near-circle fast path: central angle arctan2(y-cy, x-cx) replaces
    # stereographic phi in _build_projective_arc_window for these windows.
    # Central angle is monotone for any proper arc < 2π, so we return True
    # immediately and let the post-filter pass through without replacement.
    _ev_nc = np.linalg.eigvalsh(np.array([[A, B*0.5], [B*0.5, C]]))
    _ev_nc_abs = np.sort(np.abs(_ev_nc))
    if (_ev_nc[0] * _ev_nc[-1] > 0
            and (_ev_nc_abs[-1] - _ev_nc_abs[0])
                / (_ev_nc_abs[-1] + _ev_nc_abs[0] + _SIGN_NUG) < _EVAL_SEP_NEAR):
        det_nc = 4.0*A*C - B*B
        if abs(det_nc) > _EPS_DET:
            cx_nc = (B*E - 2*C*D) / det_nc;  cy_nc = (B*D - 2*A*E) / det_nc
            th_nc = np.arctan2(pts5_xy[:, 1] - cy_nc, pts5_xy[:, 0] - cx_nc)
            th_nc = np.unwrap(th_nc)
            d_nc  = np.diff(th_nc)
            return bool(np.all(d_nc > 0) or np.all(d_nc < 0))

    # ── Rotate to conic's principal axes ──────────────────────────────────────
    # eigh returns eigenvalues sorted ascending and orthonormal eigenvectors as
    # columns.  x'_k = pts·v1 = pts@evecs[:,0],  y'_k = pts@evecs[:,1].
    # In the rotated frame: A'=λ₁, B'=0, C'=λ₂, [D',E']=[D,E]@evecs, F'=F.
    #
    # Sort by |λ|, not λ.  fit_conic may return a conic or its sign-negation
    # (same zero-set, different normalisation).  For M and −M, ascending-λ sort
    # maps different eigenvectors to slot 0 (smallest +λ vs. most-negative −λ),
    # swapping the principal-frame axes and causing asymmetric verdicts for
    # geometrically-equivalent windows (e.g. the Rose-curve mirror pairs).
    # Ascending-|λ| sort always puts the same physical eigenvector in slot 0.
    M_mat = np.array([[A, B * 0.5], [B * 0.5, C]])
    try:
        evals, evecs = np.linalg.eigh(M_mat)
    except np.linalg.LinAlgError:
        return False
    sort_idx = np.argsort(np.abs(evals))
    evals    = evals[sort_idx]
    evecs    = evecs[:, sort_idx]

    xs_r = pts5_xy @ evecs[:, 0]        # x' coordinates
    ys_r = pts5_xy @ evecs[:, 1]        # y' coordinates
    A_r  = float(evals[0])              # λ₁  (B_r = 0 by construction)
    C_r  = float(evals[1])              # λ₂
    D_r, E_r = np.array([D, E]) @ evecs # linear terms in principal frame

    # ── P₀: point with smallest |L'/M'| in the principal frame, inner-3 ────────
    # L'_k = 2A'·x'_k + D',  M'_k = 2C'·y'_k + E'  (no B' term)
    # Restrict to indices [1,2,3] so P₀ lies in the shared inner-3 overlap zone
    # of both windows covering each blend segment — matches _build_projective_arc_window.
    Lk    = 2.0 * A_r * xs_r + D_r
    Mk    = 2.0 * C_r * ys_r + E_r
    ratio = np.where(np.abs(Mk) > _EPS_COEFF, np.abs(Lk) / np.abs(Mk), 1e30)
    k0    = [1, 2, 3][int(np.argmin(ratio[[1, 2, 3]]))]
    x0, y0 = xs_r[k0], ys_r[k0]
    L, M  = Lk[k0], Mk[k0]

    # ── x'↔y' swap if tangent at P₀ is nearly vertical in principal frame ────
    if abs(M) < _EPS_SWAP * max(abs(L), 1.0):
        A_r, C_r = C_r, A_r
        D_r, E_r = E_r, D_r
        xs_r, ys_r = ys_r, xs_r
        x0, y0 = xs_r[k0], ys_r[k0]
        L = 2.0 * A_r * x0 + D_r
        M = 2.0 * C_r * y0 + E_r

    if abs(M) < _EPS_COEFF:
        return False                        # degenerate even after swap

    s0 = -L / M

    # ── Stereographic slopes in parameter order ────────────────────────────────
    dxi      = xs_r - x0
    dyi      = ys_r - y0
    eps_dx   = _EPS_SWAP * (np.abs(dyi) + 1.0)
    mask     = np.abs(dxi) >= eps_dx
    dxi_safe = np.where(mask, dxi, 1.0)
    s        = np.where(mask, dyi / dxi_safe, np.sign(dyi + _SIGN_NUG) * 1e15)
    s[k0]    = s0                           # replace self-slope with tangent

    # Q(s) = A' + C'·s²   (B'=0 by construction)
    Qs = A_r + C_r * s**2

    if np.any(np.abs(Qs) < 1e-10):
        return False                        # control point on asymptote direction

    Q_signs = np.sign(Qs)
    n_cross = int(np.sum(np.diff(Q_signs) != 0))

    if n_cross >= 2:
        return False                        # multiple asymptote crossings

    diffs = np.diff(s)
    return bool(np.all(diffs > 0) or np.all(diffs < 0))


def _win_deriv(f, t, h):
    """Central-difference first derivative (velocity) of window function f at t."""
    fp = f(np.array([t + h]))[0]
    fm = f(np.array([t - h]))[0]
    return (fp - fm) / (2 * h)


def _win_deriv2(f, t, h):
    """Central-difference second derivative of window function f at t."""
    fp = f(np.array([t + h]))[0]
    fc = f(np.array([t]))[0]
    fm = f(np.array([t - h]))[0]
    return (fp - 2*fc + fm) / h**2


# ── DEAD CODE ─────────────────────────────────────────────────────────────────
# _make_quintic_hermite was the original per-segment fallback for segments where
# either flanking conic window was non-monotone.  It matched position, velocity,
# and acceleration from both flanking windows at the two segment endpoints, giving
# C² continuity but only over a 2-point (t0,t1) domain — not a window function.
# It was replaced by _make_conic_clamped_spline (see below) which used 4 points
# and algebraic conic-gradient BCs, and then superseded entirely when we switched
# to replacing non-monotone windows with 5-point natural spline windows (method
# 'spline') so that ALL segments use the same smoothstep blend.  The smoothstep
# guarantee makes per-segment fallbacks unnecessary: C^N continuity at every knot
# is automatic because both adjacent segments share the same window function there.
def _make_quintic_hermite(t0, t1, P0, P1, D1_l, D2_l, D1_r, D2_r):
    """Quintic Hermite segment matching position + velocity + acceleration at both endpoints.

    Gives C^2 continuity with the smoothstep-N=2 conic blend at every knot:
    both use the flanking window's pos/vel/accel at the shared endpoint, so
    derivatives agree by construction regardless of window type.
    """
    dt = t1 - t0
    dt2 = dt * dt

    def hermite(t_arr):
        s = (np.atleast_1d(t_arr) - t0) / dt
        s2, s3, s4, s5 = s**2, s**3, s**4, s**5
        h00 = 1 - 10*s3 + 15*s4 - 6*s5
        h10 =     s -  6*s3 +  8*s4 - 3*s5
        h20 = 0.5*(s2 - 3*s3 +  3*s4 - s5)
        h01 =       10*s3 - 15*s4 + 6*s5
        h11 =      -4*s3 +  7*s4 - 3*s5
        h21 = 0.5*(s3 - 2*s4 + s5)
        return (h00[:, None]*P0 + h10[:, None]*(D1_l*dt) + h20[:, None]*(D2_l*dt2)
              + h01[:, None]*P1 + h11[:, None]*(D1_r*dt) + h21[:, None]*(D2_r*dt2))

    return hermite


# ── DEAD CODE ─────────────────────────────────────────────────────────────────
# _make_conic_clamped_spline was an intermediate approach between the quintic
# Hermite fallback (above) and the current design.  It built a cubic spline
# through 4 surrounding control points [j-2, j-1, j, j+1] with algebraic
# conic-gradient BCs at both ends — giving better geometric continuity than the
# quintic Hermite while being independent of arc-tracer quality.  However, it
# still routed failing segments through a separate seg_funcs dict that bypassed
# the smoothstep blend, causing kinks (7–8°) at the spl↔blend transition knots.
# The correct fix was to replace non-monotone conic windows with 5-point spline
# windows (_make_spline_window) so all segments use the same smoothstep blend,
# eliminating kinks by construction.  See _make_spline_window above.
def _make_conic_clamped_spline(j, pts, times):
    """Cubic spline for segment j (pts[j]→pts[j+1]) through pts[j-2:j+2].

    Interpolates the 4 surrounding control points [j-2, j-1, j, j+1] with
    conic-gradient first-derivative BCs at both endpoints of that array:

      • Left  BC at pts[j-2]:  conic tangent from 5-pt window pts[j-2:j+3]
      • Right BC at pts[j+1]:  conic tangent from 5-pt window pts[j-1:j+4]

    The gradient [-Gy, Gx] at each endpoint is computed from the algebraic
    implicit-conic fit (fit_conic), independent of arc-tracer quality.
    Sign is chosen to agree with the local chord direction; magnitude is
    scaled by chord_length / dt so the BC is in position/time units.

    Falls back to natural (zero-curvature) BC at an endpoint if the conic
    fit or gradient is degenerate.

    Returns f(t_arr) → ndarray of shape (N, pts.shape[1]).
    """
    p4 = pts[j-2:j+2, :2].astype(float)   # (4, 2) world XY
    t4 = times[j-2:j+2]                   # (4,)
    ncols = pts.shape[1]

    def _conic_bc(p5_world, pt_idx, chord_a, chord_b, t_a, t_b):
        """Conic-gradient velocity at p5_world[pt_idx] in world XY.

        chord_a, chord_b: indices into p5_world for the reference chord
          direction.  Centered chord → p5_world[chord_b] - p5_world[chord_a].
        t_a, t_b: corresponding times (for chord/dt speed estimate).
        Returns (dx_dt, dy_dt) or None on failure.
        """
        try:
            ctr = p5_world.mean(axis=0)
            _, _, Vt = np.linalg.svd(p5_world - ctr, full_matrices=False)
            e1, e2 = Vt[0], Vt[1]
            p5_2d = np.column_stack([(p5_world - ctr) @ e1,
                                     (p5_world - ctr) @ e2])
            A, B, C, D, E, F = fit_conic(p5_2d)
            lx, ly = p5_2d[pt_idx]
            Gx = 2*A*lx + B*ly + D
            Gy = B*lx + 2*C*ly + E
            g = np.sqrt(Gx*Gx + Gy*Gy)
            if g < _EPS_GRAD:
                return None
            tau = np.array([-Gy, Gx]) / g        # unit conic tangent (local)
            ref = p5_2d[chord_b] - p5_2d[chord_a]  # chord reference (local)
            if float(tau @ ref) < 0:
                tau = -tau
            chord_len = np.linalg.norm(ref)
            dt_chord  = abs(t_b - t_a)
            speed = chord_len / dt_chord if dt_chord > _EPS_GRAD else 1.0
            tau_scaled = tau * speed              # velocity in local SVD coords
            return tau_scaled[0]*e1[:2] + tau_scaled[1]*e2[:2]   # world XY
        except Exception:
            return None

    # LEFT BC at pts[j-2] = index 0 of window pts[j-2:j+3]
    # Reference: one-sided chord pts[j-1] - pts[j-2] (forward)
    p5_L = pts[j-2:j+3, :2].astype(float)
    bc_L = _conic_bc(p5_L, pt_idx=0,
                     chord_a=0, chord_b=1,
                     t_a=times[j-2], t_b=times[j-1])

    # RIGHT BC at pts[j+1] = index 2 of window pts[j-1:j+4]
    # Reference: centered chord pts[j+2] - pts[j]
    p5_R = pts[j-1:j+4, :2].astype(float)
    bc_R = _conic_bc(p5_R, pt_idx=2,
                     chord_a=1, chord_b=3,
                     t_a=times[j], t_b=times[j+2])

    left_bc_x  = (1, float(bc_L[0])) if bc_L is not None else (2, 0.0)
    left_bc_y  = (1, float(bc_L[1])) if bc_L is not None else (2, 0.0)
    right_bc_x = (1, float(bc_R[0])) if bc_R is not None else (2, 0.0)
    right_bc_y = (1, float(bc_R[1])) if bc_R is not None else (2, 0.0)

    x_spl = CubicSpline(t4, p4[:, 0], bc_type=(left_bc_x, right_bc_x))
    y_spl = CubicSpline(t4, p4[:, 1], bc_type=(left_bc_y, right_bc_y))

    def seg_func(t_arr):
        t = np.atleast_1d(t_arr).astype(float)
        xy = np.column_stack([x_spl(t), y_spl(t)])
        if ncols > 2:
            return np.column_stack([xy, np.zeros((len(t), ncols - 2))])
        return xy

    return seg_func


def blend_curve(pts, times, windows, middle_start, middle_end, N_order,
                N_dense=80):
    """Build blended dense curve for given smoothstep order.

    Every segment uses the C^N smoothstep blend:
        (1 − smoothstep(s)) · w[j−2](t) + smoothstep(s) · w[j−1](t)

    Non-monotone windows are replaced with 5-point spline windows in
    `_run_blend` before this function is called, so all segments use
    the same formula and C^N continuity holds everywhere.
    """
    all_dense = []
    all_interp = []
    for j in range(middle_start, middle_end):
        t_dense = np.linspace(times[j], times[j + 1], N_dense)
        t_seg   = np.linspace(times[j], times[j + 1], 10)

        wA = windows[j - 2]
        wB = windows[j - 1]
        sd = (t_dense - times[j]) / (times[j + 1] - times[j])
        wd = smoothstep(sd, N_order)
        dA, dB = wA(t_dense), wB(t_dense)
        all_dense.append(dA * (1 - wd)[:, None] + dB * wd[:, None])
        s = (t_seg - times[j]) / (times[j + 1] - times[j])
        w = smoothstep(s, N_order)
        pA, pB = wA(t_seg), wB(t_seg)
        all_interp.append(pA * (1 - w)[:, None] + pB * w[:, None])

    return np.vstack(all_dense), np.vstack(all_interp)


def sample_curve(xy_func, t_range, n):
    """Sample n equally-spaced points from a parametric curve."""
    t = np.linspace(t_range[0], t_range[1], n)
    x, y = xy_func(t)
    pts = np.column_stack([x, y, np.zeros(n)])
    times = np.linspace(0, 1, n)
    return pts, times

def _conic_distance(coeffs, x, y):
    """Approximate distance from point (x,y) to conic Ax²+Bxy+Cy²+Dx+Ey+F=0."""
    A, B, C, D, E, F = coeffs
    fval = A*x**2 + B*x*y + C*y**2 + D*x + E*y + F
    gx = 2*A*x + B*y + D
    gy = B*x + 2*C*y + E
    gnorm = np.sqrt(gx**2 + gy**2)
    if gnorm > _EPS_GRAD:
        return np.abs(fval) / gnorm
    return np.abs(fval)


def _count_fallbacks(xy_func, t_range, n):
    """Fit all 5-pt windows for n sample points, return spline fallback count.
    Only counts windows that fall all the way to spline (tier 4)."""
    pts, times = sample_curve(xy_func, t_range, n)
    n_fb = 0
    for i in range(n - 4):
        _, _, method = fit_conic_5pt(pts[i:i+5], times[i:i+5])
        if method == 'spline':
            n_fb += 1
    return n_fb, n - 4


def adaptive_n(xy_func, t_range, max_chord_turn_deg=120, max_window_turn_deg=180,
               max_accel_turn_deg=90, max_fallback_frac=0.05):
    """[Legacy] Geometry-based adaptive sampling — superseded by `adaptive_n_budget` in blend_demo.py.

    `adaptive_n_budget` is the live path: it binary-searches
    on n until the curvature-scaled deviation max(||blended−truth||·κ) meets a
    threshold.  This function instead enforces five geometric heuristics on the
    control-point spacing itself, without ever evaluating the blended curve.

    Kept for reference.  Not called by any scan-baseline curve or test script.

    Constraints checked:
    1. Per-point chord turn ≤ max_chord_turn_deg
    2. Per-window chord turn sum ≤ max_window_turn_deg
    3. Per-segment tangent sweep ≤ max_chord_turn_deg (anti-aliasing)
    4. Per-segment acceleration turn ≤ max_accel_turn_deg (magnitude-weighted)
    5. Fallback rate: no more than max_fallback_frac of 5-pt windows
       trigger the spline fallback (non-monotonic s(t))
    """
    max_chord = np.radians(max_chord_turn_deg)
    max_win = np.radians(max_window_turn_deg)
    max_accel = np.radians(max_accel_turn_deg)

    # Dense sampling for tangent and acceleration checks
    n_dense = 5000
    t_dense = np.linspace(t_range[0], t_range[1], n_dense)
    xd, yd = xy_func(t_dense)
    dt = t_dense[1] - t_dense[0]

    tang_dense = np.unwrap(np.arctan2(np.diff(yd), np.diff(xd)))

    d2x = np.diff(xd, n=2) / dt**2
    d2y = np.diff(yd, n=2) / dt**2
    accel_mag = np.sqrt(d2x**2 + d2y**2)
    accel_ang = np.unwrap(np.arctan2(d2y, d2x))
    accel_mag_thresh = np.median(accel_mag) * 0.1

    for n in range(7, 500):
        t = np.linspace(t_range[0], t_range[1], n)
        x, y = xy_func(t)

        # ── Constraint 1: chord turn ─────────────────────────────────
        chords = np.column_stack([np.diff(x), np.diff(y)])
        chord_ang = np.unwrap(np.arctan2(chords[:, 1], chords[:, 0]))
        turns = np.abs(np.diff(chord_ang))

        if np.max(turns) > max_chord:
            continue

        # ── Constraint 2: window chord turn sum ──────────────────────
        ok = True
        for k in range(len(turns) - 2):
            if turns[k] + turns[k+1] + turns[k+2] > max_win:
                ok = False
                break
        if not ok:
            continue

        idx = np.searchsorted(t_dense, t).clip(0, n_dense - 2)

        # ── Constraint 3: dense tangent sweep ────────────────────────
        for k in range(len(idx) - 1):
            seg = tang_dense[idx[k]:idx[k+1]+1]
            if len(seg) > 1 and (np.max(seg) - np.min(seg)) > max_chord:
                ok = False
                break
        if not ok:
            continue

        for k in range(len(idx) - 4):
            win = tang_dense[idx[k]:idx[k+4]+1]
            if len(win) > 1 and (np.max(win) - np.min(win)) > max_win:
                ok = False
                break
        if not ok:
            continue

        # ── Constraint 4: acceleration direction ─────────────────────
        idx_a = idx.clip(0, len(accel_ang) - 1)
        for k in range(len(idx_a) - 1):
            sl = slice(idx_a[k], idx_a[k+1]+1)
            seg_a = accel_ang[sl]
            seg_m = accel_mag[sl]
            if len(seg_a) < 2:
                continue
            sig = seg_m > accel_mag_thresh
            if np.sum(sig) < 2:
                continue
            sig_ang = seg_a[sig]
            if (np.max(sig_ang) - np.min(sig_ang)) > max_accel:
                ok = False
                break
        if not ok:
            continue

        # ── Constraint 5: fallback rate ──────────────────────────────
        n_fb, n_win = _count_fallbacks(xy_func, t_range, n)
        if n_win > 0 and n_fb / n_win > max_fallback_frac:
            continue

        return n
    return 500


def _run_blend(pts, times, use_spline, N_order):
    """Fit 5-point conic windows to the input samples and blend them into a
    continuous orbit interpolant.

    Returns (dense, interp, costs, windows, ms, me, methods).

    ── Architecture ──────────────────────────────────────────────────────────
    Input: p[0]..p[n] with n ≥ 5 (at least 6 control points).

    Each interior segment p[j] → p[j+1]  (j in [ms, me) = [2, n-2))
    is covered by TWO overlapping 5-point windows that together span
    6 consecutive control points  p[j-2]..p[j+3]:

        wA  =  window[j-2]  fits a conic through  p[j-2], p[j-1], p[j],
                                                   p[j+1], p[j+2]
               (2 points strictly before p[j], 1 point strictly after p[j+1])

        wB  =  window[j-1]  fits a conic through  p[j-1], p[j],   p[j+1],
                                                   p[j+2], p[j+3]
               (1 point strictly before p[j], 2 points strictly after p[j+1])

    Both windows therefore "know" the same interval end-points p[j] and
    p[j+1], and together they provide matching position, tangent, and
    curvature (from the analytical conic gradient) at both endpoints.

    ── Conic blend (preferred path) ─────────────────────────────────────────
    When both wA and wB pass `_is_conic_monotone` (their 5 control points
    lie in the correct traversal order on the conic arc, with no reversal),
    the segment is rendered as a C^N smoothstep blend:

        p(t) = smoothstep(s) · wB(t) + (1 − smoothstep(s)) · wA(t)

    The blend is C^N at both endpoints because both windows agree on
    position, tangent, and curvature there (N_order derivatives matched).

    ── Monotonicity and spline replacement ──────────────────────────────────
    After all windows are fitted, `_is_conic_monotone` is applied to each
    window's 5 control points (principal-frame post-filter).  Any window
    whose φ traversal is non-monotone is replaced in-place with a 5-point
    natural cubic spline (`_make_spline_window`) before blending begins.
    Spline windows are always considered monotone (True by definition).

    Cross-branch hyperbola windows use the rational stereographic (phi-path)
    projective arc.  If the conic passes through ∞ in the blend region the
    orbit function returns NaN there; `_find_worst_interval` converts NaN
    deviation to ∞, driving `adaptive_n_budget` to add more control points
    until the asymptote leaves the blend region.  Non-monotone cross-branch
    windows fall back to 5-point spline via the post-filter above.

    After monotonicity replacement, ALL windows use the smoothstep blend
    formula — there is no per-segment quintic Hermite fallback.  C^N
    continuity at every knot is guaranteed because the same window function
    appears on both sides of each knot.
    """
    n_windows = len(pts) - 4
    windows = []
    costs = []
    methods = []
    for i in range(n_windows):
        p5 = pts[i:i+5]
        t5 = times[i:i+5]
        orb_func, cost, method = fit_conic_5pt(p5, t5, use_spline=use_spline,
                                                N_order=N_order)
        windows.append(orb_func)
        costs.append(cost)
        methods.append(method)

    # Angle-monotonicity flags: True if the 5-pt conic visits points in order
    # (projective sense, including cross-branch hyperbola traversals).
    mono_ok = [_is_conic_monotone(pts[i:i+5, :2]) for i in range(n_windows)]
    # Spline windows don't have arc-tracing issues — always treat as monotone.
    for i in range(n_windows):
        if methods[i] == 'spline':
            mono_ok[i] = True

    # Replace genuinely non-monotone windows with 5-point natural cubic spline.
    # The smoothstep blend formula is identical for conic and spline windows:
    # (1-w)*wA(t) + w*wB(t).  C^N continuity at every knot is guaranteed by
    # construction — both adjacent segments share the same window function.
    for i in range(n_windows):
        if not mono_ok[i]:
            windows[i] = _make_spline_window(pts[i:i+5], times[i:i+5])
            methods[i] = 'spline'
            mono_ok[i] = True

    middle_start = 2
    middle_end   = len(pts) - 3

    # All windows are now well-behaved; always use the smoothstep blend.
    dense, interp = blend_curve(pts, times, windows,
                                middle_start, middle_end, N_order)
    return (dense, interp, costs, windows, middle_start, middle_end,
            methods)


# ── Public API ────────────────────────────────────────────────────────────────
# Only these names are exported by "from conicspline import *".
# Internal helpers (_run_blend, _try_conic, …) remain importable explicitly,
# e.g. "from conicspline import _run_blend", but do not pollute a user's namespace.

__all__ = [
    # Core blending workflow
    'sample_curve',       # sample n uniformly-spaced control points
    'fit_conic_5pt',      # fit one 5-point window → (func, cost, method)
    'blend_curve',        # render the blended interpolant densely

    # Smoothstep blending kernel
    'smoothstep',

    # Conic fitting
    'fit_conic',
]
