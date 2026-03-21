"""
blend_demo.py — demo / visualization for the conicspline engine.

Defines eight test curves (CURVES), runs adaptive_n_budget on each,
and saves per-curve Matplotlib figures.

All fitting / blending functions are in conicspline.py, which is
imported with "from conicspline import *" so that existing diagnostic
scripts (scan_all_curves.py, diag_xbranch.py, test_cross_guard.py, …)
continue to work unchanged via importlib + bd.<name>.
"""

import os as _os
import numpy as np
from scipy.interpolate import CubicSpline
from conicspline import *  # public API re-exported for `bd.<name>` access
from conicspline import (  # private helpers needed by diagnostic scripts
    _run_blend,
    _build_projective_arc_window, _try_conic,
    _is_conic_monotone, _make_spline_window,
)

_here = _os.path.dirname(_os.path.abspath(__file__))


# ── Quality evaluation and adaptive sampling (application layer) ──────────────
# These require a ground-truth xy_func and live here, not in conicspline.

def _menger_kappa(A, B, C):
    """Menger curvature (= 1/circumradius) of three 2-D points.
    Returns 0.0 for nearly-collinear points."""
    AB = B[:2] - A[:2];  BC = C[:2] - B[:2];  AC = C[:2] - A[:2]
    cross = abs(float(AB[0]*BC[1] - AB[1]*BC[0]))
    denom = np.linalg.norm(AB) * np.linalg.norm(BC) * np.linalg.norm(AC)
    return cross / denom if denom > 1e-30 else 0.0


def _find_worst_interval(pts, times, windows, middle_start, middle_end,
                         xy_func, t_range, N_order, N_dense=80):
    """Find the blending interval with worst curvature-scaled deviation from
    ground truth.  Returns (worst_j, max_dev, dev_per_interval)."""
    dense, _ = blend_curve(pts, times, windows,
                           middle_start, middle_end, N_order, N_dense=N_dense)
    n_intervals = middle_end - middle_start
    devs = []
    for idx in range(n_intervals):
        seg = dense[idx * N_dense:(idx + 1) * N_dense]
        j = middle_start + idx
        t_dense = np.linspace(times[j], times[j + 1], N_dense)
        t_param = t_range[0] + t_dense * (t_range[1] - t_range[0])
        xg, yg = xy_func(t_param)
        gt = np.column_stack([xg, yg])
        norms = np.linalg.norm(seg[:, :2] - gt, axis=1)
        if not np.all(np.isfinite(norms)):
            devs.append(float('inf'))
            continue
        kappa = 0.5 * (_menger_kappa(pts[j-1], pts[j], pts[j+1])
                     + _menger_kappa(pts[j],   pts[j+1], pts[j+2]))
        devs.append(float(np.max(norms)) * kappa)
    worst_j = middle_start + np.argmax(devs)
    return worst_j, max(devs), devs


def sample_curve_speed(xy_func, t_range, n, alpha=1.0, n_dense=2000):
    """Sample n control points with density proportional to phase-space speed^alpha + 1.

    Concentrates points where the parametric curve moves fastest (high curvature
    transitions, spikes).  The +1 baseline prevents flat regions from being starved.

    alpha=0  → uniform (same as sample_curve)
    alpha=1  → density proportional to speed  (arc-length-ish, default)

    Useful for stiff curves like Van der Pol (mu=3) where the spike region accounts
    for most of the approximation error.  Reduces VdP n from 128 → 57 at the
    default max_dev_target=0.018.  Has no effect on easy curves (Spiral, Kepler)
    and degrades Damped oscillation, so use only for curves with genuine speed spikes.
    """
    t_dense = np.linspace(t_range[0], t_range[1], n_dense)
    x, y = xy_func(t_dense)
    dx = np.gradient(x, t_dense)
    dy = np.gradient(y, t_dense)
    speed = np.sqrt(dx**2 + dy**2) + 1e-15
    speed_mean = np.mean(speed) + 1e-15
    density = (speed / speed_mean) ** alpha + 1.0
    cdf = np.cumsum(density)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    tau = np.linspace(0.0, 1.0, n)
    t_sampled = np.interp(tau, cdf, t_dense)
    x_s, y_s = xy_func(t_sampled)
    pts   = np.column_stack([x_s, y_s, np.zeros(n)])
    times = (t_sampled - t_range[0]) / (t_range[1] - t_range[0])
    return pts, times


def adaptive_n_budget(xy_func, t_range, max_dev_target=0.018, N_order=2,
                      n_min=7, n_max=500, sampler=None):
    """Find minimum n such that blended interpolation max_dev ≤ target.

    Uses exponential search (doubling) then linear scan within the bracket.
    O(log n_converge) evaluations vs O(n_converge) for plain linear scan.

    sampler: callable (xy_func, t_range, n) -> (pts, times).  Defaults to
             sample_curve (uniform spacing).  Pass sample_curve_speed for
             speed-adaptive spacing (recommended for Van der Pol).
    """
    if sampler is None:
        sampler = sample_curve
    cache = {}
    def eval_cached(n):
        if n not in cache:
            pts, times = sampler(xy_func, t_range, n)
            _, _, _, wins, ms, me, _ = _run_blend(
                pts, times, use_spline=False, N_order=N_order)
            _, md, _ = _find_worst_interval(
                pts, times, wins, ms, me, xy_func, t_range, N_order)
            cache[n] = md
        return cache[n]

    if eval_cached(n_min) <= max_dev_target:
        return n_min
    hi = n_min
    while hi < n_max:
        hi = min(hi * 2, n_max)
        if eval_cached(hi) <= max_dev_target:
            break
    if eval_cached(hi) > max_dev_target:
        return n_max
    lo = max(hi // 2 + 1, n_min + 1)
    for n in range(lo, hi):
        if eval_cached(n) <= max_dev_target:
            return n
    return hi


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path generators ───────────────────────────────────────────────────
# Each returns (pts_Nx3, times_N) for a given parameter array t.

def _spiral_xy(t):
    r = 0.3 * np.exp(0.4 * t)
    return r * np.cos(t * 1.8), r * np.sin(t * 1.8)

def _lissajous_xy(t):
    return 2.0 * np.sin(2 * t), 2.0 * np.sin(3 * t + 0.3)

def _rose_xy(t):
    r = 2.0 * np.cos(3 * t)
    return r * np.cos(t), r * np.sin(t)

def _damped_xy(t):
    x = t / (4 * np.pi) * 4 - 2
    y = 2.0 * np.exp(-t / 8) * np.sin(t * 1.5)
    return x, y

def _flower_xy(t):
    x = 2.0 * np.cos(t) + 0.8 * np.cos(5 * t)
    y = 2.0 * np.sin(t) + 0.8 * np.sin(5 * t)
    return x, y

# ── Random smooth spline ─────────────────────────────────────────────
# Seeded RNG → deterministic but non-analytic.  A cumulative random walk
# through 10 waypoints, interpolated with a natural cubic spline.
_RNG = np.random.RandomState(42)
_N_WP = 10
_T_WP = np.linspace(0, 1, _N_WP)
_X_WP = _RNG.randn(_N_WP).cumsum()
_Y_WP = _RNG.randn(_N_WP).cumsum()
_X_SPL = CubicSpline(_T_WP, _X_WP, bc_type='natural')
_Y_SPL = CubicSpline(_T_WP, _Y_WP, bc_type='natural')

def _random_spline_xy(t):
    return _X_SPL(t), _Y_SPL(t)


# ── Van der Pol limit cycle ──────────────────────────────────────────
# Solve x'' - μ(1-x²)x' + x = 0 with μ=3 to get a limit cycle that has
# nearly flat sides and sharp curvature transitions — very different from
# any trig curve.  We precompute and cache the solution.
def _compute_vdp(mu=3.0, T_settle=50, T_trace=20, dt=0.001):
    """Integrate Van der Pol and return (t, x, y) on the limit cycle."""
    from scipy.integrate import solve_ivp
    def rhs(t, z):
        x, v = z
        return [v, mu * (1 - x**2) * v - x]
    # Settle onto limit cycle
    sol1 = solve_ivp(rhs, [0, T_settle], [0.1, 0.0],
                     max_step=dt, rtol=1e-10, atol=1e-12)
    y0 = sol1.y[:, -1]
    # Trace one+ cycle
    sol2 = solve_ivp(rhs, [0, T_trace], y0,
                     max_step=dt, rtol=1e-10, atol=1e-12)
    return sol2.t, sol2.y[0], sol2.y[1]

_VDP_T, _VDP_X, _VDP_V = _compute_vdp()
# Find one complete cycle (period ≈ 2π for μ not too large)
# Detect zero-crossings of x going positive
_vdp_cross = np.where((_VDP_X[:-1] <= 0) & (_VDP_X[1:] > 0))[0]
if len(_vdp_cross) >= 2:
    _i0, _i1 = _vdp_cross[0], _vdp_cross[1]
else:
    _i0, _i1 = 0, len(_VDP_T) - 1
_VDP_PERIOD_T = _VDP_T[_i0:_i1+1] - _VDP_T[_i0]
_VDP_PERIOD_X = _VDP_X[_i0:_i1+1]
_VDP_PERIOD_V = _VDP_V[_i0:_i1+1]
_VDP_X_SPL = CubicSpline(_VDP_PERIOD_T, _VDP_PERIOD_X)
_VDP_V_SPL = CubicSpline(_VDP_PERIOD_T, _VDP_PERIOD_V)

def _vdp_xy(t):
    return _VDP_X_SPL(t), _VDP_V_SPL(t)


def _kepler_drift_xy(t):
    """Kepler ellipse (a=2, e=0.6) with linear focus drift."""
    t = np.asarray(t, dtype=float)
    a, e = 2.0, 0.6
    b = a * np.sqrt(1 - e**2)

    # Solve Kepler's equation: M = E - e sin(E)
    M = t.copy()  # mean motion n=1 (period = 2π)
    E = M.copy()
    for _ in range(30):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < 1e-14:
            break

    # Perifocal position
    x_orb = a * (np.cos(E) - e)
    y_orb = b * np.sin(E)

    # Rotate by ω = 40°
    omega = np.radians(40)
    co, so = np.cos(omega), np.sin(omega)
    x_rot = x_orb * co - y_orb * so
    y_rot = x_orb * so + y_orb * co

    # Linear focus drift
    vx, vy = 0.3, -0.15
    return x_rot + vx * t, y_rot + vy * t


CURVES = [
    ("Logarithmic spiral", _spiral_xy, (0.5, 3.5)),
    ("Lissajous (3:2)",    _lissajous_xy, (0, 2 * np.pi * 0.9)),
    ("Rose curve (k=3)",   _rose_xy, (0.1, np.pi - 0.1)),
    ("Damped oscillation", _damped_xy, (0, 4 * np.pi)),
    ("5-petal flower",     _flower_xy, (0, 2 * np.pi * 0.95)),
    ("Kepler + drift",     _kepler_drift_xy, (0.3, 2 * np.pi - 0.3)),
    ("Random spline",      _random_spline_xy, (0.05, 0.95)),
    ("Van der Pol (mu=3)", _vdp_xy, (_VDP_PERIOD_T[1], _VDP_PERIOD_T[-2])),
]

def main():
    N_order = 2  # quintic smoothstep (C^2)
    use_spline = False  # deg-4 poly for s(t)

    # ── Sampling mode ────────────────────────────────────────────────
    # 'geometric': fast, uses angle-based constraints (no fitting needed)
    # 'budget':    slower, binary-searches on n to meet a max_dev target
    SAMPLING_MODE = 'budget'
    MAX_DEV_TARGET = 0.01        # for budget mode

    # Geometric constraint defaults.  Reasonable ranges per constraint:
    #   chord_turn:  90°–150°  (angle between consecutive chords)
    #   window_turn: 180°–270° (total chord turn across a 5-pt window)
    #   accel_turn:  90°–150°  (acceleration direction change per segment)
    # Tighter → more points, better conic approximation.
    # Which constraint binds depends on curve shape:
    #   - sharp turns / crossings: chord_turn, accel_turn
    #   - high curvature variation: window_turn, accel_turn
    #   - gentle curves: often hits the minimum (n=7)
    GEOM_KWARGS = dict(
        max_chord_turn_deg=120,
        max_window_turn_deg=180,
        max_accel_turn_deg=90,
        max_fallback_frac=0.05,
    )

    METHOD_COLORS = {
        'conic':  '#1976D2',
        'blend':  '#00897B',
        'kepler': '#7B1FA2',
        'polar':  '#2E7D32',
        'spline': '#F57C00',
    }

    fig, axes = plt.subplots(len(CURVES), 1, figsize=(14, 7 * len(CURVES)))

    for idx, (title, xy_func, t_range) in enumerate(CURVES):
        if SAMPLING_MODE == 'budget':
            n = adaptive_n_budget(xy_func, t_range,
                                  max_dev_target=MAX_DEV_TARGET,
                                  N_order=N_order)
        else:
            n = adaptive_n(xy_func, t_range, **GEOM_KWARGS)
        pts, times = sample_curve(xy_func, t_range, n)

        print(f"\n{'='*60}")
        print(f"  {title}: {n} pts")
        print(f"{'='*60}")

        ax = axes[idx]
        (dense, interp, costs, windows, ms, me,
         methods) = _run_blend(pts, times, use_spline, N_order)

        from collections import Counter
        mcounts = Counter(methods)
        worst_j, max_dev, devs = _find_worst_interval(
            pts, times, windows, ms, me, xy_func, t_range, N_order)

        print(f"  max_dev={max_dev:.3e}  "
              f"windows: {mcounts.get('conic',0)} conic, "
              f"{mcounts.get('blend',0)} blend, "
              f"{mcounts.get('kepler',0)} kepler, "
              f"{mcounts.get('polar',0)} polar, "
              f"{mcounts.get('spline',0)} spline")

        # Map each blending interval to the method of its two windows
        # Interval j blends windows[j-2] (A) and windows[j-1] (B).
        # Use the "lower tier" method of the two (spline > conic > polar).
        tier = {'conic': 0, 'blend': 1, 'kepler': 2, 'polar': 3, 'spline': 4}
        def interval_method(j):
            mA = methods[j - 2]
            mB = methods[j - 1]
            return mA if tier[mA] >= tier[mB] else mB

        # ── Ground truth ──────────────────────────────────────────
        gt_pts, _ = sample_curve(xy_func, t_range, 2000)
        ax.plot(gt_pts[:, 0], gt_pts[:, 1], color='#BBBBBB', lw=3,
                alpha=0.6, label='ground truth', zorder=1)

        # ── Blended curve, segment by segment ─────────────────────
        N_seg = 80  # must match N_dense in blend_curve
        plotted_labels = set()
        for seg_idx, j in enumerate(range(ms, me)):
            seg = dense[seg_idx * N_seg:(seg_idx + 1) * N_seg]

            m = interval_method(j)
            color = METHOD_COLORS[m]
            lbl = m if m not in plotted_labels else None
            ax.plot(seg[:, 0], seg[:, 1], color=color,
                    lw=2.5, alpha=0.85, label=lbl, zorder=3)
            if lbl:
                plotted_labels.add(m)

        # ── Control points ────────────────────────────────────────
        ax.scatter(pts[ms:me+1, 0], pts[ms:me+1, 1], c='black', s=20,
                   zorder=5, label=f'{n} control pts')

        ax.set_title(f'{title}  —  {n} pts, '
                     f'max_dev={max_dev:.2e}  '
                     f'[{mcounts.get("conic",0)}C '
                     f'{mcounts.get("blend",0)}B '
                     f'{mcounts.get("kepler",0)}K '
                     f'{mcounts.get("polar",0)}P '
                     f'{mcounts.get("spline",0)}S]',
                     fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=9, loc='best')
        ax.tick_params(labelsize=9)

    mode_label = (f'budget (target={MAX_DEV_TARGET})' if SAMPLING_MODE == 'budget'
                  else f'geometric (chord={GEOM_KWARGS["max_chord_turn_deg"]}°)')
    fig.suptitle(f'Four-tier blend: conic / kepler / polar / spline  [{mode_label}]',
                 fontsize=15, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = _os.path.join(_here, 'orbit_blend.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print(f"\nPlot saved to {out}")


if __name__ == '__main__':
    main()
