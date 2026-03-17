"""
demo_n_transitions.py
---------------------
For each curve, show 2–4 panels at the n-values where the window-type
composition transitions between all-spline, first-conic, and all-conic.

Panels (left → right):
  • all-spline      — largest n where every window is spline (may not exist)
  • first-conic     — smallest n where ≥1 window is conic
  • before-all-conic — all_conic − 1  (omitted when equal to first-conic)
  • all-conic       — smallest n where every window is conic

Output: demo_transitions_{slug}.png  (saved alongside this script)
"""
import importlib.util, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

_here = os.path.dirname(os.path.abspath(__file__))
spec  = importlib.util.spec_from_file_location('bd', os.path.join(_here, 'blend_demo.py'))
bd    = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

# ── Visual constants ────────────────────────────────────────────────────────
WIN_COLOR      = {'conic': '#1565C0', 'spline': '#E65100'}
FALLBACK_COLOR = '#6A1B9A'   # purple — distinct from blue/orange/grey
ALPHA_PEAK = 0.85
ALPHA_TAIL = 0.12
LW_ACTIVE  = 1.25
LW_TAIL    = 0.65
N_ORBIT    = 300   # samples per window orbit
N_DENSE    = 120   # samples per segment for blended curve

OUT_DIR = _here

# ── Analytic-conic extras ───────────────────────────────────────────────────
_ONE_PERIOD = (-np.pi / 2 + 0.5, 3 * np.pi / 2 - 0.5)   # one period of sec/tan, ±0.5 margin

_EXTRA_CURVES = [
    ('Ellipse arc',
     lambda t: (3.0 * np.cos(t), 2.0 * np.sin(t)),
     (0.0, 10.0)),       # ~1.6 full turns; forces spline at n=8

    # Hyperbola, one period — control points cover each branch exactly once.
    # Ground truth also uses one period → symmetric grey lines.
    ('Hyperbola (1 period)',
     lambda t: (1.0 / np.cos(t), np.tan(t)),
     _ONE_PERIOD),

    # Same curve, t_range = (-5, 5) — covers 4 asymptotes (at ±π/2, ±3π/2).
    # Two distinct phenomena are exercised here:
    #
    #   1. Non-monotone phi → spline fallback at low n.  Wide windows span such
    #      a large arc that the stereographic phi values wrap past ±π (the
    #      asymptote direction) and back, making phi non-monotone.
    #      _is_conic_monotone returns False → spline.  As n increases, windows
    #      shrink and phi becomes monotone → transition to conic.
    #
    #   2. NaN propagation at intermediate n.  Once windows are narrow enough
    #      for monotone phi, some still have an asymptote inside the blend
    #      region.  The orbit returns NaN there → ctrl_err=inf →
    #      adaptive_n_budget increases n until the asymptote clears the window.
    #
    # The plot looks unusual: most control points are NaN-clipped (near-infinite
    # near asymptotes); the grey GT shows only one period so it doesn't
    # multi-trace branches not under test.  Visible control points cluster near
    # the two vertices (|x|≈1) where t is farthest from any asymptote.
    ('Hyperbola (wide range, 1-period GT)',
     lambda t: (1.0 / np.cos(t), np.tan(t)),
     (-5.0, 5.0),
     (-np.pi / 2 + 0.02, 3 * np.pi / 2 - 0.02)),

    # Parabola y = t^2.  det(conic matrix) = 0 → _try_conic early-returns None
    # for each window, so _build_projective_arc_window is NEVER called.  The
    # arc-length fallback (_blended_conic_spline) still finds a good parabola fit
    # via the quadratic conic coefficients, so methods end up all-'conic'.
    # This exercises the parabola (degenerate-conic) path in fit_conic_5pt.
    ('Parabola (y = t²)',
     lambda t: (t, t**2),
     (-2.0, 2.0)),
]

# Override t_ranges for curves that are all-conic at n=8 on their original
# range — extend over more loops so low-n windows span wider arcs and fall
# back to spline, making the transition panels meaningful.
_RANGE_OVERRIDES = {
    # Extend at the LOW end: outer radius stays bounded (r≈7.4 at t=8),
    # inner turns added spiralling in (r≈0.06 at t=−4).  3.4 full turns total.
    # n=8 windows each span ~3/7 of the range ≈ 1.7 turns → non-monotone conic.
    'Logarithmic spiral': (-4.0, 8.0),
    # ~2.4 orbits; gives all-spline at n=8.
    'Kepler + drift':     (0.0, 15.0),
}

DEMO_CURVES = []
for name, xy_func, t_range in bd.CURVES:
    t_range = _RANGE_OVERRIDES.get(name, t_range)
    DEMO_CURVES.append((name, xy_func, t_range))
DEMO_CURVES += _EXTRA_CURVES


# ── Orbit helpers (verbatim from show_all_curves.py lines 56–101) ──────────

def _orbit_rgba_lw(t_arr, ti, color_hex):
    """Per-sample alpha and linewidth for window orbit wins[i].

    ti = times[i:i+5]  (5 values for the 5 control points)
    Peak alpha at ti[2], smoothstep-fade to 0 at ti[1] and ti[3],
    then ALPHA_TAIL in the outer tails ti[0]..ti[1] and ti[3]..ti[4].
    """
    t0, t1, t2, t3, t4 = ti[0], ti[1], ti[2], ti[3], ti[4]
    rgb = mcolors.to_rgb(color_hex)

    alpha = np.full(len(t_arr), ALPHA_TAIL)
    lw    = np.full(len(t_arr), LW_TAIL)

    # Rising half: ti[1] → ti[2]
    mask_r = (t_arr >= t1) & (t_arr <= t2)
    if np.any(mask_r) and t2 > t1:
        s = (t_arr[mask_r] - t1) / (t2 - t1)
        alpha[mask_r] = ALPHA_PEAK * bd.smoothstep(s, 2)
        lw[mask_r]    = LW_ACTIVE

    # Falling half: ti[2] → ti[3]
    mask_f = (t_arr > t2) & (t_arr <= t3)
    if np.any(mask_f) and t3 > t2:
        s = (t_arr[mask_f] - t2) / (t3 - t2)
        alpha[mask_f] = ALPHA_PEAK * (1.0 - bd.smoothstep(s, 2))
        lw[mask_f]    = LW_ACTIVE

    # Per-segment values (average of endpoint values)
    seg_alpha = (alpha[:-1] + alpha[1:]) / 2.0
    seg_lw    = (lw[:-1]    + lw[1:])   / 2.0
    rgba = np.column_stack([
        np.full(len(seg_alpha), rgb[0]),
        np.full(len(seg_alpha), rgb[1]),
        np.full(len(seg_alpha), rgb[2]),
        seg_alpha,
    ])
    return rgba, seg_lw


def _plot_orbit(ax, orb_xy, t_arr, ti, color_hex):
    """Add window orbit as a LineCollection with per-segment alpha and linewidth."""
    pts = orb_xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    rgba, seg_lw = _orbit_rgba_lw(t_arr, ti, color_hex)
    lc = LineCollection(segs, colors=rgba, linewidths=seg_lw, zorder=3)
    ax.add_collection(lc)


# ── Transition scanner ──────────────────────────────────────────────────────

def _has_consec_splines(methods):
    """Return True if any two adjacent segments are both pure spline-spline blends.

    A segment at position j is a spline-spline blend when both its contributing
    windows (j-2 and j-1) are spline — i.e. methods[j-2] and methods[j-1] are
    both 'spline'.  Two consecutive such segments means three consecutive spline
    windows: methods[i], methods[i+1], methods[i+2] all 'spline'.
    """
    return any(methods[i] == 'spline' and methods[i + 1] == 'spline'
               and methods[i + 2] == 'spline'
               for i in range(len(methods) - 2))


def find_transitions(xy_func, t_range, n_min=8, n_max=100):
    """Return list of (n, label) panels showing conic/spline transitions.

    Scans n linearly from n_min to n_max, tracking:
      last_all_spline      — largest n where 0 conic windows
      first_conic          — smallest n where ≥1 conic window
      all_conic            — smallest n where all windows are conic
      last_no_consec_spline — largest n where no two adjacent windows are spline
                              (only used when all_conic is never reached)
    """
    last_all_spline           = None
    first_conic               = None
    all_conic                 = None
    last_consec_spline        = None   # last n where ≥2 consecutive spline-spline blend segments exist
    _stable_all_conic_start   = None   # first n of the current consecutive all-conic streak

    for n in range(n_min, n_max + 1):
        pts, times = bd.sample_curve(xy_func, t_range, n)
        _, _, _, wins, ms, me, methods = bd._run_blend(pts, times, False, 2)
        nc = sum(m == 'conic' for m in methods)
        nt = len(methods)

        # Skip n values with near-infinite control points (e.g. a grid point
        # landing exactly on a conic asymptote like sec(π/2) ≈ 1.6e16).
        # Such n are a numerical artefact of the parametrisation, not a
        # genuine failure of the conic fitter.  They neither advance nor reset
        # the all-conic stability streak.
        if np.max(np.abs(pts[:, :2])) > 1e6:
            continue

        if nc == 0 and first_conic is None:   # only pre-conic all-spline counts
            last_all_spline = n
        if nc > 0 and first_conic is None:
            first_conic = n
        if _has_consec_splines(methods):
            last_consec_spline = n
        if nc == nt:
            if _stable_all_conic_start is not None:
                all_conic = _stable_all_conic_start   # first of the stable pair
                break
            _stable_all_conic_start = n
        else:
            _stable_all_conic_start = None

    # Guard: curve is all-conic from n_min
    if first_conic is None:
        first_conic = n_min

    # Collect candidate (n, label) pairs, then sort and deduplicate.
    # When two labels share the same n, the later (more specific) label wins.
    candidates = {}
    if last_all_spline is not None:
        candidates[last_all_spline] = 'all spline'
    candidates[first_conic] = 'first conic'
    if all_conic is not None:
        if all_conic - 1 > first_conic:
            candidates[all_conic - 1] = 'before all conic'
        candidates[all_conic] = 'all conic'   # overwrites 'first conic' when equal
    else:
        # all-conic never reached (e.g. inflection-point curves).
        # Show the n just before and just after consecutive spline-spline
        # blend segments disappear — the density threshold where splines
        # become permanently isolated.
        if last_consec_spline is not None and last_consec_spline + 1 <= n_max:
            candidates[last_consec_spline]     = 'last consecutive spline blend'
            candidates[last_consec_spline + 1] = 'no consecutive spline blend'
        else:
            candidates[n_max] = f'n={n_max} (all-conic not reached)'

    panels = sorted(candidates.items())   # sorted by n, no duplicates
    return panels


# ── Single-panel plot ───────────────────────────────────────────────────────

def plot_panel(ax, xy_func, t_range, n, label, gt_t_range=None):
    """Draw one panel: ground truth + window orbits + blended curve + control points.

    gt_t_range: t-range used for the ground-truth plot only (defaults to t_range).
                Use a different range to avoid multi-tracing of the same branch when
                the parameterisation has period < (t_range[1] - t_range[0]).
    """
    pts, times = bd.sample_curve(xy_func, t_range, n)
    _, _, _, wins, ms, me, methods = bd._run_blend(pts, times, False, 2)

    nc    = sum(m == 'conic' for m in methods)
    ntot  = len(methods)

    # Robust scale: median of absolute control-point coordinates.
    med_scale = max(np.median(np.abs(pts[:, 0])),
                    np.median(np.abs(pts[:, 1])), 1.0)

    # Ground truth — NaN-clip at the same bbox_thresh used for orbits/blend.
    # All three elements (GT, orbits, blend) must use the same threshold so
    # matplotlib never draws a "going-to-frame-edge" stub for one element
    # without a matching gap in the others — which would look like a
    # deviation.  Detect asymptotes from raw GT values before any clipping.
    t_gt = np.linspace(*(gt_t_range or t_range), 1200)
    xg_raw, yg_raw = xy_func(t_gt)
    # Detect asymptotes via jump discontinuities — scale/rotation-invariant.
    # A pole creates a huge jump between adjacent GT samples (ratio >> 50).
    # A spiral or other large-but-continuous curve has smoothly growing jumps.
    if not np.all(np.isfinite(xg_raw) & np.isfinite(yg_raw)):
        has_asymptote = True
    else:
        _jumps = np.hypot(np.diff(xg_raw), np.diff(yg_raw))
        _med   = float(np.median(_jumps))
        has_asymptote = bool(_med > 0 and float(np.max(_jumps)) > _med * 50)
    bbox_thresh = med_scale * 4.0 if has_asymptote else float('inf')
    xg = np.where(np.abs(xg_raw) <= bbox_thresh, xg_raw, np.nan)
    yg = np.where(np.abs(yg_raw) <= bbox_thresh, yg_raw, np.nan)
    ax.plot(xg, yg, color='#AAAAAA', lw=1.8, ls='--', zorder=1)

    # Window orbits (primary visual)
    for i in range(len(wins)):
        ti    = times[i:i + 5]
        wc    = WIN_COLOR.get(methods[i], FALLBACK_COLOR)
        t_orb = np.linspace(ti[0], ti[4], N_ORBIT)
        try:
            orb = wins[i](t_orb)[:, :2]
        except Exception:
            continue
        if has_asymptote:
            # Bbox clip only — prevents line-to-frame-edge artefacts.
            # We do NOT filter by GT deviation so library anomalies remain
            # visible (asymmetric theta-arc wrapping, wrong-direction arcs, etc.).
            bad = (np.abs(orb[:, 0]) > bbox_thresh) | (np.abs(orb[:, 1]) > bbox_thresh)
            orb[bad] = np.nan
        _plot_orbit(ax, orb, t_orb, ti, wc)

    # Blended curve
    for j in range(ms, me):
        t_d = np.linspace(times[j], times[j + 1], N_DENSE)
        s   = (t_d - times[j]) / (times[j + 1] - times[j])
        w   = bd.smoothstep(s, 2)
        sA  = wins[j - 2](t_d)[:, :2]
        sB  = wins[j - 1](t_d)[:, :2]
        seg = sA * (1 - w)[:, None] + sB * w[:, None]
        if has_asymptote:
            # Bbox clip only — keeps the anomalous blend (wrong-direction
            # theta-arc average visible so the library bug can be diagnosed.
            bad_s = ((np.abs(seg[:, 0]) > bbox_thresh) |
                     (np.abs(seg[:, 1]) > bbox_thresh))
            seg[bad_s] = np.nan
        ax.plot(seg[:, 0], seg[:, 1], color='#222222', lw=1.6,
                alpha=0.60, zorder=6)

    # Control points colored by the window whose middle they are.
    # Window i has its middle control point at pts[i+2].
    # pts[0], pts[1], pts[n-2], pts[n-1] are not any window's middle → gray.
    pt_colors = [FALLBACK_COLOR] * len(pts)
    for i in range(len(wins)):
        mid_idx = i + 2
        if mid_idx < len(pts):
            pt_colors[mid_idx] = WIN_COLOR.get(methods[i], FALLBACK_COLOR)

    for k, (px, py) in enumerate(pts[:, :2]):
        # Skip points outside the visible region (near asymptotes) — ax.scatter
        # and ax.text are not axis-clipped, so off-screen points bloat the PNG.
        if has_asymptote and (abs(px) > bbox_thresh or abs(py) > bbox_thresh):
            continue
        ax.scatter(px, py, s=50, facecolors='none',
                   edgecolors=pt_colors[k], linewidths=1.5, zorder=8)
        # Number each control point so the traversal order is clear when the
        # curve visits the same region multiple times (e.g. spiral, Lissajous).
        ax.text(px, py, str(k), fontsize=4.5, color=pt_colors[k],
                ha='center', va='center', zorder=9,
                fontweight='bold')

    # ── Bounding box ──────────────────────────────────────────────────────────
    # Curves with asymptotes (NaN in ground truth) can have 1–2 control points
    # land near a singularity, blowing the bbox.  Use the robust median-based
    # exclusion for those, but for well-behaved curves (no NaN) use ALL control
    # points so that the full spatial extent is visible (e.g. spiral outer turns).
    pad = 0.15
    if has_asymptote:
        in_bbox = ((np.abs(pts[:, 0]) <= bbox_thresh) &
                   (np.abs(pts[:, 1]) <= bbox_thresh))
        pts_bb = pts[in_bbox] if in_bbox.any() else pts
    else:
        pts_bb = pts   # show all control points

    xmin, xmax = pts_bb[:, 0].min(), pts_bb[:, 0].max()
    ymin, ymax = pts_bb[:, 1].min(), pts_bb[:, 1].max()

    # Symmetrize: equal half-widths from the curve's geometric centre.
    # For asymptotic curves the control-point centroid and (min+max)/2 of the
    # ground truth are both biased by whichever branch has more parameter
    # coverage.  Instead, use the midpoint of the INNER gap between branches:
    #   xc = (max negative finite x  +  min positive finite x) / 2
    # For the sec/tan hyperbola both vertices are at |x|=1 so xc → 0 exactly,
    # regardless of how the discrete t-grid aligns with the four asymptotes.
    # Same logic applied to y (gives yc=0 for any curve passing through y=0).
    # Falls back to the control-point centroid when only one sign is present.
    if has_asymptote:
        xg_fin = xg[np.isfinite(xg)]
        yg_fin = yg[np.isfinite(yg)]
        def _gap_centre(arr, fb):
            pos = arr[arr > 0]; neg = arr[arr < 0]
            return float((neg.max() + pos.min()) / 2.0) if (len(pos) and len(neg)) else fb
        xc = _gap_centre(xg_fin, (xmin + xmax) / 2.0)
        yc = _gap_centre(yg_fin, (ymin + ymax) / 2.0)
    else:
        xc = (xmin + xmax) / 2.0
        yc = (ymin + ymax) / 2.0
    xhalf = max(xmax - xc, xc - xmin)
    yhalf = max(ymax - yc, yc - ymin)
    xmin, xmax = xc - xhalf, xc + xhalf
    ymin, ymax = yc - yhalf, yc + yhalf

    xspan, yspan = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - pad * xspan, xmax + pad * xspan)
    ax.set_ylim(ymin - pad * yspan, ymax + pad * yspan)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.08, lw=0.3)
    ax.tick_params(labelsize=6)
    ax.set_title(f'n={n}  {nc}/{ntot} conic\n{label}',
                 fontsize=8, fontweight='bold')


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    curves_to_run = DEMO_CURVES

    for entry in curves_to_run:
        name, xy_func, t_range = entry[0], entry[1], entry[2]
        gt_t_range = entry[3] if len(entry) > 3 else None

        slug = (name.lower()
                    .replace(' ', '_')
                    .replace('(', '').replace(')', '')
                    .replace('=', '').replace(',', ''))

        print(f'\n{name} — scanning transitions …', flush=True)
        panels = find_transitions(xy_func, t_range, n_min=8, n_max=120)
        print(f'  panels: {panels}')

        ncols = len(panels)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6))
        for ax, (n, label) in zip(np.atleast_1d(axes), panels):
            print(f'  plotting n={n} ({label}) …', flush=True)
            plot_panel(ax, xy_func, t_range, n, label, gt_t_range=gt_t_range)

        fig.suptitle(name, fontsize=11, fontweight='bold')

        # Legend
        legend_elems = [
            Line2D([0], [0], color='#AAAAAA', lw=1.8, ls='--', label='Ground truth'),
            Line2D([0], [0], color='#222222', lw=1.8, alpha=0.65,
                   label='Blended curve'),
            Line2D([0], [0], color='#1565C0', lw=2.0, label='Conic orbit'),
            Line2D([0], [0], color='#E65100', lw=2.0, label='Spline orbit'),
            Line2D([0], [0], marker='o', color='#1565C0',
                   markerfacecolor='none', markersize=7,
                   label='Ctrl pt: conic window'),
            Line2D([0], [0], marker='o', color='#E65100',
                   markerfacecolor='none', markersize=7,
                   label='Ctrl pt: spline window'),
            Line2D([0], [0], marker='o', color=FALLBACK_COLOR,
                   markerfacecolor='none', markersize=7,
                   label='Ctrl pt: outer / fallback orbit'),
        ]
        fig.legend(handles=legend_elems, fontsize=7, loc='lower center',
                   ncol=4, framealpha=0.92, bbox_to_anchor=(0.5, 0.0))

        plt.tight_layout(rect=[0, 0.07, 1, 1])
        out = os.path.join(OUT_DIR, f'demo_transitions_{slug}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {out}')


if __name__ == '__main__':
    main()
