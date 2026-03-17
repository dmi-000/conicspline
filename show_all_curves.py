"""
show_all_curves.py
------------------
All 8 sample curves with the smoothstep-blend interpolation.

Visualization design (2026-03-09):

  The window orbits are the primary visual element.  Each window wins[i]
  spans pts[i:i+5] and is drawn with continuously varying alpha:

    Peak alpha  — at times[i+2] (the knot where wins[i] has weight 1 on
                  both adjacent segments; the same window instance dominates
                  on both sides)
    Fade to ~0  — at times[i+1] and times[i+3], where wins[i]'s blend
                  weight reaches 0 on both neighbouring segments
    Thin+dim    — outer tails [times[i], times[i+1]] and
                  [times[i+3], times[i+4]], where wins[i] does not
                  contribute to any blend but still passes through all 5
                  control points

  The alpha varies as smoothstep on each half (rising then falling),
  implemented per-segment via LineCollection.

  Blended curve and knot points are semi-transparent overlays — they let
  you locate the actual interpolation result without competing with the orbits.
"""
import importlib.util, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

_here = os.path.dirname(os.path.abspath(__file__))
spec  = importlib.util.spec_from_file_location('bd', os.path.join(_here, 'blend_demo.py'))
bd    = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

N_VIZ_MAX  = 40     # cap n for legibility
N_DENSE    = 120    # points per segment for the blended curve
N_ORBIT    = 300    # points per window orbit (more → smoother alpha gradient)

ALPHA_PEAK  = 0.85   # orbit alpha at the knot peak (weight = 1)
ALPHA_TAIL  = 0.12   # orbit alpha in the outer tails (weight = 0 outside)
LW_ACTIVE   = 1.25   # linewidth in the active blend region
LW_TAIL     = 0.65   # linewidth in the outer tails

WIN_COLOR = {
    'conic':  '#1565C0',   # blue
    'spline': '#E65100',   # orange
}
FALLBACK_COLOR = '#888888'


# ── Orbit helpers ──────────────────────────────────────────────────────────────

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


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
axes = axes.flatten()

for ax, (name, xy_func, t_range) in zip(axes, bd.CURVES):

    n_prod = bd.adaptive_n_budget(xy_func, t_range)
    n      = min(n_prod, N_VIZ_MAX)
    pts, times = bd.sample_curve(xy_func, t_range, n)

    _, _, _, wins, ms, me, methods = bd._run_blend(pts, times, False, 2)

    _, md, _ = bd._find_worst_interval(
        pts, times, wins, ms, me, xy_func, t_range, 2)

    # ── Ground truth ─────────────────────────────────────────────────────────
    t_gt  = np.linspace(t_range[0], t_range[1], 1200)
    xg, yg = xy_func(t_gt)
    ax.plot(xg, yg, color='#CCCCCC', lw=2.5, zorder=1)

    # ── Window orbits (primary visual) ────────────────────────────────────────
    for i in range(len(wins)):
        ti    = times[i:i + 5]
        wc    = WIN_COLOR.get(methods[i], FALLBACK_COLOR)
        t_orb = np.linspace(ti[0], ti[4], N_ORBIT)
        try:
            orb = wins[i](t_orb)[:, :2]
        except Exception:
            continue
        _plot_orbit(ax, orb, t_orb, ti, wc)

    # ── Blended curve (semi-transparent so orbits show through beneath) ───────
    for j in range(ms, me):
        t_d = np.linspace(times[j], times[j + 1], N_DENSE)
        s   = (t_d - times[j]) / (times[j + 1] - times[j])
        w   = bd.smoothstep(s, 2)
        sA  = wins[j - 2](t_d)[:, :2]
        sB  = wins[j - 1](t_d)[:, :2]
        seg = sA * (1 - w)[:, None] + sB * w[:, None]
        ax.plot(seg[:, 0], seg[:, 1], color='#222222', lw=1.6,
                alpha=0.60, zorder=6)

    # ── Knot points: blend region (colored by dominant window) ───────────────
    # At pts[k] with ms ≤ k ≤ me, wins[k-2] has weight 1 on both adjacent
    # segments — the unambiguous dominant window at that knot.
    for k in range(ms, me + 1):
        wi    = k - 2
        color = WIN_COLOR.get(methods[wi], FALLBACK_COLOR)
        ax.scatter(*pts[k, :2], s=28, color=color, alpha=0.55, zorder=8,
                   edgecolors='white', linewidths=0.4)

    # ── Control points outside the blend region ───────────────────────────────
    # pts[0:ms] and pts[me+1:] are still covered by window orbits (wins[0],
    # wins[-1], etc.) but are not knots of the blend — shown as neutral markers.
    outer = list(range(0, ms)) + list(range(me + 1, len(pts)))
    for k in outer:
        ax.scatter(*pts[k, :2], s=16, color='#888888', alpha=0.40, zorder=7,
                   edgecolors='white', linewidths=0.3)

    ax.set_aspect('equal')
    ax.autoscale_view()          # needed because LineCollection doesn't auto-scale
    ax.grid(True, alpha=0.08, lw=0.3)
    ax.tick_params(labelsize=6)
    n_label = f'n={n}' if n == n_prod else f'n={n} (prod {n_prod})'
    ax.set_title(f'{name}\n{n_label}  max_dev={md:.2%}',
                 fontsize=7.5, fontweight='bold')


# ── Legend ─────────────────────────────────────────────────────────────────────
legend_elems = [
    Line2D([0], [0], color='#CCCCCC', lw=2.5,
           label='Ground truth'),
    Line2D([0], [0], color='#222222', lw=1.8, alpha=0.65,
           label='Blended curve'),
    Line2D([0], [0], color='#1565C0', lw=2.0,
           label='Conic orbit (full alpha at knot, fades to tails)'),
    Line2D([0], [0], color='#E65100', lw=2.0,
           label='Spline orbit (full alpha at knot, fades to tails)'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='#1565C0', markersize=7, alpha=0.5,
           label='Knot: conic window'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='#E65100', markersize=7, alpha=0.5,
           label='Knot: spline window'),
]
fig.legend(handles=legend_elems, fontsize=8, loc='lower center',
           ncol=6, framealpha=0.92, bbox_to_anchor=(0.5, 0.0))

fig.suptitle(
    'Smoothstep-blend — window orbits as primary visual\n'
    'Orbit alpha peaks at the knot where weight = 1, fades via smoothstep to 0, '
    'then dim in outer tails',
    fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0.045, 1, 1])
out = os.path.join(_here, 'all_curves.png')
plt.savefig(out, dpi=170, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
