"""
tests/demo_bump_isolated_spline.py
-----------------------------------
Demonstrates a single isolated spline window surrounded entirely by conic windows.

Curve: unit circle with a narrow Gaussian bump at t=1.5.
  x(t) = cos(t) + 0.3 * exp(-(t - 1.5)² / 0.05)
  y(t) = sin(t)

The bump has two inflection points at t = 1.5 ± sqrt(0.05/2) ≈ 1.5 ± 0.158.
When the 5-point window width spans *both* inflections, just one window is
non-monotone → isolated spline.  At higher n the windows narrow enough to
resolve each inflection separately → 3 consecutive splines.

Run from repo root:
    python3 tests/demo_bump_isolated_spline.py

Saves: demo_bump_isolated_spline.png  (alongside this script)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ── Load blend_demo (for _run_blend, sample_curve, smoothstep) ──────────────
_here = os.path.dirname(os.path.abspath(__file__))
spec  = importlib.util.spec_from_file_location('bd', os.path.join(_here, '..', 'blend_demo.py'))
bd    = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

# ── Load demo_n_transitions (for plot_panel) ─────────────────────────────────
import demo_n_transitions as dn

# ── Curve definition ─────────────────────────────────────────────────────────
SIGMA2   = 0.05          # Gaussian width parameter
BUMP_T   = 1.5           # centre of bump
BUMP_AMP = 0.3           # amplitude

xy_func = lambda t: (np.cos(t) + BUMP_AMP * np.exp(-((t - BUMP_T)**2) / SIGMA2),
                     np.sin(t))
t_range = (0.0, 2 * np.pi)

# Inflection points of the Gaussian are at BUMP_T ± sqrt(SIGMA2/2)
inflection_sep = 2 * np.sqrt(SIGMA2 / 2)


# ── Scan: print all n in [8, 33] with methods summary ────────────────────────
def run_scan(n_min=8, n_max=33):
    print(f"Circular arc + Gaussian bump  (amplitude={BUMP_AMP}, σ²={SIGMA2})")
    print(f"Inflection separation: 2σ = {inflection_sep:.3f} rad")
    print()
    print(f"  {'n':>3}  {'C':>4} {'s':>4}  pattern")
    print("  " + "-" * 55)
    for n in range(n_min, n_max + 1):
        pts, times = bd.sample_curve(xy_func, t_range, n)
        _, _, _, wins, ms, me, methods = bd._run_blend(pts, times, False, 2)
        nc = sum(m == 'conic'  for m in methods)
        ns = sum(m == 'spline' for m in methods)
        abbrev = ''.join('C' if m == 'conic' else 's' for m in methods)
        isolated = [i for i in range(1, len(methods) - 1)
                    if methods[i] == 'spline'
                    and methods[i - 1] == 'conic'
                    and methods[i + 1] == 'conic']
        tag = ''
        if isolated:
            tag = f'  ← {len(isolated)} isolated spline(s)'
        elif ns == 0:
            tag = '  ← all conic'
        print(f"  {n:>3}  {nc:>4} {ns:>4}  [{abbrev}]{tag}")


# ── Visual panels ─────────────────────────────────────────────────────────────
# n=16: all conic (bump width < window width — bump not yet resolved)
# n=20: 1 isolated spline (both inflections in one window)
# n=21: all conic again (window shifts past bump inflections exactly)
# n=22: 3 consecutive splines (window narrow enough to resolve each inflection)
PANELS = [
    (16, 'all conic\n(bump unresolved)'),
    (20, 'ONE isolated spline\n(both inflections in 1 window)'),
    (21, 'all conic again\n(window just clears both inflections)'),
    (22, '3 splines\n(each inflection resolved separately)'),
]


def main():
    run_scan()

    ncols = len(PANELS)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6))

    for ax, (n, label) in zip(np.atleast_1d(axes), PANELS):
        print(f"\nPlotting n={n} …", flush=True)
        dn.plot_panel(ax, xy_func, t_range, n, label)

    fig.suptitle(
        f'Circular arc + narrow Gaussian bump  '
        f'(amp={BUMP_AMP}, σ²={SIGMA2}, inflection sep={inflection_sep:.3f} rad)',
        fontsize=10, fontweight='bold',
    )

    legend_elems = [
        Line2D([0], [0], color='#AAAAAA', lw=1.8, ls='--', label='Ground truth'),
        Line2D([0], [0], color='#222222', lw=1.8, alpha=0.65, label='Blended curve'),
        Line2D([0], [0], color='#1565C0', lw=2.0, label='Conic orbit'),
        Line2D([0], [0], color='#E65100', lw=2.0, label='Spline orbit'),
        Line2D([0], [0], marker='o', color='#1565C0',
               markerfacecolor='none', markersize=7, label='Ctrl pt: conic window'),
        Line2D([0], [0], marker='o', color='#E65100',
               markerfacecolor='none', markersize=7, label='Ctrl pt: spline window'),
    ]
    fig.legend(handles=legend_elems, fontsize=7, loc='lower center',
               ncol=3, framealpha=0.92, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(_here, 'demo_bump_isolated_spline.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
