"""
tests/demo_blend_windows.py
-----------------------------
Visualises 'blend' windows — conic+spline disagreement weighted mixes —
which appear at conic/spline transition boundaries and have not previously
been plotted explicitly.

Panels chosen to show escalating blend complexity:
  Kepler n=18  — 1 blend window in a mixed conic/spline context  [CCCCsssBCCCsss]
  Kepler n=22  — 2 blend windows, symmetric pair                 [CCCCCBsssCCCCCBsss]
  5-petal n=40 — 4 blend windows, one per petal boundary         [CCCsBCCCC...BsC]

Blend windows are shown in BLEND_COLOR (distinct from conic blue / spline orange).
Control points whose middle window is 'blend' are coloured the same way.

Run from repo root:
    python3 tests/demo_blend_windows.py

Saves: demo_blend_windows.png  (alongside this script)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_here = os.path.dirname(os.path.abspath(__file__))
spec  = importlib.util.spec_from_file_location('bd', os.path.join(_here, '..', 'blend_demo.py'))
bd    = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

import demo_n_transitions as dn

# ── Override FALLBACK_COLOR in dn so 'blend' renders distinctly ──────────────
# plot_panel uses WIN_COLOR.get(method, FALLBACK_COLOR).
# WIN_COLOR has only 'conic' and 'spline', so 'blend' falls through to FALLBACK.
# We patch WIN_COLOR to include 'blend' explicitly.
BLEND_COLOR = '#00838F'   # teal — distinct from conic blue, spline orange, fallback purple
dn.WIN_COLOR['blend'] = BLEND_COLOR

# ── Panels ────────────────────────────────────────────────────────────────────
kepler = next(e for e in dn.DEMO_CURVES if e[0] == 'Kepler + drift')
petal  = next(e for e in dn.DEMO_CURVES if e[0] == '5-petal flower')

PANELS = [
    (kepler[1], kepler[2], 18, 'Kepler n=18\n1 blend  [CCCCsssBCCCsss]'),
    (kepler[1], kepler[2], 22, 'Kepler n=22\n2 blends  [CCCCCBsssCCCCCBsss]'),
    (petal[1],  petal[2],  40, '5-petal n=40\n4 blends  [CCCsBCCCC…BsC]'),
]


def main():
    ncols = len(PANELS)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6))

    for ax, (xy_func, t_range, n, label) in zip(np.atleast_1d(axes), PANELS):
        print(f'Plotting {label.splitlines()[0]} …', flush=True)
        dn.plot_panel(ax, xy_func, t_range, n, label)

    fig.suptitle("'blend' windows — conic+spline disagreement mix",
                 fontsize=11, fontweight='bold')

    legend_elems = [
        Line2D([0], [0], color='#AAAAAA', lw=1.8, ls='--', label='Ground truth'),
        Line2D([0], [0], color='#222222', lw=1.8, alpha=0.65, label='Blended curve'),
        Line2D([0], [0], color='#1565C0', lw=2.0, label='Conic orbit'),
        Line2D([0], [0], color='#E65100', lw=2.0, label='Spline orbit'),
        Line2D([0], [0], color=BLEND_COLOR, lw=2.0, label='Blend orbit (conic+spline mix)'),
        Line2D([0], [0], marker='o', color='#1565C0',
               markerfacecolor='none', markersize=7, label='Ctrl pt: conic window'),
        Line2D([0], [0], marker='o', color='#E65100',
               markerfacecolor='none', markersize=7, label='Ctrl pt: spline window'),
        Line2D([0], [0], marker='o', color=BLEND_COLOR,
               markerfacecolor='none', markersize=7, label='Ctrl pt: blend window'),
    ]
    fig.legend(handles=legend_elems, fontsize=7, loc='lower center',
               ncol=4, framealpha=0.92, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(_here, 'demo_blend_windows.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
