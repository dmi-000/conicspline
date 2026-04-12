# conicspline

**C^N-continuous parametric curve interpolation using conic arc sections.**

Given a set of 2-D control points with associated parameter values, `conicspline` fits overlapping 5-point conic windows and blends them with a C^N smoothstep kernel to produce a curve that is exact at every control point and smooth everywhere in between. The smoothness order N is configurable; the default N=2 gives a quintic kernel and C² continuity.

Where the data lies exactly on a conic, the blend is near machine-epsilon accurate everywhere between control points, not just at them. Each conic type uses its natural intrinsic parametrization:

- **Ellipses and circles** use the *eccentric-anomaly orbit* `(cx + a·cos(E), cy + b·sin(E))` in the conic's principal frame. The center `(cx, cy)` is an algebraic invariant of the conic — every window on the same ellipse computes the same center — so the orbit functions from adjacent windows agree to floating-point precision. For circles `a = b` and E equals the central angle exactly, giving machine-epsilon accuracy (`|x²+y²−1| < 1e-15` for a unit circle arc).
- **Hyperbolas** use the *stereographic (rational) orbit* `x = x₀ + u(s)`, `y = y₀ + s·u(s)` from the conic vertex `(x₀, y₀)`. The vertex is an algebraic invariant shared by all windows on the same hyperbola → consistent orbits → near machine-epsilon blend (`|x²−y²−1| < 1e-10` for a standard hyperbola).
- **Parabolas** also use the vertex path, but the parametric angle `φ = 2·arctan(t)` is nonlinear in the sampling parameter → O(h²) accuracy at n control points rather than machine-epsilon.

For general smooth curves that are only locally conic, accuracy improves with the number of control points. A natural cubic spline window is used as a fallback where the conic fit is geometrically invalid.

---

## Successor: arcspline

[**arcspline**](https://github.com/dmi-000/arcspline) is the C++ successor to
`conicspline`.  It generalises the same overlapping-window + smoothstep-blend
architecture to higher-dimensional geometric primitives:

| Feature | conicspline | arcspline |
|---|---|---|
| Language | Python (NumPy/SciPy) | Header-only C++17 |
| Dimensions | 2D / 3D via projection | ℝᴺ, any N ≥ 2 |
| 2D window | Conic arc (SVD fit) | Conic arc (`ConicWindow<2>`) |
| 3D window | Conic arc (projects to 2D) | Cylinder geodesic (`CylinderWindow<3>`) |
| 4D+ window | — | Clifford torus (`CliffordWindow<N>`) |
| Fallback | Cubic spline | Lagrange or Floater-Hormann rational |
| Blend style | Continuous α(t) conic/spline mix | Binary gate + geometric hierarchy |

**When to use conicspline:** Python environment; 2D curves; the continuous
conic/spline disagreement blend is useful for "almost-conic" data where neither
a pure conic nor a pure spline is ideal.

**When to use arcspline:** C++ environment; 3D/4D/ND curves (helices, Clifford
tori); exact cylindrical or Clifford fits are needed; no external dependencies.

The core algorithm — 5-point algebraic conic fit, vertex-P₀ + φ = 2·arctan(s)
orbit, overlapping windows blended with a smoothstep weight — is shared by both
libraries and originated in `conicspline`.

---

## Install

```bash
pip install conicspline
```

## Quick start

```python
import numpy as np
from conicspline import sample_curve, fit_conic_5pt, blend_curve, _run_blend

# Define any parametric curve
def lissajous(t):
    return 2.0 * np.sin(2 * t), 2.0 * np.sin(3 * t + 0.3)

t_range = (0, 2 * np.pi * 0.9)

# Sample n control points
pts, times = sample_curve(lissajous, t_range, n=36)

# Fit windows and build blended interpolant
dense, interp, costs, windows, ms, me, methods = _run_blend(pts, times,
                                                            use_spline=False,
                                                            N_order=2)

# Evaluate at dense output (80 points per segment, C² blend)
output, _ = blend_curve(pts, times, windows, ms, me, N_order=2, N_dense=80)
# output is shape (n_segments * 80, 3): columns x, y, t
```

## Public API

| Function | Description |
|---|---|
| `sample_curve(xy_func, t_range, n)` | Sample *n* uniformly-spaced control points |
| `fit_conic_5pt(points, times)` | Fit one 5-point window; returns `(func, cost, method)` |
| `blend_curve(pts, times, windows, ms, me, N_order)` | Evaluate the blended interpolant densely |
| `smoothstep(s, N)` | The C^N smoothstep kernel used for blending |
| `fit_conic(points)` | Fit a conic to N≥5 2-D points (SVD, algebraic) |

The main engine `_run_blend` returns all windows and is importable explicitly:

```python
from conicspline import _run_blend
```

## Dependencies

- [NumPy](https://numpy.org/) ≥ 1.21
- [SciPy](https://scipy.org/) ≥ 1.7

## How it works

Every interior segment `p[j] → p[j+1]` is covered by two overlapping 5-point windows:

```
Window A (idx j-2):  p[j-2], p[j-1], p[j], p[j+1], p[j+2]
Window B (idx j-1):  p[j-1], p[j],   p[j+1], p[j+2], p[j+3]

Segment j = (1 − w) · A(t) + w · B(t),   w = smoothstep(s, N)
```

Because the *same* window function appears on both sides of every knot, C^N continuity is exact — not approximate. See [`conicspline_doc.md`](conicspline_doc.md) for full API documentation and mathematical properties.

## License

MIT
