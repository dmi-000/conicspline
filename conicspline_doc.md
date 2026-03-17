# conicspline — C^N Parametric Curve Interpolation

## What it does

Given a sequence of **n control points** and their **parameter values** (times),
`conicspline` constructs a smooth parametric curve that passes through all of them.
Smoothness is C^N — position, velocity, acceleration, and higher derivatives are
continuous everywhere, not just at the control points.

The curve can follow tight bends, hyperbolic branches, and near-circular arcs without
kinking, because each local segment is fitted to an **exact conic section** wherever
the geometry allows it.

---

## Core theory: overlapping 5-point windows + smoothstep blend

Every **interior segment** `p[j] → p[j+1]` is covered by two overlapping
**5-point window functions**:

```
Window A (index j−2):  p[j−2], p[j−1], p[j], p[j+1], p[j+2]
Window B (index j−1):  p[j−1], p[j],   p[j+1], p[j+2], p[j+3]
```

Each window is a parametric curve fitted to its five control points (conic arc or
spline fallback).  The blended segment is:

```
curve(t) = (1 − w(s)) · A(t)  +  w(s) · B(t)
```

where `s = (t − t_j) / (t_{j+1} − t_j) ∈ [0,1]` and `w = smoothstep(s, N)`.

### Why this gives exact C^N continuity

At knot `p[j]` (i.e. `s = 0`), `w = 0` and `dw/ds = 0` to order N, so the blended
curve reduces to `A(t)` on both sides of the knot — the **same function instance**
approaches from both the left segment (j−1) and the right segment (j).  There is no
matching condition to satisfy: continuity is an algebraic identity, not an
approximation.  The same argument applies at `p[j+1]` with `w = 1`.

### Smoothstep

`smoothstep(s, N)` is the unique polynomial that satisfies:
- `w(0) = 0`, `w(1) = 1`
- `d^k w/ds^k = 0` at `s = 0` and `s = 1` for `k = 1 … N`

For N=2 this is the quintic `6s⁵ − 15s⁴ + 10s³`.

---

## Window types

### Conic windows (preferred)

For each 5-point set, the library fits the unique conic through all five 2D
points (algebraic least-squares gives an exact fit for 5 points).  The conic is
parameterised as an arc that traverses the points in order.

**Monotonicity check** (`_is_conic_monotone`): determines whether the conic arc
moves through the five control points in a consistent direction — i.e. no
reversals.  Works in the conic's **principal axes frame** (rotates the five
points to diagonalise M = [[A, B/2],[B/2, C]]), making the result invariant
under rotation and reflection of the input.  In that frame, from the control
point with the most horizontal tangent (canonical P₀), the stereographic slope
`s_i = (y_i − y₀) / (x_i − x₀)` must be strictly monotone and `Q(s) = A′ + C′s²`
must change sign at most once (indicating a single passage through the asymptote
of a hyperbola).

**Cross-branch hyperbolas**: when five control points span both branches of a
hyperbola, the arc genuinely passes through infinity between branches.  This is
permitted — the orbit function evaluates to NaN at the asymptote.  This is the
correct geometric result; the window is accepted exactly as it is.

### Spline windows (fallback)

When the conic is non-monotone or cannot be fit (degenerate geometry), a
5-point natural cubic spline is used.  It passes through all five control points
exactly and has `.d1` / `.d2` derivative attributes so it is a drop-in replacement
wherever a conic window is expected.

### Blend windows (transition)

At conic/spline transition boundaries — where a window is nearly (but not quite)
conic-monotone — the library may return method `'blend'`: a weighted mixture of
the conic and spline orbits based on their pointwise disagreement.  These appear
at `CB` or `BC` boundaries in the methods sequence and are uncommon in practice
(~124 cases in n=8..80 across the standard test suite).

---

## API

### Sampling

```python
pts, times = sample_curve(xy_func, t_range, n)
```

Samples `n` equally-spaced parameter values from `xy_func` over `t_range`,
returning `pts` (N×3 array, z=0 for 2D curves) and `times` (N-length array,
normalised to `[0, 1]`).

### Fitting and blending

```python
dense, interp, costs, wins, ms, me, methods = _run_blend(pts, times,
                                                          use_spline=False,
                                                          N_order=2)
```

Core engine.  Fits one 5-point window per interior control point, then assembles
the smoothstep blends.  Returns:

| Return value | Meaning |
|---|---|
| `dense` | Pre-rendered array of blended curve points (n_segments × N_dense × 3) |
| `interp` | Callable `interp(t)` for evaluation at arbitrary parameter values |
| `costs` | List of per-window fit costs (float) |
| `wins` | List of window functions, each callable as `wins[i](t_arr)` returning (M×3) |
| `ms`, `me` | Index range where blending is active: segments `ms … me−1` |
| `methods` | List of strings, one per window: `'conic'`, `'spline'`, or `'blend'` |

### Rendering at arbitrary density

```python
dense, interp = blend_curve(pts, times, windows, ms, me, N_order, N_dense=80)
```

Evaluates the blended interpolant at `N_dense` points per segment.  `interp` is
a callable `interp(t)` for evaluation at arbitrary `t` values.

### Adaptive density (blend_demo.py)

```python
n = adaptive_n_budget(xy_func, t_range, max_dev_target=0.01, N_order=2,
                      n_min=7, n_max=500)
```

Exponential search (doubling `n`) to bracket the convergence threshold, then a
linear scan within the bracket to find the exact minimum `n` where the maximum
per-segment curvature-scaled deviation ≤ `max_dev_target`.  O(log n) evaluations
in the exponential phase; efficient for curves that converge at high n (e.g. VdP
uses ~21 evaluations instead of ~231 for a plain linear scan).  Defined in
`blend_demo.py`, not `conicspline.py`.

### Smoothstep

```python
w = smoothstep(s, N)   # s: array in [0,1], N: continuity order (int)
```

---

## Typical usage

```python
from conicspline import sample_curve, _run_blend, blend_curve

def my_curve(t):
    return np.cos(t), np.sin(2*t)          # returns (x_arr, y_arr)

t_range = (0.0, 2 * np.pi)
pts, times = sample_curve(my_curve, t_range, n=20)

_, _, _, wins, ms, me, methods = _run_blend(pts, times,
                                            use_spline=False,
                                            N_order=2)

dense, interp = blend_curve(pts, times, wins, ms, me, N_order=2, N_dense=80)
# dense shape: (n_segments * 200, 3)
```

To evaluate at arbitrary parameter values, use `interp(t_scalar_or_array)`.

---

## Mathematical properties

### Exact interpolation
The curve passes through every control point exactly.  This is interpolation, not
approximation — there is no smoothing or regularisation that shifts the curve away
from the given points.

### C^N continuity (exact, not approximate)
Position and the first N derivatives are continuous everywhere, including at knots.
The proof is algebraic: at any knot `p[j]` the smoothstep weight `w = 0` (and its
first N derivatives are zero), so the blended curve reduces to window A on both
sides — the same function instance, guaranteeing continuity with no tolerance.
Default N = 2 gives C² (position, velocity, acceleration).

### Local control
Modifying control point `k` affects at most **five windows** (those covering it:
`wins[k−4] … wins[k]`) and the blend segments they contribute to — at most six
consecutive segments.  The influence radius is proportional to `1/n` of the total
parameter range.  This contrasts with global splines, where adjusting one knot
shifts the entire curve.

### No global linear system
Each 5-point window is fitted independently (a single 6×6 eigenvalue problem).
There is no n×n system to solve.  A numerically degenerate region (e.g. near an
asymptote) gracefully falls back to a spline window without affecting distant parts
of the curve.

### Exact reproduction of uniformly-parameterised conics
When all control points lie on a conic, both windows covering any segment are
anchored at the *same intrinsic point*, so they compute identical orbit functions.
The smoothstep blend `(1−w)·A(t) + w·B(t) = A(t)` therefore lies exactly on the
conic.  The anchor and orbit type depend on the conic:

- **Ellipses**: eccentric-anomaly orbit `(cx + a·cos(E), cy + b·sin(E))` in the
  principal frame.  The center `(cx, cy)` is an algebraic invariant shared by all
  windows on the same ellipse.  The blend is exact when `E(t)` is linear in `t`
  (uniform eccentric-anomaly spacing ≈ uniform arc-length spacing).
- **Circles**: same eccentric-anomaly orbit with `a = b`, so E equals the central
  angle and is exactly linear for uniform parameter spacing.  The blend lies exactly
  on the circle regardless of n, with error < 1e-15 in practice.
- **Hyperbolas**: stereographic orbit from the conic vertex `(x₀, y₀)` (the point
  where the tangent gradient is zero — an algebraic invariant).  All windows on the
  same hyperbola use the same vertex → same `φ_k = 2·arctan(s_k − s₀)` at shared
  knots → identical orbits.  The blend is exact when `φ(t)` is linear in `t`.
- **Parabolas**: vertex path applies, but `φ = 2·arctan(t)` is nonlinear → O(h²)
  accuracy at n control points rather than machine-epsilon.

For general non-conic curves, the two windows produce slightly different orbits; the
residual is O(h⁴) per segment where h is the control-point spacing.

### Cross-branch hyperbola support
Standard interpolation schemes (including natural cubic splines) cannot smoothly
connect the two branches of a hyperbola: the curve passes through infinity between
branches and splines diverge there.  This library uses a rational stereographic
arc that maps the asymptote passage to a finite φ range via `tan(φ/2)`, allowing
a smooth, monotone orbit from one branch to the other.

### Approximate affine covariance
There is a unique conic through any five points in general position, and an affine
transform maps five points to five points — so each fitted conic is affine-covariant
as a geometric object.  The blend formula is linear, so the blended curve is
affine-covariant in the limit n → ∞.  At finite n, for general non-conic curves,
the two windows on each segment trace slightly different orbits, introducing an
O(1/n²) deviation from exact affine covariance.

For data lying exactly on a conic (ellipse, circle, or hyperbola), the two windows
use the same intrinsic anchor and produce identical orbits, so the blend lies exactly
on the conic and affine covariance is exact at any n.

---

## Design principles

- **No global smoothing**: the curve passes through every control point exactly.
- **No matching conditions**: C^N continuity follows algebraically from the
  overlapping-window construction, not from solving a system of equations.
- **Geometric gates only**: a window is accepted if and only if it is geometrically
  valid (monotone arc ordering).  No quality heuristics — the library does not know
  the ground truth and cannot measure approximation accuracy.
- **NaN = signal, not error**: a window that evaluates to NaN at an asymptote is
  geometrically correct.  Callers that measure accuracy against a ground truth will
  see `inf` deviation and can respond by providing denser control points.
