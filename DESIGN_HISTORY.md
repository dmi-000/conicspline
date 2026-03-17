# conicspline Design History

This document records the architectural evolution of `conicspline.py` — the original
intent behind approaches that were later removed or superseded, the missteps that
motivated each change, and why the dead code still present in the file was once thought
to be the right answer.

---

## Core architecture (stable throughout)

Every interior segment `p[j] → p[j+1]` is covered by two overlapping 5-point windows:

```
wA = wins[j-2]  fits pts[j-2:j+3]
wB = wins[j-1]  fits pts[j-1:j+4]

segment(t) = (1 − smoothstep(s)) · wA(t) + smoothstep(s) · wB(t)
```

C^N continuity at every knot is exact by construction: the *same* window function
appears on both sides of the knot, so all derivatives agree.  This principle was
established early and has never changed.  The entire history below is about how the
*window functions themselves* are built, and what happens when a window is "bad".

---

## 1. The seg_funcs era: per-segment override dict

### Original intent

Early versions could not always produce a good window function for every group of 5
control points — particularly near curve features (inflections, crossings, tight
bends) where the conic arc tracer would zig-zag to the wrong branch.  The solution
was a **fallback at the segment level**: if window wA or wB was bad for segment j,
skip the smoothstep blend entirely and substitute a bespoke per-segment function.

`_run_blend` returned a `seg_funcs` dict `{j: func}` alongside the window list.
`blend_curve` checked `if seg_funcs and j in seg_funcs:` and used the override
function directly, bypassing the smoothstep blend for that segment.

### The misstep

Bypassing the smoothstep blend at segment j meant that segment j used a *different*
function than its neighbours — the C^N continuity guarantee was broken at the two
knots bordering j.  In practice this produced visible kinks of 7–8° at
spline↔blend transition knots.

### The fix (`_make_spline_window`, 2026-03)

Instead of replacing a *segment*, replace the *window*: a non-monotone conic window
`wins[i]` is swapped out for a 5-point natural cubic spline window
(`_make_spline_window`) that covers exactly the same 5 control points and the same
parameter range.  Now *all* segments — including those adjacent to a formerly-bad
window — go through the identical smoothstep blend formula.  C^N continuity is
restored by construction, not by per-segment stitching.

`seg_funcs` is now always `{}`.  The `if seg_funcs and j in seg_funcs:` branch in
`blend_curve` is permanently False.  Both are vestigial; planned for removal.

### Dead code

- **`_make_conic_clamped_spline`** — the primary `seg_funcs` populator.  Built a
  cubic spline through the 4 surrounding control points `pts[j-2:j+2]` with
  algebraic conic-gradient BCs at both endpoints.  Better geometric continuity than
  a plain natural spline, but still bypassed the smoothstep blend → kinks.

- **`_make_quintic_hermite`** — an earlier, simpler `seg_funcs` populator.  Built a
  quintic Hermite through `p[j]..p[j+1]` using `.d1`/`.d2` from whichever adjacent
  windows were available.  Same kink problem as `_make_conic_clamped_spline`.

---

## 2. Cross-branch arc tracing: from zig-zag to projective arc

### The problem

A hyperbola has two branches.  When 5 control points span both branches, the arc
tracer (`_trace_conic_arc`) follows the gradient of the implicit conic equation to
move from one point to the next.  Near the asymptote the gradient switches sign; the
tracer snaps to the *other* branch and zig-zags.  The resulting window function is
geometrically wrong.

### Attempt 1: conic_hermite (`_build_conic_tangent_spline`, removed 2026-03-06)

The zig-zag was detected (Phase 2 of `_try_conic`), and instead of the arc-traced
conic the code substituted a quintic Hermite spline whose endpoint tangents were
taken from the *algebraic conic gradient* — the correct tangent direction at each
control point, independent of which branch the tracer was on.

**Why it seemed right:** the conic gradient gives the true tangent even at
cross-branch windows; using it as a Hermite BC guarantees tangent agreement with the
conic at the 5 knots.

**Why it was wrong:** near the spike of the Van der Pol oscillator, control points
lie near the asymptote.  The asymptote-directed conic tangent points toward infinity,
so the Hermite between two nearly-asymptote-tangent endpoints curves violently
outward.  The zig-zag problem was solved, but a worse distortion was introduced.

### Attempt 2: angle-from-center theta orbit (Cases 0–3 in `_build_projective_arc_window`)

A hyperbola in polar coordinates from its center traces both branches as `θ` sweeps
continuously: branch 1 for `θ ∈ (−π/2, π/2)`, branch 2 for `θ ∈ (π/2, 3π/2)`.
Parameterizing by θ wraps through ∞ naturally (in RP²) without switching branches.

PCHIP on `θ(t)` — with cross-branch steps corrected for direction (adding ±2π where
`cross[i]` is True) — gives a smooth orbit function that passes exactly through all
5 control points and handles the asymptote crossing geometrically.

This approach (the "theta orbit") was the primary cross-branch path for some time and
worked correctly for all 8 scan curves.

### Current approach: phi-first with theta fallback

The rational stereographic map (`x = x₀ + u(s)`, `y = y₀ + s·u(s)`) is simpler and
fully vectorised.  The parameter `φ = 2·arctan(s − s₀)` is used instead of θ; PCHIP
on `φ(t)` (after `np.unwrap`) gives the phi orbit.

`phi_mono = np.all(np.diff(phi_vals) > 0) or np.all(np.diff(phi_vals) < 0)` —
if True, use the phi orbit (Cases 0 and 1).  If False *and* cross-branch, the theta
fallback ran.

**The theta fallback is now dead** (2026-03-15, "Option C"): `_is_conic_monotone`
was made a pre-filter in `_try_conic`.  It uses the same P₀ selection and φ
computation as `_build_projective_arc_window`, so `phi_mono` is guaranteed True on
every call that reaches the function.  At converged n the theta path was already
returning None for every window anyway (ctrl_err > 1e-3 due to NaN in the blend
region) — eliminating it does not change any baseline.

### Case 2 (`through_inf`) — proven unreachable (2026-03-14)

Case 2 was a variant of the phi-orbit for windows where the asymptote is approached
from `s → ∞` rather than from a finite `s*`.  It used `v = 1/s` as the parameter to
avoid the `s → ∞` singularity.

A mathematical proof showed it is unreachable: for any conic point, the tangent slope
lies outside the asymptote cone in `s`-space but inside the asymptote cone in
`v = 1/s` space.  If `Q(s_vals)` has no sign change (ruling out Case 1), then
`Q_v(v_vals)` cannot have one either.  Verified empirically: 3,520 cross-branch
windows swept → 0 Case 2 hits.  The code block was replaced with a comment.

---

## 3. Window-fitting tiers: kepler and gen-polar

### Original intent

The tier system in `fit_conic_5pt` was designed for curves that are *not* conics but
resemble conic sections locally.  Two additional fitting families were provided:

- **Tier 2 (`_try_kepler_time`)**: fits a Keplerian ellipse/hyperbola via angle from
  focus.  Intended for near-Keplerian orbits where the conic fit might be unstable
  (e.g., near periapsis) but the Kepler parametrization (eccentric anomaly) is
  well-conditioned.

- **Tier 3 (`_try_gen_polar`)**: fits a generalized polar curve
  `r = r₀ / (1 + a·sin(kθ) + b·cos(kθ))`.  Intended for spiral and rosette-like
  curves that defy Cartesian conic fitting.

### Why they are dead

Neither tier has ever been triggered by any of the 8 scan-baseline curves or the
analytic-conic extras (ellipse, hyperbola, parabola).  Tier 1 (`_try_conic`) always
either succeeds (returns a conic or blend window) or falls back to pure spline; the
quality guard in `_blended_conic_spline` (disagreement ≥ 22% of chord → pure spline)
produces a valid `method='spline'` result before tiers 2–3 are reached.

Wrapped in `if False:` (2026-03-14) to make the dead path explicit without deletion.

---

## 4. Monotonicity detection evolution

The post-filter `_is_conic_monotone` in `_run_blend` detects whether a window's 5
control points traverse the conic arc in a consistent direction.  Non-monotone windows
are replaced by `_make_spline_window`.

### Version 1: s-difference sign check

Original check: `np.all(np.diff(s) > 0) or np.all(np.diff(s) < 0)` where `s` were
slopes from the first control point.  Simple but chose P₀ = pts[0], which is
parameter-order-dependent: mirror-image windows (e.g. Rose curve win[0] and win[7])
could get different verdicts.

### Version 2: eigh principal frame + |λ|-sort

Rotating to the conic's principal axes (B' = 0) makes the result invariant under
rigid motions.  The |λ|-sort fix (ascending by |eigenvalue|, not eigenvalue) ensures
that a conic M and its sign-negation −M (same zero-set, different `fit_conic`
normalisation) always map to the same principal frame, giving symmetric verdicts for
geometrically-equivalent windows.  Rose curve mirror pairs now agree at every n.

### Version 3: canonical P₀ = argmin|L/M|

The P₀ = pts[0] choice placed P₀ at different physical positions on the same conic
arc for mirror-image windows, giving different s-values and different monotone
verdicts.  The canonical P₀ (control point with smallest |tangent slope| = |L/M|)
is intrinsic to the conic: invariant under parameter reversal and reflection.

An additional bug was that P₀ near an asymptote direction gave spurious Q sign
changes between consecutive same-branch points (their slopes straddled s → ±∞ in
the PCHIP range).  Canonical P₀ is far from both asymptote directions by
construction, eliminating this.

### Version 4: local-frame mode for pre-filter (Option C, 2026-03-15)

Added `coeffs=None` parameter.  When coefficients are provided (called as pre-filter
from `_try_conic`), uses the local frame identical to `_build_projective_arc_window`
and checks φ-monotonicity directly.  This guarantees the phi orbit is always used —
making the theta fallback unreachable.  Principal-frame mode (post-filter in
`_run_blend`) is unchanged.

---

## 5. Quality metrics

### Version 1: Euclidean max deviation

`adaptive_n_budget` originally minimised `max(‖blended − truth‖)` with threshold
0.01 (Euclidean units).  This is scale-dependent: the same 0.01 error means very
different things on a tight spiral (large curvature) vs a gentle arc (small
curvature).

### Version 2: curvature-scaled deviation (2026-03-15)

`max(‖blended − truth‖) × κ_Menger` where κ is the Menger curvature (1/circumradius)
averaged over the two control-point triplets bracketing the segment.  This gives a
scale-invariant metric ≈ angular error in radians.  Collinear segments always score 0
(straight blended line over straight ground truth has zero angular error regardless
of offset), so they never drive n upward.

Threshold 0.018 ≈ δ·κ at the old Random spline baseline (n=28, δ·κ = 0.01777).
The mean across all 8 old baselines was 0.012, but using 0.012 caused the Random
spline to fail convergence due to an odd/even oscillation in δ·κ across the
exponential search path.

Key benefit: VdP n dropped from 238 to 133 (halved) because the high-curvature spike
region previously demanded small Euclidean errors everywhere; the new metric rewards
accurate angular tracking over absolute precision.

---

## 6. `adaptive_n` vs `adaptive_n_budget`

`adaptive_n` (still present, legacy) enforces geometric heuristics on control-point
spacing: chord turn, window chord turn sum, tangent sweep, acceleration direction, and
fallback rate.  It never evaluates the blended curve against ground truth.  It was the
original method for choosing n before ground truth became available in the demo loop.

`adaptive_n_budget` (in `blend_demo.py`) replaced it: binary-search on n, evaluating
the blended curve against the parametric ground truth at each candidate.  Direct
measurement of output quality rather than proxy geometric constraints.

`adaptive_n` is not called by any scan-baseline curve or test script.

---

## 7. Conic-exact orbits: eccentric anomaly and vertex P₀

### The problem: different windows gave different orbits for the same conic

For a window covering points on a pure ellipse, different groups of 5 control points
(e.g., win[0] and win[1]) fit the same algebraic conic but — using the first control
point as the stereographic base P₀ — computed *different* stereographic slopes.  PCHIP
on those different slopes gave slightly different orbit functions.  The two orbits agreed
at the shared knots (by construction) but diverged between them.  The blend was not
machine-epsilon accurate; it was merely continuous.

### Vertex P₀: an intrinsic anchor (sessions 5, 2026-03-17)

The vertex of a conic (the point where the gradient `∇f = 0` in the tangent direction,
i.e. `L = 2Ax + By + D = 0`) is an algebraic invariant: every window on the same conic
computes the same vertex.  Using the vertex as the stereographic base P₀ means:

- All windows produce the same `s₀ = −L/M = 0` at the vertex.
- The stereographic slopes `sᵢ = (yᵢ − y₀)/(xᵢ − x₀)` from the vertex are the same
  physical quantities for any window on the same conic.
- PCHIP on the same `φ = 2·arctan(sᵢ)` values produces the same orbit.

The **physical sign convention** (first principal axis has positive x-component in the
global frame) breaks the remaining P₀ ambiguity from the eigendecomposition sign freedom.
Together these make the orbit function a genuine conic invariant, not a per-window artifact.

**Result:** Hyperbola n=8 blend lies on the conic to `|x²−y²−1| < 5e-11`.
**Trade-off:** Lissajous n regression 29→38 — vertex P₀ is less favorable for non-conic
windows where the old first-control-point P₀ happened to give a more linear φ(t).

### Circle: equal eigenvalues → no consistent vertex direction

For a circle `A = C`, `B = 0` → both eigenvalues equal.  The eigenvectors are
arbitrary; the principal frame direction is undefined.  Vertex P₀ is not computable in
a rotationally-consistent way.

### Near-circle eccentric-anomaly orbit (session 4, 2026-03-17)

For near-circular ellipses (`eval_sep < 0.2`), the eccentric anomaly
`E = arctan2((y − cy)/b, (x − cx)/a)` is used instead of the stereographic φ:

- `x = cx + a·cos(E)`,  `y = cy + b·sin(E)`  in the principal frame.
- The center `(cx, cy)` is computed from the quadratic-form coefficients and is
  independent of the eigenvector sign ambiguity.  All windows on the same circle
  compute the same `(cx, cy)` to floating-point precision.
- For circles `a = b`, so E equals the central angle θ, and E(t) is exactly linear
  for uniformly-spaced parameter values → PCHIP interpolates it without error.
- For near-circular ellipses, E(t) is closer to linear than the stereographic φ(t).

**Result:** Circle blend: `|x²+y²−1| < 1e-15` (machine epsilon).

### Eccentric-anomaly orbit replaces theta orbit (session 6, 2026-03-17)

The original near-circle code used a polar orbit `r(θ) = √(−Fc/Q(θ))` with
`x = cx + r·cos(θ)`, `y = cy + r·sin(θ)`.  This required computing `Q(θ)` (the
quadratic form in angle-direction cosines) and taking a square root at every evaluation.

Replaced with the eccentric-anomaly form `x = cx + a·cos(E)`, `y = cy + b·sin(E)` in
the principal frame.  The two are mathematically equivalent for ellipses but the E form
is cleaner: no `Q(θ)` denominators, no `r(θ)` radicals, symmetric in the two semi-axes.
All 9 scan baselines and all exact-conic tests are unchanged.

### Geometric unification: E / D / H family

The three conic types each have a natural principal-frame orbit:

| Type | det | Parameter | Orbit |
|------|-----|-----------|-------|
| Ellipse | > 0 | E (eccentric anomaly) | `cx + a·cos(E)`,  `cy + b·sin(E)` |
| Parabola | = 0 | D (parabolic anomaly) | `x_v + t`,  `y_v + t²/(2p)` |
| Hyperbola | < 0 | φ = 2·arctan(s − s₀) (stereographic) | rational projective map |

These are related: `D = tan(E/2)` at the parabola limit `e → 1`, and the stereographic
φ for hyperbolas is the imaginary-angle analogue of E.  The code uses E for ellipses and
the stereographic map for hyperbolas/parabolas; the structural asymmetry is a
computational choice (the stereographic map handles the asymptote crossing in RP²
naturally), not a geometric one.

The `eval_sep < 0.2` guard in `_build_projective_arc_window` restricts the E orbit to
near-circles as a *computation* optimization: for elongated same-branch ellipses, `_try_conic`
would discard the E orbit anyway (falling through to the arc-length quality check), so
the guard just avoids building an orbit that will not be used.  The load-bearing
threshold is `_near_circle` in `_try_conic`.

---

## Summary table of dead code

| Symbol | File | Reason dead |
|--------|------|-------------|
| `_make_quintic_hermite` | conicspline.py | Per-segment Hermite bypassed smoothstep → kinks; superseded by `_make_spline_window` |
| `_make_conic_clamped_spline` | conicspline.py | Per-segment conic-BC spline, same kink problem; superseded by `_make_spline_window` |
| `_build_conic_tangent_spline` | conicspline.py | Hermite with asymptote-directed tangents → infinite curvature near VdP spike |
| `if False:` tiers 2–3 | conicspline.py | `_try_kepler_time`, `_try_gen_polar` never triggered; tier 1 always succeeds or falls back to spline |
| Case 2 (`through_inf`) | conicspline.py | Proven unreachable: Q_v cannot change sign when Q_s does not (canonical P₀ geometry) |
| Theta block | conicspline.py | `_is_conic_monotone` pre-filter guarantees phi_mono=True; 0 theta hits at converged n |
| `seg_funcs` dict / `if seg_funcs and j in seg_funcs:` | conicspline.py | Mechanism for per-segment override; always `{}` since `_make_spline_window` replaced it |
| `adaptive_n` | conicspline.py | Geometric heuristics approach; superseded by `adaptive_n_budget` (ground-truth deviation) |
