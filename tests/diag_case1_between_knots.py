"""
tests/diag_case1_between_knots.py
----------------------------------
For every Case 1 (finite-asymptote cross-branch) window found during a DEMO_CURVES
run, plot the per-case vs unified orbit densely between knots, alongside:
  - the true underlying curve for comparison
  - phi(t) and s(t) for both approaches
  - the deviation from the true curve at each t

The goal is to see what each approach "guesses" the curve does between control points.

Run from repo root:
    python3 tests/diag_case1_between_knots.py
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import importlib.util
import conicspline as bl

# ── Load demo_n_transitions ───────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
spec  = importlib.util.spec_from_file_location('dn', os.path.join(_here, '..', 'demo_n_transitions.py'))
dn    = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dn)

OUT_DIR = os.path.join(_here, '..', 'sandbox')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Capture Case 1 windows ────────────────────────────────────────────────────
case1_windows = []     # list of dicts with full window data + curve context

_orig_build = bl._build_projective_arc_window

def _capture(p2d, p, ts, ctr, e1, e2, cffs):
    A, B, C, D, E, F = cffs
    det = 4*A*C - B*B
    if det == 0.0:
        return _orig_build(p2d, p, ts, ctr, e1, e2, cffs)

    cx = (-2*C*D + B*E) / det
    cy = (-2*A*E + B*D) / det
    dx = p2d[:, 0] - cx
    dy = p2d[:, 1] - cy

    M_mat  = np.array([[A, B/2.0], [B/2.0, C]])
    ev, evec = np.linalg.eigh(M_mat)
    if ev[0]*ev[1] >= 0:
        return _orig_build(p2d, p, ts, ctr, e1, e2, cffs)

    pos_idx = 1 if ev[1] > 0 else 0
    et = evec[:, pos_idx]
    prj = dx*et[0] + dy*et[1]
    br  = np.sign(prj)
    cross = (br[:4] * br[1:]) < 0
    if not cross.any():
        return _orig_build(p2d, p, ts, ctr, e1, e2, cffs)

    # Replicate canonical P0 selection
    def sel_k0(pe, Ae, Ce, De, Ee):
        xs, ys = pe[:, 0], pe[:, 1]
        L = 2*Ae*xs + B*ys + De
        M = B*xs + 2*Ce*ys + Ee
        Ms = np.where(np.abs(M) >= 1e-15, M, np.sign(M+1e-300)*1e-15)
        return int(np.argmin(np.abs(L/Ms))), L, M

    pts_e = p2d.copy()
    A_e, C_e, D_e, E_e = A, C, D, E
    k0, L_all, M_all = sel_k0(pts_e, A_e, C_e, D_e, E_e)
    L_e, M_e = L_all[k0], M_all[k0]
    if abs(M_e) < 1e-9 * max(abs(L_e), 1.0):
        pts_e = pts_e[:, ::-1]; A_e, C_e, D_e, E_e = C, A, E, D
        k0, L_all, M_all = sel_k0(pts_e, A_e, C_e, D_e, E_e)
        L_e, M_e = L_all[k0], M_all[k0]
    if abs(M_e) < 1e-15:
        return _orig_build(p2d, p, ts, ctr, e1, e2, cffs)

    s0 = -L_e / M_e
    x0, y0 = pts_e[k0]
    dxi = pts_e[:, 0] - x0
    dyi = pts_e[:, 1] - y0
    mask = np.abs(dxi) >= 1e-9*(np.abs(dyi)+1.)
    dxi_s = np.where(mask, dxi, 1.0)
    s_vals = np.where(mask, dyi/dxi_s, np.sign(dyi+1e-300)*1e15)
    s_vals[k0] = s0

    Q_vals = A_e + B*s_vals + C_e*s_vals**2
    q_ch = np.where(np.diff(np.sign(Q_vals)) != 0)[0]

    if len(q_ch) >= 1:   # Case 1 only
        case1_windows.append(dict(
            pts_2d=p2d.copy(), pts_e=pts_e.copy(), pts=p.copy(), ts=ts.copy(),
            coeffs=cffs, s_vals=s_vals.copy(), s0=float(s0), k0=int(k0),
            L_e=float(L_e), M_e=float(M_e), x0=float(x0), y0=float(y0),
            center3d=ctr.copy(), e1=e1.copy(), e2=e2.copy(),
            A_e=float(A_e), C_e=float(C_e), cross=cross.copy(),
            q_changes=q_ch.copy(), Q_vals=Q_vals.copy(),
            curve_name=_current_curve[0],
            xy_func=_current_curve[1],
        ))

    return _orig_build(p2d, p, ts, ctr, e1, e2, cffs)

_current_curve = ['', None]
bl._build_projective_arc_window = _capture

for entry in dn.DEMO_CURVES:
    name, xy_func, t_range = entry[0], entry[1], entry[2]
    _current_curve[:] = [name, xy_func]
    panels = dn.find_transitions(xy_func, t_range, n_min=8, n_max=60)
    for n, _ in panels:
        pts, times = bl.sample_curve(xy_func, t_range, n)
        bl._run_blend(pts, times, False, 2)

bl._build_projective_arc_window = _orig_build
print(f"Captured {len(case1_windows)} Case 1 windows.\n")

if not case1_windows:
    print("No Case 1 windows found — exiting.")
    sys.exit(0)

# ── Classify which windows are accepted vs rejected ───────────────────────────
# A Case 1 window is accepted when the asymptote falls OUTSIDE the blend region
# [ts[1], ts[3]], so orbit_func produces no NaN there.
accepted = []
rejected = []
for ex in case1_windows:
    result = _orig_build(ex['pts_2d'], ex['pts'], ex['ts'],
                         ex['center3d'], ex['e1'], ex['e2'], ex['coeffs'])
    if result is not None:
        accepted.append(ex)
    else:
        rejected.append(ex)

print(f"  Accepted (non-None):  {len(accepted)}")
print(f"  Rejected (None/NaN):  {len(rejected)}\n")

# For accepted windows, compute |s* - s₀| as a measure of how different the
# per-case and unified parameterizations are, and sort by that.
def _s_star_for(ex):
    q_ch = ex['q_changes']
    idx = q_ch[0]
    s_lo, s_hi = sorted([ex['s_vals'][idx], ex['s_vals'][idx+1]])
    quad_roots = np.roots([ex['C_e'], ex['coeffs'][1], ex['A_e']])
    real_qr = quad_roots[np.abs(quad_roots.imag) < 1e-9*(np.abs(quad_roots.real)+1.)].real
    cands = real_qr[(real_qr >= s_lo-1e-10) & (real_qr <= s_hi+1e-10)]
    return float(np.clip(cands[0], s_lo, s_hi)) if len(cands) > 0 else np.nan

for ex in accepted:
    ex['s_star'] = _s_star_for(ex)

accepted_valid = [ex for ex in accepted if np.isfinite(ex.get('s_star', np.nan))]
accepted_valid.sort(key=lambda ex: abs(ex['s_star'] - ex['s0']), reverse=True)

print(f"Case 1 windows with largest |s* - s₀| (accepted by _orig_build):")
for ex in accepted_valid[:12]:
    print(f"  {ex['curve_name']:25s}  s*={ex['s_star']:+.4f}  s₀={ex['s0']:+.4f}  "
          f"Δ={abs(ex['s_star']-ex['s0']):.4f}  s_vals={np.round(ex['s_vals'],3)}")
print()

# Use these for plotting
plot_windows = accepted_valid  # already sorted by Δ


# ── Unified orbit builder ─────────────────────────────────────────────────────
def make_unified_orbit(ex):
    """phi = 2·arctan(s − s₀), unwrapped.  Returns (phi_u, orbit_func)."""
    s_vals, s0, ts = ex['s_vals'], ex['s0'], ex['ts']
    A_e, B_, C_e  = ex['A_e'], ex['coeffs'][1], ex['C_e']
    L_e, M_e      = ex['L_e'], ex['M_e']
    x0, y0        = ex['x0'], ex['y0']
    ctr, e1, e2   = ex['center3d'], ex['e1'], ex['e2']

    phi_vals = 2.0 * np.arctan(s_vals - s0)
    phi_u    = np.unwrap(phi_vals)
    phi_of_t = PchipInterpolator(ts, phi_u)

    def orbit(t):
        t   = np.atleast_1d(t).astype(float)
        phi = phi_of_t(t)
        s   = s0 + np.tan(phi / 2.0)
        Q   = A_e + B_*s + C_e*s**2
        v   = np.abs(Q) >= 1e-12
        Qs  = np.where(v, Q, 1.0)
        u   = np.where(v, -(L_e + s*M_e) / Qs, np.nan)
        # If pts_e was x<->y swapped, swap back
        if ex.get('swapped', False):
            lx, ly = x0 + s*u, y0 + u
        else:
            lx, ly = x0 + u, y0 + s*u
        return ctr + lx[:, None]*e1 + ly[:, None]*e2

    return phi_u, orbit


# ── Per-case orbit builder ─────────────────────────────────────────────────────
def make_percase_orbit(ex):
    """Per-case: phi = arctan(s − s*) where s* = actual asymptote root."""
    s_vals, ts = ex['s_vals'], ex['ts']
    A_e, B_, C_e = ex['A_e'], ex['coeffs'][1], ex['C_e']
    L_e, M_e     = ex['L_e'], ex['M_e']
    x0, y0       = ex['x0'], ex['y0']
    ctr, e1, e2  = ex['center3d'], ex['e1'], ex['e2']
    q_ch         = ex['q_changes']

    idx = q_ch[0]
    s_lo, s_hi = sorted([s_vals[idx], s_vals[idx+1]])
    quad_roots = np.roots([C_e, B_, A_e])
    real_qr = quad_roots[np.abs(quad_roots.imag) < 1e-9*(np.abs(quad_roots.real)+1.)].real
    cands = real_qr[(real_qr >= s_lo - 1e-10) & (real_qr <= s_hi + 1e-10)]
    if len(cands) == 0:
        return None, None
    s_star = float(np.clip(cands[0], s_lo, s_hi))

    phi_vals = np.arctan(s_vals - s_star)
    phi_of_t = PchipInterpolator(ts, phi_vals)

    _s_star = s_star

    def orbit(t):
        t   = np.atleast_1d(t).astype(float)
        phi = phi_of_t(t)
        s   = np.tan(phi) + _s_star
        Q   = A_e + B_*s + C_e*s**2
        v   = np.abs(Q) >= 1e-12
        Qs  = np.where(v, Q, 1.0)
        u   = np.where(v, -(L_e + s*M_e) / Qs, np.nan)
        lx, ly = x0 + u, y0 + s*u
        return ctr + lx[:, None]*e1 + ly[:, None]*e2

    return phi_vals, orbit


# ── True-curve deviation ───────────────────────────────────────────────────────
def true_3d(ex, t_dense):
    """Evaluate the underlying parametric curve in 3D (it's planar so z=0)."""
    xy = np.array([ex['xy_func'](tt) for tt in t_dense])
    # embed in 3D using same basis as conicspline (center3d, e1, e2)
    # The 2D points ex['pts'] were embedded as center3d + pts_2d[:,0]*e1 + pts_2d[:,1]*e2
    # So 3D true curve: center3d + xy[:,0]*e1 + xy[:,1]*e2   but we need to
    # check if the window's e1/e2/center3d match the original sampling.
    # The safest approach: just use xy directly (both orbits should match xy).
    ctr, e1, e2 = ex['center3d'], ex['e1'], ex['e2']
    # xy is in the original 2D plane; we need xy relative to the window's local frame
    # approximate: true_3d = ctr + xy[:,0]*e1 + xy[:,1]*e2
    return ctr + xy[:, 0:1]*e1 + xy[:, 1:2]*e2


# ── Plot each Case 1 window ────────────────────────────────────────────────────
MAX_PLOTS = 8    # limit output

for wi, ex in enumerate(plot_windows[:MAX_PLOTS]):
    phi_uni, orbit_uni  = make_unified_orbit(ex)
    phi_pc, orbit_pc    = make_percase_orbit(ex)
    if orbit_pc is None:
        print(f"  window {wi}: no s* found — skipping")
        continue

    ts = ex['ts']
    # Identify pairs that straddle the asymptote (cross-branch adjacent pairs)
    cross = ex['cross']

    # Dense t grid
    t_dense = np.linspace(ts[0], ts[-1], 400)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pt_uni  = orbit_uni(t_dense)
        pt_pc   = orbit_pc(t_dense)
        ct_uni  = orbit_uni(ts)
        ct_pc   = orbit_pc(ts)

    # True curve
    try:
        pt_true = true_3d(ex, t_dense)
        have_true = not np.any(np.isnan(pt_true))
    except Exception:
        have_true = False

    # Project back to 2D for plotting
    ctr, e1, e2 = ex['center3d'], ex['e1'], ex['e2']
    def to2d(p3d):
        d = p3d - ctr
        return np.stack([d @ e1, d @ e2], axis=1)

    xy_uni   = to2d(pt_uni)
    xy_pc    = to2d(pt_pc)
    xy_true  = to2d(pt_true) if have_true else None
    xy_ctrl  = to2d(ex['pts'])

    # phi(t) and s(t) along dense t
    phi_uni_d = PchipInterpolator(ts, phi_uni)(t_dense)
    phi_pc_d  = PchipInterpolator(ts, phi_pc)(t_dense)
    s0 = ex['s0']
    s_uni_d   = s0 + np.tan(phi_uni_d / 2.0)
    q_ch      = ex['q_changes']
    idx       = q_ch[0]
    s_lo, s_hi = sorted([ex['s_vals'][idx], ex['s_vals'][idx+1]])
    quad_roots = np.roots([ex['C_e'], ex['coeffs'][1], ex['A_e']])
    real_qr    = quad_roots[np.abs(quad_roots.imag) < 1e-9*(np.abs(quad_roots.real)+1.)].real
    cands      = real_qr[(real_qr >= s_lo-1e-10) & (real_qr <= s_hi+1e-10)]
    s_star     = float(np.clip(cands[0], s_lo, s_hi)) if len(cands) > 0 else np.nan
    s_pc_d     = np.tan(phi_pc_d) + s_star

    # Deviation from true
    if have_true:
        dev_uni = np.linalg.norm(xy_uni - xy_true, axis=1)
        dev_pc  = np.linalg.norm(xy_pc  - xy_true, axis=1)
    else:
        dev_uni = dev_pc = None

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    curve_name = ex['curve_name']
    fig.suptitle(f'{curve_name} — Case 1 window #{wi}  '
                 f'(s_vals = {np.round(ex["s_vals"],3)}, s*={s_star:.4f}, s₀={s0:.4f})',
                 fontsize=9)

    # ── Subplot 1: XY orbit ───────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_title('Orbit in 2D  (dense between knots)', fontsize=8)
    if have_true and xy_true is not None:
        ax.plot(xy_true[:, 0], xy_true[:, 1], color='#AAAAAA', lw=2.5,
                label='true curve', zorder=1)
    ax.plot(xy_pc[:, 0],  xy_pc[:, 1],  color='#1565C0', lw=1.5,
            label='per-case (s*)', zorder=3)
    ax.plot(xy_uni[:, 0], xy_uni[:, 1], color='#E65100', lw=1.5,
            linestyle='--', label='unified (s₀)', zorder=3, alpha=0.85)
    ax.scatter(xy_ctrl[:, 0], xy_ctrl[:, 1], s=45, c='k', zorder=5,
               label='ctrl pts')
    # Mark the asymptote-straddling pair(s)
    for ci, is_cross in enumerate(cross):
        if is_cross:
            ax.axvline(x=np.nan)   # placeholder; mark segment differently
            ax.plot(xy_ctrl[ci:ci+2, 0], xy_ctrl[ci:ci+2, 1],
                    color='red', lw=2, alpha=0.5, zorder=4)
    ax.legend(fontsize=7)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal', 'datalim')

    # ── Subplot 2: phi(t) ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title('φ(t): PCHIP parameter', fontsize=8)
    ax.plot(t_dense, np.degrees(phi_uni_d), color='#E65100', lw=1.5,
            label='unified φ (°)')
    ax.plot(t_dense, np.degrees(phi_pc_d),  color='#1565C0', lw=1.5,
            linestyle='--', label='per-case φ (°)')
    ax.scatter(ts, np.degrees(phi_uni), s=30, c='#E65100', zorder=5)
    ax.scatter(ts, np.degrees(phi_pc),  s=30, c='#1565C0', marker='^', zorder=5)
    ax.set_xlabel('t'); ax.set_ylabel('φ (degrees)')
    ax.legend(fontsize=7)
    # Draw vertical lines at ts
    for tt in ts:
        ax.axvline(tt, color='k', lw=0.4, alpha=0.3)

    # ── Subplot 3: s(t) ────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_title('s(t): stereographic slope', fontsize=8)
    # Clip for display — asymptote causes ±∞
    CLIP = 30.0
    ax.plot(t_dense, np.clip(s_uni_d, -CLIP, CLIP), color='#E65100', lw=1.5,
            label='unified s(t)')
    ax.plot(t_dense, np.clip(s_pc_d,  -CLIP, CLIP), color='#1565C0', lw=1.5,
            linestyle='--', label='per-case s(t)')
    ax.scatter(ts, np.clip(ex['s_vals'], -CLIP, CLIP), s=30, c='k', zorder=5,
               label='ctrl s_vals')
    ax.axhline(s_star, color='red',     lw=1, linestyle=':', label=f's*={s_star:.3f}')
    ax.axhline(s0,     color='#888888', lw=1, linestyle=':', label=f's₀={s0:.3f}')
    ax.set_ylim(-CLIP, CLIP)
    ax.set_xlabel('t'); ax.set_ylabel('s  (clipped to ±30)')
    ax.legend(fontsize=7)
    for tt in ts:
        ax.axvline(tt, color='k', lw=0.4, alpha=0.3)

    # ── Subplot 4: deviation from true ─────────────────────────────────────────
    ax = axes[1, 1]
    ax.set_title('Deviation from true curve', fontsize=8)
    if dev_uni is not None:
        ax.semilogy(t_dense, dev_uni + 1e-12, color='#E65100', lw=1.5,
                    label='unified')
        ax.semilogy(t_dense, dev_pc  + 1e-12, color='#1565C0', lw=1.5,
                    linestyle='--', label='per-case')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'true curve\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
    ax.set_xlabel('t'); ax.set_ylabel('‖orbit − true‖')
    for tt in ts:
        ax.axvline(tt, color='k', lw=0.4, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f'diag_case1_knots_{wi:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [{wi}] {curve_name:25s}  s*={s_star:+.4f} s₀={s0:+.4f}  "
          f"s_vals={np.round(ex['s_vals'], 3)}  →  {fname}")

print(f"\nPlots saved to {OUT_DIR}/diag_case1_knots_*.png")
