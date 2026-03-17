"""
tests/hunt_edge_cases.py
--------------------------
Investigates two theoretically possible but unobserved edge cases:

  #4  Same-branch window with non-monotone phi
      MEMORY says "same-branch windows: phi always monotone" — verify or refute.
      If non-monotone phi reaches the phi orbit (is_cross_branch=False), the
      bad orbit is only caught by ctrl_err, not by any explicit gate.

  #5  Theta fallback with overall_dir=0 → returns None
      Requires a cross-branch window where same-branch sub-point direction
      alternates.  Currently listed as "practically unreachable."

Monkey-patches _build_projective_arc_window to intercept both cases.

Run from repo root:
    python3 tests/hunt_edge_cases.py

Exit 0 always; prints a summary of any hits found.
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
spec  = importlib.util.spec_from_file_location('bd', os.path.join(_here, '..', 'blend_demo.py'))
bd    = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

import demo_n_transitions as dn

# ── Counters / hit records ────────────────────────────────────────────────────
hits = {
    'same_branch_nonmono_phi': [],   # (curve, n, window_i, phi_diffs)
    'theta_overall_dir_zero':  [],   # (curve, n, window_i)
}

_curve_label = ['?']   # mutable cell for current curve name + n

_orig_build = bd._build_projective_arc_window


def _patched_build(pts_2d, pts, ts, center3d, e1, e2, coeffs):
    A, B, C, D, E, F = coeffs

    det = 4.0 * A * C - B * B
    if det == 0.0:
        return _orig_build(pts_2d, pts, ts, center3d, e1, e2, coeffs)

    # ── Classify cross/same branch ────────────────────────────────────────
    cx = (-2*C*D + B*E) / det
    cy = (-2*A*E + B*D) / det
    dx = pts_2d[:, 0] - cx;  dy = pts_2d[:, 1] - cy
    M_mat = np.array([[A, B/2], [B/2, C]])
    ev, evec = np.linalg.eigh(M_mat)
    is_hyp = ev[0] * ev[1] < 0
    if is_hyp:
        pidx = 1 if ev[1] > 0 else 0
        et   = evec[:, pidx]
        prj  = dx*et[0] + dy*et[1]
        br   = np.sign(prj)
        cross = (br[:4] * br[1:]) < 0
        is_cross_branch = bool(cross.any())
    else:
        is_cross_branch = False
        cross = np.zeros(4, dtype=bool)

    # ── Canonical P₀ ─────────────────────────────────────────────────────
    xs, ys = pts_2d[:, 0], pts_2d[:, 1]
    L_all = 2*A*xs + B*ys + D
    M_all = B*xs + 2*C*ys + E
    Ms    = np.where(np.abs(M_all) >= 1e-15, M_all, np.sign(M_all+1e-300)*1e-15)
    k0    = int(np.argmin(np.abs(L_all / Ms)))
    L_e, M_e = L_all[k0], M_all[k0]
    if abs(M_e) < 1e-9 * max(abs(L_e), 1.0):
        pe2 = pts_2d[:, ::-1]; A2,C2,D2,E2 = C,A,E,D
        L2 = 2*A2*pe2[:,0]+B*pe2[:,1]+D2; M2 = B*pe2[:,0]+2*C2*pe2[:,1]+E2
        Ms2 = np.where(np.abs(M2)>=1e-15, M2, np.sign(M2+1e-300)*1e-15)
        k0 = int(np.argmin(np.abs(L2/Ms2))); L_e,M_e = L2[k0],M2[k0]
    s0 = -L_e / M_e if abs(M_e) > 1e-15 else 0.0
    x0, y0 = pts_2d[k0]
    dxi = pts_2d[:,0]-x0; dyi = pts_2d[:,1]-y0
    mask = np.abs(dxi) >= 1e-9*(np.abs(dyi)+1.)
    dxi_s = np.where(mask, dxi, 1.)
    s_vals = np.where(mask, dyi/dxi_s, np.sign(dyi+1e-300)*1e15)
    s_vals[k0] = s0

    # ── Phi monotonicity ──────────────────────────────────────────────────
    phi = np.unwrap(2.0 * np.arctan(s_vals - s0))
    d   = np.diff(phi)
    phi_mono = np.all(d > 0) or np.all(d < 0)

    # ── #4: same-branch non-monotone phi ─────────────────────────────────
    if not is_cross_branch and not phi_mono:
        hits['same_branch_nonmono_phi'].append(
            (_curve_label[0], list(np.round(d, 4)))
        )

    # ── #5: theta overall_dir=0 detection ────────────────────────────────
    # Replicate the theta logic to check overall_dir
    if is_cross_branch and not phi_mono:
        same_mask = ~cross
        same_mask_full = np.concatenate([[True], same_mask]) & \
                         np.concatenate([same_mask, [True]])
        same_pts = pts_2d[same_mask_full[:len(pts_2d)]] if same_mask_full.any() else pts_2d[:0]
        if len(same_pts) >= 2:
            s_same = s_vals[same_mask_full[:len(s_vals)]]
            same_diffs = np.diff(s_same)
            if len(same_diffs) > 0:
                overall_dir = int(np.sign(np.sum(same_diffs)))
                if overall_dir == 0:
                    hits['theta_overall_dir_zero'].append(_curve_label[0])

    return _orig_build(pts_2d, pts, ts, center3d, e1, e2, coeffs)


bd._build_projective_arc_window = _patched_build

# ── Scan all DEMO_CURVES, n=8..80 ────────────────────────────────────────────
all_curves = [(e[0], e[1], e[2]) for e in dn.DEMO_CURVES]

for name, xy_func, t_range in all_curves:
    for n in range(8, 81):
        _curve_label[0] = f'{name} n={n}'
        pts, times = bd.sample_curve(xy_func, t_range, n)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bd._run_blend(pts, times, False, 2)

bd._build_projective_arc_window = _orig_build

# ── Report ────────────────────────────────────────────────────────────────────
print('── #4: Same-branch non-monotone phi ──────────────────────────────────────')
sb = hits['same_branch_nonmono_phi']
if sb:
    print(f'  Found {len(sb)} case(s):')
    for label, diffs in sb[:10]:
        print(f'    {label}  diffs={diffs}')
    if len(sb) > 10:
        print(f'  … and {len(sb)-10} more')
else:
    print('  None found. Same-branch phi is always monotone across n=8..80.')

print()
print('── #5: Theta overall_dir=0 ───────────────────────────────────────────────')
td = hits['theta_overall_dir_zero']
if td:
    print(f'  Found {len(td)} case(s):')
    for label in td[:10]:
        print(f'    {label}')
else:
    print('  None found. overall_dir=0 never triggered across n=8..80.')
