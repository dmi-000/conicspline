"""
tests/branch_coverage.py
------------------------
Instruments every meaningful decision point in conicspline and reports
which branches are hit (and which are still at zero) when run over
all DEMO_CURVES at their transition n-values.

Run from the repo root:
    python3 tests/branch_coverage.py

Exit code 0  → every non-dead branch was hit at least once.
Exit code 1  → some non-dead branch was never hit (list printed).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import importlib.util
import numpy as np

# ── Load conicspline and demo_n_transitions ────────────────────────────────────
import conicspline as bl

_here  = os.path.dirname(os.path.abspath(__file__))
_dn_path = os.path.join(_here, '..', 'demo_n_transitions.py')
spec   = importlib.util.spec_from_file_location('dn', _dn_path)
dn     = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dn)


# ── Hit counters ─────────────────────────────────────────────────────────────
hits = {
    # Branch classification
    'bpaw_same_branch':           0,   # is_cross_branch=False (ellipse / same-branch hyp.)
    'bpaw_cross_branch':          0,   # is_cross_branch=True

    # x↔y swap inside _build_projective_arc_window
    'bpaw_swapped':               0,   # canonical P₀ has near-vertical tangent

    # Phi-first dispatch
    'bpaw_phi_monotone':          0,   # phi = 2·arctan(s−s₀) is monotone → use it
    'bpaw_phi_nonmono_theta':     0,   # phi non-monotone + cross-branch → theta fallback

    # Theta sub-paths ──────────────────────────────────────────────────────
    # NOTE: theta is only entered from is_cross_branch=True, so cross.any()
    # is ALWAYS True; the else-branch (bpaw_theta_no_cross) is DEAD CODE.
    'bpaw_theta_cross_any_True':  0,   # theta: cross.any() True (always)
    'bpaw_theta_cross_True_ok':   0,   # theta: direction found, returned successfully

    # Cross-branch return outcomes
    'bpaw_proj_return_none':      0,   # cross-branch _bpaw returned None (no cand root or ctrl NaN)
    'bpaw_proj_success':          0,   # cross-branch _bpaw returned is_cross_branch=True

    # Same-branch return
    'bpaw_case0_returned':        0,   # _bpaw returned is_cross_branch=False

    # _is_conic_monotone (ICM) paths ──────────────────────────────────────
    'icm_swap':                   0,   # x'↔y' swap triggered (canonical P₀ near-vertical)
    'icm_qs_on_asymptote':        0,   # |Qs| < 1e-10 at some control point → False
    'icm_n_cross_ge2':            0,   # n_cross ≥ 2 → False (multiple asymptote crossings)
    'icm_monotone_inc':           0,   # all s-diffs > 0  → True
    'icm_monotone_dec':           0,   # all s-diffs < 0  → True
    'icm_non_monotone':           0,   # s-diffs inconsistent → False
}

# Branches that are proven dead or practically unreachable (excluded from
# the zero-hit failure check):
DEAD = {
    # Case 3 internal — proven dead:
    # 'bpaw_theta_no_cross' would be the else-branch of `if cross.any():`
    # inside theta, but Case 3 is only entered when is_cross_branch=True,
    # so cross.any() is always True → else never taken.

    # Practically unreachable with smooth parameterised sampling:
    'bpaw_proj_return_none',    # Q root always exists by IVT; theta fits exactly
    'bpaw_swapped',             # requires aspect-ratio ≥ 1e9 (see tests/synthetic_coverage.py)
    'icm_swap',                 # same as bpaw_swapped (same geometry condition)
}


# ── Patch _build_projective_arc_window ───────────────────────────────────────
_orig_build = bl._build_projective_arc_window

def _patched_build(pts_2d, pts, ts, center3d, e1, e2, coeffs):
    A, B, C, D, E, F = coeffs

    # ── Replicate entry logic for classification ──────────────────────────
    det = 4.0 * A * C - B * B
    if det == 0.0:
        return _orig_build(pts_2d, pts, ts, center3d, e1, e2, coeffs)

    cx = (-2.0 * C * D + B * E) / det
    cy = (-2.0 * A * E + B * D) / det
    dx = pts_2d[:, 0] - cx
    dy = pts_2d[:, 1] - cy

    M_mat  = np.array([[A, B / 2.0], [B / 2.0, C]])
    eigvals, eigvecs = np.linalg.eigh(M_mat)
    is_hyp = eigvals[0] * eigvals[1] < 0

    if is_hyp:
        pos_idx    = 1 if eigvals[1] > 0 else 0
        e_t        = eigvecs[:, pos_idx]
        proj_eig   = dx * e_t[0] + dy * e_t[1]
        branch     = np.sign(proj_eig)
        cross          = (branch[:4] * branch[1:]) < 0
        is_cross_branch = bool(cross.any())
    else:
        is_cross_branch = False
        cross = np.zeros(4, dtype=bool)

    if is_cross_branch:
        hits['bpaw_cross_branch'] += 1
    else:
        hits['bpaw_same_branch'] += 1

    # ── Check canonical P₀ swap ───────────────────────────────────────────
    pts_e = pts_2d.copy()
    A_e, C_e, D_e, E_e = A, C, D, E

    def _sel_k0(pe, Ae, Be, Ce, De, Ee):
        xs, ys = pe[:, 0], pe[:, 1]
        L  = 2.0*Ae*xs + Be*ys + De
        M  = Be*xs + 2.0*Ce*ys + Ee
        Ms = np.where(np.abs(M) >= 1e-15, M, np.sign(M + 1e-300) * 1e-15)
        return int(np.argmin(np.abs(L / Ms))), L, M

    k0, L_all, M_all = _sel_k0(pts_e, A_e, B, C_e, D_e, E_e)
    L_e, M_e = L_all[k0], M_all[k0]
    if abs(M_e) < 1e-9 * max(abs(L_e), 1.0):
        hits['bpaw_swapped'] += 1

    # ── Phi-first dispatch (mirrors conicspline logic) ──────────────────────
    # Replicate s_vals / s0 computation to evaluate phi monotonicity.
    if abs(M_e) > 1e-15:
        s0_loc = -L_e / M_e
        x0_loc, y0_loc = pts_e[k0]
        dxi_loc = pts_e[:, 0] - x0_loc
        dyi_loc = pts_e[:, 1] - y0_loc
        mask_loc = np.abs(dxi_loc) >= 1e-9 * (np.abs(dyi_loc) + 1.0)
        dxi_s_loc = np.where(mask_loc, dxi_loc, 1.0)
        s_loc = np.where(mask_loc, dyi_loc / dxi_s_loc, np.sign(dyi_loc + 1e-300) * 1e15)
        s_loc[k0] = s0_loc
        phi_loc = np.unwrap(2.0 * np.arctan(s_loc - s0_loc))
        phi_mono_loc = (np.all(np.diff(phi_loc) > 0) or np.all(np.diff(phi_loc) < 0))

        if phi_mono_loc:
            hits['bpaw_phi_monotone'] += 1
        elif is_cross_branch:
            hits['bpaw_phi_nonmono_theta'] += 1
            hits['bpaw_theta_cross_any_True'] += 1   # cross.any() always True in theta

    # ── Run original and classify return outcome ──────────────────────────
    result = _orig_build(pts_2d, pts, ts, center3d, e1, e2, coeffs)

    if not is_cross_branch:
        if result is not None:
            hits['bpaw_case0_returned'] += 1
    else:
        if result is None:
            hits['bpaw_proj_return_none'] += 1
        else:
            _, ctrl_err, is_cb = result
            hits['bpaw_proj_success'] += 1

    return result


# ── Patch _is_conic_monotone ─────────────────────────────────────────────────
_orig_icm = bl._is_conic_monotone

def _patched_icm(pts5_xy):
    pts5_xy = np.asarray(pts5_xy, dtype=float)
    try:
        A, B, C, D, E, F = bl.fit_conic(pts5_xy)
    except Exception:
        return _orig_icm(pts5_xy)

    M_mat = np.array([[A, B * 0.5], [B * 0.5, C]])
    try:
        evals, evecs = np.linalg.eigh(M_mat)
    except np.linalg.LinAlgError:
        return _orig_icm(pts5_xy)

    sort_idx = np.argsort(np.abs(evals))
    evals    = evals[sort_idx]
    evecs    = evecs[:, sort_idx]

    xs_r = pts5_xy @ evecs[:, 0]
    ys_r = pts5_xy @ evecs[:, 1]
    A_r  = float(evals[0])
    C_r  = float(evals[1])
    D_r, E_r = np.array([D, E]) @ evecs
    Lk   = 2.0*A_r*xs_r + D_r
    Mk   = 2.0*C_r*ys_r + E_r
    ratio = np.where(np.abs(Mk) > 1e-15, np.abs(Lk) / np.abs(Mk), 1e30)
    k0    = int(np.argmin(ratio))
    L, M  = Lk[k0], Mk[k0]

    xs_w, ys_w       = xs_r.copy(), ys_r.copy()
    A_w, C_w, D_w, E_w = A_r, C_r, D_r, E_r

    if abs(M) < 1e-9 * max(abs(L), 1.0):
        hits['icm_swap'] += 1
        A_w, C_w = C_r, A_r
        D_w, E_w = E_r, D_r
        xs_w, ys_w = ys_r.copy(), xs_r.copy()
        x0_w, y0_w = xs_w[k0], ys_w[k0]
        L_w = 2.0*A_w*x0_w + D_w
        M_w = 2.0*C_w*y0_w + E_w
        if abs(M_w) < 1e-15:
            return _orig_icm(pts5_xy)
        s0_w = -L_w / M_w
    else:
        x0_w, y0_w = xs_w[k0], ys_w[k0]
        s0_w = -L / M if abs(M) > 1e-15 else 0.0

    dxi = xs_w - x0_w
    dyi = ys_w - y0_w
    eps = 1e-9 * (np.abs(dyi) + 1.0)
    mask = np.abs(dxi) >= eps
    dxi_s = np.where(mask, dxi, 1.0)
    s     = np.where(mask, dyi / dxi_s, np.sign(dyi + 1e-300) * 1e15)
    s[k0] = s0_w
    Qs    = A_w + C_w * s**2

    if np.any(np.abs(Qs) < 1e-10):
        hits['icm_qs_on_asymptote'] += 1
    else:
        n_cross = int(np.sum(np.diff(np.sign(Qs)) != 0))
        if n_cross >= 2:
            hits['icm_n_cross_ge2'] += 1
        else:
            d = np.diff(s)
            if   np.all(d > 0): hits['icm_monotone_inc'] += 1
            elif np.all(d < 0): hits['icm_monotone_dec'] += 1
            else:               hits['icm_non_monotone'] += 1

    return _orig_icm(pts5_xy)


# ── Count theta successes separately (they equal case3_theta if none return None)
# We instrument _bpaw_theta_cross_True_ok as: bpaw_case3_theta - bpaw_proj_return_none
# (computed at report time, not via a separate counter).


# ── Install patches ───────────────────────────────────────────────────────────
bl._build_projective_arc_window = _patched_build
bl._is_conic_monotone           = _patched_icm


# ── Run DEMO_CURVES at transition n-values ────────────────────────────────────
def run_curves(verbose=True):
    if verbose:
        print("Running DEMO_CURVES at transition n-values …")
    for entry in dn.DEMO_CURVES:
        name, xy_func, t_range = entry[0], entry[1], entry[2]
        panels = dn.find_transitions(xy_func, t_range, n_min=8, n_max=50)
        for n, _ in panels:
            pts, times = bl.sample_curve(xy_func, t_range, n)
            bl._run_blend(pts, times, False, 2)
        if verbose:
            print(f"  {name}: {[p for p,_ in panels]}")

run_curves()

# Derived counter: theta successes (theta entries that returned non-None)
hits['bpaw_theta_cross_True_ok'] = (
    hits['bpaw_phi_nonmono_theta'] - hits['bpaw_proj_return_none']
)


# ── Report ────────────────────────────────────────────────────────────────────
print("\n── Branch hit counts ───────────────────────────────────────────────────")
zero_non_dead = []
for k, v in sorted(hits.items()):
    dead_tag = ' [DEAD/PRACTICAL]' if k in DEAD else ''
    print(f"  {k:40s}: {v:6d}{dead_tag}")
    if v == 0 and k not in DEAD:
        zero_non_dead.append(k)

print()
if zero_non_dead:
    print(f"ZERO-HIT non-dead branches ({len(zero_non_dead)}):")
    for k in zero_non_dead:
        print(f"  ✗ {k}")
    sys.exit(1)
else:
    print("✓  All non-dead branches hit at least once.")
    sys.exit(0)
