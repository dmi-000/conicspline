"""
diag_xbranch.py — print disagreement between projective arc and spline
for cross-branch windows at n=40 for VdP and Damped oscillation.
"""
import importlib.util, os, numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('bd', os.path.join(_here, 'blend_demo.py'))
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

CURVE_MAP = {n: (f, r) for n, f, r in bd.CURVES}

def diag_curve(name, n=40):
    xy_func, t_range = CURVE_MAP[name]
    pts, times = bd.sample_curve(xy_func, t_range, n)
    N_ORDER = 2

    print(f"\n{'='*60}")
    print(f"  {name}: n={n}")
    print(f"{'='*60}")

    n_wins = len(pts) - 4
    for i in range(n_wins):
        p5 = pts[i:i+5]
        t5 = times[i:i+5]
        center = p5.mean(axis=0)
        centered = p5 - center
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        e1, e2 = Vt[0], Vt[1]
        if (centered[-1] - centered[0]) @ e1 < 0:
            e1 = -e1
        pts_2d = np.column_stack([centered @ e1, centered @ e2])

        try:
            coeffs = bd.fit_conic(pts_2d)
        except Exception:
            continue

        # ── Compute cross.sum() from conic coefficients ───────────────────────
        A, B, C = coeffs[0], coeffs[1], coeffs[2]
        det = 4.0 * A * C - B * B
        cross_sum = None
        if abs(det) > 1e-10:
            cx = (-2.0 * C * coeffs[3] + B * coeffs[4]) / det
            cy = (-2.0 * A * coeffs[4] + B * coeffs[3]) / det
            f0 = A*cx**2 + B*cx*cy + C*cy**2 + coeffs[3]*cx + coeffs[4]*cy + coeffs[5]
            if abs(f0) > 1e-15:
                dx2d = pts_2d[:, 0] - cx
                dy2d = pts_2d[:, 1] - cy
                M = np.array([[A, B / 2.0], [B / 2.0, C]])
                eigvals, eigvecs = np.linalg.eigh(M)
                if eigvals[0] * eigvals[1] < 0:  # hyperbola
                    pos_idx = 1 if eigvals[1] > 0 else 0
                    e_t = eigvecs[:, pos_idx]
                    proj2d = dx2d * e_t[0] + dy2d * e_t[1]
                    if not np.any(proj2d == 0):
                        branch = np.sign(proj2d)
                        cross_arr = (branch[:4] * branch[1:]) < 0
                        cross_sum = int(np.sum(cross_arr))

        proj = bd._build_projective_arc_window(pts_2d, p5, t5, center, e1, e2, coeffs)
        if proj is None:
            print(f"  win[{i:3d}]  xbranch=? (proj=None)  cross.sum={cross_sum}")
            continue
        proj_func, proj_err, is_cross_branch = proj
        if not is_cross_branch:
            continue

        # Check blend region
        t_check = np.linspace(t5[1], t5[3], 50)
        check = proj_func(t_check)
        if not np.all(np.isfinite(check)):
            print(f"  win[{i:3d}]  xbranch=True (NaN in blend)  cross.sum={cross_sum}")
            continue  # inner cross-branch, would fail NaN check

        # Compute disagreement vs spline
        spline_f = bd._make_spline_window(p5, t5)
        s_pts = spline_f(t_check)[:, :2]
        p_pts = check[:, :2]
        diff = np.max(np.linalg.norm(p_pts - s_pts, axis=1))
        span = float(np.sum(np.linalg.norm(np.diff(p5[:, :2], axis=0), axis=1)))
        rel = diff / span if span > 1e-12 else float('inf')

        print(f"  win[{i:3d}]  xbranch=True  cross.sum={cross_sum}  diff={diff:.4f}  span={span:.4f}  rel={rel:.3f}")

diag_curve('Van der Pol (mu=3)')
diag_curve('Damped oscillation')
diag_curve('Lissajous (3:2)')
diag_curve('Random spline')
