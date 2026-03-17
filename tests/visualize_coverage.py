"""
tests/visualize_coverage.py
-----------------------------
12-panel figure: one representative 5-point window for every code path
exercised by branch_coverage.py and synthetic_coverage.py.

Layout (3 rows × 4 cols):
  Row 0 — bpaw returns a valid arc:
    same_branch | cross_case1 | cross_case3 (theta) | parabola (succeeds, no guard)
  Row 1 — bpaw degenerate → None / DEAD synthetic:
    on_axis (→None via NaN) | at_center (→None via M_e<1e-15) | bpaw_swapped | icm_qs_on_asymptote
  Row 2 — _is_conic_monotone paths:
    icm_n_cross_ge2 | icm_non_monotone | icm_monotone_inc | icm_monotone_dec

Each panel:
  • 5 control points (filled circles, coloured by branch for cross-branch windows)
  • Conic arc from orbit_func (thick coloured line) when available
  • Background conic curve (thin dotted grey contour)
  • Title: short description + branch name(s)

Run from repo root:
    python3 tests/visualize_coverage.py
Output: coverage_paths.png  (alongside this script)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

import conicspline as bl

# ── Load demo_n_transitions ───────────────────────────────────────────────────
_here   = os.path.dirname(os.path.abspath(__file__))
_dn_path = os.path.join(_here, '..', 'demo_n_transitions.py')
spec    = importlib.util.spec_from_file_location('dn', _dn_path)
dn      = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dn)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _project_to_local(xyz, center3d, e1, e2):
    """Project 3-D orbit_func output back to the 2-D local frame."""
    E = np.column_stack([e1[:2], e2[:2]])        # (2, 2)
    return (xyz[:, :2] - center3d[:2]) @ E       # (N, 2)


def _conic_contour(ax, coeffs, xlim, ylim, **kw):
    """Draw the zero-contour of Ax²+Bxy+Cy²+Dx+Ey+F over [xlim]×[ylim]."""
    A, B, C, D, E, F = coeffs
    xs = np.linspace(xlim[0], xlim[1], 400)
    ys = np.linspace(ylim[0], ylim[1], 400)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = A*XX**2 + B*XX*YY + C*YY**2 + D*XX + E*YY + F
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            ax.contour(XX, YY, ZZ, levels=[0], colors=['#BBBBBB'],
                       linewidths=0.8, linestyles='--')
        except Exception:
            pass


def _classify_bpaw(p2d, cffs):
    """Return which bpaw branch this 5-point window falls into."""
    A, B, C, D, E, F = cffs
    det = 4*A*C - B*B
    if abs(det) < 1e-10:
        return 'parabola'
    cx = (-2*C*D + B*E)/det;  cy = (-2*A*E + B*D)/det
    dx = p2d[:,0]-cx;          dy = p2d[:,1]-cy
    if np.min(np.sqrt(dx**2+dy**2)) < 1e-10:
        return 'at_center'
    M_mat = np.array([[A, B/2],[B/2, C]])
    ev, evec = np.linalg.eigh(M_mat)
    if ev[0]*ev[1] < 0:
        pidx = 1 if ev[1]>0 else 0
        et = evec[:,pidx];  prj = dx*et[0]+dy*et[1];  br = np.sign(prj)
        if np.any(br == 0):
            return 'on_axis'
        cross = (br[:4]*br[1:]) < 0
        if not cross.any():
            return 'same_branch'
        # Cross-branch: case1 vs case3
        def sk0(pe, Ae, Be, Ce, De, Ee):
            L=2*Ae*pe[:,0]+Be*pe[:,1]+De;  M=Be*pe[:,0]+2*Ce*pe[:,1]+Ee
            Ms=np.where(np.abs(M)>=1e-15, M, np.sign(M+1e-300)*1e-15)
            return int(np.argmin(np.abs(L/Ms))), L, M
        pe=p2d.copy(); Ae,Ce,De,Ee=A,C,D,E
        k0,La,Ma=sk0(pe,Ae,B,Ce,De,Ee); Le,Me=La[k0],Ma[k0]
        if abs(Me)<1e-9*max(abs(Le),1.):
            pe=pe[:,::-1]; Ae,Ce=Ce,Ae; De,Ee=Ee,De
            k0,La,Ma=sk0(pe,Ae,B,Ce,De,Ee); Le,Me=La[k0],Ma[k0]
        if abs(Me)>1e-15:
            s0=-Le/Me; x0c,y0c=pe[k0]
            dxi=pe[:,0]-x0c; dyi=pe[:,1]-y0c
            mask=np.abs(dxi)>=1e-9*(np.abs(dyi)+1.)
            dxi_s=np.where(mask,dxi,1.)
            sv=np.where(mask,dyi/dxi_s,np.sign(dyi+1e-300)*1e15); sv[k0]=s0
            Qv=Ae+B*sv+Ce*sv**2
            return 'cross_case1' if len(np.where(np.diff(np.sign(Qv))!=0)[0])>=1 else 'cross_case3'
    return 'same_branch'


def _classify_icm(pts5_xy):
    """Return which icm branch this 5-point window falls into."""
    try:
        A, B, C, D, E, F = bl.fit_conic(pts5_xy)
    except Exception:
        return 'unknown', None
    M_mat = np.array([[A, B*0.5],[B*0.5, C]])
    try:
        evals, evecs = np.linalg.eigh(M_mat)
    except Exception:
        return 'unknown', None
    sort_idx = np.argsort(np.abs(evals));  evals = evals[sort_idx];  evecs = evecs[:,sort_idx]
    xs_r = pts5_xy @ evecs[:,0];  ys_r = pts5_xy @ evecs[:,1]
    A_r,C_r = float(evals[0]),float(evals[1])
    D_r,E_r = np.array([D,E]) @ evecs
    Lk=2*A_r*xs_r+D_r;  Mk=2*C_r*ys_r+E_r
    ratio=np.where(np.abs(Mk)>1e-15, np.abs(Lk)/np.abs(Mk), 1e30)
    k0=int(np.argmin(ratio));  L,M=Lk[k0],Mk[k0]
    xs_w,ys_w=xs_r.copy(),ys_r.copy()
    A_w,C_w,D_w,E_w=A_r,C_r,D_r,E_r
    if abs(M)<1e-9*max(abs(L),1.):
        A_w,C_w=C_r,A_r;  D_w,E_w=E_r,D_r;  xs_w,ys_w=ys_r.copy(),xs_r.copy()
        x0_w,y0_w=xs_w[k0],ys_w[k0];  L_w=2*A_w*x0_w+D_w;  M_w=2*C_w*y0_w+E_w
        if abs(M_w)<1e-15: return 'unknown', None
        s0_w=-L_w/M_w
    else:
        x0_w,y0_w=xs_w[k0],ys_w[k0];  s0_w=-L/M if abs(M)>1e-15 else 0.
    dxi=xs_w-x0_w;  dyi=ys_w-y0_w
    eps=1e-9*(np.abs(dyi)+1.);  mask=np.abs(dxi)>=eps
    dxi_s=np.where(mask,dxi,1.)
    s=np.where(mask,dyi/dxi_s,np.sign(dyi+1e-300)*1e15);  s[k0]=s0_w
    Qs=A_w+C_w*s**2
    if np.any(np.abs(Qs)<1e-10):
        return 'qs_on_asymptote', (s, Qs)
    n_cross=int(np.sum(np.diff(np.sign(Qs))!=0))
    if n_cross>=2:
        return 'n_cross_ge2', (s, Qs)
    d=np.diff(s)
    if   np.all(d>0): return 'monotone_inc', (s, Qs)
    elif np.all(d<0): return 'monotone_dec', (s, Qs)
    else:             return 'non_monotone',  (s, Qs)


# ── Capture examples via monkey-patching ──────────────────────────────────────
cap_bpaw = {}   # branch_key -> dict
cap_icm  = {}   # branch_key -> dict
_BPAW_TARGETS = {'parabola','on_axis','same_branch','cross_case1','cross_case3'}
_ICM_TARGETS  = {'qs_on_asymptote','n_cross_ge2','monotone_inc','monotone_dec','non_monotone'}

_orig_build = bl._build_projective_arc_window
_orig_icm   = bl._is_conic_monotone

def _patched_build(pts_2d, pts, ts, center3d, e1, e2, coeffs):
    key = _classify_bpaw(pts_2d, coeffs)
    if key in _BPAW_TARGETS and key not in cap_bpaw:
        result = _orig_build(pts_2d, pts, ts, center3d, e1, e2, coeffs)
        cap_bpaw[key] = dict(pts_2d=pts_2d.copy(), pts=pts.copy(), ts=ts.copy(),
                              center3d=center3d.copy(), e1=e1.copy(), e2=e2.copy(),
                              coeffs=coeffs, result=result)
        return result
    return _orig_build(pts_2d, pts, ts, center3d, e1, e2, coeffs)

def _patched_icm(pts5_xy):
    pts5_xy = np.asarray(pts5_xy, dtype=float)
    key, extra = _classify_icm(pts5_xy)
    if key in _ICM_TARGETS and key not in cap_icm:
        cap_icm[key] = dict(pts5_xy=pts5_xy.copy(), key=key, extra=extra)
    return _orig_icm(pts5_xy)

bl._build_projective_arc_window = _patched_build
bl._is_conic_monotone            = _patched_icm

print("Capturing examples from DEMO_CURVES …")
for entry in dn.DEMO_CURVES:
    name, xy_func, t_range = entry[0], entry[1], entry[2]
    panels = dn.find_transitions(xy_func, t_range, n_min=8, n_max=50)
    for n, _ in panels:
        p, t = bl.sample_curve(xy_func, t_range, n)
        bl._run_blend(p, t, False, 2)
    if (len(cap_bpaw) >= len(_BPAW_TARGETS) and
        len(cap_icm)  >= len(_ICM_TARGETS)):
        break

bl._build_projective_arc_window = _orig_build
bl._is_conic_monotone            = _orig_icm
print(f"  bpaw: {sorted(cap_bpaw.keys())}")
print(f"  icm:  {sorted(cap_icm.keys())}")


# ── Synthetic DEAD examples ───────────────────────────────────────────────────
# Test 1: bpaw_swapped (near-vertical tangent on unit circle)
eps = 1e-10
_th  = np.array([-2*eps, -eps, 0., eps, 2*eps])
_pts_swap = np.column_stack([np.cos(_th), np.sin(_th)])
_pts3_swap = np.column_stack([_pts_swap, np.zeros(5)])
_ts_swap   = np.linspace(0, 1, 5)
cap_bpaw['swapped'] = dict(
    pts_2d=_pts_swap, pts=_pts3_swap, ts=_ts_swap,
    center3d=np.zeros(3), e1=np.array([1.,0.,0.]), e2=np.array([0.,1.,0.]),
    coeffs=(1.,0.,1.,0.,0.,-1.), result=None)

# Test 2: bpaw_early_at_center (unit circle + centre)
_angs = np.array([0., np.pi/2, np.pi, 3*np.pi/2])
_circ = np.column_stack([np.cos(_angs), np.sin(_angs)])
_pts_ctr = np.vstack([_circ[:2], [[0.,0.]], _circ[2:]])
_pts3_ctr = np.column_stack([_pts_ctr, np.zeros(5)])
_ts_ctr   = np.linspace(0, 1, 5)
cap_bpaw['at_center'] = dict(
    pts_2d=_pts_ctr, pts=_pts3_ctr, ts=_ts_ctr,
    center3d=np.zeros(3), e1=np.array([1.,0.,0.]), e2=np.array([0.,1.,0.]),
    coeffs=(1.,0.,1.,0.,0.,-1.), result=None)


# ── Panel plotting helper ─────────────────────────────────────────────────────
PANEL_INFO = [
    # (row, col, key,        source,  label,            subtitle)
    (0, 0, 'same_branch',  'bpaw', 'same-branch conic arc',       'bpaw_same_branch\nbpaw_case0_returned'),
    (0, 1, 'cross_case1',  'bpaw', 'cross-branch Case 1\n(finite asymptote)', 'bpaw_case1_finite_sstar\nbpaw_proj_success'),
    (0, 2, 'cross_case3',  'bpaw', 'cross-branch Case 3\n(theta fallback)',   'bpaw_case3_theta\nbpaw_theta_cross_True_ok'),
    (0, 3, 'parabola',     'bpaw', 'parabola: succeeds\n(no guard needed)',    'same-branch arc\nQ=1 → stereographic exact'),
    (1, 0, 'on_axis',      'bpaw', 'on-axis → None\n(Q=0 → u→∞ → NaN)',      'no guard; ctrl_err=NaN\n→ returns None'),
    (1, 1, 'at_center',    'bpaw', 'at-center → None\n[synthetic]',           'M_e<1e-15 after swap\n→ returns None'),
    (1, 2, 'swapped',      'bpaw', 'swap path (ε=1e-10)\n[synthetic, returns None]', 'bpaw_swapped'),
    (1, 3, 'qs_on_asymptote', 'icm','Q ≈ 0 at control pt\n→ False (fall to spline)', 'icm_qs_on_asymptote'),
    (2, 0, 'n_cross_ge2',  'icm', '≥2 Q sign changes\n→ False (fall to spline)', 'icm_n_cross_ge2'),
    (2, 1, 'non_monotone', 'icm', 'mixed s-diffs\n→ False (fall to spline)',  'icm_non_monotone'),
    (2, 2, 'monotone_inc', 'icm', 'all s-diffs > 0\n→ True (conic accepted)', 'icm_monotone_inc'),
    (2, 3, 'monotone_dec', 'icm', 'all s-diffs < 0\n→ True (conic accepted)', 'icm_monotone_dec'),
]

# Row background colours
ROW_BG = {0: '#E8F5E9', 1: '#FFF3E0', 2: '#E3F2FD'}
# Arc and point colours
ARC_SAME   = '#1565C0'   # blue  — same-branch arc
ARC_CROSS  = '#C62828'   # red   — cross-branch arc
PT_SAME    = '#1565C0'
PT_CROSS_A = '#C62828'   # first branch
PT_CROSS_B = '#F57F17'   # second branch
PT_NEUTRAL = '#555555'   # early exits / synthetic
PT_ICM     = '#2E7D32'   # icm examples


def _scatter_pts(ax, pts_xy, colors, pad):
    """Scatter control points.  When all 5 are indistinguishably close
    (pairwise distance < 1e-4 × pad), draw concentric rings instead of
    overlapping dots so all 5 are legible."""
    n = len(pts_xy)
    dists = np.linalg.norm(pts_xy[None,:,:] - pts_xy[:,None,:], axis=2)
    too_close = dists.max() < 1e-4 * pad

    if too_close:
        cx, cy = pts_xy.mean(axis=0)
        # Centre dot for the innermost point
        c0 = colors[0] if (hasattr(colors, '__len__') and len(colors)==n) else colors
        ax.scatter(cx, cy, s=40, color=c0, zorder=6, edgecolors='white', linewidths=0.8)
        # One ring per remaining point, spaced visually
        for i in range(1, n):
            r = i * 0.14 * pad
            ci = colors[i] if (hasattr(colors, '__len__') and len(colors)==n) else colors
            circle = plt.Circle((cx, cy), r, fill=False, edgecolor=ci,
                                 linewidth=1.8, zorder=5)
            ax.add_patch(circle)
        # Annotation showing actual spacing
        ax.text(cx + 0.18*pad, cy - 0.12*pad,
                f'5 pts\nΔ≈1e-10', fontsize=5.5, color='#666666',
                ha='left', va='top', linespacing=1.3)
    else:
        if hasattr(colors, '__len__') and len(colors) == n:
            for i, (px, py) in enumerate(pts_xy):
                ax.scatter(px, py, s=55, color=colors[i], zorder=5,
                           edgecolors='white', linewidths=0.8)
        else:
            ax.scatter(pts_xy[:,0], pts_xy[:,1], s=55, color=colors,
                       zorder=5, edgecolors='white', linewidths=0.8)


def _plot_panel(ax, row, key, source, label, subtitle):
    ax.set_facecolor(ROW_BG[row])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#AAAAAA')

    cap = (cap_bpaw if source == 'bpaw' else cap_icm).get(key)
    if cap is None:
        ax.text(0.5, 0.5, f'(no example\ncaptured)', ha='center', va='center',
                transform=ax.transAxes, color='#888888', fontsize=7)
        ax.set_title(f'{label}\n{subtitle}', fontsize=7, pad=3)
        return

    if source == 'bpaw':
        pts_2d    = cap['pts_2d']
        ts        = cap['ts']
        coeffs    = cap['coeffs']
        result    = cap['result']
        center3d  = cap['center3d']
        e1, e2    = cap['e1'], cap['e2']
        E         = np.column_stack([e1[:2], e2[:2]])

        # Compute square axis limits — ensures plt.Circle stays circular.
        # Fall back to conic bounding box when pts are nearly coincident.
        raw_pad = max(np.ptp(pts_2d[:,0]), np.ptp(pts_2d[:,1]))
        if raw_pad < 1e-6:
            A_l, B_l, C_l, D_l, E_l, F_l = coeffs
            try:
                pad = max(1.5, abs(float(-F_l / A_l))**0.5 * 1.3) if abs(A_l) > 1e-12 else 1.5
            except Exception:
                pad = 1.5
        else:
            pad = raw_pad * 0.6 + 1e-6          # 0.6: keeps all extreme pts visible
        cx_, cy_ = pts_2d[:,0].mean(), pts_2d[:,1].mean()
        xlim = (cx_ - pad, cx_ + pad)
        ylim = (cy_ - pad, cy_ + pad)
        ax.set_xlim(xlim);  ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')  # circles stay circular

        # Background conic contour
        _conic_contour(ax, coeffs, xlim, ylim)

        # Plot arc (if available)
        if result is not None:
            orbit_func, ctrl_err, is_cross = result
            t_dense = np.linspace(ts[0], ts[-1], 400)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                xyz = orbit_func(t_dense)
            xy2d = (xyz[:, :2] - center3d[:2]) @ E
            col  = ARC_CROSS if is_cross else ARC_SAME
            ax.plot(xy2d[:, 0], xy2d[:, 1], color=col, lw=2.0, zorder=4,
                    solid_capstyle='round')

        # Colour control points by window type
        A, B, C, D, E_c, F = coeffs
        det = 4*A*C - B*B
        is_hyp = False
        if abs(det) >= 1e-10:
            M_mat = np.array([[A, B/2],[B/2, C]])
            ev, evec = np.linalg.eigh(M_mat)
            is_hyp = ev[0]*ev[1] < 0

        if is_hyp and key in ('cross_case1','cross_case3'):
            # Two-branch colouring: branch A (red) vs branch B (orange)
            cx = (-2*C*D + B*E_c)/det;  cy = (-2*A*E_c + B*D)/det
            dx = pts_2d[:,0]-cx;  dy = pts_2d[:,1]-cy
            pidx = 1 if ev[1]>0 else 0
            et = evec[:,pidx];  prj = dx*et[0]+dy*et[1];  br = np.sign(prj)
            pt_colors = [PT_CROSS_A if br[i] > 0 else PT_CROSS_B for i in range(5)]
            _scatter_pts(ax, pts_2d, pt_colors, pad)
        elif key in ('same_branch', 'parabola'):
            # Same-branch (including parabola, which now succeeds): blue
            _scatter_pts(ax, pts_2d, PT_SAME, pad)
        else:
            # Degenerate → None paths / synthetic: neutral grey
            _scatter_pts(ax, pts_2d, PT_NEUTRAL, pad)

    else:  # icm panel
        pts5_xy = cap['pts5_xy']
        pad = max(np.ptp(pts5_xy[:,0]), np.ptp(pts5_xy[:,1])) * 0.6 + 1e-6
        cx_, cy_ = pts5_xy[:,0].mean(), pts5_xy[:,1].mean()
        xlim = (cx_ - pad, cx_ + pad)
        ylim = (cy_ - pad, cy_ + pad)
        ax.set_xlim(xlim);  ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        try:
            coeffs_icm = bl.fit_conic(pts5_xy)
            _conic_contour(ax, coeffs_icm, xlim, ylim)
        except Exception:
            pass

        # For monotone paths (icm returns True) show the fitted conic arc too
        if key in ('monotone_inc', 'monotone_dec'):
            pts3d_icm = np.column_stack([pts5_xy, np.zeros(5)])
            ts_icm    = np.linspace(0., 1., 5)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    func_icm, _, method_icm = bl.fit_conic_5pt(pts3d_icm, ts_icm)
                    if method_icm == 'conic':
                        t_d = np.linspace(0., 1., 400)
                        xyz_icm = func_icm(t_d)
                        ax.plot(xyz_icm[:,0], xyz_icm[:,1],
                                color=ARC_SAME, lw=2.0, zorder=4, solid_capstyle='round')
                except Exception:
                    pass

        _scatter_pts(ax, pts5_xy, PT_ICM, pad)

    ax.set_title(f'{label}\n{subtitle}', fontsize=7, pad=3, color='#222222')


# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.patch.set_facecolor('#F8F8F8')

for row, col, key, source, label, subtitle in PANEL_INFO:
    _plot_panel(axes[row][col], row, key, source, label, subtitle)

# Row labels on the left
ROW_LABELS = [
    'bpaw returns\nvalid arc',
    'bpaw early exits\n& DEAD paths',
    '_is_conic_monotone\nclassification',
]
for r, rl in enumerate(ROW_LABELS):
    axes[r][0].set_ylabel(rl, fontsize=8, rotation=90, labelpad=6,
                          color='#444444', fontweight='bold')

# Legend
leg = [
    mpatches.Patch(color=ARC_SAME,   label='arc: same-branch / icm-True'),
    mpatches.Patch(color=ARC_CROSS,  label='arc: cross-branch (gap = asymptote NaN)'),
    mpatches.Patch(color=PT_SAME,    label='pts: same-branch'),
    mpatches.Patch(color=PT_CROSS_A, label='pts: cross-branch A (branch 1)'),
    mpatches.Patch(color=PT_CROSS_B, label='pts: cross-branch B (branch 2)'),
    mpatches.Patch(color=PT_NEUTRAL, label='pts: early-exit / synthetic (no arc)'),
    mpatches.Patch(color=PT_ICM,     label='pts: icm classification'),
    mpatches.Patch(color='#BBBBBB',  label='background conic contour'),
]
fig.legend(handles=leg, loc='lower center', ncol=4, fontsize=7,
           framealpha=0.85, bbox_to_anchor=(0.5, 0.0))

fig.suptitle('Coverage paths: _build_projective_arc_window  &  _is_conic_monotone',
             fontsize=10, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0.07, 1, 0.99])

out = os.path.join(_here, 'coverage_paths.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved → {out}')
