"""
tests/diag_unified_param.py
----------------------------
Compares the unified parameterisation  phi = 2·arctan(s − s₀) + np.unwrap
against the current per-case logic for one example of each case type
(Case 0 same-branch, Case 1 finite asymptote, Case 3 theta fallback).

Also benchmarks wall-time per window for both paths.

Run from repo root:
    python3 tests/diag_unified_param.py
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.interpolate import PchipInterpolator
import importlib.util
import conicspline as bl

# ── Load demo_n_transitions ───────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
spec  = importlib.util.spec_from_file_location('dn', os.path.join(_here, '..', 'demo_n_transitions.py'))
dn    = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dn)


# ── Capture one window per case by monkey-patching ───────────────────────────
examples = {}   # 'case0' / 'case1' / 'case3'  ->  dict of window data

_orig_build = bl._build_projective_arc_window

def _classify_and_capture(p2d, p, ts, ctr, e1, e2, cffs):
    A, B, C, D, E, F = cffs
    det = 4*A*C - B*B
    if det == 0.0:
        return _orig_build(p2d, p, ts, ctr, e1, e2, cffs)

    cx = (-2*C*D + B*E) / det
    cy = (-2*A*E + B*D) / det
    dx = p2d[:, 0] - cx;  dy = p2d[:, 1] - cy

    M_mat  = np.array([[A, B/2.0], [B/2.0, C]])
    ev, evec = np.linalg.eigh(M_mat)
    is_hyp = ev[0]*ev[1] < 0

    if is_hyp:
        pidx = 1 if ev[1] > 0 else 0
        et   = evec[:, pidx]
        prj  = dx*et[0] + dy*et[1]
        br   = np.sign(prj)
        cross = (br[:4] * br[1:]) < 0
        is_cb = bool(cross.any())
    else:
        is_cb = False
        cross = np.zeros(4, dtype=bool)

    # Canonical P₀
    xs, ys = p2d[:, 0], p2d[:, 1]
    L_all  = 2*A*xs + B*ys + D
    M_all  = B*xs + 2*C*ys + E
    Ms     = np.where(np.abs(M_all) >= 1e-15, M_all,
                      np.sign(M_all + 1e-300) * 1e-15)
    k0     = int(np.argmin(np.abs(L_all / Ms)))
    L_e, M_e = L_all[k0], M_all[k0]

    if abs(M_e) < 1e-9 * max(abs(L_e), 1.0):
        pe2    = p2d[:, ::-1]
        A2, C2, D2, E2 = C, A, E, D
        L2 = 2*A2*pe2[:,0] + B*pe2[:,1] + D2
        M2 = B*pe2[:,0] + 2*C2*pe2[:,1] + E2
        Ms2 = np.where(np.abs(M2) >= 1e-15, M2, np.sign(M2 + 1e-300) * 1e-15)
        k0 = int(np.argmin(np.abs(L2 / Ms2)))
        L_e, M_e = L2[k0], M2[k0]

    s0 = -L_e / M_e if abs(M_e) > 1e-15 else 0.0
    x0, y0 = p2d[k0]

    dxi    = p2d[:, 0] - x0
    dyi    = p2d[:, 1] - y0
    mask   = np.abs(dxi) >= 1e-9 * (np.abs(dyi) + 1.0)
    dxi_s  = np.where(mask, dxi, 1.0)
    s_vals = np.where(mask, dyi / dxi_s, np.sign(dyi + 1e-300) * 1e15)
    s_vals[k0] = s0

    Q_vals = A + B*s_vals + C*s_vals**2

    if is_cb:
        q_ch = np.where(np.diff(np.sign(Q_vals)) != 0)[0]
        case = 'case1' if len(q_ch) >= 1 else 'case3'
    else:
        case = 'case0'

    if case not in examples:
        examples[case] = dict(
            pts_2d=p2d.copy(), pts=p.copy(), ts=ts.copy(),
            coeffs=cffs, s_vals=s_vals.copy(), s0=float(s0),
            k0=int(k0), L_e=float(L_e), M_e=float(M_e),
            center3d=ctr.copy(), e1=e1.copy(), e2=e2.copy(),
        )

    return _orig_build(p2d, p, ts, ctr, e1, e2, cffs)


bl._build_projective_arc_window = _classify_and_capture

for entry in dn.DEMO_CURVES:
    if len(examples) == 3:
        break
    xy_func, t_range = entry[1], entry[2]
    panels = dn.find_transitions(xy_func, t_range, n_min=8, n_max=60)
    for n, _ in panels:
        pts, times = bl.sample_curve(xy_func, t_range, n)
        bl._run_blend(pts, times, False, 2)
        if len(examples) == 3:
            break

bl._build_projective_arc_window = _orig_build
print(f"Captured: {sorted(examples.keys())}\n")


# ── Unified parameterisation ──────────────────────────────────────────────────
def _eval_unified(ex):
    """phi = 2·arctan(s − s₀) + np.unwrap.  Returns ctrl_err (nan if failed)."""
    A, B, C, D, E, F = ex['coeffs']
    s_vals, s0, k0   = ex['s_vals'], ex['s0'], ex['k0']
    L_e, M_e         = ex['L_e'], ex['M_e']
    x0, y0           = ex['pts_2d'][k0]
    ts               = ex['ts']

    phi_vals = 2.0 * np.arctan(s_vals - s0)
    phi_u    = np.unwrap(phi_vals)

    phi_of_t = PchipInterpolator(ts, phi_u)

    def orbit_unified(t):
        t   = np.atleast_1d(t).astype(float)
        phi = phi_of_t(t)
        s   = s0 + np.tan(phi / 2.0)
        Q   = A + B*s + C*s**2
        v   = np.abs(Q) >= 1e-12
        Qs  = np.where(v, Q, 1.0)
        u   = np.where(v, -(L_e + s*M_e) / Qs, np.nan)
        x2d = x0 + u
        y2d = y0 + s * u
        ctr, e1, e2 = ex['center3d'], ex['e1'], ex['e2']
        return ctr + x2d[:, None]*e1 + y2d[:, None]*e2

    pred     = orbit_unified(ts)
    ctrl_err = float(np.nanmax(np.linalg.norm(pred - ex['pts'], axis=1)))
    return phi_u, ctrl_err, orbit_unified


# ── Report ────────────────────────────────────────────────────────────────────
print(f"{'Case':<8}  {'s_vals (rounded)':<40}  {'phi_u (°)':<40}  "
      f"{'mono':>5}  {'unified err':>12}  {'current err':>12}")
print("-" * 130)

for case in ('case0', 'case1', 'case3'):
    if case not in examples:
        print(f"{case:<8}  (not captured)"); continue
    ex = examples[case]

    phi_u, err_uni, _ = _eval_unified(ex)
    mono = np.all(np.diff(phi_u) > 0) or np.all(np.diff(phi_u) < 0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = _orig_build(
            ex['pts_2d'], ex['pts'], ex['ts'],
            ex['center3d'], ex['e1'], ex['e2'], ex['coeffs'])
    err_cur = result[1] if result is not None else float('nan')

    sv_str  = str(np.round(ex['s_vals'], 4))[:38]
    phi_str = str(np.round(np.degrees(phi_u), 2))[:38]
    print(f"{case:<8}  {sv_str:<40}  {phi_str:<40}  "
          f"{'yes' if mono else 'NO':>5}  {err_uni:>12.3e}  {err_cur:>12.3e}")

print()


# ── Benchmark: per-case vs unified, over all captured windows ────────────────
# Collect ~500 windows via full DEMO_CURVES sweep
all_windows = []

def _collect(p2d, p, ts, ctr, e1, e2, cffs):
    # Rebuild s_vals quickly
    A, B, C, D, E, F = cffs
    det = 4*A*C - B*B
    if det == 0.0:
        return _orig_build(p2d, p, ts, ctr, e1, e2, cffs)
    xs, ys = p2d[:,0], p2d[:,1]
    L = 2*A*xs+B*ys+D;  M = B*xs+2*C*ys+E
    Ms = np.where(np.abs(M)>=1e-15, M, np.sign(M+1e-300)*1e-15)
    k0 = int(np.argmin(np.abs(L/Ms)))
    L_e, M_e = L[k0], M[k0]
    if abs(M_e) < 1e-9*max(abs(L_e),1.):
        pe2=p2d[:,::-1]; A2,C2,D2,E2=C,A,E,D
        L2=2*A2*pe2[:,0]+B*pe2[:,1]+D2; M2=B*pe2[:,0]+2*C2*pe2[:,1]+E2
        Ms2=np.where(np.abs(M2)>=1e-15,M2,np.sign(M2+1e-300)*1e-15)
        k0=int(np.argmin(np.abs(L2/Ms2))); L_e,M_e=L2[k0],M2[k0]
    s0 = -L_e/M_e if abs(M_e)>1e-15 else 0.
    x0,y0 = p2d[k0]
    dxi=p2d[:,0]-x0; dyi=p2d[:,1]-y0
    mask=np.abs(dxi)>=1e-9*(np.abs(dyi)+1.)
    dxi_s=np.where(mask,dxi,1.)
    s_v=np.where(mask,dyi/dxi_s,np.sign(dyi+1e-300)*1e15); s_v[k0]=s0
    all_windows.append(dict(
        pts_2d=p2d.copy(), pts=p.copy(), ts=ts.copy(), coeffs=cffs,
        s_vals=s_v, s0=float(s0), k0=int(k0), L_e=float(L_e), M_e=float(M_e),
        center3d=ctr.copy(), e1=e1.copy(), e2=e2.copy()))
    return _orig_build(p2d, p, ts, ctr, e1, e2, cffs)

bl._build_projective_arc_window = _collect
for entry in dn.DEMO_CURVES:
    xy_func, t_range = entry[1], entry[2]
    for n, _ in dn.find_transitions(xy_func, t_range, n_min=8, n_max=50):
        pts, times = bl.sample_curve(xy_func, t_range, n)
        bl._run_blend(pts, times, False, 2)
bl._build_projective_arc_window = _orig_build

print(f"Collected {len(all_windows)} windows for benchmark.\n")

REPS = 3

t0 = time.perf_counter()
for _ in range(REPS):
    for ex in all_windows:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _orig_build(ex['pts_2d'], ex['pts'], ex['ts'],
                        ex['center3d'], ex['e1'], ex['e2'], ex['coeffs'])
t_current = (time.perf_counter() - t0) / REPS

t0 = time.perf_counter()
for _ in range(REPS):
    for ex in all_windows:
        _eval_unified(ex)
t_unified = (time.perf_counter() - t0) / REPS

print(f"Per-case (current):  {t_current*1e3:.1f} ms  for {len(all_windows)} windows"
      f"  ({t_current/len(all_windows)*1e6:.1f} µs/window)")
print(f"Unified (2·arctan):  {t_unified*1e3:.1f} ms  for {len(all_windows)} windows"
      f"  ({t_unified/len(all_windows)*1e6:.1f} µs/window)")
print(f"Ratio unified/current: {t_unified/t_current:.2f}x")
