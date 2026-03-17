"""
check_devs.py  —  Compare segment deviations for n=36 Lissajous (3:2)
                  between blend_demo.py.bak4 (old) and blend_demo.py (new).
"""
import importlib.machinery, importlib.util, os, numpy as np

_here = os.path.dirname(os.path.abspath(__file__))

def load(path, name):
    loader = importlib.machinery.SourceFileLoader(name, path)
    return loader.load_module()

# bd_old: point at a local .bak snapshot (e.g. blend_demo.py.bak4) to compare against
bd_old = load(os.path.join(_here, 'blend_demo.py.bak4'), 'bd_old')
spec   = importlib.util.spec_from_file_location('bd_new', os.path.join(_here, 'blend_demo.py'))
bd_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd_new)

# Use the CURVES entry so t_range is correct
_, xy_func, t_range = next(c for c in bd_new.CURVES if 'Lissajous' in c[0])
n = 36

pts, times = bd_new.sample_curve(xy_func, t_range, n)
N_SEG = 80

def seg_dev(dense, j, ms):
    """Max deviation of segment j from GT, as % of chord length.
    times are in [0,1] normalized; convert to actual param via t_range."""
    t_s = np.linspace(times[j], times[j+1], N_SEG)
    t_a = t_range[0] + t_s * (t_range[1] - t_range[0])
    xg, yg = xy_func(t_a)
    gt    = np.column_stack([xg, yg])
    seg   = dense[(j - ms) * N_SEG : (j - ms + 1) * N_SEG, :2]
    chord = float(np.linalg.norm(pts[j+1, :2] - pts[j, :2])) or 1.0
    return np.max(np.linalg.norm(seg - gt, axis=1)) / chord * 100

for label, bd in [('OLD (bak4)', bd_old), ('NEW (current)', bd_new)]:
    dense, interp, costs, windows, ms, me, methods = \
        bd._run_blend(pts, times, False, 2)
    devs = [(seg_dev(dense, j, ms), j) for j in range(ms, me)]
    devs.sort(reverse=True)
    print(f"\n{label}:")
    for pct, j in devs[:5]:
        mA, mB = methods[j-2], methods[j-1]
        print(f"  j={j:3d}  {pct:8.3f}%  mA={mA}  mB={mB}")
    d = dict((jj, pp) for pp, jj in devs)
    print(f"  j=15: {d.get(15,-1):.3f}%   j=18: {d.get(18,-1):.3f}%")
