"""P5 FEM code check: full-width film with exact 1D three-slab Dirichlet data -> 2D solution equals 1D solution."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src.fem import fem2d
class Slab:
    """One-dimensional three-slab analogue of P5 (film of thickness t, contrast eps2) with a closed-form solution."""
    eps1 = 1.0
    def __init__(s, k, t): s.eps2 = k; s.t = t; s.D = 2 / ((2 - t) + t / k)
    def side_of(s, X):
        """Material label for each point: 2 inside the film |y| < t/2, otherwise 1."""
        return np.where(np.abs(X[:, 1]) < s.t / 2, 2, 1)
    def phi(s, y):
        """Exact piecewise-linear potential across the three slabs as a function of y."""
        t, D, k = s.t, s.D, s.eps2; a = 1 - D * (1 - t / 2); b = a - D / k * t
        return np.where(y < -t / 2, 1 - D * (y + 1), np.where(y < t / 2, a - D / k * (y + t / 2), b - D * (y - t / 2)))
    def h_train_np(s, X):
        """Dirichlet boundary data for the FEM solver: the exact potential at the points X."""
        return s.phi(X[:, 1])
lines = []
for k in (80.0, 2000.0):
    for t in (0.02, 0.005):
        p = Slab(k, t); sol = fem2d.solve(p, 0.1, geometry=lambda occ, t=t: [occ.addRectangle(-1, -t / 2, 0, 2, t)], grade=(t / 4, t, 0.3))
        rng = np.random.default_rng(0)
        X = np.c_[rng.uniform(-.99, .99, 4000), np.r_[rng.uniform(-t / 2, t / 2, 2000) * 0.999, rng.uniform(-.99, .99, 2000)]]
        s = p.side_of(X); ph, g = sol(X, s); Ey = p.D / np.where(s == 2, k, 1.0)
        lines.append(f"kappa={k:g} t={t}: elems={sol.n_elements} dofs={sol.dofs} max|phi err|={np.abs(ph - p.phi(X[:, 1])).max():.1e} "
                     f"max rel E_y err={np.max(np.abs(-g[:, 1] - Ey) / Ey):.1e} max|E_x|={np.abs(g[:, 0]).max():.1e}")
out = "\n".join(lines); print(out)
(pathlib.Path(__file__).resolve().parents[1] / "results/validation/p5_slab_fem_check.txt").write_text(out + "\n")
