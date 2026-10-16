"""Regression test for the P2 gradient bug found in Phase 7 (lambda gradients were shifted by one)."""
import numpy as np
from src.problems.p2d import P2Cylinder
from src.fem import fem2d

def test_p2_element_reproduces_quadratic_values_and_gradients():
    p = P2Cylinder(80.0); s = fem2d.solve(p, 0.2, geometry=lambda occ: [occ.addDisk(0, 0, 0, p.R, p.R)])
    f = lambda X: 1 + 2 * X[:, 0] - X[:, 1] + 3 * X[:, 0]**2 - X[:, 0] * X[:, 1] + 0.5 * X[:, 1]**2
    s.u = f(s.basis.doflocs.T); X = np.random.default_rng(0).uniform(-.95, .95, (400, 2)); ph, gr = s(X, p.side_of(X))
    assert abs(ph - f(X)).max() < 1e-12
    assert abs(gr - np.c_[2 + 6 * X[:, 0] - X[:, 1], -1 - X[:, 0] + X[:, 1]]).max() < 1e-12
