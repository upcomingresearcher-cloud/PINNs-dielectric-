"""P1: 1D layered dielectric capacitor (protocol Sec. 2).

Prediction interface used throughout the project
-------------------------------------------------
A *prediction* is any callable ``pred(x, side) -> (phi, dphi_dx)`` where ``x`` is a
float64 array and ``side`` is an int array of the same shape with values 1 or 2,
naming the material whose representation is evaluated. Away from the interface the
evaluator always passes the material containing x; at x == a it passes each side in
turn to obtain one-sided limits. Fields follow E = -dphi/dx and D = eps*E.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class P1Problem:
    kappa: float
    a: float = 0.5
    V: float = 1.0          # phi(0) = V, phi(1) = 0
    L: float = 1.0
    band_halfwidth: float = 0.05
    eps1: float = 1.0
    problem_id: str = "P1"

    @property
    def eps2(self):
        return float(self.kappa) * self.eps1

    def side_of(self, x):
        """Material index of each point (points exactly at a are assigned to 2; the
        evaluator never uses this for x == a)."""
        return np.where(np.asarray(x) < self.a, 1, 2)

    def eps_side(self, side):
        return np.where(np.asarray(side) == 1, self.eps1, self.eps2)

    @property
    def D_exact(self):
        """D = V / (a/eps1 + (L-a)/eps2)  (series-capacitor flux)."""
        return self.V / (self.a / self.eps1 + (self.L - self.a) / self.eps2)

    # ---------------- exact solution ----------------
    def exact(self, x, side):
        x = np.asarray(x, dtype=np.float64); side = np.asarray(side)
        D = self.D_exact
        phi1 = self.V - D * x / self.eps1
        phi2 = D * (self.L - x) / self.eps2
        d1 = np.full_like(x, -D / self.eps1); d2 = np.full_like(x, -D / self.eps2)
        return np.where(side == 1, phi1, phi2), np.where(side == 1, d1, d2)

    # deliberately wrong: solution of the uniform-permittivity problem
    def wrong_uniform(self, x, side):
        x = np.asarray(x, dtype=np.float64)
        return self.V * (1.0 - x / self.L), np.full_like(x, -self.V / self.L)


@dataclass(frozen=True)
class P1SProblem(P1Problem):
    """P1-S: manufactured variant -(eps phi')' = f, f = 1, same BCs/interface.
    Used ONLY to verify FEM convergence order (protocol Sec. 9)."""
    f: float = 1.0
    problem_id: str = "P1S"

    def _coeffs(self):
        # phi_i = -f x^2/(2 eps_i) + A_i x + B_i ; unknowns A1,B1,A2,B2
        e1, e2, a, f, V, L = self.eps1, self.eps2, self.a, self.f, self.V, self.L
        M = np.array([[0, 1, 0, 0],                  # phi1(0) = V
                      [0, 0, L, 1],                  # phi2(L) = 0
                      [a, 1, -a, -1],                # continuity
                      [e1, 0, -e2, 0]], float)       # flux: eps1 phi1' = eps2 phi2'
        rhs = np.array([V,
                        f * L**2 / (2 * e2),
                        f * a**2 / (2 * e1) - f * a**2 / (2 * e2),
                        0.0])                        # -f a + f a cancels in flux row
        return np.linalg.solve(M, rhs)

    def exact(self, x, side):
        x = np.asarray(x, dtype=np.float64); side = np.asarray(side)
        A1, B1, A2, B2 = self._coeffs()
        e1, e2, f = self.eps1, self.eps2, self.f
        phi1 = -f * x**2 / (2 * e1) + A1 * x + B1; d1 = -f * x / e1 + A1
        phi2 = -f * x**2 / (2 * e2) + A2 * x + B2; d2 = -f * x / e2 + A2
        return np.where(side == 1, phi1, phi2), np.where(side == 1, d1, d2)

    @property
    def D_exact(self):  # D varies with x in P1-S; scale for relative jumps = D at interface
        A1 = self._coeffs()[0]
        return -self.eps1 * (-self.f * self.a / self.eps1 + A1)
