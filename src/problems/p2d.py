"""2D electrostatic interface problems on the box [-1,1]^2 (protocol Sec. 8; change log v1.2.0).

Conventions shared by all 2D problems
- material 1: eps1 (= 1), material 2: eps2 = kappa (P2: the inclusion). psi(xy) > 0 in material 2,
  psi = 0 on the interface; interface normal n = grad psi/|grad psi| points from material 1 into 2.
- prediction interface: pred(XY, side) -> (phi (n,), grad_phi (n,2)), side in {1,2}.
- E = -grad phi, D = eps E. Normal-D jump [D_n] = D_n(2) - D_n(1).
"""
import numpy as np
import jax.numpy as jnp

GL = {n: np.polynomial.legendre.leggauss(n) for n in (4, 6, 8, 12, 16, 24, 32)}


def gl(a, b, n):
    t, w = GL[n]; return 0.5 * (b - a) * t + 0.5 * (b + a), 0.5 * (b - a) * w


def coons(h, x, y):
    """Transfinite (Coons) interpolation on [-1,1]^2 of boundary data h(x, y) (JAX)."""
    u, v = (x + 1) / 2, (y + 1) / 2
    return ((1 - u) * h(-1.0, y) + u * h(1.0, y) + (1 - v) * h(x, -1.0) + v * h(x, 1.0)
            - ((1 - u) * (1 - v) * h(-1.0, -1.0) + u * (1 - v) * h(1.0, -1.0) + (1 - u) * v * h(-1.0, 1.0) + u * v * h(1.0, 1.0)))


def bubble(x, y):
    return (1 - x**2) * (1 - y**2)


class Problem2D:
    eps1 = 1.0; band_halfwidth = 0.05; problem_id = "P2D"
    def __init__(self, kappa): self.kappa = float(kappa); self.eps2 = float(kappa)
    def eps_side(self, side): return np.where(np.asarray(side) == 1, self.eps1, self.eps2)
    def h_np(self, X): return np.asarray(self.h(X[:, 0], X[:, 1]), np.float64)   # h uses only arithmetic

    def boundary_points(self, rng, n_edge=64):
        s = rng.uniform(-1, 1, (4, n_edge))
        return np.concatenate([np.c_[s[0], -np.ones(n_edge)], np.c_[np.ones(n_edge), s[1]],
                               np.c_[s[2], np.ones(n_edge)], np.c_[-np.ones(n_edge), s[3]]])


class P2Cylinder(Problem2D):
    """Dielectric cylinder (radius R) in a uniform field E0 x-hat; exact isolated-cylinder solution
    used as Dirichlet data on the box, so the reference is exact in the box (verified symbolically)."""
    problem_id = "P2"; R = 0.25; E0 = 1.0; Lc = 2.0; geometry_label = "2d_cylinder_R0.25_box2"
    @property
    def Vc(self): return self.E0 * self.Lc
    @property
    def K(self): return (self.eps2 - self.eps1) / (self.eps2 + self.eps1)
    @property
    def E_in(self): return 2 * self.eps1 / (self.eps2 + self.eps1) * self.E0

    # --- geometry. Smooth level set (R^2 - r^2)/(2R): zero on the circle, > 0 inside, |grad| = 1 on the
    # interface. The distance R - r is NOT used: its Laplacian -1/r is singular at the centre.
    @staticmethod
    def psi_jax(x, y, pp):          # pp = [R]
        return (pp[0]**2 - x**2 - y**2) / (2 * pp[0])
    @staticmethod
    def h_jax(x, y, hp):            # hp = [E0, K, R]; Dirichlet data (box boundary lies in material 1)
        return -hp[0] * x + hp[0] * hp[1] * hp[2]**2 * x / (x**2 + y**2)
    def jax_params(self): return dict(pp=[self.R], hp=[self.E0, self.K, self.R])
    def side_of(self, X): return np.where(np.hypot(X[:, 0], X[:, 1]) < self.R, 2, 1)

    # --- exact solution (numpy)
    def exact(self, X, side):
        x, y = X[:, 0], X[:, 1]; r2 = x**2 + y**2; side = np.asarray(side)
        pin = -self.E_in * x; gin = np.c_[-self.E_in * np.ones_like(x), np.zeros_like(x)]
        c = self.E0 * self.K * self.R**2
        pout = -self.E0 * x + c * x / r2
        gout = np.c_[-self.E0 + c * (y**2 - x**2) / r2**2, -2 * c * x * y / r2**2]
        return np.where(side == 2, pin, pout), np.where((side == 2)[:, None], gin, gout)

    def h(self, x, y): return self.h_jax(x, y, [self.E0, self.K, self.R])

    def exact_jax(self, z, side):
        x, y = z[0], z[1]
        if side == 2: return -self.E_in * x
        return -self.E0 * x + self.E0 * self.K * self.R**2 * x / (x**2 + y**2)

    def wrong_uniform_jax(self, z, side): return -self.E0 * z[0]

    def wrong_uniform(self, X, side):
        return -self.E0 * X[:, 0], np.c_[-self.E0 * np.ones(len(X)), np.zeros(len(X))]

    # --- training points
    def sample(self, seed, n_per_material=512, n_interface=128, n_edge=64):
        rng = np.random.default_rng([seed, 2027])
        rr = self.R * np.sqrt(rng.uniform(size=n_per_material)); th = rng.uniform(0, 2 * np.pi, n_per_material)
        X2 = np.c_[rr * np.cos(th), rr * np.sin(th)]
        X1 = []
        while len(X1) < n_per_material:
            p = rng.uniform(-1, 1, (n_per_material, 2)); X1.extend(p[np.hypot(p[:, 0], p[:, 1]) > self.R])
        X1 = np.asarray(X1[:n_per_material])
        t = (np.arange(n_interface) + rng.uniform()) * 2 * np.pi / n_interface
        XI = self.R * np.c_[np.cos(t), np.sin(t)]; NI = -np.c_[np.cos(t), np.sin(t)]   # grad psi points inward
        XB = self.boundary_points(rng, n_edge)
        return dict(X1=X1, X2=X2, XI=XI, NI=NI, QI=np.zeros(n_interface), XB=XB, sideB=self.side_of(XB),
                    A1=4 - np.pi * self.R**2, A2=np.pi * self.R**2, LG=2 * np.pi * self.R)

    # --- evaluation quadrature (conforming to circle and band edges)
    def quadrature(self, nr=8, nth=16):
        R, hb = self.R, self.band_halfwidth
        X, W, S = [], [], []
        def ring(r0, r1, side, npan):
            for k in range(8):
                th, wt = gl(k * np.pi / 4, (k + 1) * np.pi / 4, nth)
                for j in range(npan):
                    a, b = r0 + (r1 - r0) * j / npan, r0 + (r1 - r0) * (j + 1) / npan
                    r, wr = gl(a, b, nr)
                    T, Rr = np.meshgrid(th, r); WT, WR = np.meshgrid(wt, wr)
                    X.append(np.c_[(Rr * np.cos(T)).ravel(), (Rr * np.sin(T)).ravel()]); W.append((WT * WR * Rr).ravel()); S.append(np.full(Rr.size, side))
        ring(0.0, R - hb, 2, 2); ring(R - hb, R, 2, 1); ring(R, R + hb, 1, 1)
        for k in range(8):                                   # outer region up to the box, sector-wise
            th, wt = gl(k * np.pi / 4, (k + 1) * np.pi / 4, nth)
            for t_, w_ in zip(th, wt):
                rmax = 1.0 / max(abs(np.cos(t_)), abs(np.sin(t_)))
                for j in range(4):
                    a = R + hb + (rmax - R - hb) * j / 4; b = R + hb + (rmax - R - hb) * (j + 1) / 4
                    r, wr = gl(a, b, nr)
                    X.append(np.c_[r * np.cos(t_), r * np.sin(t_)]); W.append(w_ * wr * r); S.append(np.full(r.size, 1))
        X = np.concatenate(X); W = np.concatenate(W); S = np.concatenate(S)
        band = np.abs(np.hypot(X[:, 0], X[:, 1]) - R) < hb
        t, wt = gl(0, 2 * np.pi, 32); ti = np.concatenate([t / 8 + k * np.pi / 4 for k in range(8)]); wi = np.tile(wt / 8, 8) * R
        XI = R * np.c_[np.cos(ti), np.sin(ti)]; NI = -np.c_[np.cos(ti), np.sin(ti)]
        tb, wb = gl(-1, 1, 32)
        XB = np.concatenate([np.c_[tb, -np.ones(32)], np.c_[np.ones(32), tb], np.c_[tb, np.ones(32)], np.c_[-np.ones(32), tb]])
        return dict(X=X, W=W, side=S, band=band, XI=XI, WI=wi, NI=NI, XB=XB, sideB=np.ones(len(XB), int))

    def extra_metrics(self, X, W, side, grad_pred):
        inner = (side == 2) & (np.hypot(X[:, 0], X[:, 1]) < self.R - self.band_halfwidth)
        Ex = -np.sum(W[inner] * grad_pred[inner, 0]) / np.sum(W[inner])
        keff = 2 * self.E0 / Ex - 1 if Ex > 0 else np.nan
        return dict(E_in_mean=float(Ex), kappa_eff=float(keff),
                    log_kappa_eff_ratio=float(np.log(keff / self.kappa)) if (np.isfinite(keff) and keff > 0) else float("nan"))
