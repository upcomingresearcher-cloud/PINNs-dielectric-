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
    def h_train_np(self, X): return self.h_np(X)          # training-variable boundary data (differs for P3)
    def exact_train(self, X, side): return self.exact(X, side)
    def to_physical(self, X, phi, grad, side=None): return phi, grad

    def boundary_points(self, rng, n_edge=64):
        s = rng.uniform(-1, 1, (4, n_edge))
        return np.concatenate([np.c_[s[0], -np.ones(n_edge)], np.c_[np.ones(n_edge), s[1]],
                               np.c_[s[2], np.ones(n_edge)], np.c_[-np.ones(n_edge), s[3]]])


class P2Cylinder(Problem2D):
    """Dielectric cylinder (radius R) in a uniform field E0 x-hat; exact isolated-cylinder solution
    used as Dirichlet data on the box, so the reference is exact in the box (verified symbolically)."""
    problem_id = "P2"; R = 0.25; E0 = 1.0; Lc = 2.0; geometry_label = "2d_cylinder_R0.25_box2"; refined_quadrature = dict(nr=16, nth=32)
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
        return dict(X1=X1, X2=X2, XI=XI, NI=NI, QI=np.zeros(n_interface), PI=np.zeros(n_interface), XB=XB, sideB=self.side_of(XB),
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


class P3LineCharge(Problem2D):
    """2D line charge lambda = 2*pi at (0, H) in material 1 (y > 0, eps1 = 1) above material 2 (y < 0, eps2 = kappa).
    Exact (image) solution, verified symbolically. Singular subtraction: phi_s = -ln r1 (lambda/(2 pi eps1) = 1);
    networks learn w = phi - phi_s, which is harmonic in each half, continuous at y = 0, with flux jump
    QI = eps1 dw1/dn - eps2 dw2/dn = H (eps1 - eps2)/(x^2 + H^2), n = (0, -1). Evaluation excludes r1 < 0.05."""
    problem_id = "P3"; H = 0.25; Lc = 2.0; Vc = 1.0; R_EXCL = 0.05; R_NEAR = 0.15; geometry_label = "2d_linecharge_h0.25_planar"; refined_quadrature = dict(n=16, nth=32)
    @property
    def K(self): return (self.eps1 - self.eps2) / (self.eps1 + self.eps2)

    @staticmethod
    def psi_jax(x, y, pp): return -y                      # > 0 in material 2 (lower half)
    @staticmethod
    def h_jax(x, y, hp):                                   # training boundary data for w (hp = [K, H])
        r2 = jnp.log(x**2 + (y + hp[1])**2); r1 = jnp.log(x**2 + (y - hp[1])**2)
        return jnp.where(y >= 0, -0.5 * hp[0] * r2, -0.5 * hp[0] * r1)
    def jax_params(self): return dict(pp=[0.0], hp=[self.K, self.H])
    def side_of(self, X): return np.where(X[:, 1] < 0, 2, 1)

    def phis(self, X):
        dx, dy = X[:, 0], X[:, 1] - self.H; r2 = dx**2 + dy**2
        return -0.5 * np.log(r2), np.c_[-dx / r2, -dy / r2]
    def exact_w(self, X, side):
        x, y = X[:, 0], X[:, 1]; s = np.asarray(side)
        yy = np.where(s == 1, y + self.H, y - self.H); r2 = x**2 + yy**2
        return -0.5 * self.K * np.log(r2), np.c_[-self.K * x / r2, -self.K * yy / r2]
    def exact(self, X, side):                              # physical potential
        w, gw = self.exact_w(X, side); s, gs = self.phis(X); return w + s, gw + gs
    def exact_train(self, X, side): return self.exact_w(X, side)
    def to_physical(self, X, phi, grad, side=None):
        s, gs = self.phis(X); return phi + s, grad + gs
    def h_np(self, X):                                      # PHYSICAL boundary data (evaluator)
        return self.exact(X, self.side_of(X))[0]
    def h_train_np(self, X): return self.exact_w(X, self.side_of(X))[0]
    def exact_jax(self, z, side):                           # w (training variable)
        yy = z[1] + self.H if side == 1 else z[1] - self.H
        return -0.5 * self.K * jnp.log(z[0]**2 + yy**2)
    def wrong_uniform_jax(self, z, side):                   # uniform-eps answer for w: phi_s alone -> w = 0
        return 0.0 * z[0]
    def wrong_uniform(self, X, side):
        return self.phis(X)

    def sample(self, seed, n_per_material=512, n_interface=128, n_edge=64):
        rng = np.random.default_rng([seed, 3031])
        def half(sign):
            out = []
            while len(out) < n_per_material:
                p = np.c_[rng.uniform(-1, 1, n_per_material), sign * rng.uniform(0, 1, n_per_material)]
                out.extend(p[np.hypot(p[:, 0], p[:, 1] - self.H) > self.R_EXCL])
            return np.asarray(out[:n_per_material])
        X1, X2 = half(1.0), half(-1.0)
        xi = -1 + (np.arange(n_interface) + rng.uniform(size=n_interface)) * 2 / n_interface
        XI = np.c_[xi, np.zeros(n_interface)]; NI = np.tile([0.0, -1.0], (n_interface, 1))
        QI = self.H * (self.eps1 - self.eps2) / (xi**2 + self.H**2)
        XB = self.boundary_points(rng, n_edge)
        return dict(X1=X1, X2=X2, XI=XI, NI=NI, QI=QI, PI=np.zeros(n_interface), XB=XB, sideB=self.side_of(XB), A1=2.0, A2=2.0, LG=2.0)

    def quadrature(self, n=8, nth=16):
        """Tensor GL on rectangles; a polar patch around the source on the square S = [-0.2,0.2] x [0.05,0.45]."""
        hb = self.band_halfwidth; X, W, S = [], [], []
        def rect(x0, x1, y0, y1, nx, ny, side):
            for i in range(nx):
                for j in range(ny):
                    xs, wx = gl(x0 + (x1 - x0) * i / nx, x0 + (x1 - x0) * (i + 1) / nx, n)
                    ys, wy = gl(y0 + (y1 - y0) * j / ny, y0 + (y1 - y0) * (j + 1) / ny, n)
                    A, B = np.meshgrid(xs, ys); WA, WB = np.meshgrid(wx, wy)
                    X.append(np.c_[A.ravel(), B.ravel()]); W.append((WA * WB).ravel()); S.append(np.full(A.size, side))
        rect(-1, 1, -1, -hb, 16, 8, 2); rect(-1, 1, -hb, 0, 16, 1, 2); rect(-1, 1, 0, hb, 16, 1, 1)
        rect(-1, -0.2, hb, 1, 8, 8, 1); rect(0.2, 1, hb, 1, 8, 8, 1); rect(-0.2, 0.2, 0.45, 1, 4, 5, 1)
        cx, cy, a = 0.0, self.H, 0.2
        for k in range(8):                                  # square S minus disc r < R_EXCL, polar about the source
            th, wt = gl(k * np.pi / 4, (k + 1) * np.pi / 4, nth)   # sector edges at the square corners (kinks of rmax)
            for t_, w_ in zip(th, wt):
                rmax = a / max(abs(np.cos(t_)), abs(np.sin(t_)))
                for (r0, r1) in ((self.R_EXCL, self.R_NEAR), (self.R_NEAR, rmax)):
                    for jj in range(2):
                        aa, bb = r0 + (r1 - r0) * jj / 2, r0 + (r1 - r0) * (jj + 1) / 2
                        r, wr = gl(aa, bb, n)
                        X.append(np.c_[cx + r * np.cos(t_), cy + r * np.sin(t_)]); W.append(w_ * wr * r); S.append(np.full(r.size, 1))
        X = np.concatenate(X); W = np.concatenate(W); S = np.concatenate(S)
        band = np.abs(X[:, 1]) < hb
        xi, wi = [], []
        for i in range(16):
            a_, b_ = -1 + i / 8, -1 + (i + 1) / 8; t, w = gl(a_, b_, n); xi.append(t); wi.append(w)
        xi = np.concatenate(xi); wi = np.concatenate(wi)
        tb, wb = gl(-1, 1, 32)
        XB = np.concatenate([np.c_[tb, -np.ones(32)], np.c_[np.ones(32), tb], np.c_[tb, np.ones(32)], np.c_[-np.ones(32), tb]])
        return dict(X=X, W=W, side=S, band=band, XI=np.c_[xi, np.zeros_like(xi)], WI=wi, NI=np.tile([0.0, -1.0], (len(xi), 1)),
                    XB=XB, sideB=self.side_of(XB))

    def interface_load(self, x):                       # Q(x) on y = 0 (x: (2, ...) quadrature coordinates)
        return self.H * (self.eps1 - self.eps2) / (x[0]**2 + self.H**2)
    def interface_facets(self, mesh):
        mid = mesh.p[:, mesh.facets].mean(1)
        return np.nonzero((np.abs(mid[1]) < 1e-12) & (np.abs(mid[0]) < 1 - 1e-12))[0]

    def extra_metrics(self, X, W, side, grad_pred):
        _, ge = self.exact(X, side); r1 = np.hypot(X[:, 0], X[:, 1] - self.H)
        out = {}
        for nm, mk in (("near_source", r1 < self.R_NEAR), ("far_field", r1 >= self.R_NEAR)):
            e = np.sum((grad_pred[mk] - ge[mk])**2, 1); ref = np.sum(ge[mk]**2, 1)
            out[f"relL2_E_{nm}"] = float(np.sqrt(np.sum(W[mk] * e) / np.sum(W[mk] * ref)))
        return out



class P3MatSub(P3LineCharge):
    """PRE-REGISTERED SENSITIVITY VARIANT (v1.3.0; F2 only): material-wise subtraction. Training variable
    w1 = phi - phi_s in material 1 (y > 0), w2 = phi in material 2. Interface: w2 - w1 = phi_s (PI), and
    eps1 dw1/dn - eps2 dw2/dn = -eps1 dphi_s/dn (QI), n = (0,-1). Evaluation identical to P3 (physical phi)."""
    problem_id = "P3"; geometry_label = "2d_linecharge_h0.25_planar_matsub"
    @staticmethod
    def h_jax(x, y, hp):
        K, H = hp[0], hp[1]
        up = -0.5 * K * jnp.log(x**2 + (y + H)**2)
        dn = -(1 + K) * 0.5 * jnp.log(x**2 + (y - H)**2)        # phi = -2 eps1/(eps1+eps2) ln r1 = -(1+K) ln r1
        return jnp.where(y >= 0, up, dn)
    def exact_train(self, X, side):
        w, gw = self.exact_w(X, side); s = np.asarray(side) == 2
        ph, gp = self.exact(X, side)
        return np.where(s, ph, w), np.where(s[:, None], gp, gw)
    def h_train_np(self, X): return self.exact_train(X, self.side_of(X))[0]
    def to_physical(self, X, phi, grad, side=None):
        s, gs = self.phis(X); m = (np.asarray(side) == 1) if side is not None else np.ones(len(X), bool)
        return phi + m * s, grad + m[:, None] * gs
    def exact_jax(self, z, side):
        if side == 1: return -0.5 * self.K * jnp.log(z[0]**2 + (z[1] + self.H)**2)
        return -(1 + self.K) * 0.5 * jnp.log(z[0]**2 + (z[1] - self.H)**2)
    def sample(self, seed, *a, **k):
        S = super().sample(seed, *a, **k); xi = S["XI"][:, 0]
        S["PI"] = -0.5 * np.log(xi**2 + self.H**2)                     # phi_s on y = 0
        S["QI"] = self.eps1 * self.H / (xi**2 + self.H**2)             # -eps1 dphi_s/dn with dphi_s/dn = -H/r^2 at y=0, n = (0,-1)
        return S


def corner_lambda(k):
    """Smallest positive root of kappa tan(l pi/4) + tan(3 l pi/4) = 0 for the bisector-symmetric mode."""
    from scipy.optimize import brentq
    if k == 1: return 1.0
    f = lambda l: k * np.tan(l * np.pi / 4) + np.tan(3 * l * np.pi / 4)
    return brentq(f, 2 / 3 + 1e-12, 1.0, xtol=1e-15) if k > 1 else brentq(f, 1.0, 4 / 3 - 1e-12, xtol=1e-15)


class P4Corner(Problem2D):
    """Two-material corner: material 2 (eps kappa) = quadrant x > 0, y > 0 (90 deg); material 1 elsewhere (270 deg).
    Exact solution phi = r^lam Phi(theta): Phi2 = A cos(lam(theta - pi/4)), Phi1 = cos(lam(theta - 5pi/4)),
    A = cos(3 lam pi/4)/cos(lam pi/4); verified (results/validation/p4_eigen_checks.txt). Used as Dirichlet data on
    the box, so it is exact. Field ~ r^(lam-1): singular for kappa > 1. Corner disc r < 0.02 excluded from all
    metrics; band = strips |y| < 0.05 (x > -0.05) and |x| < 0.05 (y > -0.05). Level set psi = x y."""
    problem_id = "P4"; Lc = 2.0; Vc = 1.0; R_EXCL = 0.02; geometry_label = "2d_corner_quadrant"; refined_quadrature = dict(n=16, nth=32)
    def __init__(self, kappa):
        super().__init__(kappa); self.lam = corner_lambda(self.kappa)
        self.A = np.cos(3 * self.lam * np.pi / 4) / np.cos(self.lam * np.pi / 4)
    @staticmethod
    def psi_jax(x, y, pp): return x * y
    @staticmethod
    def h_jax(x, y, hp):                                   # hp = [lam, A]
        lam, A = hp[0], hp[1]
        r = jnp.sqrt(x**2 + y**2); t = jnp.arctan2(y, x); t = jnp.where(t < 0, t + 2 * jnp.pi, t)
        in2 = (x > 0) & (y > 0)
        return r**lam * jnp.where(in2, A * jnp.cos(lam * (t - jnp.pi / 4)), jnp.cos(lam * (t - 5 * jnp.pi / 4)))
    def jax_params(self): return dict(pp=[0.0], hp=[self.lam, self.A])
    def side_of(self, X): return np.where((X[:, 0] > 0) & (X[:, 1] > 0), 2, 1)
    def _theta(self, X, side):
        t = np.arctan2(X[:, 1], X[:, 0]); s = np.asarray(side)
        t2 = np.clip(t, 0, np.pi / 2); t1 = np.mod(t, 2 * np.pi); t1 = np.where(t1 < np.pi / 2 - 1e-14, t1 + 2 * np.pi, t1)
        return np.where(s == 2, t2, t1)
    def exact(self, X, side):
        s = np.asarray(side); r = np.hypot(X[:, 0], X[:, 1]); t = self._theta(X, s); lam = self.lam
        c = np.where(s == 2, self.A, 1.0); t0 = np.where(s == 2, np.pi / 4, 5 * np.pi / 4)
        Ph = c * np.cos(lam * (t - t0)); dPh = -c * lam * np.sin(lam * (t - t0))
        phi = r**lam * Ph
        gr = lam * r**(lam - 1) * Ph; gt = r**(lam - 1) * dPh             # radial, angular components
        g = np.c_[gr * np.cos(t) - gt * np.sin(t), gr * np.sin(t) + gt * np.cos(t)]
        return phi, g
    def exact_jax(self, z, side):
        x, y = z[0], z[1]; r = jnp.sqrt(x**2 + y**2); t = jnp.arctan2(y, x)
        if side == 2: return r**self.lam * self.A * jnp.cos(self.lam * (t - jnp.pi / 4))   # no clip: its derivative vanishes at the bound theta = 0
        t = jnp.where(t < 0, t + 2 * jnp.pi, t); t = jnp.where(t < jnp.pi / 2 - 1e-14, t + 2 * jnp.pi, t)
        return r**self.lam * jnp.cos(self.lam * (t - 5 * jnp.pi / 4))
    def h_np(self, X): return self.exact(X, self.side_of(X))[0]
    def wrong_uniform_jax(self, z, side): return -(z[0] + z[1]) / np.sqrt(2.0)
    def wrong_uniform(self, X, side):
        return -(X[:, 0] + X[:, 1]) / np.sqrt(2), np.tile([-1 / np.sqrt(2), -1 / np.sqrt(2)], (len(X), 1))

    def sample(self, seed, n_per_material=512, n_interface=128, n_edge=64):
        rng = np.random.default_rng([seed, 4049])
        def draw(cond):
            out = []
            while len(out) < n_per_material:
                p = rng.uniform(-1, 1, (4 * n_per_material, 2)); m = cond(p) & (np.hypot(p[:, 0], p[:, 1]) > self.R_EXCL)
                out.extend(p[m])
            return np.asarray(out[:n_per_material])
        X2 = draw(lambda p: (p[:, 0] > 0) & (p[:, 1] > 0)); X1 = draw(lambda p: ~((p[:, 0] > 0) & (p[:, 1] > 0)))
        m = n_interface // 2
        s = self.R_EXCL + (np.arange(m) + rng.uniform(size=m)) * (1 - self.R_EXCL) / m
        XI = np.r_[np.c_[s, np.zeros(m)], np.c_[np.zeros(m), s]]
        NI = np.r_[np.tile([0.0, 1.0], (m, 1)), np.tile([1.0, 0.0], (m, 1))]          # from material 1 into 2
        XB = self.boundary_points(rng, n_edge)
        return dict(X1=X1, X2=X2, XI=XI, NI=NI, QI=np.zeros(2 * m), PI=np.zeros(2 * m), XB=XB, sideB=self.side_of(XB),
                    A1=3.0 - 0.75 * np.pi * self.R_EXCL**2, A2=1.0 - 0.25 * np.pi * self.R_EXCL**2, LG=2 * (1 - self.R_EXCL))

    def quadrature(self, n=8, nth=16):
        hb = self.band_halfwidth; X, W, S = [], [], []
        def rect(x0, x1, y0, y1, nx, ny):
            for i in range(nx):
                for j in range(ny):
                    xs, wx = gl(x0 + (x1 - x0) * i / nx, x0 + (x1 - x0) * (i + 1) / nx, n)
                    ys, wy = gl(y0 + (y1 - y0) * j / ny, y0 + (y1 - y0) * (j + 1) / ny, n)
                    A_, B_ = np.meshgrid(xs, ys); WA, WB = np.meshgrid(wx, wy)
                    P = np.c_[A_.ravel(), B_.ravel()]; X.append(P); W.append((WA * WB).ravel()); S.append(self.side_of(P))
        cuts = [-1, -hb, 0, hb, 1]; nseg = {(-1, -hb): 8, (-hb, 0): 1, (0, hb): 1, (hb, 1): 8}
        for xa, xb in zip(cuts[:-1], cuts[1:]):
            for ya, yb in zip(cuts[:-1], cuts[1:]):
                if abs(xa) <= hb and abs(xb) <= hb and abs(ya) <= hb and abs(yb) <= hb: continue     # corner square: polar
                rect(xa, xb, ya, yb, nseg[(xa, xb)], nseg[(ya, yb)])
        for k in range(8):                                  # square [-hb, hb]^2 minus disc r < R_EXCL, polar about the corner
            th, wt = gl(k * np.pi / 4, (k + 1) * np.pi / 4, nth)
            for t_, w_ in zip(th, wt):
                rmax = hb / max(abs(np.cos(t_)), abs(np.sin(t_)))
                for jj in range(3):
                    a_, b_ = self.R_EXCL + (rmax - self.R_EXCL) * jj / 3, self.R_EXCL + (rmax - self.R_EXCL) * (jj + 1) / 3
                    r, wr = gl(a_, b_, n); P = np.c_[r * np.cos(t_), r * np.sin(t_)]
                    X.append(P); W.append(w_ * wr * r); S.append(self.side_of(P))
        X = np.concatenate(X); W = np.concatenate(W); S = np.concatenate(S)
        band = ((np.abs(X[:, 1]) < hb) & (X[:, 0] > -hb)) | ((np.abs(X[:, 0]) < hb) & (X[:, 1] > -hb))
        si, wi = [], []
        for a_, b_ in ((self.R_EXCL, hb),) + tuple((hb + (1 - hb) * i / 8, hb + (1 - hb) * (i + 1) / 8) for i in range(8)):
            t, w = gl(a_, b_, n); si.append(t); wi.append(w)
        si = np.concatenate(si); wi = np.concatenate(wi)
        XI = np.r_[np.c_[si, np.zeros_like(si)], np.c_[np.zeros_like(si), si]]
        NI = np.r_[np.tile([0.0, 1.0], (len(si), 1)), np.tile([1.0, 0.0], (len(si), 1))]
        tb, wb = gl(-1, 1, 32)
        XB = np.concatenate([np.c_[tb, -np.ones(32)], np.c_[np.ones(32), tb], np.c_[tb, np.ones(32)], np.c_[-np.ones(32), tb]])
        return dict(X=X, W=W, side=S, band=band, XI=XI, WI=np.r_[wi, wi], NI=NI, XB=XB, sideB=self.side_of(XB))

    def corner_metrics(self, pred):
        """H10: signed relative |E| error at fixed r on fixed rays; fitted r^lam coefficient on r = 0.05."""
        out = {}
        for nm, th, side in (("bisector2", np.pi / 4, 2), ("bisector1", 5 * np.pi / 4, 1), ("ray_pi", np.pi, 1)):
            for r in (0.02, 0.05, 0.1):
                X = np.array([[r * np.cos(th), r * np.sin(th)]]); s = np.array([side])
                _, gp = pred(X, s); _, ge = self.exact(X, s)
                out[f"signedE_{nm}_r{r:g}"] = float((np.linalg.norm(gp) - np.linalg.norm(ge)) / np.linalg.norm(ge))
        t, w = gl(0, 2 * np.pi, 32); t = np.concatenate([t / 8 + k * np.pi / 4 for k in range(8)]); w = np.tile(w / 8, 8)
        X = 0.05 * np.c_[np.cos(t), np.sin(t)]; s = self.side_of(X + 1e-15)
        php, _ = pred(X, s); phe, _ = self.exact(X, s)
        out["coef_ratio_r0.05"] = float(np.sum(w * php * phe) / np.sum(w * phe * phe))
        return out


class P5Film(Problem2D):
    """Thin high-permittivity film: material 2 = rectangle |x| < 0.5, |y| < t/2 (eps kappa); material 1 elsewhere.
    Applied field via phi = -E0 y on the box. No closed form: reference = converged body-fitted P2-FEM (graded mesh,
    h = t/8 near the film), cached in results/reference/. Corner squares of half-width e = t/2 around the four film
    corners are excluded from all metrics (90-degree high-eps corners are singular, cf. P4). Band = |x| < 0.55 and
    |y| < t/2 + 0.05 (max-norm dilation of the film by 0.05; contains the whole film). Interface metrics on the long
    faces only (the short faces lie inside the corner exclusion)."""
    problem_id = "P5"; reference_type = "fem"; E0 = 1.0; Lc = 2.0; geometry_label = "2d_thin_film"; refined_quadrature = dict(n=12); T = 0.02
    REF_LEVEL = dict(h_far=0.05, frac_in=1 / 8)
    def __init__(self, kappa):
        super().__init__(kappa); self.t = self.T; self.e = self.t / 2; self._ref = None; self._memo = {}
    @property
    def Vc(self): return self.E0 * self.Lc
    @staticmethod
    def psi_jax(x, y, pp): return (0.25 - x**2) * (pp[0]**2 / 4 - y**2) / (0.25 * pp[0])
    @staticmethod
    def h_jax(x, y, hp): return -hp[0] * y
    def jax_params(self): return dict(pp=[self.t], hp=[self.E0])
    def side_of(self, X): return np.where((np.abs(X[:, 0]) < 0.5) & (np.abs(X[:, 1]) < self.t / 2), 2, 1)
    def h_np(self, X): return -self.E0 * X[:, 1]
    def exact_jax(self, z, side):
        if self.kappa != 1: raise NotImplementedError("P5 has no closed form for kappa != 1 (FEM reference)")
        return -self.E0 * z[1]
    def wrong_uniform_jax(self, z, side): return -self.E0 * z[1]
    def wrong_uniform(self, X, side): return -self.E0 * X[:, 1], np.tile([0.0, -self.E0], (len(X), 1))
    def geometry(self): return lambda occ: [occ.addRectangle(-0.5, -self.t / 2, 0, 1, self.t)]

    # ---------------- reference (cached converged FEM)
    def ref_path(self, level=None):
        import pathlib
        lv = level or self.REF_LEVEL
        return pathlib.Path(__file__).resolve().parents[2] / f"results/reference/P5_t{self.t:g}_k{self.kappa:g}_hf{lv['h_far']:g}_fi{lv['frac_in']:g}.npz"
    def build_reference(self, level=None, force=False):
        import time, skfem
        from src.fem import fem2d
        lv = level or self.REF_LEVEL; path = self.ref_path(lv)
        if path.exists() and not force: return self.load_reference(lv)
        path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        s = fem2d.solve(self, lv["h_far"], geometry=self.geometry(), grade=(self.t * lv["frac_in"], self.t, 0.3))
        np.savez_compressed(path, p=s.mesh.p, tri=s.mesh.t, M=s.M, u=s.u, seconds=time.perf_counter() - t0)
        return s
    def load_reference(self, level=None):
        import skfem
        from src.fem.fem2d import FEM2DSolution
        z = np.load(self.ref_path(level)); mesh = skfem.MeshTri(z["p"], z["tri"]); basis = skfem.Basis(mesh, skfem.ElementTriP2())
        return FEM2DSolution(mesh, basis, z["u"], z["M"], True, 2, 0.0, float(z["seconds"]))
    def exact(self, X, side):
        if self.kappa == 1.0:                      # uniform field is exact at kappa = 1
            return -self.E0 * X[:, 1], np.tile([0.0, -self.E0], (len(X), 1))
        key = (X.tobytes(), np.asarray(side).tobytes())
        if key not in self._memo:
            if self._ref is None: self._ref = self.build_reference()
            self._memo[key] = self._ref(X, side)
        return self._memo[key]
    def exact_train(self, X, side): return self.exact(X, side)
    def h_train_np(self, X): return self.h_np(X)

    # ---------------- training points
    def sample(self, seed, n_per_material=512, n_interface=128, n_edge=64, n_film=None):
        rng = np.random.default_rng([seed, 5051]); t = self.t; nf = n_film or n_per_material
        X2 = np.c_[rng.uniform(-0.5, 0.5, nf), rng.uniform(-t / 2, t / 2, nf)]
        X1 = []
        while len(X1) < n_per_material:
            p = rng.uniform(-1, 1, (2 * n_per_material, 2)); X1.extend(p[self.side_of(p) == 1])
        X1 = np.asarray(X1[:n_per_material])
        ns = 2; nl = (n_interface - 2 * ns) // 2                 # long faces get the rest, >= 2 points per short face
        xl = -0.5 + (np.arange(nl) + rng.uniform(size=nl)) / nl
        ys = -t / 2 + (np.arange(ns) + rng.uniform(size=ns)) * t / ns
        XI = np.r_[np.c_[xl, np.full(nl, t / 2)], np.c_[xl, np.full(nl, -t / 2)], np.c_[np.full(ns, 0.5), ys], np.c_[np.full(ns, -0.5), ys]]
        NI = np.r_[np.tile([0., -1.], (nl, 1)), np.tile([0., 1.], (nl, 1)), np.tile([-1., 0.], (ns, 1)), np.tile([1., 0.], (ns, 1))]
        XB = self.boundary_points(rng, n_edge); m = len(XI)
        return dict(X1=X1, X2=X2, XI=XI, NI=NI, QI=np.zeros(m), PI=np.zeros(m), XB=XB, sideB=self.side_of(XB),
                    A1=4 - t, A2=t, LG=2 * (1 + t))

    # ---------------- evaluation quadrature (cell-aligned with film faces, band edges, corner exclusions)
    def quadrature(self, n=6):
        t, e, hb = self.t, self.e, self.band_halfwidth
        def breaks(core, outer=1.0, pan=0.125):
            b = sorted(set(core)); out = []
            for a_, b_ in zip(b[:-1], b[1:]):
                k = max(1, int(np.ceil((b_ - a_) / pan - 1e-12))); out.extend(np.linspace(a_, b_, k + 1)[:-1])
            return np.r_[out, b[-1]]
        xb = breaks([-1, -0.5 - hb, -0.5 - e, -0.5, -0.5 + e, 0.5 - e, 0.5, 0.5 + e, 0.5 + hb, 1], pan=0.1)
        yb = breaks([-1, -t / 2 - hb, -t / 2 - e, -t / 2, 0.0, t / 2, t / 2 + e, t / 2 + hb, 1])
        X, W = [], []
        for x0, x1 in zip(xb[:-1], xb[1:]):
            for y0, y1 in zip(yb[:-1], yb[1:]):
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if abs(abs(cx) - 0.5) < e and abs(abs(cy) - t / 2) < e: continue          # corner exclusion square
                xs, wx = gl(x0, x1, n); ys, wy = gl(y0, y1, n)
                A_, B_ = np.meshgrid(xs, ys); WA, WB = np.meshgrid(wx, wy)
                X.append(np.c_[A_.ravel(), B_.ravel()]); W.append((WA * WB).ravel())
        X = np.concatenate(X); W = np.concatenate(W); S = self.side_of(X)
        band = (np.abs(X[:, 0]) < 0.5 + hb) & (np.abs(X[:, 1]) < t / 2 + hb)
        xi, wi = [], []
        for i in range(10):
            a_, b_ = -0.5 + e + (1 - 2 * e) * i / 10, -0.5 + e + (1 - 2 * e) * (i + 1) / 10; q, w = gl(a_, b_, 8); xi.append(q); wi.append(w)
        xi = np.concatenate(xi); wi = np.concatenate(wi)
        XI = np.r_[np.c_[xi, np.full_like(xi, t / 2)], np.c_[xi, np.full_like(xi, -t / 2)]]
        NI = np.r_[np.tile([0., -1.], (len(xi), 1)), np.tile([0., 1.], (len(xi), 1))]
        tb, _ = gl(-1, 1, 32)
        XB = np.concatenate([np.c_[tb, -np.ones(32)], np.c_[np.ones(32), tb], np.c_[tb, np.ones(32)], np.c_[-np.ones(32), tb]])
        return dict(X=X, W=W, side=S, band=band, XI=XI, WI=np.r_[wi, wi], NI=NI, XB=XB, sideB=np.ones(len(XB), int))

    def extra_metrics(self, X, W, side, grad_pred):
        _, ge = self.exact(X, side); m = side == 2
        e = np.sum((grad_pred[m] - ge[m])**2, 1); r = np.sum(ge[m]**2, 1)
        Ein = -np.sum(W[m] * grad_pred[m, 1]) / np.sum(W[m]); Ein_ref = -np.sum(W[m] * ge[m, 1]) / np.sum(W[m])
        return dict(relL2_E_film=float(np.sqrt(np.sum(W[m] * e) / np.sum(W[m] * r))), Ey_film_mean=float(Ein), Ey_film_mean_ref=float(Ein_ref))


class P5FilmThick(P5Film):
    T = 0.1; geometry_label = "2d_thin_film_t0.1"
class P5FilmThin(P5Film):
    T = 0.005; geometry_label = "2d_thin_film_t0.005"
