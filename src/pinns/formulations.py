"""PINN formulations for P1 (protocol Sec. 5).

Each formulation = representation + loss functional.
  rep(params, x, side, d) -> phi (scalar)       side is a Python int (1 or 2)
  terms(phi_fn, d) -> dict of unweighted loss terms, phi_fn(x, side) -> scalar
  term weights are an array `w` in TERM order, supplied by `weights(d)`.
`d` (data) holds arrays: x1, x2 (interior points per material), eps1, eps2, a, V, L.
The loss functional accepts ANY differentiable phi_fn, so analytic solutions can be
substituted (loss-visibility diagnostic) through exactly the training code path.
"""
import jax, jax.numpy as jnp
from src.pinns.nets import init_mlp, mlp


def _d1(f, side):
    return jax.grad(lambda x: f(x, side))


def _d2(f, side):
    return jax.grad(_d1(f, side))


def pde_terms(phi_fn, d):
    """Strong-form residual r_i = eps_i * phi'' at interior points of material i."""
    r1 = d["eps1"] * jax.vmap(_d2(phi_fn, 1))(d["x1"])
    r2 = d["eps2"] * jax.vmap(_d2(phi_fn, 2))(d["x2"])
    return jnp.mean(r1**2), jnp.mean(r2**2)


def bc_terms(phi_fn, d):
    return (phi_fn(jnp.zeros_like(d["a"]), 1) - d["V"])**2, phi_fn(d["L"], 2)**2


def cont_term(phi_fn, d):
    return (phi_fn(d["a"], 1) - phi_fn(d["a"], 2))**2


def flux_term(phi_fn, d):
    return (d["eps1"] * _d1(phi_fn, 1)(d["a"]) - d["eps2"] * _d1(phi_fn, 2)(d["a"]))**2


def energy_term(phi_fn, d):
    """Ritz energy J = sum_i |Omega_i| * mean_{Omega_i} 0.5 eps_i phi'^2.
    If quadrature weights q1, q2 are present (extra band points, non-uniform density), J = sum_j q_j f(x_j) with
    q_j = 1/rho(x_j) (bug fix 2026-10-16: an equal-weight mean over non-uniform points biased the functional)."""
    g1 = jax.vmap(_d1(phi_fn, 1))(d["x1"]); g2 = jax.vmap(_d1(phi_fn, 2))(d["x2"])
    if "q1" in d:
        return jnp.sum(d["q1"] * 0.5 * d["eps1"] * g1**2) + jnp.sum(d["q2"] * 0.5 * d["eps2"] * g2**2)
    return d["a"] * jnp.mean(0.5 * d["eps1"] * g1**2) + (d["L"] - d["a"]) * jnp.mean(0.5 * d["eps2"] * g2**2)


class Formulation:
    name = ""; TERMS = (); n_inputs = 1; n_nets = 1
    uses_interface_point = True

    def init(self, key, width, depth, dtype):
        keys = jax.random.split(key, self.n_nets)
        return {f"net{i}": init_mlp(k, self.n_inputs, width, depth, dtype) for i, k in enumerate(keys)}

    def weights(self, d):
        return jnp.ones(len(self.TERMS), dtype=d["eps1"].dtype)

    def term_vector(self, phi_fn, d):
        t = self.terms(phi_fn, d)
        return jnp.stack([t[k] for k in self.TERMS])

    def __hash__(self): return hash(self.name)
    def __eq__(self, o): return isinstance(o, Formulation) and o.name == self.name


class F1(Formulation):
    """Single network, strong form with piecewise eps. No interface term."""
    name = "F1"; TERMS = ("pde1", "pde2", "bc_left", "bc_right"); uses_interface_point = False
    def rep(self, p, x, side, d): return mlp(p["net0"], jnp.stack([x]))
    def terms(self, f, d):
        p1, p2 = pde_terms(f, d); b0, b1 = bc_terms(f, d)
        return dict(pde1=p1, pde2=p2, bc_left=b0, bc_right=b1)
    def weights(self, d):  # mean over all 256 interior points (equal counts) = 0.5*pde1 + 0.5*pde2
        return jnp.array([0.5, 0.5, 1.0, 1.0], dtype=d["eps1"].dtype)


class F2(Formulation):
    """Two subdomain networks, soft continuity + soft flux, unit weights."""
    name = "F2"; TERMS = ("pde1", "pde2", "bc_left", "bc_right", "cont", "flux"); n_nets = 2
    def rep(self, p, x, side, d): return mlp(p["net0" if side == 1 else "net1"], jnp.stack([x]))
    def terms(self, f, d):
        p1, p2 = pde_terms(f, d); b0, b1 = bc_terms(f, d)
        return dict(pde1=p1, pde2=p2, bc_left=b0, bc_right=b1, cont=cont_term(f, d), flux=flux_term(f, d))


class F6(F2):
    """F2 with frozen physics-derived weights: each residual divided by its physical scale.
    D_s = eps_h V / L, eps_h = 2 eps1 eps2 / (eps1 + eps2)."""
    name = "F6"
    def weights(self, d):
        eh = 2 * d["eps1"] * d["eps2"] / (d["eps1"] + d["eps2"])
        Ds = eh * d["V"] / d["L"]
        wp = (d["L"] / Ds)**2; wv = 1 / d["V"]**2; wf = 1 / Ds**2
        return jnp.stack([wp, wp, wv, wv, wv, wf])


class F3(Formulation):
    """Cusp-capturing single network N(x, |x-a|). One-sided |x-a| = sgn(side)*(x-a)."""
    name = "F3"; TERMS = ("pde1", "pde2", "bc_left", "bc_right", "flux"); n_inputs = 2
    def rep(self, p, x, side, d):
        s = -1.0 if side == 1 else 1.0
        return mlp(p["net0"], jnp.stack([x, s * (x - d["a"])]))
    def terms(self, f, d):
        p1, p2 = pde_terms(f, d); b0, b1 = bc_terms(f, d)
        return dict(pde1=p1, pde2=p2, bc_left=b0, bc_right=b1, flux=flux_term(f, d))
    def weights(self, d):
        return jnp.array([0.5, 0.5, 1.0, 1.0, 1.0], dtype=d["eps1"].dtype)


class F4(Formulation):
    """Hard BCs + hard continuity: phi = V(1-x/L) + x(L-x)[N_A(x) + max(x-a,0) N_B(x)].
    Flux continuity is NOT built in; it is penalised."""
    name = "F4"; TERMS = ("pde1", "pde2", "flux"); n_nets = 2
    def rep(self, p, x, side, d):
        z = jnp.stack([x]); r = 0.0 if side == 1 else (x - d["a"])
        return d["V"] * (1 - x / d["L"]) + x * (d["L"] - x) * (mlp(p["net0"], z) + r * mlp(p["net1"], z))
    def terms(self, f, d):
        p1, p2 = pde_terms(f, d); return dict(pde1=p1, pde2=p2, flux=flux_term(f, d))


class F5(Formulation):
    """Ritz energy with cusp feature and hard BCs: phi = V(1-x/L) + x(L-x) N(x, |x-a|)."""
    name = "F5"; TERMS = ("energy",); n_inputs = 2; uses_interface_point = False
    def rep(self, p, x, side, d):
        s = -1.0 if side == 1 else 1.0
        return d["V"] * (1 - x / d["L"]) + x * (d["L"] - x) * mlp(p["net0"], jnp.stack([x, s * (x - d["a"])]))
    def terms(self, f, d): return dict(energy=energy_term(f, d))


class F4Full(Formulation):
    """Intervention D only (1D): N_i + linear buffer c_i0 + c_i1 x solving BCs, continuity
    and flux exactly (4x4 system, det = a*eps2 + (L-a)*eps1 > 0). Only the PDE is penalised."""
    name = "F4full"; TERMS = ("pde1", "pde2"); n_nets = 2
    def _coef(self, p, d):
        N1 = lambda x: mlp(p["net0"], jnp.stack([x])); N2 = lambda x: mlp(p["net1"], jnp.stack([x]))
        a, L, V, e1, e2 = d["a"], d["L"], d["V"], d["eps1"], d["eps2"]
        one, zero = jnp.ones_like(a), jnp.zeros_like(a)
        M = jnp.array([[one, zero, zero, zero], [zero, zero, one, L], [one, a, -one, -a], [zero, e1, zero, -e2]])
        rhs = jnp.stack([V - N1(zero), -N2(L), N2(a) - N1(a), e2 * jax.grad(N2)(a) - e1 * jax.grad(N1)(a)])
        return jnp.linalg.solve(M, rhs)
    def rep(self, p, x, side, d):
        c = self._coef(p, d)
        if side == 1: return mlp(p["net0"], jnp.stack([x])) + c[0] + c[1] * x
        return mlp(p["net1"], jnp.stack([x])) + c[2] + c[3] * x
    def terms(self, f, d):
        p1, p2 = pde_terms(f, d); return dict(pde1=p1, pde2=p2)


def pde_terms_resnorm(phi_fn, d):
    """Intervention B: material-i strong-form residual divided by eps_i, r_i/eps_i = phi''."""
    r1 = jax.vmap(_d2(phi_fn, 1))(d["x1"]); r2 = jax.vmap(_d2(phi_fn, 2))(d["x2"])
    return jnp.mean(r1**2), jnp.mean(r2**2)


class F1Norm(F1):
    """Intervention B on F1: PDE residuals r_i/eps_i; BCs unchanged (F1 has no interface terms)."""
    name = "F1norm"
    def terms(self, f, d):
        p1, p2 = pde_terms_resnorm(f, d); b0, b1 = bc_terms(f, d)
        return dict(pde1=p1, pde2=p2, bc_left=b0, bc_right=b1)


class F2Norm(F2):
    """Intervention B on F2: PDE residuals r_i/eps_i. Continuity and flux are interface residuals not indexed
    by a single material and are left unchanged (decision logged, protocol change log v1.6.0)."""
    name = "F2norm"
    def terms(self, f, d):
        p1, p2 = pde_terms_resnorm(f, d); b0, b1 = bc_terms(f, d)
        return dict(pde1=p1, pde2=p2, bc_left=b0, bc_right=b1, cont=cont_term(f, d), flux=flux_term(f, d))


REGISTRY = {c.name: c() for c in (F1, F2, F3, F4, F5, F6, F4Full, F1Norm, F2Norm)}
INTERVENTION_B = ("F1norm", "F2norm")
PRIMARY = ("F1", "F2", "F3", "F4", "F5", "F6")


# ---------------- Regression-only baselines (protocol Sec. 7.1) ----------------
class Regression(Formulation):
    """Supervised fit of phi_exact using an unchanged representation (NOT a PINN).
    Loss: MSE of phi at the same interior collocation points, per material. Targets y1, y2
    are supplied in the data dict."""
    TERMS = ("data1", "data2")
    def __init__(self, base):
        self.base = REGISTRY[base]; self.name = f"REG_{base}"
        self.n_inputs, self.n_nets = self.base.n_inputs, self.base.n_nets
        self.uses_interface_point = False
    def rep(self, p, x, side, d): return self.base.rep(p, x, side, d)
    def terms(self, f, d):
        e1 = jax.vmap(lambda x: f(x, 1))(d["x1"]) - d["y1"]; e2 = jax.vmap(lambda x: f(x, 2))(d["x2"]) - d["y2"]
        return dict(data1=jnp.mean(e1**2), data2=jnp.mean(e2**2))


for _b in ("F1", "F2", "F3", "F4"):     # four distinct architectures (protocol Sec. 7.1)
    REGISTRY[f"REG_{_b}"] = Regression(_b)
REGRESSION = ("REG_F1", "REG_F2", "REG_F3", "REG_F4")


class RegressionField(Regression):
    """EXPLORATORY (Phase 6, not pre-registered): supervised fit of phi AND of D = -eps phi',
    D-error normalised by D_exact so both materials carry equal weight. Same architecture/budget."""
    TERMS = ("data1", "data2", "flux1", "flux2")
    def __init__(self, base):
        super().__init__(base); self.name = f"REGD_{base}"
    def terms(self, f, d):
        t = super().terms(f, d)
        g1 = jax.vmap(jax.grad(lambda x: f(x, 1)))(d["x1"]); g2 = jax.vmap(jax.grad(lambda x: f(x, 2)))(d["x2"])
        Dex = 1.0 / (d["a"] / d["eps1"] + (d["L"] - d["a"]) / d["eps2"]) * d["V"]
        t["flux1"] = jnp.mean(((-d["eps1"] * g1) - Dex)**2) / Dex**2
        t["flux2"] = jnp.mean(((-d["eps2"] * g2) - Dex)**2) / Dex**2
        return t


for _b in ("F1", "F2", "F3", "F4"):
    REGISTRY[f"REGD_{_b}"] = RegressionField(_b)
