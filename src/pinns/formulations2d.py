"""2D formulations F1-F6 (+ regression) on [-1,1]^2 (protocol Sec. 5 generalised; change log v1.2.0).

Problem hooks (static methods, JAX-traceable, parameters passed through data so one compiled
executable serves all contrasts): psi(x, y, pp) level set (> 0 in material 2), h(x, y, hp) Dirichlet data.
Data d: X1, X2 (interior), XB1, HB1, XB2, HB2 (boundary points by material), XI, NI, QI (interface
points, unit normal from material 1 to 2, exact flux jump eps1 dphi1/dn - eps2 dphi2/dn), eps1, eps2,
Vc, Lc, A1, A2, LG (interface length), pp, hp, Y1, Y2 (regression targets only).
"""
import jax, jax.numpy as jnp
from src.pinns.nets import init_mlp, mlp


def _grad(f, side): return jax.grad(lambda z: f(z, side))
def _lap(f, side): return lambda z: jnp.trace(jax.hessian(lambda q: f(q, side))(z))


class Form2D:
    """Base class for 2D formulations on a problem class (shared PDE, boundary and interface terms)."""
    name = ""; TERMS = (); n_inputs = 2; n_nets = 1; WEIGHTS = None
    def __init__(self, prob_cls): self.P = prob_cls
    def init(self, key, width, depth, dtype):
        """Initialise the parameters of the formulation's networks."""
        ks = jax.random.split(key, self.n_nets)
        return {f"net{i}": init_mlp(k, self.n_inputs, width, depth, dtype) for i, k in enumerate(ks)}
    def weights(self, d):
        """Loss weights in TERMS order."""
        return jnp.asarray(self.WEIGHTS if self.WEIGHTS else [1.0] * len(self.TERMS), d["eps1"].dtype)
    def psi(self, z, d):
        """Signed level-set function of the interface (positive in material 2)."""
        return self.P.psi_jax(z[0], z[1], d["pp"])
    def lift(self, z, d):
        """Transfinite (Coons) interpolant of the boundary data, used to impose boundary conditions exactly."""
        from src.problems.p2d import coons
        return coons(lambda x, y: self.P.h_jax(x, y, d["hp"]), z[0], z[1])
    def term_vector(self, f, d):
        """Loss terms stacked as a vector in TERMS order."""
        t = self.terms(f, d); return jnp.stack([t[k] for k in self.TERMS])
    def __hash__(self): return hash((self.name, self.P.__name__))
    def __eq__(self, o): return isinstance(o, Form2D) and (o.name, o.P) == (self.name, self.P)

    # ---- shared loss pieces
    def pde(self, f, d):
        """Mean squared PDE residuals eps_i * Laplacian(phi_i) in each material."""
        r1 = d["eps1"] * jax.vmap(_lap(f, 1))(d["X1"]); r2 = d["eps2"] * jax.vmap(_lap(f, 2))(d["X2"])
        return jnp.mean(r1**2), jnp.mean(r2**2)
    def bc(self, f, d):
        """Mean squared Dirichlet boundary residual."""
        e1 = jax.vmap(lambda z: f(z, 1))(d["XB1"]) - d["HB1"]; e2 = jax.vmap(lambda z: f(z, 2))(d["XB2"]) - d["HB2"]
        return (jnp.sum(e1**2) + jnp.sum(e2**2)) / (e1.shape[0] + e2.shape[0])
    def cont(self, f, d):
        """Mean squared potential jump at interface points (minus any prescribed jump)."""
        return jnp.mean((jax.vmap(lambda z: f(z, 1))(d["XI"]) - jax.vmap(lambda z: f(z, 2))(d["XI"]) + d["PI"])**2)
    def flux(self, f, d):
        """Mean squared normal-displacement jump at interface points (minus any interface load)."""
        g1 = jax.vmap(_grad(f, 1))(d["XI"]); g2 = jax.vmap(_grad(f, 2))(d["XI"])
        r = d["eps1"] * jnp.sum(g1 * d["NI"], 1) - d["eps2"] * jnp.sum(g2 * d["NI"], 1) - d["QI"]
        return jnp.mean(r**2)


class F1(Form2D):
    """F1 in 2D: one network, PDE and boundary terms only."""
    name = "F1"; TERMS = ("pde1", "pde2", "bc"); WEIGHTS = (0.5, 0.5, 1.0)
    def rep(self, p, z, side, d):
        """Single network for both materials."""
        return mlp(p["net0"], z)
    def terms(self, f, d):
        """PDE and boundary terms."""
        a, b = self.pde(f, d); return dict(pde1=a, pde2=b, bc=self.bc(f, d))


class F2(Form2D):
    """F2 in 2D: one network per material with soft continuity and flux terms."""
    name = "F2"; TERMS = ("pde1", "pde2", "bc", "cont", "flux"); n_nets = 2
    def rep(self, p, z, side, d):
        """Separate network for each material."""
        return mlp(p["net0" if side == 1 else "net1"], z)
    def terms(self, f, d):
        """PDE, boundary, continuity and flux terms."""
        a, b = self.pde(f, d); return dict(pde1=a, pde2=b, bc=self.bc(f, d), cont=self.cont(f, d), flux=self.flux(f, d))


class F6(F2):
    """F6 in 2D: F2 with physics-scaled weights."""
    name = "F6"
    def weights(self, d):
        """Physics-scaled weights from the harmonic-mean displacement scale."""
        eh = 2 * d["eps1"] * d["eps2"] / (d["eps1"] + d["eps2"]); Ds = eh * d["Vc"] / d["Lc"]
        wp = (d["Lc"] / Ds)**2; wv = 1 / d["Vc"]**2
        return jnp.stack([wp, wp, wv, wv, 1 / Ds**2])


class F3(Form2D):
    """F3 in 2D: one network with a signed level-set input."""
    name = "F3"; TERMS = ("pde1", "pde2", "bc", "flux"); n_inputs = 3; WEIGHTS = (0.5, 0.5, 1.0, 1.0)
    def rep(self, p, z, side, d):
        """Network with the signed level-set value as an extra input."""
        s = -1.0 if side == 1 else 1.0
        return mlp(p["net0"], jnp.concatenate([z, jnp.stack([s * self.psi(z, d)])]))
    def terms(self, f, d):
        """PDE, boundary and flux terms."""
        a, b = self.pde(f, d); return dict(pde1=a, pde2=b, bc=self.bc(f, d), flux=self.flux(f, d))


class F4(Form2D):
    """Hard BC + hard continuity; flux penalised (NOT exact)."""
    name = "F4"; TERMS = ("pde1", "pde2", "flux"); n_nets = 2
    def rep(self, p, z, side, d):
        """Boundary data and potential continuity built into the representation via a bubble function."""
        from src.problems.p2d import bubble
        r = 0.0 if side == 1 else self.psi(z, d)
        return self.lift(z, d) + bubble(z[0], z[1]) * (mlp(p["net0"], z) + r * mlp(p["net1"], z))
    def terms(self, f, d):
        """PDE and flux terms."""
        a, b = self.pde(f, d); return dict(pde1=a, pde2=b, flux=self.flux(f, d))


class F5(Form2D):
    """Ritz energy, hard BC, cusp feature: J = sum_i A_i mean_i 0.5 eps_i |grad phi|^2 - int_Gamma QI phi ds."""
    name = "F5"; TERMS = ("energy",); n_inputs = 3
    def rep(self, p, z, side, d):
        """Level-set-input network with the boundary data built in."""
        from src.problems.p2d import bubble
        s = -1.0 if side == 1 else 1.0
        return self.lift(z, d) + bubble(z[0], z[1]) * mlp(p["net0"], jnp.concatenate([z, jnp.stack([s * self.psi(z, d)])]))
    def terms(self, f, d):
        """Monte Carlo Ritz energy over the two materials (minus the interface-load work, if any)."""
        g1 = jax.vmap(_grad(f, 1))(d["X1"]); g2 = jax.vmap(_grad(f, 2))(d["X2"])
        J = d["A1"] * jnp.mean(0.5 * d["eps1"] * jnp.sum(g1**2, 1)) + d["A2"] * jnp.mean(0.5 * d["eps2"] * jnp.sum(g2**2, 1))
        J = J - d["LG"] * jnp.mean(d["QI"] * jax.vmap(lambda z: f(z, 1))(d["XI"]))
        return dict(energy=J)


class Reg2D(Form2D):
    """phi-only regression baseline (NOT a PINN)."""
    TERMS = ("data1", "data2")
    def __init__(self, prob_cls, base):
        super().__init__(prob_cls); self.base = base(prob_cls); self.name = f"REG_{self.base.name}"
        self.n_inputs, self.n_nets = self.base.n_inputs, self.base.n_nets
    def rep(self, p, z, side, d):
        """Same representation as the wrapped formulation."""
        return self.base.rep(p, z, side, d)
    def terms(self, f, d):
        """Supervised misfit of the potential in each material."""
        e1 = jax.vmap(lambda z: f(z, 1))(d["X1"]) - d["Y1"]; e2 = jax.vmap(lambda z: f(z, 2))(d["X2"]) - d["Y2"]
        return dict(data1=jnp.mean(e1**2), data2=jnp.mean(e2**2))


CLASSES = {"F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6}


def make(name, prob_cls):
    """Construct a 2D formulation by name for a problem class ('REG_<name>' gives its supervised baseline)."""
    if name.startswith("REG_"): return Reg2D(prob_cls, CLASSES[name[4:]])
    return CLASSES[name](prob_cls)
