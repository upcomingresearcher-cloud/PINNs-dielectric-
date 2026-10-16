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
    name = ""; TERMS = (); n_inputs = 2; n_nets = 1; WEIGHTS = None
    def __init__(self, prob_cls): self.P = prob_cls
    def init(self, key, width, depth, dtype):
        ks = jax.random.split(key, self.n_nets)
        return {f"net{i}": init_mlp(k, self.n_inputs, width, depth, dtype) for i, k in enumerate(ks)}
    def weights(self, d):
        return jnp.asarray(self.WEIGHTS if self.WEIGHTS else [1.0] * len(self.TERMS), d["eps1"].dtype)
    def psi(self, z, d): return self.P.psi_jax(z[0], z[1], d["pp"])
    def lift(self, z, d):
        from src.problems.p2d import coons
        return coons(lambda x, y: self.P.h_jax(x, y, d["hp"]), z[0], z[1])
    def term_vector(self, f, d):
        t = self.terms(f, d); return jnp.stack([t[k] for k in self.TERMS])
    def __hash__(self): return hash((self.name, self.P.__name__))
    def __eq__(self, o): return isinstance(o, Form2D) and (o.name, o.P) == (self.name, self.P)

    # ---- shared loss pieces
    def pde(self, f, d):
        r1 = d["eps1"] * jax.vmap(_lap(f, 1))(d["X1"]); r2 = d["eps2"] * jax.vmap(_lap(f, 2))(d["X2"])
        return jnp.mean(r1**2), jnp.mean(r2**2)
    def bc(self, f, d):
        e1 = jax.vmap(lambda z: f(z, 1))(d["XB1"]) - d["HB1"]; e2 = jax.vmap(lambda z: f(z, 2))(d["XB2"]) - d["HB2"]
        return (jnp.sum(e1**2) + jnp.sum(e2**2)) / (e1.shape[0] + e2.shape[0])
    def cont(self, f, d):
        return jnp.mean((jax.vmap(lambda z: f(z, 1))(d["XI"]) - jax.vmap(lambda z: f(z, 2))(d["XI"]))**2)
    def flux(self, f, d):
        g1 = jax.vmap(_grad(f, 1))(d["XI"]); g2 = jax.vmap(_grad(f, 2))(d["XI"])
        r = d["eps1"] * jnp.sum(g1 * d["NI"], 1) - d["eps2"] * jnp.sum(g2 * d["NI"], 1) - d["QI"]
        return jnp.mean(r**2)


class F1(Form2D):
    name = "F1"; TERMS = ("pde1", "pde2", "bc"); WEIGHTS = (0.5, 0.5, 1.0)
    def rep(self, p, z, side, d): return mlp(p["net0"], z)
    def terms(self, f, d):
        a, b = self.pde(f, d); return dict(pde1=a, pde2=b, bc=self.bc(f, d))


class F2(Form2D):
    name = "F2"; TERMS = ("pde1", "pde2", "bc", "cont", "flux"); n_nets = 2
    def rep(self, p, z, side, d): return mlp(p["net0" if side == 1 else "net1"], z)
    def terms(self, f, d):
        a, b = self.pde(f, d); return dict(pde1=a, pde2=b, bc=self.bc(f, d), cont=self.cont(f, d), flux=self.flux(f, d))


class F6(F2):
    name = "F6"
    def weights(self, d):
        eh = 2 * d["eps1"] * d["eps2"] / (d["eps1"] + d["eps2"]); Ds = eh * d["Vc"] / d["Lc"]
        wp = (d["Lc"] / Ds)**2; wv = 1 / d["Vc"]**2
        return jnp.stack([wp, wp, wv, wv, 1 / Ds**2])


class F3(Form2D):
    name = "F3"; TERMS = ("pde1", "pde2", "bc", "flux"); n_inputs = 3; WEIGHTS = (0.5, 0.5, 1.0, 1.0)
    def rep(self, p, z, side, d):
        s = -1.0 if side == 1 else 1.0
        return mlp(p["net0"], jnp.concatenate([z, jnp.stack([s * self.psi(z, d)])]))
    def terms(self, f, d):
        a, b = self.pde(f, d); return dict(pde1=a, pde2=b, bc=self.bc(f, d), flux=self.flux(f, d))


class F4(Form2D):
    """Hard BC + hard continuity; flux penalised (NOT exact)."""
    name = "F4"; TERMS = ("pde1", "pde2", "flux"); n_nets = 2
    def rep(self, p, z, side, d):
        from src.problems.p2d import bubble
        r = 0.0 if side == 1 else self.psi(z, d)
        return self.lift(z, d) + bubble(z[0], z[1]) * (mlp(p["net0"], z) + r * mlp(p["net1"], z))
    def terms(self, f, d):
        a, b = self.pde(f, d); return dict(pde1=a, pde2=b, flux=self.flux(f, d))


class F5(Form2D):
    """Ritz energy, hard BC, cusp feature: J = sum_i A_i mean_i 0.5 eps_i |grad phi|^2 - int_Gamma QI phi ds."""
    name = "F5"; TERMS = ("energy",); n_inputs = 3
    def rep(self, p, z, side, d):
        from src.problems.p2d import bubble
        s = -1.0 if side == 1 else 1.0
        return self.lift(z, d) + bubble(z[0], z[1]) * mlp(p["net0"], jnp.concatenate([z, jnp.stack([s * self.psi(z, d)])]))
    def terms(self, f, d):
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
    def rep(self, p, z, side, d): return self.base.rep(p, z, side, d)
    def terms(self, f, d):
        e1 = jax.vmap(lambda z: f(z, 1))(d["X1"]) - d["Y1"]; e2 = jax.vmap(lambda z: f(z, 2))(d["X2"]) - d["Y2"]
        return dict(data1=jnp.mean(e1**2), data2=jnp.mean(e2**2))


CLASSES = {"F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6}


def make(name, prob_cls):
    if name.startswith("REG_"): return Reg2D(prob_cls, CLASSES[name[4:]])
    return CLASSES[name](prob_cls)
