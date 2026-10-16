"""Loss-visibility diagnostic (protocol Sec. 7.2): substitute analytic functions into the
exact training loss functional of each formulation."""
import jax, jax.numpy as jnp
from src.pinns.formulations import REGISTRY


def analytic_fns(problem):
    D, a, V, L, e2 = problem.D_exact, problem.a, problem.V, problem.L, problem.eps2
    exact = lambda x, side: (V - D * x / problem.eps1) if side == 1 else D * (L - x) / e2
    wrong = lambda x, side: V * (1 - x / L)
    return {"exact": exact, "wrong_uniform": wrong}


def loss_of(fname, phi_fn, d):
    F = REGISTRY[fname]
    t = F.terms(phi_fn, d); w = F.weights(d)
    vals = jnp.stack([t[k] for k in F.TERMS])
    return float(jnp.sum(w * vals)), {k: float(v) for k, v in zip(F.TERMS, vals)}


def f1_smoothed_loss(phi_fn, d, width, x_band):
    """F1 with sigmoid-smoothed permittivity eps_s(x) = eps1 + (eps2-eps1) sigmoid((x-a)/width).
    Residual (eps_s phi')' = eps_s phi'' + eps_s' phi' over interior + extra band points.
    The analytic candidates are evaluated with the side given by location."""
    e1, e2, a = d["eps1"], d["eps2"], d["a"]
    eps_s = lambda x: e1 + (e2 - e1) * jax.nn.sigmoid((x - a) / width)
    xs = jnp.concatenate([d["x1"], d["x2"], x_band])
    def r_at(x, side):
        g = jax.grad(lambda t: phi_fn(t, side)); h = jax.grad(g)
        return eps_s(x) * h(x) + jax.grad(eps_s)(x) * g(x)
    left = xs[xs < a]; right = xs[xs >= a]
    r = jnp.concatenate([jax.vmap(lambda x: r_at(x, 1))(left), jax.vmap(lambda x: r_at(x, 2))(right)])
    bc = (phi_fn(jnp.zeros_like(a), 1) - d["V"])**2 + phi_fn(d["L"], 2)**2
    return float(jnp.mean(r**2) + bc), {"pde": float(jnp.mean(r**2)), "bc": float(bc)}


def smoothed_problem_solution(problem, width):
    """Exact solution of (eps_s phi')' = 0 with the same BCs, eps_s sigmoid-smoothed.
    phi(x) = V - D_s * G(x), G(x) = int_0^x dt/eps_s(t) in closed form:
    with z = (t-a)/w:  int dt/eps_s = w [ z/eps1 + (1/eps2 - 1/eps1) ln(eps2 e^z + eps1) ]."""
    e1, e2, a, V, L, w = problem.eps1, problem.eps2, problem.a, problem.V, problem.L, width
    H = lambda x: w * (((x - a) / w) / e1 + (1 / e2 - 1 / e1) * jnp.logaddexp(jnp.log(e2) + (x - a) / w, jnp.log(e1)))
    G = lambda x: H(x) - H(0.0)
    Ds = V / G(L)
    return (lambda x, side: V - Ds * G(x)), float(Ds)
