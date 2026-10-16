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
