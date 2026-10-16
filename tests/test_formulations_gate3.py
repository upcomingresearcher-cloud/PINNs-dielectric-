"""Gate 3: mathematical correctness of each formulation before any training result counts."""
import numpy as np, pytest, yaml, pathlib
import jax, jax.numpy as jnp
from src.pinns import train as T
from src.pinns.formulations import REGISTRY, PRIMARY
from src.pinns.diagnostics import analytic_fns, loss_of
from src.pinns.nets import n_params
from src.problems.p1 import P1Problem

CFG = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "configs/protocol_v1.yaml"))
KAPPAS = [1.0, 80.0, 2000.0]
EXPECTED_PARAMS = {"F1": 2209, "F2": 4418, "F3": 2241, "F4": 4418, "F5": 2241, "F6": 4418, "F4full": 4418,
                   "F1norm": 2209, "F2norm": 4418}
EXPECTED_PARAMS.update({f"{p}_{b}": EXPECTED_PARAMS[b] for b in ("F1", "F2", "F3", "F4") for p in ("REG", "REGD")})   # same architecture as parent


def _data(k, dtype=jnp.float64, seed=0):
    p = P1Problem(k); x1, x2 = T.collocation(seed, p, 128); return p, T.make_data(p, x1, x2, dtype)


@pytest.mark.parametrize("f", list(REGISTRY))
def test_param_counts(f):
    params = REGISTRY[f].init(jax.random.PRNGKey(0), 32, 3, jnp.float64)
    assert n_params(params) == EXPECTED_PARAMS[f]


@pytest.mark.parametrize("f", PRIMARY + ("F1norm", "F2norm"))
@pytest.mark.parametrize("k", KAPPAS)
def test_loss_at_exact_solution(f, k):
    p, d = _data(k)
    L, terms = loss_of(f, analytic_fns(p)["exact"], d)
    if f == "F5":
        assert abs(L - p.D_exact / 2) < 1e-12 * max(1, p.D_exact), (L, p.D_exact / 2)   # J = W = D/2
    else:
        assert L < 1e-20, terms


@pytest.mark.parametrize("f", PRIMARY + ("F1norm", "F2norm"))
@pytest.mark.parametrize("k", KAPPAS)
def test_loss_at_wrong_solution(f, k):
    p, d = _data(k); fn = analytic_fns(p)
    Lw, terms = loss_of(f, fn["wrong_uniform"], d); Le, _ = loss_of(f, fn["exact"], d)
    if k == 1.0:
        assert abs(Lw - Le) < 1e-12, (Lw, Le)       # at kappa=1 the 'wrong' solution IS the exact one
    elif f in ("F1", "F1norm"):
        assert Lw < 1e-20, terms                    # F1 cannot see the interface (zero loss for the wrong answer)
    else:
        assert Lw > Le + 1e-6, (Lw, Le, terms)       # interface-aware losses must prefer the exact solution


@pytest.mark.parametrize("k", KAPPAS)
def test_hard_constraints_at_random_init(k):
    p, d = _data(k)
    for f, checks in (("F3", ("cont",)), ("F4", ("cont", "bc")), ("F5", ("cont", "bc")), ("F4full", ("cont", "bc", "flux"))):
        F = REGISTRY[f]; params = F.init(jax.random.PRNGKey(3), 32, 3, jnp.float64)
        phi = lambda x, s: F.rep(params, x, s, d); a = d["a"]
        if "cont" in checks: assert abs(phi(a, 1) - phi(a, 2)) < 1e-12, f
        if "bc" in checks:
            assert abs(phi(jnp.zeros_like(a), 1) - 1.0) < 1e-12 and abs(phi(d["L"], 2)) < 1e-12, f
        if "flux" in checks:
            g1 = jax.grad(lambda x: phi(x, 1))(a); g2 = jax.grad(lambda x: phi(x, 2))(a)
            assert abs(d["eps1"] * g1 - d["eps2"] * g2) < 1e-9 * max(1.0, k), f
    # soft formulations must NOT satisfy continuity at random init (otherwise the test above is vacuous)
    F = REGISTRY["F2"]; params = F.init(jax.random.PRNGKey(3), 32, 3, jnp.float64)
    assert abs(F.rep(params, d["a"], 1, d) - F.rep(params, d["a"], 2, d)) > 1e-6


def test_F6_equals_F2_at_unit_contrast():
    p, d = _data(1.0)
    assert np.allclose(np.asarray(REGISTRY["F6"].weights(d)), np.asarray(REGISTRY["F2"].weights(d)))


@pytest.mark.parametrize("f", list(REGISTRY))
def test_gradient_logger_finite_and_dtype(f):
    for prec, dt in (("float32", jnp.float32), ("float64", jnp.float64)):
        cfg = dict(CFG); row, params = T.train_run(f, 80.0, 0, cfg, overrides=dict(steps=50, precision=prec), out_dir=None)
        leaves = jax.tree_util.tree_leaves(params)
        assert all(l.dtype == dt for l in leaves), (f, prec)
        assert row["grad_evals"] == 50 and row["step_budget"] == 50
        assert not row["nonfinite"]
        F, opt, update, term_grads, phi_dphi, sched = T.compiled_fns(f, 1e-3, 1e-4, 50)
        p_, d = _data(80.0, dt)
        vals, norms = term_grads(params, d)
        assert np.all(np.isfinite(np.asarray(norms))) and len(norms) == len(F.TERMS)
        assert np.asarray(norms).dtype == np.dtype(dt)


def test_collocation_seed_dependence_only():
    p = P1Problem(80.0)
    a1, b1 = T.collocation(0, p, 128); a2, b2 = T.collocation(0, P1Problem(5.0), 128); a3, _ = T.collocation(1, p, 128)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2) and not np.array_equal(a1, a3)
    assert np.all((a1 > 0) & (a1 < 0.5)) and np.all((b1 > 0.5) & (b1 < 1.0))
    assert np.all(np.diff(a1) > 0)    # one point per stratum
