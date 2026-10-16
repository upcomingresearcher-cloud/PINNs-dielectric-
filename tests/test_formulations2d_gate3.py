"""Gate 3 for 2D formulations (P2): exact-solution losses, hard constraints at random init, param counts."""
import numpy as np, pytest, jax, jax.numpy as jnp
from src.problems.p2d import P2Cylinder
from src.pinns import train2d as T
from src.pinns.formulations2d import make

FORMS = ["F1", "F2", "F3", "F4", "F5", "F6"]
PARAMS = {"F1": 2241, "F2": 4482, "F3": 2273, "F4": 4482, "F5": 2273, "F6": 4482}


def _d(k):
    p = P2Cylinder(k); return p, T.make_data(p, p.sample(0), jnp.float64)


@pytest.mark.parametrize("f", FORMS)
@pytest.mark.parametrize("k", [1.0, 80.0, 0.0125])
def test_exact_loss_and_params(f, k):
    p, d = _d(k); F = make(f, P2Cylinder)
    from src.pinns.nets import n_params
    assert n_params(F.init(jax.random.PRNGKey(0), 32, 3, jnp.float64)) == PARAMS[f]
    L = float(jnp.sum(F.weights(d) * F.term_vector(p.exact_jax, d)))
    if f == "F5": assert 0 < L < 10
    else: assert L < 1e-20


@pytest.mark.parametrize("k", [80.0, 0.0125])
def test_hard_constraints(k):
    p, d = _d(k)
    XB = jnp.asarray(p.quadrature()["XB"]); hb = p.h_np(np.asarray(XB))
    for f in ("F3", "F4", "F5"):
        F = make(f, P2Cylinder); prm = F.init(jax.random.PRNGKey(5), 32, 3, jnp.float64)
        a = jax.vmap(lambda z: F.rep(prm, z, 1, d))(d["XI"]); b = jax.vmap(lambda z: F.rep(prm, z, 2, d))(d["XI"])
        assert float(jnp.max(jnp.abs(a - b))) < 1e-12, f
        if f in ("F4", "F5"):
            vb = np.asarray(jax.vmap(lambda z: F.rep(prm, z, 1, d))(XB)); assert np.max(np.abs(vb - hb)) < 1e-12, f
    F = make("F2", P2Cylinder); prm = F.init(jax.random.PRNGKey(5), 32, 3, jnp.float64)
    assert float(jnp.max(jnp.abs(jax.vmap(lambda z: F.rep(prm, z, 1, d))(d["XI"]) - jax.vmap(lambda z: F.rep(prm, z, 2, d))(d["XI"])))) > 1e-6


def test_F6_weights_follow_frozen_rule():
    """F6 == F2 at kappa=1 holds only when V_c = L_c = 1 (P1). In P2 (V_c = L_c = 2) the frozen rule gives (4,4,1/4,1/4,1)."""
    for k in (1.0, 80.0):
        p, d = _d(k); eh = 2 * k / (1 + k); Ds = eh * p.Vc / p.Lc
        exp = [(p.Lc / Ds)**2] * 2 + [1 / p.Vc**2] * 2 + [1 / Ds**2]
        assert np.allclose(np.asarray(make("F6", P2Cylinder).weights(d)), exp)


@pytest.mark.parametrize("cls_name,forms", [("P3LineCharge", FORMS), ("P3MatSub", ["F2"])])
@pytest.mark.parametrize("k", [80.0, 2000.0, 0.0125])
def test_p3_exact_training_variable_has_zero_strong_form_loss(cls_name, forms, k):
    from src.problems import p2d
    cls = getattr(p2d, cls_name); p = cls(k); d = T.make_data(p, p.sample(0), jnp.float64)
    for f in forms:
        if f == "F5": continue
        F = make(f, cls); assert float(jnp.sum(F.weights(d) * F.term_vector(p.exact_jax, d))) < 1e-20, (cls_name, f, k)
    X = p.quadrature()["X"]; s = p.side_of(X)
    ph, g = p.to_physical(X, *p.exact_train(X, s), s); pe, ge = p.exact(X, s)
    assert np.allclose(ph, pe, atol=1e-12) and np.allclose(g, ge, atol=1e-10)
