"""Phase 11 (Interventions A-E) correctness tests.

1. The default (baseline) code path is unchanged: it reproduces stored primary rows bit-for-bit (CSV round-trip).
2. Intervention A weight update matches its definition; PDE weights stay fixed; counts are reported.
3. Intervention B formulations equal F1/F2 at kappa = 1 and rescale only the PDE terms otherwise.
4. Intervention E counts every function+gradient evaluation (cross-checked with a host callback counter)
   and never exceeds the budget.
"""
from conftest import reproduction_tolerance
import json, pathlib
import numpy as np, pandas as pd, pytest, yaml
import jax, jax.numpy as jnp, optax
from src.pinns import train as T
from src.pinns.formulations import REGISTRY
from src.problems.p1 import P1Problem

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/protocol_v1.yaml"))


def _data(k, dtype=jnp.float64, seed=0):
    p = P1Problem(k); x1, x2 = T.collocation(seed, p, 128); return p, T.make_data(p, x1, x2, dtype)


@pytest.mark.parametrize("f,k,s", [("F2", 80.0, 0), ("F3", 2000.0, 3)])
def test_baseline_path_reproduces_primary_rows(f, k, s):
    db = pd.read_csv(ROOT / "results/raw/runs.csv", low_memory=False)
    ref = db[db.run_id == f"P1-{f}-k{k:g}-s{s}-pinn-base"].iloc[0]
    tol = reproduction_tolerance(ref.hardware)
    row, _ = T.train_run(f, k, s, CFG)
    for c in ("primary_field_error", "final_loss", "relL2_phi_global", "D_jump_rel"):
        assert abs(float(row[c]) - float(ref[c])) <= tol * max(1.0, abs(float(ref[c]))), c
    assert row["optimiser"] == "adam" and row["grad_evals"] == 3000 and "rebal" not in row["notes"]


def test_rebalance_weights_definition():
    F = REGISTRY["F3"]                                  # TERMS: pde1, pde2, bc_left, bc_right, flux; base w .5 .5 1 1 1
    rng = np.random.default_rng(0); J = rng.normal(size=(5, 7)); w0 = np.array([0.5, 0.5, 1, 1, 1.0])
    lam = T.rebalance_weights(F, w0, w0, None, J)
    gp = np.linalg.norm(0.5 * J[0] + 0.5 * J[1])
    assert np.allclose(lam[:2], w0[:2])
    for i in (2, 3, 4):
        assert np.isclose(lam[i], 0.9 * 1.0 + 0.1 * gp / np.linalg.norm(J[i]))
    J[4] = 0.0                                           # vanishing term gradient: weight kept
    assert T.rebalance_weights(F, w0, lam, None, J)[4] == lam[4]


def test_rebalanced_run_counts_and_weights():
    row, _ = T.train_run("F2", 80.0, 0, CFG, overrides=dict(steps=50, rebalance=1))
    n = row["notes"]
    assert "rebal_updates=5;" in n and "rebal_term_grads=30;" in n and row["grad_evals"] == 50
    lam = json.loads(n.split("final_lambda=")[1])
    assert lam["pde1"] == 1.0 and lam["pde2"] == 1.0 and lam["cont"] != 1.0 and not row["nonfinite"]


def test_rebalance_with_unit_ratio_is_identity():
    """If every |grad L_i| equals |grad L_pde| the update leaves weights at 1 -> run identical to baseline."""
    F = REGISTRY["F2"]; J = np.zeros((6, 3)); J[0, 0] = 1.0; J[1, 0] = 1.0; J[2:, 0] = 2.0
    assert np.allclose(T.rebalance_weights(F, np.ones(6), np.ones(6), None, J), 1.0)


@pytest.mark.parametrize("pair", [("F1", "F1norm"), ("F2", "F2norm")])
def test_resnorm_terms(pair):
    base, norm = (REGISTRY[x] for x in pair)
    for k in (1.0, 80.0):
        p, d = _data(k)
        params = base.init(jax.random.PRNGKey(5), 32, 3, jnp.float64)
        tb = np.asarray(base.term_vector(lambda x, s: base.rep(params, x, s, d), d))
        tn = np.asarray(norm.term_vector(lambda x, s: norm.rep(params, x, s, d), d))
        scale = np.ones_like(tb); scale[1] = 1.0 / k**2          # pde2 divided by eps2^2 = kappa^2; pde1 eps1 = 1
        assert np.allclose(tn, tb * scale, rtol=1e-12, atol=0), (k, tb, tn)


@pytest.mark.parametrize("f", ["F1norm", "F2norm"])
def test_resnorm_training_finite(f):
    row, params = T.train_run(f, 2000.0, 0, CFG, overrides=dict(steps=50))
    assert not row["nonfinite"] and np.isfinite(row["final_loss"])
    assert all(np.all(np.isfinite(np.asarray(l))) for l in jax.tree_util.tree_leaves(params))


def test_lbfgs_eval_count_matches_host_counter():
    """Same optimiser construction as T.compiled_extra, with a host callback inside the loss."""
    F = REGISTRY["F2"]; p, d = _data(80.0, jnp.float32); w = F.weights(d)
    params = F.init(jax.random.PRNGKey(0), 32, 3, jnp.float32)
    cnt = {"n": 0}
    def bump(_): cnt["n"] += 1
    def f(pp):
        jax.debug.callback(bump, 0.0)
        return jnp.sum(w * F.term_vector(lambda x, s: F.rep(pp, x, s, d), d))
    opt = optax.lbfgs(memory_size=50, linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=20, initial_guess_strategy="one"))
    @jax.jit
    def step(pp, st):
        v, g = optax.value_and_grad_from_state(f)(pp, state=st)
        u, st = opt.update(g, st, pp, value=v, grad=g, value_fn=f)
        return optax.apply_updates(pp, u), st, v, optax.tree_utils.tree_get(st, "num_linesearch_steps")
    st = opt.init(params); counted = 0
    for it in range(25):
        params, st, v, nls = step(params, st); jax.block_until_ready(v); jax.effects_barrier()
        counted += int(nls) + (1 if it == 0 else 0)
    assert counted == cnt["n"], (counted, cnt["n"])


@pytest.mark.parametrize("opt,extra", [("lbfgs", {}), ("adam_lbfgs", dict(adam_steps=60))])
def test_lbfgs_budget_respected(opt, extra):
    row, _ = T.train_run("F3", 80.0, 0, CFG, overrides=dict(steps=100, optimiser=opt, **extra))
    assert row["grad_evals"] <= 100 and row["grad_evals"] >= 100 - 21
    assert ("lbfgs" in row["optimiser"]) and f"fn_evals={row['grad_evals']}" in row["notes"]
    if opt == "adam_lbfgs":
        assert "adam_steps=60" in row["notes"] and row["optimiser"].startswith("adam60")


def test_lbfgs_reaches_low_loss_on_easy_case():
    """Sanity: at kappa = 1, L-BFGS on F2 (float64) must reduce the loss far below its initial value."""
    row0, _ = T.train_run("F2", 1.0, 0, CFG, overrides=dict(steps=1, precision="float64"))
    row, _ = T.train_run("F2", 1.0, 0, CFG, overrides=dict(steps=300, optimiser="lbfgs", precision="float64"))
    assert row["final_loss"] < 1e-3 * row0["final_loss"]


def test_rebalanced_training_matches_independent_reference():
    """Independent implementation (separate jax.grad per term, no Jacobian, no AOT) gives the same lambdas."""
    import optax
    N = 30; F = REGISTRY["F2"]; p, d = _data(1.0, jnp.float32)
    params = F.init(jax.random.PRNGKey(0), 32, 3, jnp.float32)
    opt = optax.adam(optax.cosine_decay_schedule(1e-3, decay_steps=N, alpha=0.1), b1=0.9, b2=0.999, eps=1e-8); st = opt.init(params)
    term = lambda pp, i: F.terms(lambda x, s: F.rep(pp, x, s, d), d)[F.TERMS[i]]
    flat = lambda g: jnp.concatenate([l.ravel() for l in jax.tree_util.tree_leaves(g)])
    lam = np.ones(6)
    for step in range(1, N + 1):
        if (step - 1) % 10 == 0:
            gs = [np.asarray(flat(jax.grad(term)(params, i)), np.float64) for i in range(6)]
            gp = np.linalg.norm(gs[0] + gs[1])
            for i in range(2, 6): lam[i] = 0.9 * lam[i] + 0.1 * gp / np.linalg.norm(gs[i])
        w = jnp.asarray(lam, jnp.float32)
        g = jax.grad(lambda pp: sum(w[i] * term(pp, i) for i in range(6)))(params)
        u, st = opt.update(g, st, params); params = optax.apply_updates(params, u)
    row, _ = T.train_run("F2", 1.0, 0, CFG, overrides=dict(steps=N, rebalance=1))
    lt = json.loads(row["notes"].split("final_lambda=")[1])
    assert np.allclose(lam, [lt[t] for t in F.TERMS], rtol=1e-5)


@pytest.mark.parametrize("f", ["F1", "F2", "F3", "F4", "F5", "F6", "F4full", "F1norm", "F2norm"])
def test_float64_init_equals_float32_init(f):
    """Regression (bug 2026-10-16): precision must not change the initial weights."""
    F = REGISTRY[f]
    a = jax.tree_util.tree_leaves(F.init(jax.random.PRNGKey(7), 32, 3, jnp.float32))
    b = jax.tree_util.tree_leaves(F.init(jax.random.PRNGKey(7), 32, 3, jnp.float64))
    assert all(y.dtype == jnp.float64 for y in b)
    assert all(np.array_equal(np.asarray(x, np.float64), np.asarray(y)) for x, y in zip(a, b))


def test_float32_init_unchanged_by_fix():
    """The fix must leave float32 draws bit-identical (the primary data)."""
    net_key = jax.random.split(jax.random.PRNGKey(0), 1)[0]          # Formulation.init: one key per network
    k = jax.random.split(net_key)[1]                                  # init_mlp: first layer key
    old = jax.random.normal(k, (1, 32), jnp.float32) * jnp.sqrt(2.0 / (1 + 32)).astype(jnp.float32)
    new = REGISTRY["F1"].init(jax.random.PRNGKey(0), 32, 3, jnp.float32)["net0"][0][0]
    assert np.array_equal(np.asarray(old), np.asarray(new))
