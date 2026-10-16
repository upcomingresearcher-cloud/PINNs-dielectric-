"""Instrumented training for P1 (protocol Secs. 6, 15, 16).

Budget accounting: one gradient evaluation = one call of the compiled
value-and-gradient update (Adam). Counted in Python, not assumed.
Diagnostic per-term gradients (jacrev every `grad_log_interval` steps) are NOT part of
the training budget; they are counted separately (diag_grad_evals) and excluded from
wall_clock_s. Wall clock = sum of blocked update times after AOT compilation.
"""
import json, resource, time, functools
import numpy as np
import jax, jax.numpy as jnp, optax

jax.config.update("jax_enable_x64", True)   # dtype is then controlled explicitly per run

from src.pinns.formulations import REGISTRY
from src.pinns.nets import n_params
from src.problems.p1 import P1Problem
from src.evaluator.evaluator1d import evaluate
from src.experiments import records as R, provenance as P


def collocation(seed, problem, n_per_material):
    """Stratified jitter: one uniform draw per equal stratum, per material. Seed-only dependence."""
    rng = np.random.default_rng([seed, 7919])
    def strat(lo, hi, n):
        edges = np.linspace(lo, hi, n + 1); return edges[:-1] + rng.uniform(size=n) * np.diff(edges)
    return strat(0.0, problem.a, n_per_material), strat(problem.a, problem.L, n_per_material)


def make_data(problem, x1, x2, dtype):
    """y1, y2: exact potential at the interior points; used ONLY by regression baselines."""
    c = lambda v: jnp.asarray(v, dtype)
    y1 = problem.exact(x1, np.ones_like(x1, int))[0]; y2 = problem.exact(x2, np.full_like(x2, 2, int))[0]
    return dict(x1=c(x1), x2=c(x2), y1=c(y1), y2=c(y2), eps1=c(problem.eps1), eps2=c(problem.eps2),
                a=c(problem.a), V=c(problem.V), L=c(problem.L))


@functools.lru_cache(None)
def compiled_fns(fname, lr_init, lr_final, steps):
    F = REGISTRY[fname]
    sched = optax.cosine_decay_schedule(lr_init, decay_steps=steps, alpha=lr_final / lr_init)
    opt = optax.adam(sched, b1=0.9, b2=0.999, eps=1e-8)

    def terms(params, d):
        return F.term_vector(lambda x, s: F.rep(params, x, s, d), d)

    def total(params, d, w):
        return jnp.sum(w * terms(params, d))

    @jax.jit
    def update(params, opt_state, d, w):
        loss, g = jax.value_and_grad(total)(params, d, w)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    @jax.jit
    def term_grads(params, d):
        vals = terms(params, d)
        J = jax.jacrev(terms)(params, d)              # pytree with leading dim = n_terms
        sq = sum(jnp.sum(x.reshape(x.shape[0], -1)**2, axis=1) for x in jax.tree_util.tree_leaves(J))
        return vals, jnp.sqrt(sq)

    @functools.partial(jax.jit, static_argnums=2)
    def phi_dphi(params, xs, side, d):
        f = lambda x: F.rep(params, x, side, d)
        return jax.vmap(f)(xs), jax.vmap(jax.grad(f))(xs)

    return F, opt, update, term_grads, phi_dphi, sched


_AOT = {}   # compiled executables keyed by formulation/optimiser/shape/dtype (independent of kappa and seed)


def make_pred(phi_dphi, params, d, dtype):
    def pred(x, side):
        x = np.asarray(x, np.float64); side = np.asarray(side)
        phi = np.empty_like(x); dphi = np.empty_like(x)
        for s in (1, 2):
            m = side == s
            if m.any():
                a, b = phi_dphi(params, jnp.asarray(x[m], dtype), s, d)
                phi[m] = np.asarray(a, np.float64); dphi[m] = np.asarray(b, np.float64)
        return phi, dphi
    return pred


def train_run(fname, kappa, seed, cfg, run_type="pinn", variant="base", out_dir=None, overrides=None):
    """Train one configuration; return the database row (and write logs if out_dir)."""
    o = dict(overrides or {})
    tr, ar, col = cfg["training"], cfg["architecture"], cfg["collocation"]
    lr_init = float(o.get("lr_init", tr["learning_rate"]["init"])); lr_final = float(tr["learning_rate"]["final"])
    steps = int(o.get("steps", tr["gradient_evaluations"])); width = int(o.get("width", ar["width"]))
    depth = int(o.get("depth", ar["hidden_layers"])); npm = int(o.get("n_interior", col["interior_per_material"]))
    precision = o.get("precision", tr["precision"]); dtype = jnp.float32 if precision == "float32" else jnp.float64
    gli = int(cfg["gradient_logging"]["interval_steps"]); mli = int(cfg["evaluation"]["metric_log_interval_steps"])

    problem = P1Problem(float(kappa))
    F, opt, update, term_grads, phi_dphi, sched = compiled_fns(fname, lr_init, lr_final, steps)
    x1, x2 = collocation(seed, problem, npm)
    d = make_data(problem, x1, x2, dtype); w = F.weights(d)
    params = F.init(jax.random.PRNGKey(seed), width, depth, dtype)
    opt_state = opt.init(params)
    rid = R.run_id("P1", fname, float(kappa), seed, run_type, variant)

    key = (fname, lr_init, lr_final, steps, precision, npm, width, depth)
    t0 = time.perf_counter()
    if key not in _AOT:
        _AOT[key] = (update.lower(params, opt_state, d, w).compile(), term_grads.lower(params, d).compile())
        cached = False
    else:
        cached = True
    upd_c, tg_c = _AOT[key]
    compile_s = time.perf_counter() - t0

    gradlog, traj = [], []
    def log_grads(step, params):
        vals, norms = (np.asarray(v, np.float64) for v in tg_c(params, d))
        wn = np.asarray(w, np.float64) * norms
        fin = np.isfinite(norms); mx = np.max(norms[fin]) if fin.any() else np.nan
        for i, t in enumerate(F.TERMS):
            gradlog.append(dict(step=step, term=t, loss_value=vals[i], grad_norm_unweighted=norms[i], grad_norm_weighted=wn[i],
                                weight=float(w[i]), learning_rate=float(sched(step)), precision=precision,
                                nonfinite=not np.isfinite(norms[i]) or not np.isfinite(vals[i]),
                                effectively_zero=bool(np.isfinite(norms[i]) and norms[i] < 1e-12 * mx)))
    def log_metrics(step, params):
        m = evaluate(problem, make_pred(phi_dphi, params, d, dtype))
        traj.append(dict(step=step, **{k: m[k] for k in ("primary_field_error", "strict_field_error", "relL2_phi_global",
                                                          "relL2_E_global", "D_jump_rel", "phi_jump_rel", "nonfinite")}))

    log_grads(0, params); log_metrics(0, params)
    grad_evals = 0; diag_evals = 1; train_time = 0.0; loss = jnp.nan
    for step in range(1, steps + 1):
        ts = time.perf_counter()
        params, opt_state, loss = upd_c(params, opt_state, d, w)
        jax.block_until_ready(loss); train_time += time.perf_counter() - ts
        grad_evals += 1
        if step % gli == 0 or step == steps: log_grads(step, params); diag_evals += 1
        if step % mli == 0 or step == steps: log_metrics(step, params)

    vals = np.asarray(tg_c(params, d)[0], np.float64)
    final_loss = float(np.sum(np.asarray(w, np.float64) * vals))
    m = evaluate(problem, make_pred(phi_dphi, params, d, dtype))
    n_loss_pts = 2 * npm + 2 + (1 if F.uses_interface_point else 0)
    row = dict(run_id=rid, run_type=run_type, status=R.STATUS_COMPLETE, problem_id="P1", geometry="1d_two_slab_a0.5",
               formulation=fname, variant=variant, kappa=float(kappa), seed=int(seed),
               architecture=f"{depth}x{width} tanh x{F.n_nets} net(s), inputs={F.n_inputs}", hidden_layers=depth, width=width,
               activation="tanh", n_params=n_params(params), optimiser="adam", lr_schedule=f"cosine {lr_init:g}->{lr_final:g}/{steps}",
               lr_init=lr_init, precision=precision, n_interior_per_material=npm, n_boundary=2,
               n_interface=1 if F.uses_interface_point else 0, n_extra_band=0, step_budget=steps, grad_evals=grad_evals,
               collocation_evals=grad_evals * n_loss_pts, final_loss=final_loss,
               final_loss_terms_json=json.dumps(dict(zip(F.TERMS, map(float, vals)))), **m,
               wall_clock_s=train_time, compile_s=compile_s,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
               config_id=R.config_id(dict(f=fname, k=kappa, s=seed, rt=run_type, v=variant, o=o, pv=cfg["protocol_version"])),
               protocol_version=cfg["protocol_version"], software_env=P.software_env(), hardware=P.hardware(),
               git_commit=P.git_commit(), timestamp=P.now(), notes=f"diag_grad_evals={diag_evals}; compile_cached={cached}")
    if out_dir is not None:
        import csv, pathlib
        for sub, rows, cols in (("gradlogs", gradlog, R.GRADLOG_COLUMNS), ("trajectories", traj, list(traj[0]))):
            p = pathlib.Path(out_dir) / sub; p.mkdir(parents=True, exist_ok=True)
            with open(p / f"{rid}.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
    return row, params
