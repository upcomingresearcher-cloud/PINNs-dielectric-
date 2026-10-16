"""Instrumented training for P1 (protocol Secs. 6, 15, 16).

Budget accounting: one gradient evaluation = one call of the compiled
value-and-gradient update (Adam). Counted in Python, not assumed.
Diagnostic per-term gradients (jacrev every `grad_log_interval` steps) are NOT part of
the training budget; they are counted separately (diag_grad_evals) and excluded from
wall_clock_s. Wall clock = sum of blocked update times after AOT compilation.

Phase 11 additions (protocol Sec. 10; the default path is unchanged and bit-identical):
  overrides["rebalance"]=1     Intervention A: every 10 steps lambda_i <- 0.9 lambda_i + 0.1 |grad L_pde|/|grad L_i|
                               for every non-PDE term (L2 norms). The per-term Jacobians are part of the method:
                               counted in notes (rebal_updates, rebal_term_grads) and included in wall_clock_s.
  overrides["optimiser"]       "adam" (default) | "lbfgs" | "adam_lbfgs" (Intervention E). L-BFGS: optax.lbfgs,
                               memory 50, zoom line search (max 20 steps). Every function+gradient evaluation is
                               counted (1 at the first iteration + num_linesearch_steps per iteration; verified
                               against a host-callback counter in tests). An iteration is started only if its
                               worst case fits in the remaining budget, so grad_evals <= step_budget always.
                               For L-BFGS rows, log "step" = cumulative function+gradient evaluations.
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


@functools.lru_cache(None)
def compiled_extra(fname, memory=50, max_ls=20):
    """Intervention A Jacobian and Intervention E L-BFGS step for formulation `fname`."""
    F = REGISTRY[fname]
    opt = optax.lbfgs(memory_size=memory, linesearch=optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_ls, initial_guess_strategy="one"))

    def terms(params, d):
        return F.term_vector(lambda x, s: F.rep(params, x, s, d), d)

    @jax.jit
    def term_jac(params, d):
        """(term values, flattened per-term gradients [n_terms, n_params])."""
        J = jax.jacrev(terms)(params, d)
        flat = jnp.concatenate([x.reshape(x.shape[0], -1) for x in jax.tree_util.tree_leaves(J)], axis=1)
        return terms(params, d), flat

    @jax.jit
    def lbfgs_step(params, state, d, w):
        f = lambda p: jnp.sum(w * terms(p, d))
        value, grad = optax.value_and_grad_from_state(f)(params, state=state)
        upd, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=f)
        nls = optax.tree_utils.tree_get(state, "num_linesearch_steps")
        return optax.apply_updates(params, upd), state, value, nls

    return opt, term_jac, lbfgs_step


def rebalance_weights(F, w_base, lam, vals, J):
    """Intervention A update (pure numpy; float64). lam: current weights (array, TERMS order).
    PDE terms keep their base weights; |grad L_pde| is the norm of the gradient of the weighted PDE sum."""
    J = np.asarray(J, np.float64); w_base = np.asarray(w_base, np.float64); lam = np.array(lam, np.float64)
    pde = np.array([t.startswith("pde") for t in F.TERMS])
    g_pde = np.linalg.norm((w_base[pde][:, None] * J[pde]).sum(axis=0))
    for i in np.where(~pde)[0]:
        n_i = np.linalg.norm(J[i])
        if np.isfinite(n_i) and n_i > 1e-30 and np.isfinite(g_pde):    # skip (keep lambda) if the term gradient vanishes
            lam[i] = 0.9 * lam[i] + 0.1 * g_pde / n_i
    return lam


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

    rebalance = bool(int(float(o.get("rebalance", 0))))
    optimiser = str(o.get("optimiser", "adam"))
    if optimiser not in ("adam", "lbfgs", "adam_lbfgs"): raise ValueError(optimiser)
    adam_steps = {"adam": steps, "lbfgs": 0, "adam_lbfgs": int(float(o.get("adam_steps", 2000)))}[optimiser]
    if rebalance and optimiser != "adam": raise ValueError("rebalancing is defined for Adam only")
    if adam_steps not in (0, steps):     # hybrid: the Adam phase gets its own complete cosine schedule
        F, opt, update, term_grads, phi_dphi, sched = compiled_fns(fname, lr_init, lr_final, adam_steps)
        opt_state = opt.init(params)

    key = (fname, lr_init, lr_final, steps, precision, npm, width, depth)
    if adam_steps not in (0, steps): key = key + (("adam_steps", adam_steps),)
    t0 = time.perf_counter()
    if key not in _AOT:
        _AOT[key] = (update.lower(params, opt_state, d, w).compile(), term_grads.lower(params, d).compile())
        cached = False
    else:
        cached = True
    upd_c, tg_c = _AOT[key]
    if rebalance or optimiser != "adam":
        lopt, term_jac, lbfgs_step = compiled_extra(fname)
        kx = key + ("extra",)
        if kx not in _AOT:
            lstate0 = lopt.init(params)
            _AOT[kx] = (term_jac.lower(params, d).compile(), lbfgs_step.lower(params, lstate0, d, w).compile())
        jac_c, lb_c = _AOT[kx]
    compile_s = time.perf_counter() - t0

    gradlog, traj = [], []
    def log_grads(step, params):
        vals, norms = (np.asarray(v, np.float64) for v in tg_c(params, d))
        wn = np.asarray(w, np.float64) * norms
        fin = np.isfinite(norms); mx = np.max(norms[fin]) if fin.any() else np.nan
        for i, t in enumerate(F.TERMS):
            gradlog.append(dict(step=step, term=t, loss_value=vals[i], grad_norm_unweighted=norms[i], grad_norm_weighted=wn[i],
                                weight=float(w[i]), learning_rate=float(sched(step)) if (adam_steps and step <= adam_steps) else np.nan,
                                precision=precision,
                                nonfinite=not np.isfinite(norms[i]) or not np.isfinite(vals[i]),
                                effectively_zero=bool(np.isfinite(norms[i]) and norms[i] < 1e-12 * mx)))
    def log_metrics(step, params):
        m = evaluate(problem, make_pred(phi_dphi, params, d, dtype))
        traj.append(dict(step=step, **{k: m[k] for k in ("primary_field_error", "strict_field_error", "relL2_phi_global",
                                                          "relL2_E_global", "D_jump_rel", "phi_jump_rel", "nonfinite")}))

    log_grads(0, params); log_metrics(0, params)
    grad_evals = 0; diag_evals = 1; train_time = 0.0; loss = jnp.nan
    w_base = np.asarray(w, np.float64); lam = w_base.copy(); n_rebal = 0
    for step in range(1, adam_steps + 1):
        ts = time.perf_counter()
        if rebalance and (step - 1) % 10 == 0:          # update before steps 1, 11, ..., 2991 (300 updates)
            vals_j, J = jac_c(params, d)
            lam = rebalance_weights(F, w_base, lam, vals_j, J); w = jnp.asarray(lam, dtype); n_rebal += 1
        params, opt_state, loss = upd_c(params, opt_state, d, w)
        jax.block_until_ready(loss); train_time += time.perf_counter() - ts
        grad_evals += 1
        if step % gli == 0 or step == steps: log_grads(step, params); diag_evals += 1
        if step % mli == 0 or step == steps: log_metrics(step, params)

    n_iter = 0; MAXLS = 20
    if optimiser != "adam":
        lstate = lopt.init(params); first = True
        while True:
            worst = MAXLS + (1 if first else 0)
            if grad_evals + worst > steps: break
            ts = time.perf_counter()
            params, lstate, loss, nls = lb_c(params, lstate, d, w)
            jax.block_until_ready(loss); train_time += time.perf_counter() - ts
            used = int(nls) + (1 if first else 0); first = False; n_iter += 1
            prev = grad_evals; grad_evals += used
            if grad_evals // gli > prev // gli: log_grads(grad_evals, params); diag_evals += 1
            if grad_evals // mli > prev // mli: log_metrics(grad_evals, params)
        if not traj or traj[-1]["step"] != grad_evals:
            log_grads(grad_evals, params); diag_evals += 1; log_metrics(grad_evals, params)

    vals = np.asarray(tg_c(params, d)[0], np.float64)
    final_loss = float(np.sum(np.asarray(w, np.float64) * vals))
    m = evaluate(problem, make_pred(phi_dphi, params, d, dtype))
    opt_label = {"adam": "adam", "lbfgs": "lbfgs(m=50,zoom20)", "adam_lbfgs": f"adam{adam_steps}+lbfgs(m=50,zoom20)"}[optimiser]
    lr_label = (f"cosine {lr_init:g}->{lr_final:g}/{adam_steps}" if adam_steps else "none (line search)") + \
               (" then lbfgs" if optimiser == "adam_lbfgs" else "")
    extra_notes = ""
    if rebalance:
        extra_notes += (f"; rebal_updates={n_rebal}; rebal_term_grads={n_rebal * len(F.TERMS)}"
                        f"; final_lambda={json.dumps(dict(zip(F.TERMS, map(float, lam))))}")
    if optimiser != "adam":
        extra_notes += f"; adam_steps={adam_steps}; lbfgs_iters={n_iter}; fn_evals={grad_evals}; grad_evals={grad_evals}"
    n_loss_pts = 2 * npm + 2 + (1 if F.uses_interface_point else 0)
    row = dict(run_id=rid, run_type=run_type, status=R.STATUS_COMPLETE, problem_id="P1", geometry="1d_two_slab_a0.5",
               formulation=fname, variant=variant, kappa=float(kappa), seed=int(seed),
               architecture=f"{depth}x{width} tanh x{F.n_nets} net(s), inputs={F.n_inputs}", hidden_layers=depth, width=width,
               activation="tanh", n_params=n_params(params), optimiser=opt_label, lr_schedule=lr_label,
               lr_init=lr_init, precision=precision, n_interior_per_material=npm, n_boundary=2,
               n_interface=1 if F.uses_interface_point else 0, n_extra_band=0, step_budget=steps, grad_evals=grad_evals,
               collocation_evals=grad_evals * n_loss_pts, final_loss=final_loss,
               final_loss_terms_json=json.dumps(dict(zip(F.TERMS, map(float, vals)))), **m,
               wall_clock_s=train_time, compile_s=compile_s,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
               config_id=R.config_id(dict(f=fname, k=kappa, s=seed, rt=run_type, v=variant, o=o, pv=cfg["protocol_version"])),
               protocol_version=cfg["protocol_version"], software_env=P.software_env(), hardware=P.hardware(),
               git_commit=P.git_commit(), timestamp=P.now(), notes=f"diag_grad_evals={diag_evals}; compile_cached={cached}" + extra_notes)
    if out_dir is not None:
        import csv, pathlib
        for sub, rows, cols in (("gradlogs", gradlog, R.GRADLOG_COLUMNS), ("trajectories", traj, list(traj[0]))):
            p = pathlib.Path(out_dir) / sub; p.mkdir(parents=True, exist_ok=True)
            with open(p / f"{rid}.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
    return row, params
