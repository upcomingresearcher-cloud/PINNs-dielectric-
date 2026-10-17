"""Instrumented 2D training (same accounting as src/pinns/train.py; change log v1.2.0)."""
import json, resource, time, functools, csv, pathlib
import numpy as np
import jax, jax.numpy as jnp, optax
jax.config.update("jax_enable_x64", True)
from src.pinns.formulations2d import make
from src.pinns.nets import n_params
from src.evaluator.evaluator2d import evaluate, METRIC2D_COLUMNS
from src.experiments import records as R, provenance as P

RUNS2D_COLUMNS = (R.RUN_ID_COLUMNS + R.RUN_SETTING_COLUMNS + R.RUN_OUTCOME_COLUMNS + METRIC2D_COLUMNS
                  + R.RUN_COST_COLUMNS + R.RUN_PROV_COLUMNS)


def make_data(prob, S, dtype):
    """Convert a 2D collocation sample into the device arrays used by the loss functions."""
    c = lambda v: jnp.asarray(v, dtype)
    sb = S["sideB"]; jp = prob.jax_params()
    XB1, XB2 = S["XB"][sb == 1], S["XB"][sb == 2]
    y1 = prob.exact_train(S["X1"], np.ones(len(S["X1"]), int))[0]; y2 = prob.exact_train(S["X2"], np.full(len(S["X2"]), 2))[0]
    return dict(X1=c(S["X1"]), X2=c(S["X2"]), XB1=c(XB1.reshape(-1, 2)), HB1=c(prob.h_train_np(XB1) if len(XB1) else np.zeros(0)),
                XB2=c(XB2.reshape(-1, 2)), HB2=c(prob.h_train_np(XB2) if len(XB2) else np.zeros(0)),
                XI=c(S["XI"]), NI=c(S["NI"]), QI=c(S["QI"]), PI=c(S.get("PI", np.zeros(len(S["XI"])))), Y1=c(y1), Y2=c(y2),
                eps1=c(prob.eps1), eps2=c(prob.eps2), Vc=c(prob.Vc), Lc=c(prob.Lc), A1=c(S["A1"]), A2=c(S["A2"]),
                LG=c(S["LG"]), pp=c(jp["pp"]), hp=c(jp["hp"]))


@functools.lru_cache(None)
def compiled_fns(fname, prob_cls, lr_init, lr_final, steps):
    """Build the jitted update, per-term gradient and prediction functions for a 2D formulation."""
    F = make(fname, prob_cls)
    sched = optax.cosine_decay_schedule(lr_init, decay_steps=steps, alpha=lr_final / lr_init)
    opt = optax.adam(sched, b1=0.9, b2=0.999, eps=1e-8)
    terms = lambda p, d: F.term_vector(lambda z, s: F.rep(p, z, s, d), d)
    total = lambda p, d, w: jnp.sum(w * terms(p, d))

    @jax.jit
    def update(p, st, d, w):
        """One Adam step; returns updated parameters, optimiser state and the loss."""
        l, g = jax.value_and_grad(total)(p, d, w); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l

    @jax.jit
    def term_grads(p, d):
        """Per-term loss values and per-term gradient norms."""
        v = terms(p, d); J = jax.jacrev(terms)(p, d)
        sq = sum(jnp.sum(x.reshape(x.shape[0], -1)**2, axis=1) for x in jax.tree_util.tree_leaves(J))
        return v, jnp.sqrt(sq)

    @functools.partial(jax.jit, static_argnums=2)
    def phi_grad(p, Z, side, d):
        """Potential and gradient at points Z on one side of the interface."""
        f = lambda z: F.rep(p, z, side, d); return jax.vmap(f)(Z), jax.vmap(jax.grad(f))(Z)
    return F, opt, update, term_grads, phi_grad, sched


@functools.lru_cache(None)
def compiled_lbfgs(fname, prob_cls, memory=50, max_ls=20):
    """Phase 13 remedy: same construction as src/pinns/train.py compiled_extra (Phase 11 Intervention E)."""
    F = make(fname, prob_cls)
    opt = optax.lbfgs(memory_size=memory, linesearch=optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_ls, initial_guess_strategy="one"))
    terms = lambda p, d: F.term_vector(lambda z, s: F.rep(p, z, s, d), d)

    @jax.jit
    def step(p, st, d, w):
        """One L-BFGS iteration with zoom line search; also returns the number of line-search evaluations."""
        f = lambda q: jnp.sum(w * terms(q, d))
        v, g = optax.value_and_grad_from_state(f)(p, state=st)
        u, st = opt.update(g, st, p, value=v, grad=g, value_fn=f)
        return optax.apply_updates(p, u), st, v, optax.tree_utils.tree_get(st, "num_linesearch_steps")
    return opt, step


_AOT = {}


def make_pred(phi_grad, p, d, dtype, prob=None):
    """Wrap trained 2D parameters as a prediction function (potential, gradient) in physical variables."""
    def pred(X, side):
        """Prediction at points X on the given sides, evaluated material by material."""
        X = np.asarray(X, np.float64); side = np.asarray(side); phi = np.empty(len(X)); g = np.empty((len(X), 2))
        for s in (1, 2):
            m = side == s
            if m.any():
                a, b = phi_grad(p, jnp.asarray(X[m], dtype), s, d); phi[m] = np.asarray(a, np.float64); g[m] = np.asarray(b, np.float64)
        return prob.to_physical(X, phi, g, side) if prob is not None else (phi, g)
    return pred


def train_run(fname, prob_cls, kappa, seed, cfg2d, run_type="pinn", variant="base", out_dir=None, overrides=None):
    """Train one 2D configuration, evaluate it and return the database row and trained parameters."""
    o = dict(overrides or {})
    steps = int(o.get("steps", cfg2d["steps"])); lr_init = float(o.get("lr_init", cfg2d["lr_init"])); lr_final = float(cfg2d["lr_final"])
    width = int(o.get("width", cfg2d["width"])); depth = int(o.get("depth", cfg2d["depth"]))
    precision = o.get("precision", cfg2d["precision"]); dtype = jnp.float32 if precision == "float32" else jnp.float64
    npm = int(o.get("n_interior", cfg2d["n_interior"])); ni = int(cfg2d["n_interface"]); ne = int(cfg2d["n_edge"])
    prob = prob_cls(kappa)
    S = prob.sample(seed, npm, ni, ne, n_film=int(o["n_film"])) if "n_film" in o else prob.sample(seed, npm, ni, ne)
    d = make_data(prob, S, dtype)
    F, opt, update, term_grads, phi_grad, sched = compiled_fns(fname, prob_cls, lr_init, lr_final, steps)
    w = F.weights(d); params = F.init(jax.random.PRNGKey(seed), width, depth, dtype); st = opt.init(params)
    rid = R.run_id(prob.problem_id, fname, float(kappa), seed, run_type, variant)
    key = (fname, prob_cls.__name__, lr_init, lr_final, steps, precision, width, depth,
           tuple(sorted((k_, tuple(v_.shape)) for k_, v_ in d.items())))   # shapes vary by seed when the boundary crosses the interface
    t0 = time.perf_counter(); cached = key in _AOT
    if not cached: _AOT[key] = (update.lower(params, st, d, w).compile(), term_grads.lower(params, d).compile())
    upd, tg = _AOT[key]; compile_s = time.perf_counter() - t0
    quad = prob.quadrature(); gl_, tr_ = [], []
    def log_g(step, p):
        """Record per-term loss values and gradient norms at a training step."""
        v, nrm = (np.asarray(x, np.float64) for x in tg(p, d)); wn = np.asarray(w, np.float64) * nrm
        fin = np.isfinite(nrm); mx = nrm[fin].max() if fin.any() else np.nan
        for i, t in enumerate(F.TERMS):
            gl_.append(dict(step=step, term=t, loss_value=v[i], grad_norm_unweighted=nrm[i], grad_norm_weighted=wn[i], weight=float(w[i]),
                            learning_rate=float(sched(step)) if optimiser == "adam" else float("nan"), precision=precision, nonfinite=not (np.isfinite(nrm[i]) and np.isfinite(v[i])),
                            effectively_zero=bool(np.isfinite(nrm[i]) and nrm[i] < 1e-12 * mx)))
    def log_m(step, p):
        """Evaluate all metrics at a training step and append them to the trajectory."""
        m = evaluate(prob, make_pred(phi_grad, p, d, dtype, prob), quad)
        tr_.append(dict(step=step, **{k: m[k] for k in ("primary_field_error", "strict_field_error", "relL2_phi_global", "relL2_E_global",
                                                         "D_jump_rel", "phi_jump_rel", "nonfinite")}))
    gli, mli = int(cfg2d["grad_log_interval"]), int(cfg2d["metric_log_interval"])
    optimiser = str(o.get("optimiser", "adam"))
    if optimiser not in ("adam", "lbfgs"): raise ValueError(optimiser)
    log_g(0, params); log_m(0, params); ge = 0; tt = 0.0; n_iter = 0
    if optimiser == "lbfgs":                      # <= steps function+gradient evaluations; "step" in logs = evaluations
        lopt, lstep = compiled_lbfgs(fname, prob_cls); lst = lopt.init(params); first = True
        kx = key + ("lbfgs",)
        if kx not in _AOT: _AOT[kx] = lstep.lower(params, lst, d, w).compile()
        lb = _AOT[kx]
        while ge + 20 + (1 if first else 0) <= steps:
            ts = time.perf_counter(); params, lst, loss, nls = lb(params, lst, d, w); jax.block_until_ready(loss)
            tt += time.perf_counter() - ts; used = int(nls) + (1 if first else 0); first = False; n_iter += 1
            prev = ge; ge += used
            if ge // gli > prev // gli: log_g(ge, params)
            if ge // mli > prev // mli: log_m(ge, params)
        if tr_[-1]["step"] != ge: log_g(ge, params); log_m(ge, params)
    for step in range(1, (steps if optimiser == "adam" else 0) + 1):
        ts = time.perf_counter(); params, st, loss = upd(params, st, d, w); jax.block_until_ready(loss); tt += time.perf_counter() - ts; ge += 1
        if step % gli == 0 or step == steps: log_g(step, params)
        if step % mli == 0 or step == steps: log_m(step, params)
    vals = np.asarray(tg(params, d)[0], np.float64)
    m = evaluate(prob, make_pred(phi_grad, params, d, dtype, prob), quad)
    npts = 2 * npm + 4 * ne + ni
    row = dict(run_id=rid, run_type=run_type, status=R.STATUS_COMPLETE, problem_id=prob.problem_id, geometry=prob.geometry_label,
               formulation=fname, variant=variant, kappa=float(kappa), seed=int(seed),
               architecture=f"{depth}x{width} tanh x{F.n_nets} net(s), inputs={F.n_inputs}", hidden_layers=depth, width=width, activation="tanh",
               n_params=n_params(params), optimiser="adam" if optimiser == "adam" else "lbfgs(m=50,zoom20)",
               lr_schedule=f"cosine {lr_init:g}->{lr_final:g}/{steps}" if optimiser == "adam" else "none (line search)", lr_init=lr_init,
               precision=precision, n_interior_per_material=npm, n_boundary=4 * ne, n_interface=ni, n_extra_band=0, step_budget=steps,
               grad_evals=ge, collocation_evals=ge * npts, final_loss=float(np.sum(np.asarray(w, np.float64) * vals)),
               final_loss_terms_json=json.dumps(dict(zip(F.TERMS, map(float, vals)))), **m, wall_clock_s=tt, compile_s=compile_s,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
               config_id=R.config_id(dict(f=fname, p=prob.problem_id, k=kappa, s=seed, rt=run_type, v=variant, o=o, c=cfg2d)),
               protocol_version=cfg2d["protocol_version"], software_env=P.software_env(), hardware=P.hardware(), git_commit=P.git_commit(),
               timestamp=P.now(), notes=f"compile_cached={cached}" + (f"; lbfgs_iters={n_iter}; fn_evals={ge}" if optimiser == "lbfgs" else ""))
    if out_dir is not None:
        for sub, rows, cols in (("gradlogs", gl_, R.GRADLOG_COLUMNS), ("trajectories", tr_, list(tr_[0]))):
            pth = pathlib.Path(out_dir) / sub; pth.mkdir(parents=True, exist_ok=True)
            with open(pth / f"{rid}.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
    return row, params
