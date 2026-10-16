"""Loss-visibility diagnostic (protocol Sec. 7.2) -> results/raw/diagnostics_loss_visibility.csv.
Deterministic (no training); collocation from primary seed 0; float64."""
import sys, pathlib, json
root = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
import numpy as np, yaml, jax.numpy as jnp
from src.pinns import train as T
from src.pinns.diagnostics import analytic_fns, loss_of, f1_smoothed_loss, smoothed_problem_solution
from src.evaluator.evaluator1d import evaluate
import jax
from src.problems.p1 import P1Problem
from src.experiments import records as R, provenance as P

cfg = yaml.safe_load(open(root / "configs/protocol_v1.yaml"))
out = root / "results/raw/diagnostics_loss_visibility.csv"
if out.exists(): out.unlink()                         # deterministic: regenerated in full
kappas = cfg["contrasts"]["primary"] + cfg["contrasts"]["extension_preregistered"]
for k in kappas:
    p = P1Problem(float(k)); x1, x2 = T.collocation(0, p, 128); d = T.make_data(p, x1, x2, jnp.float64)
    fns = analytic_fns(p)
    for f in ("F1", "F2", "F3", "F4", "F5", "F6", "F4full"):
        for cand, fn in fns.items():
            if f == "F4full":    # F4full's loss is the PDE residual only (all conditions in the ansatz)
                from src.pinns.formulations import pde_terms
                a_, b_ = pde_terms(fn, d); L, t = float(a_ + b_), {"pde1": float(a_), "pde2": float(b_)}
                if cand == "wrong_uniform" and k != 1:
                    t["note"] = "candidate violates flux continuity -> NOT representable by the F4full ansatz; visibility N/A"
            else:
                L, t = loss_of(f, fn, d)
            R.append_row(out, dict(diag_id=f"LV-{f}-k{k:g}-{cand}", problem_id="P1", formulation=f, variant="base", kappa=k,
                                   candidate=cand, loss_total=L, loss_terms_json=json.dumps(t), smoothing_width="", n_extra_band=0,
                                   git_commit=P.git_commit(), timestamp=P.now()), R.LOSS_VIS_COLUMNS)
    hb = p.band_halfwidth
    xb = np.concatenate([np.linspace(p.a - hb, p.a, 17)[:-1] + hb / 32, np.linspace(p.a, p.a + hb, 17)[:-1] + hb / 32])
    for wdt in (0.01, 0.002):
        fs, Ds = smoothed_problem_solution(p, wdt)
        cands = dict(fns); cands["exact_smoothed_problem"] = fs
        for cand, fn in cands.items():
            L, t = f1_smoothed_loss(fn, d, wdt, jnp.asarray(xb))
            if cand == "exact_smoothed_problem":   # modelling error of smoothing, scored against the SHARP problem
                g = jax.vmap(jax.grad(lambda x: fn(x, 1)))
                pred = lambda x, side: (np.asarray(jax.vmap(lambda x: fn(x, 1))(jnp.asarray(x))), np.asarray(g(jnp.asarray(x))))
                me = evaluate(p, pred)
                t.update(D_s=Ds, smoothing_model_error_primary=me["primary_field_error"], smoothing_model_error_phi=me["relL2_phi_global"])
            R.append_row(out, dict(diag_id=f"LV-F1smooth-w{wdt:g}-k{k:g}-{cand}", problem_id="P1", formulation="F1",
                                   variant=f"smoothed_eps_w{wdt:g}+16band", kappa=k, candidate=cand, loss_total=L,
                                   loss_terms_json=json.dumps(t), smoothing_width=wdt, n_extra_band=32,
                                   git_commit=P.git_commit(), timestamp=P.now()), R.LOSS_VIS_COLUMNS)
import pandas as pd
df = pd.read_csv(out)
pv = df.pivot_table(index=["formulation", "variant"], columns=["kappa", "candidate"], values="loss_total")
print(pv.map(lambda v: f"{v:.2e}").to_string())
