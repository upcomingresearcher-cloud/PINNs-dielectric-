"""P2 analysis (Phase 7). Reads results/raw/runs_2d.csv and runs.csv; writes
results/processed/p2_*.csv/json, results/figures/p2_*.png, results/raw/diagnostics_loss_visibility_2d.csv,
results/raw/fem_runs_2d.csv (FEM computed here if absent)."""
import json, pathlib, sys, time
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import kendalltau, beta
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
FORMS = ["F1", "F2", "F3", "F4", "F5", "F6"]


def cp(k, n):
    return (0.0 if k == 0 else beta.ppf(.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(.975, k + 1, n - k))


def loss_visibility():
    import jax.numpy as jnp
    from src.problems.p2d import P2Cylinder
    from src.pinns import train2d as T
    from src.pinns.formulations2d import make
    rows = []
    for k in (1.0, 80.0, 2000.0, 0.0005):
        p = P2Cylinder(k); d = T.make_data(p, p.sample(0), jnp.float64)
        for f in FORMS:
            F = make(f, P2Cylinder); w = F.weights(d)
            for cand, fn in (("exact", p.exact_jax), ("wrong_uniform", p.wrong_uniform_jax)):
                L = float(jnp.sum(w * F.term_vector(fn, d)))
                note = "wrong candidate violates Dirichlet data -> not representable (hard BC); N/A" if (f in ("F4", "F5") and cand != "exact") else ""
                rows.append(dict(problem_id="P2", formulation=f, kappa=k, candidate=cand, loss_total=L, note=note))
    out = pd.DataFrame(rows); out.to_csv(ROOT / "results/raw/diagnostics_loss_visibility_2d.csv", index=False); return out


def fem_records():
    out = ROOT / "results/raw/fem_runs_2d.csv"
    if out.exists(): return pd.read_csv(out)
    from src.problems.p2d import P2Cylinder
    from src.fem import fem2d
    from src.evaluator.evaluator2d import evaluate
    from src.experiments import provenance as P
    rows = []
    for k in (1.0, 80.0, 2000.0, 0.0005):
        p = P2Cylinder(k); geo = lambda occ: [occ.addDisk(0, 0, 0, p.R, p.R)]
        for conf in (True, False):
            for h in (0.2, 0.1, 0.05, 0.025, 0.0125):
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter(); s = fem2d.solve(p, h, geometry=geo, conforming=conf); ts.append(time.perf_counter() - t0)
                m = evaluate(p, s)
                rows.append(dict(fem_id=f"P2-FEM-k{k:g}-h{h:g}-{'conf' if conf else 'nonconf'}", problem_id="P2", kappa=k, h=h,
                                 conforming=conf, order=2, n_elements=s.n_elements, dofs=s.dofs, mesh_plus_solve_s=float(np.median(ts)),
                                 solve_only_s=s.t_solve, **{c: m[c] for c in ("primary_field_error", "strict_field_error", "relL2_phi_global",
                                 "relL2_E_global", "D_jump_rel", "phi_jump_rel", "log_kappa_eff_ratio", "fail_primary")},
                                 git_commit=P.git_commit(), hardware=P.hardware()))
    df = pd.DataFrame(rows); df.to_csv(out, index=False); return df


def main():
    d = pd.read_csv(ROOT / "results/raw/runs_2d.csv"); d = d[(d.status == "complete") & (d.problem_id == "P2")]
    P = d[d.run_type == "pinn"]; reg = d[d.run_type == "regression"]
    rows = []
    for (f, k), g in P.groupby(["formulation", "kappa"]):
        n = len(g); nf = int(g.fail_primary.sum()); lo, hi = cp(nf, n)
        ex = [json.loads(e) for e in g.extra_json]
        rows.append(dict(formulation=f, kappa=k, n_seeds=n, n_failed=nf, cp95_lo=lo, cp95_hi=hi, config_fails=nf >= 3,
                         n_failed_strict=int(g.fail_strict.sum()), n_failed_potential=int(g.fail_potential.sum()),
                         median_primary=g.primary_field_error.median(), q25=g.primary_field_error.quantile(.25), q75=g.primary_field_error.quantile(.75),
                         median_phi=g.relL2_phi_global.median(), median_D_jump=g.D_jump_rel.median(), median_phi_jump=g.phi_jump_rel.median(),
                         median_log_keff=g.log_kappa_eff_ratio.median(), n_sec_fail_keff=int(g.sec_fail_effective_contrast.sum()),
                         median_wall_s=g.wall_clock_s.median()))
    t = pd.DataFrame(rows); t.to_csv(ROOT / "results/processed/p2_config_summary.csv", index=False)
    res = {}
    Q = P[P.kappa != 1].copy(); Q["R"] = Q.primary_field_error / Q.relL2_phi_global
    fl = Q[Q.fail_primary]
    res["H2_on_P2"] = dict(n=len(Q), median_R=float(Q.R.median()), frac_R_gt1=float((Q.R > 1).mean()), n_fail=len(fl),
                           concealed_fraction=float((fl.relL2_phi_global < .05).mean()) if len(fl) else None,
                           median_R_by_formulation=Q.groupby("formulation").R.median().to_dict())
    p1 = pd.read_csv(ROOT / "results/raw/runs.csv"); p1 = p1[(p1.run_type == "pinn") & (p1.status == "complete")]
    shared = [1.0, 80.0, 2000.0]
    c1 = p1[p1.kappa.isin(shared)].groupby("formulation").fail_primary.sum().reindex(FORMS)
    c2 = P[P.kappa.isin(shared)].groupby("formulation").fail_primary.sum().reindex(FORMS)
    tau = kendalltau(c1.values, c2.values)
    m1 = p1[p1.kappa.isin(shared)].groupby("formulation").primary_field_error.median().reindex(FORMS)
    m2 = P[P.kappa.isin(shared)].groupby("formulation").primary_field_error.median().reindex(FORMS)
    tau_m = kendalltau(m1.values, m2.values)
    res["H8_P1_vs_P2"] = dict(shared_kappas=shared, failed_P1=c1.to_dict(), failed_P2=c2.to_dict(), kendall_tau_failcount=float(tau.statistic),
                              p=float(tau.pvalue), kendall_tau_median_error=float(tau_m.statistic),
                              verdict_partial="tau >= 0.6" if tau.statistic >= 0.6 else "tau < 0.6",
                              note="H8 also requires H2 median R > 3 on each of P2-P5; final verdict after P5")
    res["regression"] = reg.groupby(["formulation", "kappa"]).primary_field_error.median().unstack().to_dict()
    res["loss_visibility"] = loss_visibility().to_dict("records")
    fem = fem_records()
    res["fem_conforming_kappa80"] = fem[(fem.kappa == 80) & fem.conforming][["h", "dofs", "primary_field_error", "relL2_phi_global", "mesh_plus_solve_s"]].to_dict("records")
    json.dump(res, open(ROOT / "results/processed/p2_results.json", "w"), indent=2, default=float)
    ks = sorted(t.kappa.unique())
    for val, fn, ttl, fmt, lg in (("n_failed", "p2_failure_map.png", "P2 cylinder: failed seeds of 5 (band field error > 5%)", lambda v: f"{int(v)}/5", False),
                                  ("median_primary", "p2_median_band_field_error.png", "P2 cylinder: median interface-band field error (log10 colour)", lambda v: f"{v:.1e}", True)):
        pv = t.pivot(index="formulation", columns="kappa", values=val).reindex(FORMS)[ks]
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        dat = np.log10(pv.values.astype(float)) if lg else pv.values.astype(float)
        im = ax.imshow(np.ma.masked_invalid(dat), cmap="RdYlGn_r", aspect="auto", vmin=None if lg else 0, vmax=None if lg else 5)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"{k:g}" for k in ks]); ax.set_yticks(range(6)); ax.set_yticklabels(FORMS)
        for i in range(6):
            for j in range(len(ks)):
                v = pv.values[i, j]; ax.text(j, i, "not run" if np.isnan(v) else fmt(v), ha="center", va="center", fontsize=7)
        ax.set_xlabel("κ = ε_inclusion / ε_matrix (κ<1: inverse contrast)"); ax.set_title(ttl + "\nF3–F5: reduced grid by pre-registered design", fontsize=9)
        fig.colorbar(im, ax=ax); fig.savefig(ROOT / "results/figures" / fn, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    for k, c in ((80.0, "tab:orange"), (2000.0, "tab:red"), (0.0005, "tab:blue")):
        for conf, ls in ((True, "-o"), (False, "--s")):
            g = fem[(fem.kappa == k) & (fem.conforming == conf)].sort_values("dofs")
            ax.loglog(g.dofs, g.primary_field_error, ls, color=c, ms=4, label=f"κ={k:g}, {'body-fitted' if conf else 'non-conforming'}")
    ax.axhline(.05, c="k", ls=":"); ax.set_xlabel("FEM degrees of freedom (P2 elements)"); ax.set_ylabel("interface-band field error")
    ax.set_title("P2 FEM convergence (body-fitted vs non-conforming)", fontsize=9); ax.legend(fontsize=6); ax.grid(alpha=.3, which="both")
    fig.savefig(ROOT / "results/figures/p2_fem_convergence.png", dpi=150); plt.close(fig)
    return t, res


if __name__ == "__main__":
    t, r = main()
    print(t.pivot(index="formulation", columns="kappa", values="n_failed").to_string()); print()
    print(t.pivot(index="formulation", columns="kappa", values="median_primary").map(lambda v: f"{v:.1e}").to_string()); print()
    for k in ("H2_on_P2", "H8_P1_vs_P2", "regression"): print(k, json.dumps(r[k], default=float)[:700])
    print("FEM k=80:", json.dumps(r["fem_conforming_kappa80"], default=float)[:500])
