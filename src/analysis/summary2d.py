"""Generic 2D problem analysis (P3-P5). Usage: python src/analysis/summary2d.py P3
Reads results/raw/runs_2d.csv (+ runs.csv for H8); writes results/processed/{pid}_*.{csv,json},
results/figures/{pid}_*.png, results/raw/fem_runs_2d.csv rows (if missing for pid),
results/raw/diagnostics_loss_visibility_2d.csv rows (if missing for pid)."""
import json, pathlib, sys, time
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import kendalltau, beta
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
FORMS = ["F1", "F2", "F3", "F4", "F5", "F6"]
GEOM = {"P3": lambda occ: [occ.addRectangle(-1, -1, 0, 2, 1)], "P4": lambda occ: [occ.addRectangle(0, 0, 0, 1, 1)]}


def cp(k, n): return (0.0 if k == 0 else beta.ppf(.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(.975, k + 1, n - k))


def problem_class(pid):
    from src.problems import p2d
    return {getattr(p2d, n).problem_id: getattr(p2d, n) for n in dir(p2d)
            if isinstance(getattr(p2d, n), type) and issubclass(getattr(p2d, n), p2d.Problem2D) and getattr(p2d, n) is not p2d.Problem2D
            and not any(x in n for x in ("MatSub", "Thick", "Thin"))}[pid]


def fem_rows(pid, cls, kappas=(1.0, 80.0, 2000.0, 0.0005), hs=(0.2, 0.1, 0.05, 0.025, 0.0125)):
    """P5: graded meshes (h_far, h_in = t*frac) as in the reference; other problems: uniform h."""
    out = ROOT / "results/raw/fem_runs_2d.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old) and (old.problem_id == pid).any(): return old[old.problem_id == pid]
    from src.fem import fem2d
    from src.evaluator.evaluator2d import evaluate
    from src.experiments import provenance as P
    rows = []
    for k in kappas:
        p = cls(k)
        levels = [(0.4, 1.0), (0.2, 0.5), (0.1, 0.25), (0.05, 0.125)] if pid == "P5" else [(h, None) for h in hs]
        for h, fr in levels:
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                s = fem2d.solve(p, h, geometry=p.geometry(), grade=(p.t * fr, p.t, 0.3)) if fr else fem2d.solve(p, h, geometry=GEOM[pid], conforming=True)
                ts.append(time.perf_counter() - t0)
            m = evaluate(p, s)
            rows.append(dict(fem_id=f"{pid}-FEM-k{k:g}-h{h:g}-conf", problem_id=pid, kappa=k, h=h, conforming=True, order=2,
                             n_elements=s.n_elements, dofs=s.dofs, mesh_plus_solve_s=float(np.median(ts)), solve_only_s=s.t_solve,
                             **{c: m[c] for c in ("primary_field_error", "strict_field_error", "relL2_phi_global", "relL2_E_global",
                                                  "D_jump_rel", "phi_jump_rel", "log_kappa_eff_ratio", "fail_primary")},
                             git_commit=P.git_commit(), hardware=P.hardware()))
    new = pd.DataFrame(rows); pd.concat([old, new]).to_csv(out, index=False); return new


def loss_vis(pid, cls, kappas=(1.0, 80.0, 2000.0, 0.0005)):
    if pid == "P5": kappas = (1.0,)          # no closed form for kappa != 1
    import jax.numpy as jnp
    from src.pinns import train2d as T
    from src.pinns.formulations2d import make
    out = ROOT / "results/raw/diagnostics_loss_visibility_2d.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old) and (old.problem_id == pid).any(): return old[old.problem_id == pid]
    rows = []
    for k in kappas:
        p = cls(k); d = T.make_data(p, p.sample(0), jnp.float64)
        for f in FORMS:
            F = make(f, cls); w = F.weights(d)
            for cand, fn in (("exact", p.exact_jax), ("wrong_uniform", p.wrong_uniform_jax)):
                rows.append(dict(problem_id=pid, formulation=f, kappa=k, candidate=cand, loss_total=float(jnp.sum(w * F.term_vector(fn, d))),
                                 note="hard-BC formulation: wrong candidate violates Dirichlet data -> N/A" if (f in ("F4", "F5") and cand != "exact") else ""))
    new = pd.DataFrame(rows); pd.concat([old, new]).to_csv(out, index=False); return new


def main(pid):
    cls = problem_class(pid)
    d = pd.read_csv(ROOT / "results/raw/runs_2d.csv"); d = d[(d.status == "complete") & (d.problem_id == pid)]
    d = d.sort_values("timestamp").drop_duplicates("run_id", keep="first")   # concurrent-execution duplicates (identical results) kept in raw data
    P = d[(d.run_type == "pinn") & (d.variant == "base")]; reg = d[d.run_type == "regression"]; var = d[(d.run_type == "pinn") & (d.variant != "base")]
    rows = []
    for (f, k), g in P.groupby(["formulation", "kappa"]):
        n = len(g); nf = int(g.fail_primary.sum()); lo, hi = cp(nf, n); ex = pd.DataFrame([json.loads(e) for e in g.extra_json])
        r = dict(formulation=f, kappa=k, n_seeds=n, n_failed=nf, cp95_lo=lo, cp95_hi=hi, config_fails=nf >= 3,
                 n_failed_strict=int(g.fail_strict.sum()), n_failed_potential=int(g.fail_potential.sum()),
                 median_primary=g.primary_field_error.median(), q25=g.primary_field_error.quantile(.25), q75=g.primary_field_error.quantile(.75),
                 median_phi=g.relL2_phi_global.median(), median_D_jump=g.D_jump_rel.median(), median_phi_jump=g.phi_jump_rel.median(),
                 median_wall_s=g.wall_clock_s.median())
        for c in ex.columns: r[f"median_{c}"] = ex[c].median()
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(ROOT / f"results/processed/{pid}_config_summary.csv", index=False)
    res = {}
    Q = P[P.kappa != 1].copy(); Q["R"] = Q.primary_field_error / Q.relL2_phi_global; fl = Q[Q.fail_primary]
    res["H2"] = dict(n=len(Q), median_R=float(Q.R.median()), frac_R_gt1=float((Q.R > 1).mean()), n_fail=len(fl),
                     concealed_fraction=float((fl.relL2_phi_global < .05).mean()) if len(fl) else None,
                     median_R_by_formulation=Q.groupby("formulation").R.median().to_dict())
    p1 = pd.read_csv(ROOT / "results/raw/runs.csv"); p1 = p1[(p1.run_type == "pinn") & (p1.status == "complete")]
    sh = [1.0, 80.0, 2000.0]
    c1 = p1[p1.kappa.isin(sh)].groupby("formulation").fail_primary.sum().reindex(FORMS); c2 = P[P.kappa.isin(sh)].groupby("formulation").fail_primary.sum().reindex(FORMS)
    m1 = p1[p1.kappa.isin(sh)].groupby("formulation").primary_field_error.median().reindex(FORMS); m2 = P[P.kappa.isin(sh)].groupby("formulation").primary_field_error.median().reindex(FORMS)
    res["H8_vs_P1"] = dict(failed_P1=c1.to_dict(), failed_this=c2.to_dict(), tau_failcount=float(kendalltau(c1, c2).statistic),
                           tau_median_error=float(kendalltau(m1, m2).statistic))
    res["regression_median_primary"] = reg.groupby(["formulation", "kappa"]).primary_field_error.median().unstack().to_dict()
    if len(var):
        cmp_ = []
        for k, g in var.groupby("kappa"):
            b = P[(P.formulation == "F2") & (P.kappa == k)]
            cmp_.append(dict(kappa=k, variant=g.variant.iloc[0], variant_failed=int(g.fail_primary.sum()), variant_median=float(g.primary_field_error.median()),
                             base_F2_failed=int(b.fail_primary.sum()), base_F2_median=float(b.primary_field_error.median())))
        res["variant_vs_base"] = cmp_
    res["loss_visibility"] = loss_vis(pid, cls).to_dict("records")
    res["fem"] = fem_rows(pid, cls)[["kappa", "h", "dofs", "primary_field_error", "relL2_phi_global", "mesh_plus_solve_s"]].to_dict("records")
    json.dump(res, open(ROOT / f"results/processed/{pid}_results.json", "w"), indent=2, default=float)
    ks = sorted(t.kappa.unique())
    for val, fn, ttl, fmt, lg in (("n_failed", f"{pid}_failure_map.png", f"{pid}: failed seeds of 5 (band field error > 5%)", lambda v: f"{int(v)}/5", False),
                                  ("median_primary", f"{pid}_median_band_field_error.png", f"{pid}: median interface-band field error (log10 colour)", lambda v: f"{v:.1e}", True)):
        pv = t.pivot(index="formulation", columns="kappa", values=val).reindex(FORMS)[ks]
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        dat = np.log10(pv.values.astype(float)) if lg else pv.values.astype(float)
        im = ax.imshow(np.ma.masked_invalid(dat), cmap="RdYlGn_r", aspect="auto", vmin=None if lg else 0, vmax=None if lg else 5)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"{k:g}" for k in ks]); ax.set_yticks(range(6)); ax.set_yticklabels(FORMS)
        for i in range(6):
            for j in range(len(ks)):
                v = pv.values[i, j]; ax.text(j, i, "not run" if np.isnan(v) else fmt(v), ha="center", va="center", fontsize=7)
        ax.set_xlabel("κ = ε₂/ε₁ (κ<1: inverse contrast)"); ax.set_title(ttl + "\nF3–F5: reduced grid by pre-registered design", fontsize=9)
        fig.colorbar(im, ax=ax, label="log10 median primary error (–)" if lg else "failed seeds (of 5)"); fig.savefig(ROOT / "results/figures" / fn, dpi=150); plt.close(fig)
    return t, res


if __name__ == "__main__":
    pid = sys.argv[1]; t, r = main(pid)
    print(t.pivot(index="formulation", columns="kappa", values="n_failed").to_string()); print()
    print(t.pivot(index="formulation", columns="kappa", values="median_primary").map(lambda v: f"{v:.1e}").to_string()); print()
    ec = [c for c in t.columns if c.startswith("median_relL2_E_")]
    if ec: print(t[t.kappa.isin([1.0, 80.0, 2000.0])][["formulation", "kappa"] + ec].to_string(index=False)); print()
    for k in ("H2", "H8_vs_P1", "variant_vs_base", "regression_median_primary"):
        if k in r: print(k, json.dumps(r[k], default=float)[:600])
    print("FEM:", json.dumps([x for x in r["fem"] if x["kappa"] in (80.0, 2000.0)], default=float)[:700])


def h10(pid="P4"):
    d = pd.read_csv(ROOT / "results/raw/runs_2d.csv"); d = d[(d.status == "complete") & (d.problem_id == pid) & (d.run_type == "pinn") & (d.variant == "base")]
    d = d.sort_values("timestamp").drop_duplicates("run_id", keep="first")
    ex = pd.DataFrame([json.loads(e) for e in d.extra_json], index=d.index); d = pd.concat([d, ex], axis=1)
    cols = [f"signedE_{ray}_r{r}" for ray in ("bisector2", "bisector1") for r in ("0.02", "0.05")]
    tab = d.groupby(["formulation", "kappa"])[cols + ["coef_ratio_r0.05"]].median().reset_index()
    sing = tab[(tab.kappa > 1) & tab.formulation.isin(["F1", "F2"])]
    below = bool((sing[cols] < -0.10).all().all()); within_or_pos = bool(((sing[cols] > -0.10)).any().any())
    verdict = "supported" if below else ("refuted" if within_or_pos else "inconclusive")
    tab.to_csv(ROOT / f"results/processed/{pid}_H10_corner_metrics.csv", index=False)
    return verdict, tab


def p5_variants():
    d = pd.read_csv(ROOT / "results/raw/runs_2d.csv"); d = d[(d.status == "complete") & (d.problem_id == "P5") & (d.run_type == "pinn")]
    d = d[d.formulation.isin(["F2", "F6"])].copy()
    d["film_err"] = [json.loads(e).get("relL2_E_film", np.nan) for e in d.extra_json]
    d["setting"] = d.variant.map({"base": "t=0.02, 512 film pts", "t0.1": "t=0.1", "t0.005": "t=0.005", "nfilm128": "t=0.02, 128 film pts", "nfilm2048": "t=0.02, 2048 film pts"})
    g = d[d.kappa.isin([1.0, 80.0, 2000.0])].groupby(["formulation", "setting", "kappa"]).agg(
        n=("seed", "size"), failed=("fail_primary", "sum"), median_band=("primary_field_error", "median"),
        median_film=("film_err", "median"), median_phi=("relL2_phi_global", "median")).reset_index()
    g.to_csv(ROOT / "results/processed/P5_thickness_density.csv", index=False); return g
