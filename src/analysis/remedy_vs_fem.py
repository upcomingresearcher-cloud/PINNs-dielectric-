"""Phase 13: PINN remedies vs interface-conforming FEM on P2 and P5 (H7; design in change log v1.6.0, Phase 13).

(A) error at equal cost: PINN median wall-clock vs the conforming FEM mesh whose mesh_plus_solve_s is closest (log).
(B) cost to reach 5% / 1% primary band field error:
    PINN per run: first trajectory record with primary_field_error <= target; t = wall_clock_s * step / grad_evals
    (constant per-step cost assumed; compile time excluded). Not reached -> inf; configuration = median of 5 seeds.
    FEM: smallest conforming mesh meeting the target; P5 finest level (reference itself) excluded.
Outputs: results/processed/p13_{pinn_time_to_target,comparison}.csv, p13_h7_verdict.json, results/figures/p13_*.png
"""
import json, pathlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBLEMS = ["P2", "P5"]; KAPPAS = [0.0005, 1.0, 80.0, 2000.0]; TARGETS = [0.05, 0.01]
P5_REFERENCE_H = 0.05


def load_pinn():
    d = pd.read_csv(ROOT / "results/raw/runs_2d.csv", low_memory=False)
    d = d[(d.status == "complete") & d.problem_id.isin(PROBLEMS) & d.kappa.isin(KAPPAS) & d.seed.isin(range(5))]
    d = d[((d.run_type == "pinn") & (d.variant == "base")) | (d.run_type == "remedy")]
    d = d.sort_values("timestamp").drop_duplicates("run_id", keep="first").copy()
    d["arm"] = np.where(d.run_type == "remedy", d.formulation + "/" + d.variant, d.formulation + "/base")
    return d


def load_fem():
    f = pd.read_csv(ROOT / "results/raw/fem_runs_2d.csv", low_memory=False)
    f = f[f.problem_id.isin(PROBLEMS) & f.kappa.isin(KAPPAS)].copy()
    f = f[~((f.problem_id == "P5") & np.isclose(f.h, P5_REFERENCE_H))]          # reference level excluded
    return f


def run_time_to_target(row, target):
    tr = pd.read_csv(ROOT / "results/raw/trajectories" / f"{row.run_id}.csv")
    hit = tr[(tr.primary_field_error <= target) & np.isfinite(tr.primary_field_error)]
    if hit.empty: return np.inf, np.nan, np.inf
    i = hit.index[0]; step = float(tr.step.loc[i]); prev = float(tr.step.loc[i - 1]) if i > 0 else 0.0
    scale = float(row.wall_clock_s) / float(row.grad_evals)
    return scale * step, step, scale * prev        # upper bound (first hit) and lower bound (previous record)


def fem_time_to_target(F, p, k, target, conforming=True):
    g = F[(F.problem_id == p) & (F.kappa == k) & (F.conforming == conforming)].sort_values("dofs")
    hit = g[g.primary_field_error <= target]
    if hit.empty: return np.inf, np.nan, np.nan
    r = hit.iloc[0]; return float(r.mesh_plus_solve_s), float(r.h), float(r.solve_only_s)


def main():
    D, F = load_pinn(), load_fem()
    rows = []
    for r in D.itertuples():
        t5, s5, l5 = run_time_to_target(r, 0.05); t1, s1, l1 = run_time_to_target(r, 0.01)
        rows.append(dict(problem_id=r.problem_id, arm=r.arm, formulation=r.formulation, variant="base" if r.run_type == "pinn" else r.variant,
                         kappa=r.kappa, seed=r.seed, final_err=r.primary_field_error, wall_s=r.wall_clock_s, compile_s=r.compile_s,
                         grad_evals=r.grad_evals, t_5pct=t5, step_5pct=s5, t_5pct_lower=l5, t_1pct=t1, step_1pct=s1, t_1pct_lower=l1, hardware=r.hardware))
    T = pd.DataFrame(rows)
    C = []
    for (p, arm, k), g in T.groupby(["problem_id", "arm", "kappa"]):
        if len(g) != 5: continue
        fe = F[(F.problem_id == p) & (F.kappa == k) & F.conforming]
        if fe.empty: continue
        med_wall = float(g.wall_s.median()); med_err = float(g.final_err.median())
        idx = (np.log(fe.mesh_plus_solve_s) - np.log(med_wall)).abs().idxmin(); fm = fe.loc[idx]
        rec = dict(problem_id=p, arm=arm, kappa=k, n_seeds=5, pinn_median_err=med_err,
                   pinn_err_q25=float(g.final_err.quantile(.25)), pinn_err_q75=float(g.final_err.quantile(.75)),
                   pinn_median_wall_s=med_wall, pinn_median_compile_s=float(g.compile_s.median()), pinn_grad_evals=float(g.grad_evals.median()),
                   fem_equalcost_h=float(fm.h), fem_equalcost_dofs=int(fm.dofs), fem_equalcost_s=float(fm.mesh_plus_solve_s),
                   fem_equalcost_err=float(fm.primary_field_error), fem_max_s=float(fe.mesh_plus_solve_s.max()))
        for tgt, lab in ((0.05, "5pct"), (0.01, "1pct")):
            tp = g[f"t_{lab}"].values; n_reach = int(np.isfinite(tp).sum()); med_t = float(np.median(tp))
            ft, fh, fso = fem_time_to_target(F, p, k, tgt)
            ftn, _, _ = fem_time_to_target(F, p, k, tgt, conforming=False)
            rec.update({f"pinn_n_reach_{lab}": n_reach, f"pinn_median_t_{lab}": med_t, f"fem_t_{lab}": ft, f"fem_h_{lab}": fh,
                        f"fem_solve_only_t_{lab}": fso, f"fem_nonconforming_t_{lab}": ftn,
                        f"ratio_pinn_over_fem_{lab}": med_t / ft if np.isfinite(med_t) and np.isfinite(ft) else np.inf})
        rec["pinn_median_t_1pct_lower"] = float(np.median(g.t_1pct_lower.values))
        rec["pinn_median_step_1pct"] = float(np.median(g.step_1pct.fillna(np.inf).values))
        rec["clause1"] = (not np.isfinite(rec["pinn_median_t_1pct"])) or rec["pinn_median_t_1pct"] >= 10 * rec["fem_t_1pct"]
        # logging every 250 steps: the true first-hit time lies in [lower, upper]; clause 1 is resolution-safe only
        # if it also holds at the lower bound
        rec["clause1_resolution_safe"] = (not np.isfinite(rec["pinn_median_t_1pct"])) or rec["pinn_median_t_1pct_lower"] >= 10 * rec["fem_t_1pct"]
        rec["clause2"] = rec["fem_equalcost_err"] < rec["pinn_median_err"]
        rec["refutes"] = bool(np.isfinite(rec["pinn_median_t_1pct"]) and rec["pinn_median_t_1pct"] < 10 * rec["fem_t_1pct"])
        rec["degenerate_kappa1"] = k == 1.0
        C.append(rec)
    C = pd.DataFrame(C)
    def verdict(S):
        if S.refutes.any(): return "refuted"
        if (S.clause1 & S.clause2).all():
            return "supported" if S.clause1_resolution_safe.all() else "supported at logging resolution (clause 1 not resolvable in some configurations)"
        return "inconclusive"
    nd = C[~C.degenerate_kappa1]
    V = dict(verdict_all_configs=verdict(C), verdict_excluding_degenerate_kappa1=verdict(nd), n_configs=len(C),
             n_clause1=int(C.clause1.sum()), n_clause2=int(C.clause2.sum()),
             not_resolution_safe=C[~C.clause1_resolution_safe][["problem_id", "arm", "kappa", "pinn_median_t_1pct_lower",
                                                                "pinn_median_t_1pct", "fem_t_1pct"]].to_dict("records"),
             refuting=C[C.refutes][["problem_id", "arm", "kappa", "pinn_median_t_1pct", "fem_t_1pct"]].to_dict("records"),
             clause2_failures=C[~C.clause2][["problem_id", "arm", "kappa", "pinn_median_err", "fem_equalcost_err"]].to_dict("records"),
             pinn_reaching_1pct=C[np.isfinite(C.pinn_median_t_1pct)][["problem_id", "arm", "kappa", "pinn_median_t_1pct", "fem_t_1pct",
                                                                        "ratio_pinn_over_fem_1pct"]].to_dict("records"),
             hardware=sorted(set(T.hardware)) + sorted(set(F.hardware)),
             notes=["PINN time-to-target assumes constant per-step cost (t = wall x step / grad_evals); compile excluded",
                    "kappa = 1 is degenerate for FEM (exact to round-off at every mesh)",
                    "P5 FEM errors are relative to the finest FEM level (excluded); P5 PINN errors use the same reference"])
    out = ROOT / "results/processed"
    T.to_csv(out / "p13_pinn_time_to_target.csv", index=False); C.to_csv(out / "p13_comparison.csv", index=False)
    json.dump(V, open(out / "p13_h7_verdict.json", "w"), indent=1, default=str)
    fig_cost(C, F, "p13_error_vs_cost.png")
    return T, C, V


def fig_cost(C, F, fname):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for i, p in enumerate(PROBLEMS):
        for j, k in enumerate([1.0, 80.0, 2000.0]):
            ax = axes[i, j]; fe = F[(F.problem_id == p) & (F.kappa == k)]
            for conf, st in ((True, "k-o"), (False, "k--x")):
                g = fe[fe.conforming == conf].sort_values("dofs")
                if len(g): ax.plot(g.mesh_plus_solve_s, np.maximum(g.primary_field_error, 1e-16), st, lw=1,
                                   label="FEM conforming" if conf else "FEM non-conforming")
            g = C[(C.problem_id == p) & (C.kappa == k)]
            for r, mk in zip(g.itertuples(), "sD^v<>ph*"):
                ax.errorbar(r.pinn_median_wall_s, r.pinn_median_err, yerr=[[r.pinn_median_err - r.pinn_err_q25], [r.pinn_err_q75 - r.pinn_median_err]],
                            fmt=mk, ms=6, capsize=2, label=f"PINN {r.arm}")
            for t in TARGETS: ax.axhline(t, color="gray", ls=":", lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("wall-clock (s): FEM mesh+solve; PINN training")
            ax.set_ylabel("primary band field error (–)"); ax.set_title(f"{p}, κ = {k:g}" + (" (FEM exact: degenerate)" if k == 1 else ""), fontsize=9)
            if i == 0 and j == 0: ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Phase 13: error vs cost, PINN (median, IQR over 5 seeds) vs FEM refinement; dotted = 5% and 1%", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "results/figures" / fname, dpi=130); plt.close(fig)


if __name__ == "__main__":
    T, C, V = main()
    pd.set_option("display.width", 250)
    print(C[["problem_id", "arm", "kappa", "pinn_median_err", "pinn_median_wall_s", "fem_equalcost_err", "pinn_n_reach_1pct",
             "pinn_median_t_1pct", "fem_t_1pct", "ratio_pinn_over_fem_1pct", "clause1", "clause2"]].to_string(float_format=lambda v: f"{v:.3g}"))
    print(json.dumps({k: v for k, v in V.items() if k not in ("clause2_failures",)}, indent=1, default=str))
