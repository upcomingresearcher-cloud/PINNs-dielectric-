"""Phase 6: tests of H1, H2, H3, H4, H11 on P1 data with the pre-registered wording (protocol Sec. 1).
Reads results/raw only; writes results/processed/p1_hypothesis_tests.json and figures."""
import json, pathlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(12345); B = 10000
FORMS = ["F1", "F2", "F3", "F4", "F5", "F6"]


def boot(fn, *arrays):
    n = len(arrays[0]); vals = []
    for _ in range(B):
        i = RNG.integers(0, n, n); vals.append(fn(*[a[i] for a in arrays]))
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


def early_ratio(run_id, weighted=True, max_step=300):
    g = pd.read_csv(ROOT / f"results/raw/gradlogs/{run_id}.csv")
    g = g[g.step <= max_step]; col = "grad_norm_weighted" if weighted else "grad_norm_unweighted"
    r = []
    for _, s in g.groupby("step"):
        v = s[col].values; v = v[np.isfinite(v)]; v = v[v >= 1e-12 * v.max()] if len(v) else v
        if len(v) >= 2 and v.min() > 0: r.append(np.log10(v.max() / v.min()))
    return float(np.median(r)) if r else np.nan


def main():
    d = pd.read_csv(ROOT / "results/raw/runs.csv"); d = d[d.status == "complete"]
    P = d[(d.run_type == "pinn") & (d.variant == "base")].copy()
    reg = d[d.run_type == "regression"]
    lv = pd.read_csv(ROOT / "results/raw/diagnostics_loss_visibility.csv"); lv = lv[lv.variant == "base"]
    res = {}

    # ---------------- H1
    f1 = P[(P.formulation == "F1") & (P.kappa >= 5)].groupby("kappa").fail_primary.sum()
    c1 = bool((f1 >= 4).all())
    g2 = P[P.formulation == "F2"]; x, y = np.log10(g2.kappa.values), g2.primary_field_error.values
    rho = spearmanr(x, y).statistic; ci = boot(lambda a, b: spearmanr(a, b).statistic, x, y)
    c2 = ci[0] > 0
    tot = P[P.kappa > 1].groupby("formulation").fail_primary.sum().to_dict()
    c3 = all(tot[f] < tot["F1"] for f in ("F3", "F4", "F5", "F6"))
    ref = bool((P[(P.formulation == "F1") & (P.kappa >= 20)].groupby("kappa").fail_primary.sum() <= 3).any()) or ci[1] <= 0 \
        or any(tot[f] >= tot["F1"] for f in ("F3", "F4", "F5", "F6"))
    res["H1"] = dict(F1_failed_by_kappa=f1.to_dict(), clause1_F1_fails_ge4_all_kappa_ge5=c1,
                     F2_spearman_logk_error=float(rho), F2_spearman_ci95=ci, clause2=c2,
                     failed_seeds_kappa_gt1=tot, clause3_F3toF6_fewer_than_F1=c3,
                     verdict="supported" if (c1 and c2 and c3) else ("refuted" if ref else "inconclusive"),
                     note="F3/F4/F5 satisfy clause 3 only narrowly (19-20 vs 25 failed seeds); they fail from kappa=20, earlier than F2.")

    # ---------------- H2
    Q = P[P.kappa > 1].copy(); Q["R"] = Q.primary_field_error / Q.relL2_phi_global
    medR = float(Q.R.median()); ciR = boot(np.median, Q.R.values); fracR = float((Q.R > 1).mean())
    fails = Q[Q.fail_primary]; conceal = float((fails.relL2_phi_global < 0.05).mean())
    byF = Q.groupby("formulation").R.median().to_dict()
    res["H2"] = dict(n_runs=len(Q), median_R=medR, median_R_ci95=ciR, frac_R_gt1=fracR, n_failures=len(fails),
                     concealed_failure_fraction=conceal, n_concealed=int((fails.relL2_phi_global < 0.05).sum()),
                     median_R_by_formulation=byF,
                     verdict="supported" if (medR > 3 and fracR >= 0.8 and conceal >= 0.2) else ("refuted" if medR <= 1 else "inconclusive"))

    # ---------------- H3
    s = P[P.formulation.isin(["F2", "F6"]) & P.fail_primary]
    flux_dom = (s.D_jump_rel > 0.05) & (s.phi_jump_rel < 0.01)
    fr = float(flux_dom.mean()) if len(s) else np.nan
    res["H3"] = dict(n_failing_runs=len(s), n_flux_dominant=int(flux_dom.sum()), frac_flux_dominant=fr,
                     median_D_jump_rel_failing=float(s.D_jump_rel.median()), median_phi_jump_rel_failing=float(s.phi_jump_rel.median()),
                     median_D_jump_rel_passing=float(P[P.formulation.isin(["F2", "F6"]) & ~P.fail_primary].D_jump_rel.median()),
                     verdict="supported" if fr >= 0.8 else ("refuted" if fr < 0.5 else "inconclusive"))

    # ---------------- H4 (intervention clause pending Phase 11)
    reg_ok = bool((reg.primary_field_error < 0.01).all())
    piv = lv.pivot_table(index=["formulation", "kappa"], columns="candidate", values="loss_total").reset_index()
    f1v = piv[piv.formulation == "F1"]; c_f1 = bool((f1v.wrong_uniform <= f1v.exact).all())
    o = piv[piv.formulation.isin(["F2", "F3", "F4", "F5", "F6"]) & (piv.kappa > 1)]
    ratio = (o.wrong_uniform / o.exact.replace(0, np.nan)).fillna(np.inf)
    c_vis = bool((ratio > 10).all()); low = o[ratio <= 10][["formulation", "kappa"]].values.tolist()
    f2fail = P[P.formulation == "F2"].groupby("kappa").fail_primary.sum(); f2k = f2fail[f2fail >= 4].index
    regF2 = reg[reg.formulation == "REG_F2"].groupby("kappa").primary_field_error.median()
    refuted_repr = bool((regF2.loc[f2k] > 0.05).any())
    pinn_vs_reg = {float(k): dict(PINN_F2_median=float(P[(P.formulation == "F2") & (P.kappa == k)].primary_field_error.median()),
                                  REG_F2_median=float(regF2.loc[k])) for k in regF2.index}
    res["H4"] = dict(clause_regression_lt1pct_all=reg_ok, regression_max=float(reg.primary_field_error.max()),
                     clause_F1_invisible_all_kappa=c_f1, clause_visibility_ratio_gt10_F2toF6=c_vis,
                     visibility_ratio_le10_cases=low, refutation_regression_fails_where_F2_fails=refuted_repr,
                     PINN_vs_regression_same_architecture=pinn_vs_reg, intervention_clause="pending Phase 11",
                     verdict="refuted (pre-registered criterion)" if refuted_repr else "pending",
                     interpretation=("Pre-registered refutation criterion met: phi-only regression with the F2 architecture exceeds 5% "
                                     "where F2 fails. But PINN-trained F2 with the SAME architecture is more accurate than regression at "
                                     "every kappa >= 5, so phi-only regression underestimates representation capacity; the refutation does "
                                     "not by itself establish a representation limit. An exploratory field-supervised regression is run separately."))

    # ---------------- H11
    S = P[P.formulation.isin(["F2", "F3", "F4", "F6"])].copy()
    S["lr_w"] = [early_ratio(r, True) for r in S.run_id]; S["lr_u"] = [early_ratio(r, False) for r in S.run_id]
    lab = S.fail_primary.values.astype(bool)
    out = {}
    for c in ("lr_w", "lr_u"):
        sc = S[c].values; a = auc(sc, lab)
        ci = boot(lambda s_, l_: auc(s_, l_.astype(bool)), sc, lab.astype(float))
        out[c] = dict(AUC=a, ci95=ci)
    res["H11"] = dict(n_runs=len(S), n_failed=int(lab.sum()), weighted=out["lr_w"], unweighted_sensitivity=out["lr_u"],
                      definition="median over logged steps 0-300 of log10(max/min per-term gradient norm), weighted norms primary",
                      verdict="supported" if (out["lr_w"]["AUC"] >= 0.75 and out["lr_w"]["ci95"][0] > 0.5) else
                              ("refuted" if out["lr_w"]["ci95"][0] <= 0.5 else "inconclusive"))
    S[["run_id", "formulation", "kappa", "seed", "lr_w", "lr_u", "fail_primary", "primary_field_error"]].to_csv(
        ROOT / "results/processed/p1_early_gradient_ratio.csv", index=False)

    # ---------------- 2D selection rule (protocol Sec. 8.2)
    t = P.groupby("formulation").fail_primary.sum().sort_values()
    m2000 = P[P.kappa == 2000].groupby("formulation").primary_field_error.median()
    order = sorted(FORMS, key=lambda f: (t[f], m2000[f]))
    res["selection_2d"] = dict(total_failed_primary=t.to_dict(), median_at_2000=m2000.to_dict(),
                               ranking=order, strongest_two=[f for f in order if f != "F1"][:2])
    json.dump(res, open(ROOT / "results/processed/p1_hypothesis_tests.json", "w"), indent=2, default=float)

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(6.4, 5), constrained_layout=True)
    for f, mk in zip(FORMS, "osD^vP"):
        q = Q[Q.formulation == f]
        ax.loglog(q.relL2_phi_global, q.primary_field_error, mk, ms=5, alpha=.7, label=f)
    lim = [1e-6, 1e3]; ax.plot(lim, lim, "k-", lw=.8, label="equal error")
    ax.axhline(.05, ls=":", c="r"); ax.axvline(.05, ls=":", c="r")
    ax.text(1.5e-6, .07, "field-failure threshold", color="r", fontsize=7)
    ax.set_xlabel("global potential error (relative L2)"); ax.set_ylabel("interface-band field error (primary, relative L2)")
    ax.set_title(f"P1, κ>1, all formulations (n={len(Q)}): field vs potential error\nmedian ratio R = {medR:.1f}; "
                 f"{conceal:.0%} of field failures have φ error < 5%", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
    fig.savefig(ROOT / "results/figures/p1_global_vs_local_error.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, f in zip(axes, ("F2", "F3", "F6")):
        for k, c in zip((5, 80, 2000), ("tab:blue", "tab:orange", "tab:red")):
            g = pd.read_csv(ROOT / f"results/raw/gradlogs/P1-{f}-k{k:g}-s0-pinn-base.csv")
            for term, ls in zip(sorted(g.term.unique()), ["-", "--", ":", "-.", (0, (1, 3)), (0, (5, 1))]):
                s_ = g[g.term == term]; ax.semilogy(s_.step, s_.grad_norm_weighted, ls=ls, color=c, lw=1,
                                                   label=f"κ={k} {term}" if True else None)
        ax.set_title(f"{f}, seed 0: weighted per-term gradient norms", fontsize=9); ax.set_xlabel("training step")
        ax.set_ylabel("‖∇θ (w_i L_i)‖₂"); ax.grid(alpha=.3)
    axes[-1].legend(fontsize=5, ncol=2)
    fig.savefig(ROOT / "results/figures/p1_gradient_trajectories.png", dpi=150); plt.close(fig)

    t_ = pd.read_csv(ROOT / "results/processed/p1_config_summary.csv"); t_ = t_[t_.grid == "primary"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, (cols, ttl) in zip(axes, ((("median_E_band1", "median_E_band2"), "interface-band field error by side"),
                                      (("median_E_mat1", "median_E_mat2"), "field error by material"),
                                      (("median_D_jump_rel", "median_phi_jump_rel"), "interface jumps |[D]|/D and |[φ]|/V"))):
        for f, c in zip(FORMS, plt.cm.tab10.colors):
            g = t_[t_.formulation == f]
            ax.loglog(g.kappa, g[cols[0]].clip(lower=1e-9), "-o", color=c, ms=3, label=f"{f} {cols[0].split('_')[-1]}")
            ax.loglog(g.kappa, g[cols[1]].clip(lower=1e-9), "--s", color=c, ms=3, label=f"{f} {cols[1].split('_')[-1]}")
        ax.axhline(.05, c="k", ls=":"); ax.set_xlabel("κ"); ax.set_ylabel("median over 5 seeds (clipped at 1e-9)"); ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=.3, which="both")
    axes[0].legend(fontsize=5, ncol=2)
    fig.savefig(ROOT / "results/figures/p1_material_and_jumps.png", dpi=150); plt.close(fig)
    return res


if __name__ == "__main__":
    r = main()
    for h in ("H1", "H2", "H3", "H4", "H11", "selection_2d"):
        print(h, json.dumps(r[h], default=float)[:900]); print()
