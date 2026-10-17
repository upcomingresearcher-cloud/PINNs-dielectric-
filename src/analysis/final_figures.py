"""Phase 14: figure types not produced by the per-phase modules (brief Sec. 21 list items 6, 7, 9, 11, 14, 15).

Reads results/raw only. Outputs results/figures/final_*.png and results/processed/final_*.csv.
  final_capacitance_energy.png     (6)  P1 capacitance/energy relative errors; 2D energy relative errors
  final_gradratio_vs_error.png     (7)  early (steps 0-300) log10 gradient ratio vs primary error, P1 soft formulations (H11)
  final_convergence_curves.png     (9)  primary error vs training step, median and IQR over seeds
  final_critical_contrast.png      (11) smallest kappa > 1 at which >= 3/5 seeds fail, per formulation and problem
  final_fem_convergence.png        (14) conforming FEM band error vs DOFs, P2-P5 (P1 exact: degenerate)
  final_cross_geometry.png         (15) failures at kappa in {80, 2000} and median R per problem x formulation
"""
import json, pathlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.p1_hypotheses import early_ratio

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures"; OUT = ROOT / "results/processed"
FORMS = ["F1", "F2", "F3", "F4", "F5", "F6"]; PROBS = ["P1", "P2", "P3", "P4", "P5"]
COL = dict(zip(FORMS, plt.get_cmap("tab10").colors))


def load_primary():
    a = pd.read_csv(ROOT / "results/raw/runs.csv", low_memory=False)
    a = a[(a.run_type == "pinn") & (a.variant == "base") & (a.status == "complete")]
    b = pd.read_csv(ROOT / "results/raw/runs_2d.csv", low_memory=False)
    b = b[(b.run_type == "pinn") & (b.variant == "base") & (b.status == "complete")]
    d = pd.concat([a, b], ignore_index=True).sort_values("timestamp").drop_duplicates("run_id", keep="first")
    d["fail"] = d.fail_primary.astype(bool); d["R"] = d.primary_field_error / d.relL2_phi_global
    return d[d.seed.isin(range(5))]


def band(ax, x, g, col, label, marker="o"):
    y = [c for c in g.columns if c != x][0]
    q = g.groupby(x)[y].quantile([.25, .5, .75]).unstack()
    ax.plot(q.index, q[.5], marker=marker, color=col, label=label, ms=4)
    ax.fill_between(q.index, q[.25], q[.75], color=col, alpha=0.18, lw=0)


def fig_capacitance_energy(D):
    P1 = D[(D.problem_id == "P1") & (D.kappa >= 1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, col, ttl in ((axes[0], "C_flux_relerr", "P1: capacitance from interface flux"),
                         (axes[1], "C_energy_relerr", "P1: capacitance from field energy")):
        for f in FORMS:
            g = P1[P1.formulation == f]
            band(ax, "kappa", g[["kappa", col]].set_index("kappa")[col].abs().reset_index(), COL[f], f)
        ax.set_title(ttl, fontsize=10); ax.set_ylabel("|relative error| (–)")
    ax = axes[2]; Q = D[(D.problem_id != "P1") & D.kappa.isin([1.0, 80.0, 2000.0])]
    w = 0.13
    for i, f in enumerate(FORMS):
        for j, p in enumerate(PROBS[1:]):
            g = Q[(Q.formulation == f) & (Q.problem_id == p)].groupby("kappa").energy_relerr.apply(lambda s: s.abs().median())
            ax.scatter(np.log10(g.index) + (i - 2.5) * w * 0.4 + j * 0.0, g.values, color=COL[f], marker="osD^"[j], s=18,
                       label=f if j == 0 else None)
    ax.set_xticks([0, np.log10(80), np.log10(2000)]); ax.set_xticklabels(["1", "80", "2000"])
    ax.set_title("P2–P5: field-energy relative error (median of 5 seeds)\nmarkers: ○ P2, □ P3, ◇ P4, △ P5", fontsize=9)
    ax.set_ylabel("|relative error| (–)")
    for a in axes:
        a.set_yscale("log"); a.set_xlabel("contrast κ = ε₂/ε₁ (–)"); a.legend(fontsize=7, ncol=2)
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    fig.suptitle("Integral quantities: capacitance and energy errors (median, shaded = IQR over 5 seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "final_capacitance_energy.png", dpi=130); plt.close(fig)


def fig_gradratio(D):
    P1 = D[(D.problem_id == "P1") & D.formulation.isin(["F2", "F3", "F4", "F6"]) & (D.kappa >= 1)].copy()
    P1["early_log_ratio"] = [early_ratio(r) for r in P1.run_id]
    P1[["run_id", "formulation", "kappa", "seed", "early_log_ratio", "primary_field_error", "fail"]].to_csv(OUT / "final_gradratio.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    cm = plt.get_cmap("viridis"); ks = sorted(P1.kappa.unique()); kc = {k: cm(i / (len(ks) - 1)) for i, k in enumerate(ks)}
    for ax, f in zip(axes, ["F2", "F3", "F4", "F6"]):
        g = P1[P1.formulation == f]
        for k in ks:
            h = g[g.kappa == k]
            ax.scatter(h.early_log_ratio, h.primary_field_error, color=kc[k], s=22, marker="o", edgecolor="k", lw=0.3, label=f"κ={k:g}")
        ax.axhline(0.05, color="k", ls="--", lw=0.8); ax.set_yscale("log"); ax.set_title(f, fontsize=10)
        ax.set_xlabel("early log₁₀ gradient ratio (–)")
    axes[0].set_ylabel("final primary band field error (–)"); axes[0].legend(fontsize=7)
    fig.supxlabel("early log₁₀ gradient ratio = median over steps 0–300 of log₁₀(max / min weighted per-term gradient norm)", fontsize=9)
    fig.suptitle("P1: early gradient imbalance vs final error (H11; dashed = 5% failure threshold). AUC 0.85 pooled, confounded with κ", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "final_gradratio_vs_error.png", dpi=130); plt.close(fig)


def fig_convergence(D):
    panels = [("P1", "F2"), ("P1", "F3"), ("P1", "F6"), ("P2", "F2"), ("P5", "F2"), ("P5", "F6")]
    cm = {1.0: "tab:green", 80.0: "tab:orange", 2000.0: "tab:red"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, (p, f) in zip(axes.ravel(), panels):
        for k, c in cm.items():
            ids = D[(D.problem_id == p) & (D.formulation == f) & (D.kappa == k)].run_id
            T = pd.concat([pd.read_csv(ROOT / "results/raw/trajectories" / f"{r}.csv")[["step", "primary_field_error"]] for r in ids])
            band(ax, "step", T, c, f"κ = {k:g} ({len(ids)} seeds)")
        ax.axhline(0.05, color="k", ls="--", lw=0.8); ax.set_yscale("log"); ax.set_title(f"{p} {f}", fontsize=10)
        ax.set_xlabel("Adam step (–)"); ax.set_ylabel("primary band field error (–)"); ax.legend(fontsize=7)
    fig.suptitle("Training convergence: primary error every 250 steps (median, shaded = IQR over seeds; dashed = 5%)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "final_convergence_curves.png", dpi=130); plt.close(fig)


def critical_table(D):
    rows = []
    for p in PROBS:
        for f in FORMS:
            g = D[(D.problem_id == p) & (D.formulation == f) & (D.kappa > 1)].groupby("kappa").fail.sum()
            g = g[g.index <= 2000]
            fails = g[g >= 3]
            rows.append(dict(problem_id=p, formulation=f, kappas_tested=",".join(f"{k:g}" for k in g.index),
                             critical_contrast=float(fails.index.min()) if len(fails) else np.inf,
                             monotone=bool((g[g.index >= fails.index.min()] >= 3).all()) if len(fails) else True))
    T = pd.DataFrame(rows); T.to_csv(OUT / "final_critical_contrast.csv", index=False); return T


def fig_critical(T):
    fig, ax = plt.subplots(figsize=(9, 4.6)); grid = [5, 20, 80, 400, 2000]
    for j, p in enumerate(PROBS):
        g = T[T.problem_id == p]
        for r in g.itertuples():
            i = FORMS.index(r.formulation); y = np.log10(r.critical_contrast) if np.isfinite(r.critical_contrast) else np.log10(6000)
            ax.scatter(i + (j - 2) * 0.12, y, color=plt.get_cmap("Set1").colors[j], marker="os^Dv"[j], s=40,
                       facecolor="none" if not r.monotone else plt.get_cmap("Set1").colors[j], label=p if r.formulation == "F1" else None)
    ax.set_yticks([np.log10(k) for k in grid] + [np.log10(6000)]); ax.set_yticklabels([str(k) for k in grid] + ["none ≤ 2000"])
    ax.set_xticks(range(6)); ax.set_xticklabels(FORMS); ax.set_ylabel("critical contrast κ_c (–)")
    ax.set_title("Smallest κ > 1 at which ≥ 3/5 seeds fail (primary criterion); hollow = non-monotone above κ_c\n"
                 "F3/F4/F5 were run on 2D problems at κ ∈ {1, 80, 2000} only, so their κ_c there is 80 or 2000 at best resolution", fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "final_critical_contrast.png", dpi=130); plt.close(fig)


def fig_fem():
    f = pd.read_csv(ROOT / "results/raw/fem_runs_2d.csv"); f = f[f.conforming.astype(bool)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, p in zip(axes, PROBS[1:]):
        g = f[f.problem_id == p]
        for k, h in g.groupby("kappa"):
            h = h.sort_values("dofs"); h = h[h.primary_field_error > 0]     # P5 finest level = reference (error 0) omitted
            if k == 1.0 or h.primary_field_error.max() < 1e-10: continue
            ax.plot(h.dofs, h.primary_field_error, "o-", ms=4, label=f"κ = {k:g}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(0.01, color="gray", ls=":")
        ax.set_xlabel("degrees of freedom (–)"); ax.set_ylabel("primary band field error (–)")
        ax.set_title(f"{p}: conforming FEM (quadratic elements)" + ("\nerrors vs finest level (reference)" if p == "P5" else ""), fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("FEM convergence in the interface band (κ = 1 omitted: exact to round-off; P1 exact for any mesh, degenerate; dotted = 1%)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "final_fem_convergence.png", dpi=130); plt.close(fig)


def fig_cross(D):
    Q = D[D.kappa.isin([80.0, 2000.0])]
    F = Q.groupby(["problem_id", "formulation"]).fail.sum().unstack().reindex(index=PROBS, columns=FORMS)
    N = Q.groupby(["problem_id", "formulation"]).fail.size().unstack().reindex(index=PROBS, columns=FORMS)
    Rm = D[D.kappa != 1].groupby("problem_id").R.median().reindex(PROBS)     # as in H2/H8 (P1 has no kappa < 1 runs)
    F.to_csv(OUT / "final_cross_geometry_failcounts.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    im = a1.imshow(F.values / N.values, cmap="Reds", vmin=0, vmax=1)
    for i in range(len(PROBS)):
        for j in range(6):
            fr = F.values[i, j] / N.values[i, j]
            a1.text(j, i, f"{int(F.values[i, j])}/{int(N.values[i, j])}", ha="center", va="center", fontsize=8, color="w" if fr > .6 else "k")
    a1.set_xticks(range(6)); a1.set_xticklabels(FORMS); a1.set_yticks(range(5)); a1.set_yticklabels(PROBS)
    a1.set_title("Failed runs at κ ∈ {80, 2000} (5 seeds each; P1 primary rows)", fontsize=9); plt.colorbar(im, ax=a1, label="failure fraction")
    a2.bar(PROBS, Rm.values, color="0.6"); a2.axhline(3, color="k", ls="--", lw=0.8, label="H2 threshold R = 3")
    a2.set_ylabel("median R = band field error / global potential error (–)"); a2.set_title("Masking ratio, all κ ≠ 1 runs (H2/H8 definition)", fontsize=9)
    a2.legend(fontsize=8)
    fig.suptitle("Cross-geometry comparison (P1 1D slab; P2 cylinder; P3 line charge; P4 corner; P5 thin film)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "final_cross_geometry.png", dpi=130); plt.close(fig)
    return F, Rm


def main():
    D = load_primary()
    fig_capacitance_energy(D); fig_gradratio(D); fig_convergence(D)
    T = critical_table(D); fig_critical(T); fig_fem(); F, Rm = fig_cross(D)
    return T, F, Rm


if __name__ == "__main__":
    T, F, Rm = main()
    print(T.pivot(index="problem_id", columns="formulation", values="critical_contrast").to_string())
    print(F.to_string()); print(Rm.round(2).to_dict())
