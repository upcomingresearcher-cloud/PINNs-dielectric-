"""Phase 12 analysis: robustness of P1 conclusions (protocol Sec. 11; H6; operationalisation in change log).

Reference = primary rows (run_type pinn, variant base), seeds 0-4, kappa in {1, 80, 2000}.
Score per formulation = failed runs (primary criterion) out of 15. Kendall tau-b (tie-corrected) vs the reference;
undefined tau counts as < 0.6. R = primary_field_error / relL2_phi_global, median over kappa > 1 runs (as H2).
Outputs: results/processed/p1_robustness_{counts,summary}.csv, p1_robustness_verdict.json, results/figures/p1_robustness_*.png
"""
import json, pathlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import kendalltau

ROOT = pathlib.Path(__file__).resolve().parents[2]
FORMS = ["F1", "F2", "F3", "F4", "F5", "F6"]
KAPPAS = [1.0, 80.0, 2000.0]; SEEDS = [0, 1, 2, 3, 4]
VARIATIONS = ["width16", "width64", "depth2", "depth5", "n32", "n512", "lr3e-3", "steps12000"]
EXTRAS = ["pm_width22", "band16"]
BOOT_N, BOOT_SEED = 10_000, 12345


def load():
    d = pd.read_csv(ROOT / "results/raw/runs.csv", low_memory=False)
    d = d[(d.status == "complete") & (d.problem_id == "P1")].sort_values("timestamp").drop_duplicates("run_id", keep="first")
    d = d[d.kappa.isin(KAPPAS) & d.seed.isin(SEEDS)]
    ref = d[(d.run_type == "pinn") & (d.variant == "base")].assign(setting="baseline")
    rob = d[d.run_type == "robustness"].assign(setting=lambda x: x.variant)
    out = pd.concat([ref, rob], ignore_index=True)
    out["fail"] = out.fail_primary.astype(bool)
    out["R"] = out.primary_field_error / out.relL2_phi_global
    return out


def expected_n(setting):
    return 45 if setting == "pm_width22" else 90


def settings_table(D):
    """For the pm_width22 extra, F1/F3/F5 come from the baseline (they are not re-run): mixed ranking, descriptive."""
    rows = []
    base = D[D.setting == "baseline"]
    for s in ["baseline"] + VARIATIONS + EXTRAS:
        g = D[D.setting == s]
        if s == "pm_width22": g = pd.concat([g, base[base.formulation.isin(["F1", "F3", "F5"])]])
        if len(g) != 90:
            rows.append(dict(setting=s, complete=False, n_runs=len(g))); continue
        fails = g.groupby("formulation").fail.sum().reindex(FORMS)
        med = g.groupby("formulation").primary_field_error.median().reindex(FORMS)
        bf = base.groupby("formulation").fail.sum().reindex(FORMS); bm = base.groupby("formulation").primary_field_error.median().reindex(FORMS)
        tau = kendalltau(bf.values, fails.values).statistic if s != "baseline" else 1.0
        tau_med = kendalltau(bm.values, med.values).statistic if s != "baseline" else 1.0
        q = g[g.kappa > 1].R
        rng = np.random.default_rng(BOOT_SEED); qq = q.values
        bm_R = np.median(qq[rng.integers(0, len(qq), (BOOT_N, len(qq)))], axis=1)
        rows.append(dict(setting=s, complete=True, n_runs=len(g), role="variation" if s in VARIATIONS else ("extra" if s in EXTRAS else "reference"),
                         **{f"fails_{f}": int(fails[f]) for f in FORMS}, total_fails=int(fails.sum()),
                         kendall_tau_b=float(tau), tau_ge_0p6=bool(np.isfinite(tau) and tau >= 0.6),
                         kendall_tau_b_median_error=float(tau_med),
                         median_R=float(q.median()), R_ci_lo=float(np.quantile(bm_R, .025)), R_ci_hi=float(np.quantile(bm_R, .975)),
                         frac_R_gt1=float((q > 1).mean()), R_gt_3=bool(q.median() > 3),
                         n_concealed=int(((g.kappa > 1) & g.fail & (g.relL2_phi_global < 0.05)).sum()),
                         n_fail_kappa_gt1=int((g.fail & (g.kappa > 1)).sum()),
                         median_wall_s=float(g.wall_clock_s.median()),
                         **{f"median_err_{f}": float(med[f]) for f in FORMS}))
    return pd.DataFrame(rows)


def counts_table(D):
    return (D.groupby(["setting", "formulation", "kappa"])
             .agg(n=("fail", "size"), n_failed=("fail", "sum"), median_err=("primary_field_error", "median"),
                  q25=("primary_field_error", lambda x: x.quantile(.25)), q75=("primary_field_error", lambda x: x.quantile(.75)),
                  median_R=("R", "median"), median_wall_s=("wall_clock_s", "median"), n_params=("n_params", "first"))
             .reset_index())


def verdict(S):
    V = S[S.role == "variation"]
    if len(V) < len(VARIATIONS) or not V.complete.all():
        return dict(verdict="incomplete", n_variations_complete=int(V.complete.sum()))
    n_tau = int(V.tau_ge_0p6.sum()); n_R = int(V.R_gt_3.sum())
    if len(V) - n_tau >= 3: v = "refuted"
    elif n_tau >= 6 and n_R >= 6: v = "supported"
    else: v = "inconclusive"
    return dict(verdict=v, n_tau_ge_0p6=n_tau, n_R_gt_3=n_R, n_variations=len(V),
                failing_tau=V[~V.tau_ge_0p6].setting.tolist(), failing_R=V[~V.R_gt_3].setting.tolist(),
                per_variation=V[["setting", "kendall_tau_b", "median_R", "total_fails"]].to_dict("records"))


def fig_fails(S, fname):
    S = S[S.complete].copy(); order = ["baseline"] + [s for s in VARIATIONS + EXTRAS if s in set(S.setting)]
    S = S.set_index("setting").loc[order]
    M = S[[f"fails_{f}" for f in FORMS]].values.astype(float)
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(order) + 1.6))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=15, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=8, color="k" if M[i, j] < 10 else "w")
    ax.set_xticks(range(6)); ax.set_xticklabels(FORMS)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{s}  (τ={S.loc[s, 'kendall_tau_b']:.2f}, R={S.loc[s, 'median_R']:.1f})" for s in order], fontsize=8)
    ax.set_title("P1 robustness: failed runs out of 15 (κ ∈ {1, 80, 2000} × 5 seeds)\nτ = Kendall τ-b vs baseline ranking; R = median band-field / global-potential error (κ > 1)", fontsize=9)
    plt.colorbar(im, ax=ax, label="failed runs (of 15)"); fig.tight_layout()
    fig.savefig(ROOT / "results/figures" / fname, dpi=130); plt.close(fig)


def fig_errors(C, fname):
    order = ["baseline"] + VARIATIONS + EXTRAS
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, k in zip(axes, KAPPAS):
        for j, f in enumerate(FORMS):
            g = C[(C.formulation == f) & (C.kappa == k)].set_index("setting").reindex(order)
            x = np.arange(len(order)) + (j - 2.5) * 0.12
            ax.errorbar(x, g.median_err, yerr=[g.median_err - g.q25, g.q75 - g.median_err], fmt="o", ms=3, capsize=1.5, label=f)
        ax.axhline(0.05, color="k", ls="--", lw=0.8); ax.set_yscale("log")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=60, fontsize=7)
        ax.set_title(f"κ = {k:g}", fontsize=10)
    axes[0].set_ylabel("primary band field error (–)\nmedian, bars = IQR, 5 seeds"); axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("P1 robustness: primary error per formulation and setting (dashed = 5% failure threshold)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "results/figures" / fname, dpi=130); plt.close(fig)


def main():
    D = load(); C = counts_table(D); S = settings_table(D); V = verdict(S)
    V["extras"] = S[S.role == "extra"][["setting", "kendall_tau_b", "median_R", "total_fails"]].to_dict("records") if "role" in S else []
    V["notes"] = ["pm_width22 ranking uses baseline F1/F3/F5 with width-22 F2/F4/F6 (descriptive extra)",
                  "boundary sampling: not applicable in 1D (fixed endpoints)", "precision and optimiser: see Phase 11 (C, E)"]
    out = ROOT / "results/processed"
    C.to_csv(out / "p1_robustness_counts.csv", index=False); S.to_csv(out / "p1_robustness_summary.csv", index=False)
    json.dump(V, open(out / "p1_robustness_verdict.json", "w"), indent=1, default=str)
    if S.complete.sum() > 1:
        fig_fails(S, "p1_robustness_failcounts.png")
        if V.get("verdict") != "incomplete": fig_errors(C, "p1_robustness_errors.png")
    return D, C, S, V


if __name__ == "__main__":
    D, C, S, V = main()
    pd.set_option("display.width", 250)
    print(S[[c for c in S.columns if c in ("setting", "n_runs", "total_fails", "kendall_tau_b", "kendall_tau_b_median_error", "median_R",
                                            "R_ci_lo", "R_ci_hi", "n_concealed", "median_wall_s") or c.startswith("fails_")]].to_string(float_format=lambda v: f"{v:.3g}"))
    print(json.dumps({k: v for k, v in V.items() if k != "per_variation"}, indent=1, default=str))
