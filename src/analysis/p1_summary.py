"""P1 primary-sweep analysis (brief Sec. 13 D-H; protocol Sec. 13). Reads results/raw only.
Outputs: results/processed/p1_*.csv and results/figures/p1_*.png."""
import pathlib, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import beta

ROOT = pathlib.Path(__file__).resolve().parents[2]
FORMS = ["F1", "F2", "F3", "F4", "F5", "F6"]


def cp_interval(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def load():
    d = pd.read_csv(ROOT / "results/raw/runs.csv")
    d = d[d.status == "complete"]
    main = d[d.run_type.isin(["pinn", "extension"]) & (d.variant == "base")].copy()
    reg = d[d.run_type == "regression"].copy()
    return main, reg


def config_table(main):
    rows = []
    for (f, k), g in main.groupby(["formulation", "kappa"]):
        n = len(g); nf = int(g.fail_primary.sum()); lo, hi = cp_interval(nf, n)
        q = g.primary_field_error.quantile([.25, .5, .75])
        rows.append(dict(formulation=f, kappa=k, grid="extension" if k == 10000 else "primary", n_seeds=n, n_failed=nf,
                         failure_rate=nf / n, cp95_lo=lo, cp95_hi=hi, config_fails=nf >= 3,
                         n_failed_strict=int(g.fail_strict.sum()), n_failed_potential=int(g.fail_potential.sum()),
                         median_primary=q[.5], q25_primary=q[.25], q75_primary=q[.75],
                         min_primary=g.primary_field_error.min(), max_primary=g.primary_field_error.max(),
                         median_phi_global=g.relL2_phi_global.median(), median_E_global=g.relL2_E_global.median(),
                         median_E_mat1=g.relL2_E_mat1.median(), median_E_mat2=g.relL2_E_mat2.median(),
                         median_E_band1=g.relL2_E_band1.median(), median_E_band2=g.relL2_E_band2.median(),
                         median_phi_jump_rel=g.phi_jump_rel.median(), median_D_jump_rel=g.D_jump_rel.median(),
                         median_C_flux_relerr=g.C_flux_relerr.median(), median_C_energy_relerr=g.C_energy_relerr.median(),
                         median_log_kappa_eff=g.log_kappa_eff_ratio.median(),
                         n_sec_fail_eff_contrast=int(g.sec_fail_effective_contrast.sum()), n_sec_fail_Djump=int(g.sec_fail_D_jump.sum()),
                         median_wall_s=g.wall_clock_s.median(), n_params=int(g.n_params.iloc[0]), grad_evals=int(g.grad_evals.iloc[0])))
    return pd.DataFrame(rows)


def critical_contrast(t):
    out = []
    for f in FORMS:
        g = t[(t.formulation == f) & (t.grid == "primary")].sort_values("kappa")
        fails = g[g.config_fails]
        if fails.empty: kc, mono = np.nan, True
        else:
            kc = fails.kappa.min(); mono = bool(g[g.kappa > kc].config_fails.all())
        out.append(dict(formulation=f, critical_contrast=kc, monotone_after_critical=mono,
                        total_failed_seeds_primary=int(g.n_failed.sum()),
                        median_primary_at_2000=float(g[g.kappa == 2000].median_primary.iloc[0]),
                        median_wall_s=float(g.median_wall_s.median())))
    return pd.DataFrame(out)


def heatmap(t, value, fname, title, cbar, fmt, log=False, vmax=None):
    pv = t.pivot(index="formulation", columns="kappa", values=value).reindex(FORMS)
    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    data = np.log10(pv.values.astype(float)) if log else pv.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0 if not log else None, vmax=vmax)
    ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels([f"{k:g}" + ("*" if k == 10000 else "") for k in pv.columns])
    ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]; ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8)
    ax.set_xlabel("permittivity contrast κ = ε₂/ε₁   (* = pre-registered extension)"); ax.set_ylabel("formulation")
    ax.set_title(title, fontsize=10); fig.colorbar(im, ax=ax, label=cbar)
    fig.savefig(ROOT / "results/figures" / fname, dpi=150); plt.close(fig)


def main():
    (ROOT / "results/processed").mkdir(parents=True, exist_ok=True); (ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
    main_, reg = load()
    t = config_table(main_); t.to_csv(ROOT / "results/processed/p1_config_summary.csv", index=False)
    cc = critical_contrast(t); cc.to_csv(ROOT / "results/processed/p1_critical_contrast.csv", index=False)
    rt = reg.groupby(["formulation", "kappa"]).agg(n=("seed", "size"), median_primary=("primary_field_error", "median"),
                                                   max_primary=("primary_field_error", "max"), median_phi=("relL2_phi_global", "median"),
                                                   n_failed=("fail_primary", "sum")).reset_index()
    rt.to_csv(ROOT / "results/processed/p1_regression_baseline.csv", index=False)
    t["fail_label"] = t.n_failed
    heatmap(t, "n_failed", "p1_failure_map.png",
            "P1 failure map: failed seeds out of 5 (primary criterion: interface-band field error > 5%)\n"
            "3000 Adam gradient evaluations, float32, 3×32 tanh",
            "failed seeds (of 5)", lambda v: f"{int(v)}/5", vmax=5)
    heatmap(t, "median_primary", "p1_median_band_field_error.png",
            "P1 median interface-band field error over 5 seeds (log10 colour; 5% threshold = -1.3)",
            "log10 median primary error", lambda v: f"{v:.1e}", log=True)
    return t, cc, rt


if __name__ == "__main__":
    t, cc, rt = main()
    pd.set_option("display.width", 200)
    print(t.pivot(index="formulation", columns="kappa", values="n_failed").to_string()); print()
    print(t.pivot(index="formulation", columns="kappa", values="median_primary").map(lambda v: f"{v:.2e}").to_string()); print()
    print(cc.to_string(index=False)); print()
    print(rt.pivot(index="formulation", columns="kappa", values="median_primary").map(lambda v: f"{v:.2e}").to_string())
