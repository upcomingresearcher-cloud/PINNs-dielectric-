"""Phase 11 analysis: Interventions A-E on P1 (protocol Sec. 10; operationalisation in change log v1.6.0).

Reads results/raw/runs.csv only. Outputs:
  results/processed/phase11_paired.csv          every paired comparison (pre-specified and descriptive)
  results/processed/phase11_failure_counts.csv  failure counts per arm x kappa (10 or 5 seeds)
  results/processed/phase11_critical_contrast.csv
  results/processed/phase11_precision.csv       Intervention C fold changes and flag changes
  results/processed/phase11_verdicts.json       H5 per intervention, H4 intervention clause, C and D criteria
  results/figures/p1_interventions_*.png
Errors: primary field error (max of band relL2 E and D). Non-finite errors are +inf (worst) for signs and counts;
bootstrap CIs use finite pairs only (count reported). Ties count against the prediction.
"""
import json, pathlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import binom
from src.analysis.p1_summary import cp_interval

ROOT = pathlib.Path(__file__).resolve().parents[2]
KAPPAS = [1.0, 5.0, 20.0, 80.0, 400.0, 2000.0]
K3 = [1.0, 80.0, 2000.0]
SEEDS10 = list(range(10)); SEEDS5 = list(range(5))
BOOT_N, BOOT_SEED = 10_000, 12345

# arm = (formulation, variant, precision-label) ; "base" merges primary rows (seeds 0-4) and intervention/base (5-9)
ARMS = {
    "base":     dict(variant="base"),
    "A":        dict(variant="A_rebal"),
    "B":        dict(variant="B_resnorm"),
    "D":        dict(variant="D_hard"),
    "C64":      dict(variant="C_f64v2"),            # after the init-precision bug fix (change log v1.6.0)
    "C64lbfgs": dict(variant="C_f64v2_lbfgs"),
    "C64old":   dict(variant="C_f64"),              # SUPERSEDED: float64 init differed from float32 init
    "C64lbfgsold": dict(variant="C_f64_lbfgs"),     # SUPERSEDED
    "Elbfgs":   dict(variant="E_lbfgs"),
    "Ehyb":     dict(variant="E_adam2k_lbfgs1k"),
}


def load():
    d = pd.read_csv(ROOT / "results/raw/runs.csv", low_memory=False)
    d = d[d.status == "complete"].sort_values("timestamp").drop_duplicates("run_id", keep="first")
    d = d[d.problem_id == "P1"]
    base = d[((d.run_type == "pinn") & (d.variant == "base") & d.seed.isin(SEEDS5)) |
             ((d.run_type == "intervention") & (d.variant == "base") & d.seed.isin(range(5, 10)))].copy()
    base["arm"] = "base"
    iv = d[(d.run_type == "intervention") & (d.variant != "base")].copy()
    iv["arm"] = iv.variant.map({v["variant"]: k for k, v in ARMS.items()})
    out = pd.concat([base, iv], ignore_index=True)
    out["err"] = np.where(np.isfinite(out.primary_field_error) & ~out.nonfinite.astype(bool), out.primary_field_error, np.inf)
    out["fail"] = out.fail_primary.astype(bool)
    return out


def series(D, f, arm, k, seeds):
    g = D[(D.formulation == f) & (D.arm == arm) & (D.kappa == k) & D.seed.isin(seeds)].set_index("seed")
    return g.reindex(seeds)


def boot_median_ci(x, rng_seed=BOOT_SEED):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(rng_seed)
    m = np.median(x[rng.integers(0, len(x), (BOOT_N, len(x)))], axis=1)
    return float(np.median(x)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def verdict(k_pred, k_opp, n):
    if n == 10:
        if k_pred >= 9: return "supported"
        if k_pred <= 5 or k_opp >= 9: return "refuted"
        return "inconclusive"
    return "descriptive"


def paired(D, intervention, label, fx, ax, fy, ay, k, seeds, prespecified):
    """Prediction is always: error(x) < error(y)."""
    X, Y = series(D, fx, ax, k, seeds), series(D, fy, ay, k, seeds)
    if X.err.isna().any() or Y.err.isna().any():
        raise RuntimeError(f"missing runs for {label} k={k}: {X.err.isna().sum()} / {Y.err.isna().sum()}")
    x, y = X.err.values, Y.err.values; n = len(seeds)
    k_pred = int(np.sum(x < y)); k_opp = int(np.sum(x > y)); ties = n - k_pred - k_opp
    p = float(binom.sf(k_pred - 1, n, 0.5))          # P(K >= k_pred) under H0
    diff = x - y; fin = np.isfinite(diff)
    med, lo, hi = boot_median_ci(diff)
    lr = np.log10(x[fin] / y[fin]) if fin.any() else np.array([np.nan])
    return dict(intervention=intervention, comparison=label, x=f"{fx}/{ax}", y=f"{fy}/{ay}", kappa=k, n_seeds=n,
                prespecified=prespecified, n_pred_sign=k_pred, n_opposite=k_opp, n_ties=ties, p_sign_one_sided=p,
                verdict=verdict(k_pred, k_opp, n) if prespecified else "descriptive",
                median_x=float(np.median(x)), q25_x=float(np.quantile(x, .25)), q75_x=float(np.quantile(x, .75)),
                median_y=float(np.median(y)), q25_y=float(np.quantile(y, .25)), q75_y=float(np.quantile(y, .75)),
                median_diff=med, diff_ci_lo=lo, diff_ci_hi=hi, n_nonfinite_pairs=int((~fin).sum()),
                median_log10_ratio=float(np.median(lr)), fails_x=int(X.fail.sum()), fails_y=int(Y.fail.sum()))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def build_paired(D):
    R = []
    for f in ("F2", "F3"):                                   # A
        for k in KAPPAS: R.append(paired(D, "A", f"{f} rebal vs base", f, "A", f, "base", k, SEEDS10, True))
    for f, fn in (("F2", "F2norm"), ("F1", "F1norm")):       # B
        for k in KAPPAS:
            R.append(paired(D, "B", f"{f} resnorm vs base", fn, "B", f, "base", k, SEEDS10, f == "F2" and k >= 80))
    for k in KAPPAS:                                         # D
        R.append(paired(D, "D", "F4 vs F2", "F4", "base", "F2", "base", k, SEEDS10, k >= 400))
        R.append(paired(D, "D", "F4full vs F2", "F4full", "D", "F2", "base", k, SEEDS10, False))
    for f in ("F2", "F3", "F5"):                             # E
        for arm, lab in (("Elbfgs", "L-BFGS"), ("Ehyb", "Adam2k+L-BFGS1k")):
            for k in K3: R.append(paired(D, "E", f"{f} {lab} vs Adam", f, arm, f, "base", k, SEEDS10, f in ("F3", "F5")))
    for f in ("F1", "F2", "F3", "F4", "F5", "F6"):            # C replica (5 seeds)
        for k in KAPPAS: R.append(paired(D, "C", f"{f} f64 vs f32 (Adam, replica)", f, "C64", f, "base", k, SEEDS5, False))
    for f in ("F2", "F3"):                                   # C 2x2 (10 seeds)
        for k in K3:
            R.append(paired(D, "C", f"{f} f64 vs f32 (Adam)", f, "C64", f, "base", k, SEEDS10, False))
            R.append(paired(D, "C", f"{f} f64 vs f32 (L-BFGS)", f, "C64lbfgs", f, "Elbfgs", k, SEEDS10, False))
    P = pd.DataFrame(R)
    P["p_holm_within_intervention"] = np.nan
    for iv, g in P[P.prespecified].groupby("intervention"):
        P.loc[g.index, "p_holm_within_intervention"] = holm(g.p_sign_one_sided.values)
    return P


def failure_counts(D):
    rows = []
    for (f, arm, k), g in D.groupby(["formulation", "arm", "kappa"]):
        for lab, seeds in (("10seed", SEEDS10), ("5seed", SEEDS5)):
            h = g[g.seed.isin(seeds)]
            if len(h) != len(seeds): continue
            nf = int(h.fail.sum()); lo, hi = cp_interval(nf, len(h))
            rows.append(dict(formulation=f, arm=arm, kappa=k, seed_set=lab, n=len(h), n_failed=nf, cp95_lo=lo, cp95_hi=hi,
                             median_err=float(h.err.median()), q25=float(h.err.quantile(.25)), q75=float(h.err.quantile(.75)),
                             median_phi_jump_rel=float(h.phi_jump_rel.median()), median_D_jump_rel=float(h.D_jump_rel.median()),
                             median_wall_s=float(h.wall_clock_s.median()), median_grad_evals=float(h.grad_evals.median())))
    return pd.DataFrame(rows)


def critical(FC, f, arm, thr, seed_set="10seed"):
    g = FC[(FC.formulation == f) & (FC.arm == arm) & (FC.seed_set == seed_set) & FC.kappa.isin(KAPPAS)].sort_values("kappa")
    if len(g) < len(KAPPAS): return None
    fails = g[g.n_failed >= thr]
    if fails.empty: return dict(critical=">2000", index=len(KAPPAS), monotone=True)
    kc = float(fails.kappa.min()); return dict(critical=kc, index=KAPPAS.index(kc),
                                              monotone=bool((g[g.kappa >= kc].n_failed >= thr).all()))


def precision_table(D, FC):
    rows = []
    for f in ("F1", "F2", "F3", "F4", "F5", "F6"):
        for k in KAPPAS:
            a = series(D, f, "base", k, SEEDS5); b = series(D, f, "C64", k, SEEDS5)
            r = b.err.median() / a.err.median()
            rows.append(dict(design="replica_adam_5seed", formulation=f, kappa=k, median_f32=a.err.median(), median_f64=b.err.median(),
                             ratio=r, fold=max(r, 1 / r), fails_f32=int(a.fail.sum()), fails_f64=int(b.fail.sum()),
                             config_fails_f32=a.fail.sum() >= 3, config_fails_f64=b.fail.sum() >= 3))
    for f in ("F2", "F3"):
        for k in K3:
            for opt, a_arm, b_arm in (("adam", "base", "C64"), ("lbfgs", "Elbfgs", "C64lbfgs")):
                a = series(D, f, a_arm, k, SEEDS10); b = series(D, f, b_arm, k, SEEDS10)
                r = b.err.median() / a.err.median()
                rows.append(dict(design=f"2x2_{opt}_10seed", formulation=f, kappa=k, median_f32=a.err.median(),
                                 median_f64=b.err.median(), ratio=r, fold=max(r, 1 / r), fails_f32=int(a.fail.sum()),
                                 fails_f64=int(b.fail.sum()), config_fails_f32=a.fail.sum() >= 6, config_fails_f64=b.fail.sum() >= 6))
    return pd.DataFrame(rows)


def verdicts(P, FC, PT):
    V = {}
    for iv in ("A", "B", "D", "E"):
        g = P[(P.intervention == iv) & P.prespecified]
        vs = g.verdict.tolist()
        overall = "supported" if all(v == "supported" for v in vs) else "refuted" if all(v == "refuted" for v in vs) else "mixed"
        V[f"H5_{iv}"] = dict(overall=overall, n_configs=len(vs), counts={v: vs.count(v) for v in set(vs)},
                             per_config=[dict(comparison=r.comparison, kappa=r.kappa, n_pred=r.n_pred_sign, n_opp=r.n_opposite,
                                              p=r.p_sign_one_sided, p_holm=r.p_holm_within_intervention, verdict=r.verdict)
                                         for r in g.itertuples()])
    # A: critical contrast of F2 (and F3)
    cc = {}
    for f in ("F2", "F3"):
        for thr in (6, 5):
            b, a = critical(FC, f, "base", thr), critical(FC, f, "A", thr)
            cc[f"{f}_thr{thr}"] = dict(base=b, A=a, raised_by_ge1_step=a["index"] > b["index"])
    V["A_critical_contrast"] = cc
    # B: F1 "no change" (per-seed failure flag identical in >= 9/10 at every kappa > 1)
    # H4 intervention clause
    h4 = {}
    for thr_lab, base_set, thr in (("10seed_ge8", "10seed", 8), ("5seed_ge4", "5seed", 4)):
        g = FC[(FC.formulation == "F2") & (FC.arm == "base") & (FC.seed_set == base_set) & FC.kappa.isin(KAPPAS)]
        ks = g[g.n_failed >= thr].kappa.tolist(); res = []
        for k in ks:
            nb = int(FC[(FC.formulation == "F2") & (FC.arm == "base") & (FC.seed_set == "10seed") & (FC.kappa == k)].n_failed.iloc[0])
            for arm, fn in (("A", "F2"), ("B", "F2norm")):
                ni = int(FC[(FC.formulation == fn) & (FC.arm == arm) & (FC.seed_set == "10seed") & (FC.kappa == k)].n_failed.iloc[0])
                res.append(dict(kappa=k, arm=arm, base_failed_10=nb, int_failed_10=ni, removed=nb - ni, meets=nb - ni >= 3))
        h4[thr_lab] = dict(eligible_kappas=ks, results=res, clause_met=any(r["meets"] for r in res))
    V["H4_intervention_clause"] = h4
    # C criteria
    rep = PT[PT.design == "replica_adam_5seed"]
    flagchg = int((rep.config_fails_f32 != rep.config_fails_f64).sum())
    fa = PT[PT.design == "2x2_adam_10seed"].fold; fl = PT[PT.design == "2x2_lbfgs_10seed"].fold
    V["C"] = dict(replica_median_fold=float(rep.fold.median()), replica_max_fold=float(rep.fold.max()),
                  replica_n_configs=len(rep), replica_config_flag_changes=flagchg,
                  adam_prediction_holds=bool(rep.fold.median() <= 1.25 and flagchg <= 1),
                  twobytwo_median_fold_adam=float(fa.median()), twobytwo_median_fold_lbfgs=float(fl.median()),
                  lbfgs_larger_effect=bool(fl.median() > fa.median()))
    # D criteria
    def nf(f, arm, k): return int(FC[(FC.formulation == f) & (FC.arm == arm) & (FC.seed_set == "10seed") & (FC.kappa == k)].n_failed.iloc[0])
    dd = {str(k): dict(F2=nf("F2", "base", k), F4=nf("F4", "base", k), F4full=nf("F4full", "D", k)) for k in KAPPAS}
    f4full_med = {str(k): float(FC[(FC.formulation == "F4full") & (FC.arm == "D") & (FC.seed_set == "10seed") & (FC.kappa == k)].median_err.iloc[0]) for k in KAPPAS}
    V["D"] = dict(failure_counts_10seed=dd,
                  F4_fails_less_than_F2_at_400_and_2000=all(dd[str(k)]["F4"] < dd[str(k)]["F2"] for k in (400.0, 2000.0)),
                  F4full_median=f4full_med, F4full_all_medians_below_1pct=all(v < 0.01 for v in f4full_med.values()))
    return V


ARM_LABEL = {"base": "baseline (Adam, f32)", "A": "A: rebalancing", "B": "B: residual norm.", "D": "D: F4full",
             "C64": "C: float64", "Elbfgs": "E: L-BFGS", "Ehyb": "E: Adam2k+L-BFGS1k", "C64lbfgs": "C: L-BFGS f64"}


def fig_arms(FC, panels, fname, title):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0), squeeze=False)
    for ax, (f_title, series_list) in zip(axes[0], panels):
        for (f, arm, lab), mk in zip(series_list, "osd^v"):
            g = FC[(FC.formulation == f) & (FC.arm == arm) & (FC.seed_set == "10seed")].sort_values("kappa")
            ax.errorbar(g.kappa, g.median_err, yerr=[g.median_err - g.q25, g.q75 - g.median_err], marker=mk, capsize=3, label=lab)
        ax.axhline(0.05, color="k", ls="--", lw=0.8, label="5% failure threshold")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("contrast κ = ε₂/ε₁ (–)")
        ax.set_ylabel("primary band field error (–)\nmedian, bars = IQR, 10 seeds"); ax.set_title(f_title, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=11); fig.tight_layout()
    fig.savefig(ROOT / "results/figures" / fname, dpi=130); plt.close(fig)


def fig_seed_scatter(D, P, fname):
    sel = P[P.prespecified]
    groups = list(sel.intervention.unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 4.2), squeeze=False)
    for ax, iv in zip(axes[0], groups):
        g = sel[sel.intervention == iv]
        for r, mk in zip(g.itertuples(), "osd^v<>ph*Xo"):
            fx, ax_ = r.x.split("/"); fy, ay = r.y.split("/")
            x = series(D, fx, ax_, r.kappa, SEEDS10).err.values; y = series(D, fy, ay, r.kappa, SEEDS10).err.values
            ax.scatter(np.clip(y, 1e-7, 1e2), np.clip(x, 1e-7, 1e2), s=14, marker=mk, label=f"{r.comparison.split(' vs')[0]} κ={r.kappa:g}")
        ax.plot([1e-7, 1e2], [1e-7, 1e2], "k-", lw=0.7); ax.axhline(.05, color="gray", ls=":"); ax.axvline(.05, color="gray", ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("reference error (same seed) (–)"); ax.set_ylabel("intervention error (–)")
        ax.set_title(f"Intervention {iv}: below diagonal = predicted sign", fontsize=9); ax.legend(fontsize=5.5, ncol=2)
    fig.suptitle("Phase 11 pre-specified paired comparisons (P1, primary band field error, one point per seed)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "results/figures" / fname, dpi=130); plt.close(fig)


def fig_precision(PT, fname):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for des, mk, lab in (("replica_adam_5seed", "o", "Adam, replica (5 seeds, 36 configs)"),
                         ("2x2_adam_10seed", "s", "Adam, F2/F3 (10 seeds)"), ("2x2_lbfgs_10seed", "^", "L-BFGS, F2/F3 (10 seeds)")):
        g = PT[PT.design == des]; ax.scatter(g.median_f32, g.ratio, marker=mk, label=lab, alpha=.8)
    ax.axhspan(1 / 1.25, 1.25, color="green", alpha=.12, label="±1.25× band (prediction)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("median primary error, float32 (–)")
    ax.set_ylabel("median error ratio float64 / float32 (–)"); ax.legend(fontsize=7)
    ax.set_title("Intervention C: effect of float64 on median primary band field error (P1)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "results/figures" / fname, dpi=130); plt.close(fig)


def main():
    D = load()
    P = build_paired(D); FC = failure_counts(D); PT = precision_table(D, FC)
    cc = []
    for f, arms in (("F1", ["base"]), ("F2", ["base", "A", "C64"]), ("F3", ["base", "A", "C64"]), ("F4", ["base", "C64"]),
                    ("F5", ["base", "C64"]), ("F6", ["base", "C64"]), ("F1norm", ["B"]), ("F2norm", ["B"]), ("F4full", ["D"])):
        for arm in arms:
            for ss, thrs in (("10seed", (6, 5)), ("5seed", (3,))):
                for thr in thrs:
                    c = critical(FC, f, arm, thr, ss)
                    if c: cc.append(dict(formulation=f, arm=arm, seed_set=ss, threshold=thr, **c))
    CC = pd.DataFrame(cc)
    V = verdicts(P, FC, PT)
    # B, F1: per-seed flag identity at every kappa > 1
    same = {}
    for k in KAPPAS[1:]:
        a = series(D, "F1", "base", k, SEEDS10).fail.values; b = series(D, "F1norm", "B", k, SEEDS10).fail.values
        same[str(k)] = int((a == b).sum())
    V["B_F1_no_change"] = dict(identical_flags_per_kappa=same, holds=all(v >= 9 for v in same.values()))
    # A cost accounting
    A = D[D.arm == "A"]
    V["A_cost"] = dict(median_wall_s_A=float(A.wall_clock_s.median()),
                       median_wall_s_base_same_configs=float(D[(D.arm == "base") & D.formulation.isin(["F2", "F3"])].wall_clock_s.median()),
                       extra_term_grads_per_run=sorted(set(int(n.split("rebal_term_grads=")[1].split(";")[0]) for n in A.notes)))
    E = D[D.arm.isin(["Elbfgs", "Ehyb", "C64lbfgs"])]
    V["E_budget"] = dict(min_evals=int(E.grad_evals.min()), max_evals=int(E.grad_evals.max()))
    out = ROOT / "results/processed"; out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "phase11_paired.csv", index=False); FC.to_csv(out / "phase11_failure_counts.csv", index=False)
    CC.to_csv(out / "phase11_critical_contrast.csv", index=False); PT.to_csv(out / "phase11_precision.csv", index=False)
    json.dump(V, open(out / "phase11_verdicts.json", "w"), indent=1, default=str)
    (ROOT / "results/figures").mkdir(exist_ok=True)
    fig_arms(FC, [("F2", [("F2", "base", ARM_LABEL["base"]), ("F2", "A", ARM_LABEL["A"]), ("F2norm", "B", ARM_LABEL["B"])]),
                  ("F3", [("F3", "base", ARM_LABEL["base"]), ("F3", "A", ARM_LABEL["A"])]),
                  ("F1", [("F1", "base", ARM_LABEL["base"]), ("F1norm", "B", ARM_LABEL["B"])])],
             "p1_interventions_AB.png", "Interventions A (gradient rebalancing) and B (residual normalisation), P1")
    fig_arms(FC, [("hard interface (D)", [("F2", "base", "F2 (soft continuity + flux)"), ("F4", "base", "F4 (hard continuity)"),
                                          ("F4full", "D", "F4full (all conditions hard)")])],
             "p1_interventions_D.png", "Intervention D: hard interface conditions, P1")
    fig_arms(FC, [(f, [(f, "base", "Adam 3000"), (f, "Elbfgs", "L-BFGS ≤3000 evals"), (f, "Ehyb", "Adam 2000 → L-BFGS ≤1000")])
                  for f in ("F2", "F3", "F5")], "p1_interventions_E.png", "Intervention E: optimiser (equal function+gradient budget), P1")
    fig_precision(PT, "p1_interventions_C.png"); fig_seed_scatter(D, P, "p1_interventions_paired.png")
    return P, FC, CC, PT, V


if __name__ == "__main__":
    P, FC, CC, PT, V = main()
    print(json.dumps({k: (v if k not in ("H5_A", "H5_B", "H5_D", "H5_E") else {kk: vv for kk, vv in v.items() if kk != "per_config"})
                      for k, v in V.items()}, indent=1, default=str))
