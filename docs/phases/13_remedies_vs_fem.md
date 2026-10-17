# Phase 13 — Remedies vs interface-conforming FEM on P2 and P5 (findings; H7)

- **Code:** `python -m src.analysis.remedy_vs_fem`.
- **Outputs:** `results/processed/p13_{pinn_time_to_target,comparison}.csv`, `results/processed/p13_h7_verdict.json`, `results/figures/p13_error_vs_cost.png`.
- **Design:** frozen in `docs/protocol_change_log.md` (Phase 13) before any remedy run. One analysis addition, the logging-resolution check, was logged before the verdict.

## Runs

There are 110 new remedy rows, all complete, with no duplicates and no technical failures.

| Remedy | Formulations | κ | Runs |
|---|---|---|---|
| L-BFGS (≤ 3,000 function+gradient evaluations; 2,981–2,982 used) | F2, F3, F6 | 1, 80, 2000 | 90 |
| Adam for 20,000 steps | F2, F6 | 80 | 20 |

Each set covers P2 and P5 with 5 seeds.
- **Baseline PINN rows:** taken from Phases 7 and 10 (Adam 3,000 steps, all six formulations).
- **FEM:** the conforming records in `fem_runs_2d.csv`. P5's finest level (h = 0.05) is its reference and is excluded.
- **Configurations analysed:** 64, i.e. (problem, formulation/remedy, κ) wherever both FEM and the PINN configuration exist, with κ ∈ {0.0005, 1, 80, 2000}.

## H7 verdict

| Scope | Verdict |
|---|---|
| All 64 configurations | **supported at logging resolution**: clause 1 holds in 64/64 at the logged first-hit times and clause 2 in 64/64; no configuration refutes |
| Excluding the degenerate κ = 1 (FEM exact to round-off) | **supported** |

The "logging resolution" qualifier affects 10 configurations, all at κ = 1. PINN errors are logged every 250 steps, so a first-hit time is an upper bound. In these 10, the PINN reaches 1% by the first logged record, so the lower bound is 0 and the ≥ 10× margin cannot be resolved; at the logged times the ratios are 21–139×. At κ = 1 conforming FEM is exact at every mesh, costing 0.0135 s on P2 and 0.038 s on P5, so this case is degenerate in any event.

## Measured results

1. **No PINN configuration reaches 1% band field error at any κ ≠ 1.** This holds for the baseline, L-BFGS and 20,000-step Adam alike, in all 5 seeds of every configuration. Conforming FEM reaches 1% in 0.65–4.75 s (P2) and 0.10–0.58 s (P5).
2. **Reaching 5% at κ = 80, P5 only.** L-BFGS with F2 or F6 reaches it in 5/5 seeds (median 4.81 s and 4.96 s). 20,000-step Adam reaches it in 4/5 seeds (median 46.7 s and 48.5 s). Conforming FEM reaches 5% in 0.042 s, so the PINN takes about 115× longer with L-BFGS and about 1,100× with 20,000-step Adam. On P2, no PINN reaches 5% at any κ ≠ 1.
3. **Remedies on P2 at κ ≥ 80: no rescue.** F2/F6 median errors are 0.634 (Adam 3,000), 0.635 (L-BFGS) and 0.635 (Adam 20,000) at κ = 80, and 0.676/0.707 → 0.645 with L-BFGS at κ = 2000. F3 goes from 1.00 to 0.89 at κ = 80. This contrasts with P1, where L-BFGS cut F2 failures at κ = 2000 from 9/10 to 4/10 and rescued F3 at κ = 80 (Phase 11 E).
4. **Remedies on P5:**
   - **κ = 80:** L-BFGS lowers F2/F6 from 0.070/0.069 to 0.049/0.049, just below the 5% failure threshold (5/5 seeds pass).
   - **κ = 2000:** L-BFGS lowers F2 from 1.21 to 0.072 and F6 from 1.13 to 0.114; both still fail.
   - **F3:** improves at κ = 1 (0.95 → 0.019) but not at κ ≥ 80.
5. **κ = 1:** L-BFGS lowers the error 54–116× for F2, F3 and F6 (for example P2 F2 from 1.8×10⁻³ to 2.4×10⁻⁵). FEM remains exact.
6. **Error at equal cost (clause 2).** Every conforming FEM mesh costs at most 4.75 s, less than every PINN's median training time (2.6–95 s), so the comparison mesh is always the finest. Its error is lower than the PINN median error in all 64 configurations. Away from κ = 1, the gap ranges from 86× (P5 F2 L-BFGS at κ = 80: 0.049 against 5.6×10⁻⁴) to 1.7×10⁵×.
7. **Non-conforming FEM** (P2, descriptive) does not converge in the band at κ ≠ 1 (errors 5–580). This is the known failure of interface-unaware meshes, and it is why H7 names interface-conforming FEM.

## Timing method and caveats

- **PINN time to target:** wall_clock_s × (first-hit step ÷ total gradient evaluations), assuming a constant cost per step; for L-BFGS, steps are function+gradient evaluations. JIT compile time is excluded and reported separately, as is evaluation/logging overhead. Most runs reuse a cached compilation (compile_s ≈ 0); uncached runs take about 2–6 s.
- **FEM time:** gmsh meshing + scikit-fem assembly + sparse direct solve (`mesh_plus_solve_s`); `solve_only_s` is also recorded.
- **Hardware:** single-threaded CPU. P2 FEM ran on a 2.10 GHz Xeon and P5 FEM on a 2.80 GHz Xeon; PINN rows record their own hardware string. The margins (≥ 21× at κ = 1 and "never reached" elsewhere) far exceed plausible hardware differences.
- **Different computational structures:** PINN and FEM work differently, and equal gradient evaluations do not imply equal wall-clock. P5 FEM errors are measured against the finest FEM level, as are P5 PINN errors.

## Interpretation (separate from measurements)

On curved (P2) and thin-film (P5) dielectric interfaces at κ ≥ 80, none of the PINN variants tested reaches 1% band field accuracy. That covers six formulations, a stronger optimiser and a 6.7× longer budget. Interface-conforming quadratic FEM reaches 1% in under 5 s.

The P1 result that the optimiser is the bottleneck does not carry over to P2. There, F2/F6 plateau at a band error of about 0.63 whatever the optimiser or budget, which points to a representation or loss-landscape limit rather than an optimisation-budget limit on this geometry. This is consistent with, but does not prove, the hypothesis that interface curvature, not contrast alone, is the obstacle.

## Limitations

- Only two remedies were tested on 2D, and A/B were not carried over because they were refuted on P1.
- The 20,000-step budget was run at κ = 80 only.
- κ = 5/20, where Adam P5 F2/F6 pass the 5% criterion, has no FEM refinement records, so the PINN's most favourable regime is absent from the time-to-target comparison.
- Time-to-target resolution is 250 steps.
- The comparison uses CPU only; GPU PINN training would change the absolute times, not whether 1% is reached.
