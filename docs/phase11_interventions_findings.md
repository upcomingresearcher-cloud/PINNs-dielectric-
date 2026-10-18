# Phase 11 — Controlled interventions A–E on P1 (findings)

All numbers come from `results/raw/runs.csv`, processed by `python -m src.analysis.interventions`. Outputs: `results/processed/phase11_*.csv`, `results/processed/phase11_verdicts.json`, and `results/figures/p1_interventions_{AB,C,D,E,paired}.png`. The operationalisation (configuration sets, thresholds, L-BFGS budget rule) was frozen in `docs/protocol_change_log.md` (v1.6.0) before any intervention run.

## Runs

There are 1,155 complete intervention rows, with no technical failures, no duplicates and no non-finite results.

| Run set | Rows |
|---|---|
| Baseline, seeds 5–9 | 135 |
| A (gradient rebalancing) | 120 |
| B (residual normalisation) | 120 |
| D (F4full) | 60 |
| E, L-BFGS | 90 |
| E, Adam 2000 → L-BFGS 1000 | 90 |
| C, float64 Adam (corrected) | 210 |
| C, float64 L-BFGS (corrected) | 60 |
| C, superseded rows (kept) | 270 |

- **Paired baseline:** seeds 0–4 are the primary rows. They were reproduced bit-for-bit by the unchanged code path, and this is a regression test.
- **L-BFGS budget:** runs used 2,981–3,000 function+gradient evaluations each, never more than 3,000.
- **Cost of A:** each A run adds 1,500 (F3) or 1,800 (F2) per-term gradient evaluations, through 300 Jacobians. Median wall-clock is 3.7 s versus 2.1 s for the baseline F2/F3 runs. A is therefore not cost-matched.

## Failure counts (primary criterion, 10 matched seeds)

| Formulation / arm | κ=1 | 5 | 20 | 80 | 400 | 2000 |
|---|---|---|---|---|---|---|
| F2 baseline | 0 | 0 | 0 | 2 | 5 | 9 |
| F2 + A (rebalancing) | **10** | **10** | 4 | 2 | 6 | 9 |
| F2norm (B) | 0 | 0 | 0 | 2 | 7 | 9 |
| F2 + L-BFGS (E) | 0 | – | – | 0 | – | 4 |
| F2 + Adam2k→L-BFGS1k (E) | 0 | – | – | 0 | – | 9 |
| F3 baseline | 0 | 0 | 5 | 10 | 10 | 10 |
| F3 + A | 4 | 7 | 8 | 6 | 3 | 9 |
| F3 + L-BFGS (E) | 0 | – | – | 0 | – | 6 |
| F3 + Adam2k→L-BFGS1k (E) | 0 | – | – | 0 | – | 10 |
| F5 baseline | 0 | – | – | 10 | – | 10 |
| F5 + L-BFGS (E) | **8** | – | – | 5 | – | 3 |
| F5 + Adam2k→L-BFGS1k (E) | 5 | – | – | 2 | – | 2 |
| F4 (D comparator) | 0 | 0 | 10 | 10 | 10 | 10 |
| F4full (D) | 0 | 0 | 0 | 0 | 0 | 0 |
| F1 baseline / F1norm (B) | 0 / 0 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 |

"–" means not part of the pre-registered design. Medians, IQRs, Clopper–Pearson intervals and bootstrap CIs of the paired median differences are in the processed CSVs.

## Pre-registered verdicts

| Item | Verdict | Evidence |
|---|---|---|
| **H5-A** (rebalancing lowers error, F2 and F3 × 6 κ) | **mixed**: 3 supported, 9 refuted | F2 is refuted at all 6 κ. F3 is refuted at κ ≤ 20 and supported at κ = 80, 400 and 2000, with 9/10, 10/10 and 10/10 seeds improving. At κ = 400 the F3 median falls from 1.06 to 0.033. |
| A: F2 critical contrast raised by ≥1 grid step | **no** | F2 at 10 seeds, ≥6/10 rule: 2000 → 1 (non-monotone). The ≥5/10 sensitivity rule gives the same result. |
| **H5-B** (F2norm lowers error at κ = 80, 400, 2000) | **refuted** (3/3) | 1, 1 and 2 seeds out of 10 improve. |
| B: F1 "no change" | **holds** | Per-seed failure flags are identical in 10/10 seeds at every κ > 1. |
| **H5-D** (F4 beats F2 at κ = 400, 2000) | **refuted** (2/2) | F4 fails 10/10 seeds at both contrasts, against F2's 5/10 and 9/10. |
| D: F4full < 1% at all κ | **holds** | Medians are 1.2×10⁻⁶ to 9.9×10⁻⁴. Trivially expected: the exact P1 solution lies in F4full's buffer space (protocol Sec. 5 caveat). |
| **H5-E** (L-BFGS variants beat Adam, F3 and F5 × 3 κ × 2 variants) | **mixed**: 9 supported, 2 refuted, 1 inconclusive | F3: all 6 supported, 10/10 seeds each; the κ = 80 median falls from 0.55 to 3.5×10⁻⁵. F5: refuted at κ = 1 for both variants; L-BFGS at κ = 80 is inconclusive (8/10); the other 3 are supported. |
| **C** (Adam: median fold ≤1.25×, flag change in ≤1 configuration) | **holds** | Median fold change over 36 configurations is 1.0001 (max 1.07). No configuration-level failure flag changes. |
| C: larger effect under L-BFGS | **holds** | Median fold change 3.4 under L-BFGS against 1.0008 under Adam. float64 is lower in all 6 L-BFGS cells, with failures at κ = 2000 going from 4 to 1 (F2) and 6 to 3 (F3). |
| **H4 intervention clause** (A or B removes ≥3 F2 failures where the baseline fails ≥8/10) | **not met** | The only eligible κ is 2000 (9/10 baseline failures). A and B both leave 9/10 failing. The 5-seed reading gives the same κ and outcome. |

**Updated H4 status: refuted.** Its regression criterion was already refuted in Phase 6, and now the intervention clause also fails: neither rebalancing nor residual normalisation rescues F2 at high contrast. The change of optimiser does. L-BFGS cuts F2 failures at κ = 2000 from 9/10 to 4/10 (descriptive, F2 not pre-specified under E), and to 1/10 in float64. This is consistent with H4's claim that F2's high-κ failures are optimisation failures, but the pre-registered criterion named A/B, so the verdict stays "refuted by criterion". The E evidence is reported separately.

## Investigations of unexpected results

None of the following were bugs, except item 4.

1. **A degrades F2 at low κ.** The rule λ̂_i = ‖∇L_pde‖/‖∇L_i‖ gives the largest weights to terms that are already satisfied. Under it, flux reaches λ ≈ 5×10⁴ and bc_left ≈ 3×10⁴, while the still-violated continuity term is pushed down to λ ≈ 0.05. Runs end with a potential jump of about 30%. The implementation was checked against an independent per-term `jax.grad` reference (agreement 6×10⁻⁸), and this is now a regression test. This is a property of the pre-registered L2-norm variant. It should not be generalised to the original max/mean rule of Wang et al., which was not tested.
2. **F5 + L-BFGS fails at κ = 1.** In the 7 failing seeds, the Monte Carlo energy on the 256 fixed points falls to 10⁻⁶–0.1 of the true minimum D/2. Meanwhile the evaluator-grid energy is 370–880 times too large. This is quadrature overfitting, the mechanism already identified for F5 in P2, here triggered by a stronger optimiser in 1D. Under Adam the Monte Carlo energy stays at the true value.
3. **F1norm magnitudes.** F1norm converges to the uniform-ε (wrong) solution at every κ: its loss is 2.147×10⁻⁶ at all κ and its band-2 D error is about κ/2. Baseline F1's smaller errors at κ ≥ 400 come from an incompletely minimised ε₂²-weighted PDE term, not from interface information.
4. **BUG: float64 initialisation differed from float32.** It was found in the first C analysis and fixed. All 270 C rows were rerun; the originals are kept as "superseded". Details are in the change log. The superseded comparison is itself informative: changing only the initial draw produced a median fold change of 1.38, a maximum of 20.7, and 2 configuration-level flag changes. The genuine precision effect under Adam is 1.0001 and 1.07. **Initialisation variability exceeds the precision effect under Adam by more than an order of magnitude on this problem.** This supports the protocol's no-best-seed rule.

## Interpretation (separate from the measurements)

- **Optimisation versus formulation.** The interventions that changed the optimiser (E) or fixed the representation (D, F4full) moved outcomes far more than re-weighting the loss (A, B). This is consistent with the high-κ failures of F2/F3 on P1 being largely optimisation failures within a fixed 3000-evaluation budget. It is not proof: L-BFGS changes both the step direction and the effective step size.
- **What it does not show.** That L-BFGS rescues F3 at κ = 80 on P1, whose exact solution is piecewise linear, does not show that it will on the 2D problems. The Phase 13 remedy comparison should test it there before any general claim.
- **F5.** Its gains under L-BFGS at high κ coexist with quadrature overfitting at κ = 1. A stronger optimiser exposes the fixed-quadrature weakness rather than removing it.
- **Precision.** Under Adam, precision is immaterial here. Under L-BFGS, float64 consistently helps, matching the direction reported in the precision literature cited in the gap report. The magnitudes are problem-specific.

## Limitations

- A is not cost-matched (extra Jacobians), and only the L2-norm variant of the rebalancing rule was tested.
- B normalises PDE residuals only. Scaling the flux by one scalar would reproduce F6's weighting.
- E uses one line-search configuration, optax zoom with at most 20 steps.
- D's F4full result is trivial on P1.
- The H5 configuration sets were fixed by the change log before any run, not in the original protocol.
- No multiplicity correction is used for verdicts. Holm-adjusted p-values (within each intervention) are reported in `phase11_paired.csv`. All 9 supported E configurations stay below 0.05 after adjustment (0.012–0.043). For A, F3 at κ = 400 and 2000 stay below 0.05 (0.012), but **F3 at κ = 80 (9/10 seeds, p = 0.011) does not survive Holm adjustment (0.107)**. The minimum one-sided p for 10/10 seeds is 0.00098.
