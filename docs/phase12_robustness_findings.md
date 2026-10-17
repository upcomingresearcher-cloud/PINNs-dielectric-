# Phase 12 — Robustness of the P1 conclusions (findings)

This study varies one factor at a time on P1, for all six formulations, κ ∈ {1, 80, 2000} and seeds 0–4.
- **Reference:** the primary rows on the same subset.
- **Code:** `python -m src.analysis.robustness`.
- **Outputs:** `results/processed/p1_robustness_{counts,summary}.csv`, `p1_robustness_verdict.json`, and `results/figures/p1_robustness_{failcounts,errors}.png`.
- **Operationalisation:** frozen in the change log before any run.

## Runs

There are 870 robustness rows, all complete, with no duplicates and no non-finite results:
- **8 variations:** 720 runs (90 each).
- **Parameter-matched two-network extra:** 45 runs (F2, F4 and F6 at width 22, giving 2,158 parameters, against 2,209 for F1 and 2,241 for F3/F5).
- **16 extra band points per side:** 90 runs, plus 15 F5 reruns after a bug fix (below). The original 15 F5 rows are kept and marked superseded.

Boundary sampling cannot vary in 1D because the boundary is the two fixed endpoints. Precision and optimiser were covered in Phase 11 (C, E).

## Failed runs out of 15, per formulation (κ ∈ {1, 80, 2000} × 5 seeds)

| Setting | F1 | F2 | F3 | F4 | F5 | F6 | τ-b vs baseline | median R (95% CI) | concealed failures | median wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 10 | 6 | 10 | 10 | 10 | 3 | 1 | 2.24 (1.49–3.11) | 7 | 1.9 |
| width 16 | 10 | 9 | 10 | 10 | 10 | 5 | 1.00 | 1.84 (1.30–2.76) | 7 | 1.8 |
| width 64 | 10 | 5 | 10 | 10 | 10 | 0 | 1.00 | 2.37 (1.66–3.15) | 8 | 4.3 |
| depth 2 | 10 | 7 | 10 | 10 | 10 | 5 | 1.00 | 2.18 (1.44–2.86) | 7 | 1.6 |
| depth 5 | 10 | 5 | 10 | 10 | 10 | 2 | 1.00 | 2.25 (1.39–3.15) | 10 | 4.0 |
| 32 points/material | 10 | 6 | 10 | 10 | 11 | 3 | 0.87 | 2.28 (1.49–3.10) | 7 | 1.4 |
| 512 points/material | 10 | 6 | 10 | 10 | 10 | 3 | 1.00 | 2.24 (1.45–3.12) | 7 | 6.2 |
| lr 3×10⁻³ | 10 | 5 | 9 | 10 | 10 | 1 | 0.87 | 2.49 (1.71–3.41) | 10 | 2.4 |
| 12,000 steps | 10 | **1** | 7 | 10 | 7 | 2 | **0.65** | **5.62 (3.64–19.6)** | **16** | 9.9 |
| *extra:* width 22, matched params | 10 | 6 | 10 | 10 | 10 | 4 | 1.00 | 2.28 (1.47–2.84) | 6 | 2.2 |
| *extra:* 16 band points/side | 10 | 6 | 10 | 10 | 10 | 3 | 1.00 | 2.02 (1.33–3.16) | 7 | 2.6 |

Two definitions apply to this table:
- **R** is the median over κ > 1 runs of the primary band field error divided by the global relative L2 potential error (the H2 definition).
- **Concealed failures** are runs with κ > 1 that fail the primary criterion while their global potential error is below 5%.

For the width-22 extra, the ranking combines the width-22 F2/F4/F6 runs with the baseline F1/F3/F5 runs. Per-κ medians and IQRs are in `p1_robustness_counts.csv`.

## H6 verdict (pre-registered rule): **inconclusive**

- **Ranking criterion: met in 8/8 variations** (needed ≥ 6/8). τ-b is ≥ 0.6 everywhere; it is 1.00 in 5 variations and lowest at 12,000 steps (0.65). The order holds throughout: F1, F3, F4 and F5 fail most, and F6 and F2 fail least. F6 has no more failures than F2 in every variation except 12,000 steps, where F2 has 1 and F6 has 2.
- **Masking criterion (median R > 3): met in 1/8** (12,000 steps only; needed ≥ 6/8).
- **Refutation rule (τ < 0.6 in ≥ 3/8): not triggered.**

**Context for the R shortfall** (measured, not a re-interpretation of the rule):
- **The baseline also falls short of 3.** On this κ subset the baseline reference has R = 2.24, CI 1.49–3.11. In Phase 6, H2 measured R = 2.39 (CI 1.73–3.18) over all κ > 1, and H2 was inconclusive for the same reason.
- **R shrinks as κ grows.** Baseline medians by κ are 3.38 (κ = 5), 3.40 (20), 2.48 (80), 2.29 (400) and 2.08 (2000). The robustness subset contains only κ = 80 and 2000.
- **R is stable across settings.** In 7 of 8 variations it lies within 1.84–2.49, inside or overlapping the reference CI. The masking effect therefore does not depend on the setting. What fails is the absolute threshold: R > 3 is not reached on this problem at κ ≥ 80, whatever the setting.

## Notable measured effects

1. **Longer training (12,000 steps) changes the most.** It does not reverse the ranking.
   - F2 failures drop from 6 to 1 and F3/F5 from 10 to 7.
   - Concealed failures double, from 7 to 16, and median R rises to 5.6. The runs that still fail have a lower global potential error but not a proportionally lower band field error.
   - Measured implication: with more training, the global potential error becomes a *less* reliable indicator of band field accuracy.
2. **Capacity matters most for F2/F6.** Width 16 raises their failures (F2 from 6 to 9, F6 from 3 to 5), and width 64 lowers them (F2 from 6 to 5, F6 from 3 to 0). F1, F3, F4 and F5 fail 10/15 in every capacity setting; their failures sit at κ = 80 and 2000 and are not capacity-limited at these sizes.
3. **Parameter matching does not explain F2/F6's advantage.** At width 22 (2,158 parameters, below F1's 2,209), F2 has 6 failures and F6 has 4, against 10 for every single-network formulation.
4. **Collocation density (32 or 512 points per material) and extra band points** leave failure counts essentially unchanged. The one difference is a single F5 failure at κ = 1 with 32 points.

## Bug found and fixed during this phase (details in the change log)

- **Cause:** the F5 Ritz energy used |Ωᵢ| times an equal-weight mean. That is an unbiased integral only for uniform sampling. With the extra band points, the band (10% of each material) held 20% of the points, so F5 minimised a band-overweighted functional.
- **Symptom:** at κ = 1, all 5 seeds failed with primary errors of 0.51–0.52, a Monte Carlo energy of 0.94 J (below the true minimum) and a grid energy of 1.03 J.
- **Fix:** importance weights 1/ρ(x) when extra points are present; the path without extra points is unchanged and bit-identical.
- **Tests and reruns:** three regression tests were added and the 15 F5 runs were rerun. After the fix, F5 with band points passes at κ = 1 (errors 0.001–0.012) and its failure counts match the baseline.
- **Affected rows:** only F5 band-point rows; residual formulations are unaffected.
- **Note:** a fixed-point Monte Carlo energy is sensitive to how the points are distributed, a third instance after F5's quadrature overfitting in P2 and in Phase 11 E.

## Interpretation (separate from measurements)

The formulation ranking is a robust property of P1 under these eight settings. The size of the masking effect is also robust, but it sits around R ≈ 2, not above 3. The paper should report the masking effect as stable at about 2×, not as meeting the pre-registered 3× threshold.

The 12,000-step result suggests that some of the "soft-constraint" failures at 3,000 steps are optimisation-budget failures. This matches Phase 11 E. The longer budget does not remove the high-contrast failures of F1, F3, F4 or F5.

## Limitations

- All results are on P1 only, with 5 seeds per cell (failure counts out of 15 per formulation), and one factor changed at a time, so no interactions were tested.
- The masking statistic covers only κ = 80 and 2000.
- The width-22 extra mixes width-22 two-network runs with width-32 single-network baseline runs in one ranking (descriptive).
- Wall-clock times come from a shared CPU sandbox and are comparable only within a setting.
