# Phase 9: P4 Two-Material Corner — Results
Runs (complete, unique): 165 full-sweep (F1, F2, F6 × 11 contrasts × 5) + 45 reduced (F3–F5 × {1,80,2000} × 5) + 60 regression.
5 duplicate rows from a concurrent-execution incident (identical results) kept in raw data, de-duplicated in analysis; timing of
8 runs flagged (change log). FEM: 20 body-fitted records. Gates: P4 self-tests pass; exact-loss consistency ~1e-25 (after fixing a
clip-derivative bug in the JAX verification function; no training/evaluation path was affected).

## Measured
- Every formulation fails at every kappa != 1 (5/5 seeds), including all inverse contrasts; F5 also fails at kappa = 1 (median 0.69; the same fixed-point quadrature-overfitting mechanism seen in P2).
- Median band field error at kappa = 80: F1 14, F2 0.69, F3 1.0, F4 17, F5 1.7, F6 0.70. Best inverse-contrast result: F2 0.070 at kappa 0.05 (still failing).
- **H10 (pre-registered, operationalised before computing): REFUTED.** F2 underestimates the near-corner field at every kappa > 1 on both bisectors (median signed |E| error -0.58 to -0.88 at r = 0.02/0.05), consistent with the hypothesis; but F1 OVERestimates it on the high-eps bisector (+5.7 at kappa = 80, +18.6 at 2000) while underestimating on the low-eps bisector (-0.78 to -0.98). The pre-registered rule requires both F1 and F2 below -10% everywhere, so H10 is refuted. F4 and F5 also overestimate on the high-eps bisector at high kappa.
- H2 (P4): median field/potential ratio R = 4.0 (> 3); R > 1 in 100% of runs; concealed failures 11.7%.
- H8 vs P1: Kendall tau on failure counts 0.30; on median error 0.60.
- Regression (phi-only) at kappa = 80: REG_F2 0.44, REG_F4 7.4, REG_F3 9.9, REG_F1 13.9.
- Body-fitted P2-FEM at kappa = 80: band error 0.10 (557 DOFs), 0.040 (7645), 0.013 (29957), 0.0024 (118853) despite the singularity (the corner disc r < 0.02 is excluded from the metric).

## Interpretation (not demonstrated)
"Smooth networks underestimate the corner field" is not a general statement: the sign of the near-corner error depends on the formulation and on which material the measurement ray lies in. Two-network soft coupling (F2/F6) uniformly underestimates; single-network/hard-constrained designs overshoot in the high-eps wedge.
