# Phase 8: P3 Line Charge Near a Planar Interface — Results
Runs (complete): 165 full-sweep (F1, F2, F6 × 11 contrasts × 5) + 45 reduced (F3–F5 × {1,80,2000} × 5) + 15 pre-registered
sensitivity (F2, material-wise subtraction) + 60 regression. 60 earlier rows are status=failed_technical (AOT shape bug,
see change log); kept, excluded from statistics. FEM: 20 body-fitted records. Gates: P3 self-tests 17/17; 2D Gate 3 27/27.

## Measured
- kappa > 1 (material below has higher eps): every formulation fails every contrast, 5/5 seeds.
- Inverse contrast (kappa < 1): F2 passes at kappa 0.2 and 0.05 (0/5 failed), 1/5 at 0.0125, 5/5 at 0.0025 and 0.0005. F6 fails all inverse contrasts (median error 0.40 at 0.2 up to 52 at 0.0005). F1 fails all.
- Median band field error at kappa = 80: F1 31, F2 6.1, F3 3.5, F4 8.5, F5 1.3e4, F6 11.
- Near-source ring vs far field (kappa = 80, median E error): F3 0.07 vs 0.31; F2 0.72 vs 2.7.
- **Pre-registered decomposition sensitivity (F2):** material-wise subtraction reduces median error from 6.1 to 0.61 (kappa 80) and 121 to 21 (kappa 2000), but still fails 5/5; at kappa 0.0125 it is slightly worse (0.047 vs 0.032; 2/5 vs 1/5 failed).
- H2 (P3): median R = 3.95; R > 1 in 100% of runs; concealed failures 2.4%.
- H8 vs P1: failure counts identical (10/10) for all formulations on shared kappa -> tau undefined; tau on median error 0.07.
- Regression (phi-only, training variable w) at kappa = 80: 4.6–19.6; at kappa = 1: ~1e-3.
- FEM (body-fitted, w with interface load): kappa = 80 band error 0.28 (2013 DOFs), 0.080 (7621), 0.021 (29929, 0.68 s).

## Interpretation (not demonstrated)
The global singular subtraction is a substantial, measured contributor to high-kappa failure (an order of magnitude in error for F2), consistent with the pre-run cancellation analysis, but not the only cause. The asymmetry between kappa > 1 and kappa < 1 for F2 suggests difficulty grows when the high-eps material does not contain the source; this is not tested further.
