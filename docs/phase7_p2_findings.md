# Phase 7: P2 Dielectric Cylinder — Results
Runs: 165 full-sweep (F1, F2, F6 × 11 contrasts incl. 5 inverse × 5 seeds) + 45 reduced (F3, F4, F5 × {1, 80, 2000} × 5)
+ 60 regression + 1 pilot-labelled bug check; 0 technical failures. FEM: 40 records (4 contrasts × 5 meshes × conforming/non-conforming).
Validation: 2D Gate 1 17/17, 2D Gate 3 22/22, FEM P2-element reproduction test (after fixing a gradient bug).

## Measured
- **Every formulation fails at every kappa != 1** (5/5 seeds), including all inverse contrasts. At kappa = 1 all pass except F5.
- Median band field error at kappa = 80: F1 30, F2 0.63, F3 1.0, F4 0.84, F5 1.2, F6 0.63.
- **F5 fails even at kappa = 1**: its Monte-Carlo energy estimate on the fixed points (0.59–1.59) falls BELOW the true minimum energy (2.0), while the true energy of its predictions is 4.2–75. The Ritz loss overfits the fixed quadrature points.
- **F2/F6 plateau at a stationary point**: at kappa = 5, a float64 run of 20000 steps stays at primary error 0.486 from step 4000 to 20000. Per-term gradients are O(0.1) but cancel. Continuity is the unsatisfied condition (phi jump ~7% of V_c), as in P1 (H3).
- **Not a loss bug**: minimising the F2 loss over a 3-parameter family containing the exact solution (BFGS) recovers the exact parameters to 6 digits with loss ~1e-31 (kappa = 5 and 80).
- H2 on P2: median field/potential error ratio 7.5 (> 3); R > 1 in 97% of runs; but only 1.7% of failures are concealed (potential error is also large).
- H8 partial (P1 vs P2 on kappa {1, 80, 2000}): Kendall tau on failure counts 0.30 (< 0.6, ties dominate since nearly everything fails in P2); on median error 0.73.
- Regression (phi-only) at kappa = 80: REG_F2 0.63, REG_F4 3.9, REG_F1 17, REG_F3 12.8.
- Body-fitted P2-FEM at kappa = 80: 7.5% band error at 2093 DOFs (0.03 s), 3.0% at 7869 DOFs (0.13 s), 1.0% at 30337 DOFs; non-conforming FEM does not reach 5% on the tested meshes at high contrast.

## Interpretation (not demonstrated)
The P1 ranking (F6 > F2 > others) does not survive the curved interface at this budget: all fail. The two-network failure mode (stationary point with unsatisfied continuity) is common to P1 and P2, and is the target for Interventions A and B. F5's failure is a quadrature-overfitting mechanism specific to fixed-point energy estimates; resampling would be the natural remedy, but it is outside the frozen design and is NOT tried here.
