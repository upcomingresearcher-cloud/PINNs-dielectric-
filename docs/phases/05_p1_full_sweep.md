# Phase 5 Progress Report — Full P1 Sweep, Regression Baselines, Loss Visibility
Date: 2026-10-16. Code commit for all Phase 5 training rows: 6ccd902 (loss-visibility: 3846d02; src identical).

## Executed (all complete, 0 technical failures, 0 non-finite)
| Set | Runs | Spec |
|---|---|---|
| Primary sweep | 180 | F1–F6 × kappa {1,5,20,80,400,2000} × seeds 0–4; every run exactly 3000 counted gradient evaluations |
| Pre-registered extension | 30 | F1–F6 × kappa 1e4 × seeds 0–4 (reported separately) |
| Regression-only baselines | 120 | 4 architectures × 6 kappa × 5 seeds (NOT PINN results) |
| Loss visibility | 196 diagnostic rows | 7 formulations × 7 kappa × 2 candidates + F1 smoothing (2 widths × 3 candidates) |
Every training run has a gradient log (per-term norms every 25 steps) and a metric trajectory (every 250 steps).

## Results (measured; hypothesis tests follow in Phase 6)
Failed seeds out of 5 (primary criterion):

| | 1 | 5 | 20 | 80 | 400 | 2000 | 1e4* |
|---|---|---|---|---|---|---|---|
| F1 | 0 | 5 | 5 | 5 | 5 | 5 | 5 |
| F2 | 0 | 0 | 0 | 1 | 2 | 5 | 5 |
| F3 | 0 | 0 | 4 | 5 | 5 | 5 | 5 |
| F4 | 0 | 0 | 5 | 5 | 5 | 5 | 5 |
| F5 | 0 | 0 | 5 | 5 | 5 | 5 | 5 |
| F6 | 0 | 0 | 0 | 0 | 0 | 3 | 5 |

- Critical contrast (smallest failing kappa, >= 3/5): F1 = 5; F3, F4, F5 = 20; F2, F6 = 2000. All boundaries monotone.
- F1 converges to the uniform-permittivity solution (median error 27.8 at kappa = 80, vs 27.9 for that solution).
- The interface-aware designs F3–F5 fail from kappa = 20, earlier than the plain two-network F2. Pilot bug checks showed F3/F5 reach < 5% with 20000 steps, so this is a fixed-budget optimisation outcome; the 12000-step robustness variation will test it.
- Loss visibility: F1 loss = 0 for both exact and wrong solutions at every kappa (blind); F2/F3/F4/F6 exact ~1e-32 vs wrong 16 to 1e8; F5 energy lower for exact. F4-full: wrong candidate is not representable by its ansatz (N/A).
- Smoothing check: smoothed eps restores visibility (smoothed-problem solution loss ~1e-20) but the smoothed problem's solution has 36% band field error at kappa = 80 (width 0.002) against the sharp problem, with only 1.7% potential error.
- Regression baselines (phi-only MSE): band field error grows with kappa for every architecture (e.g. two-network 31% at kappa = 400, where PINN-trained F2 gets 4.75% with the same architecture). The regression baseline as designed therefore underestimates representation capacity; analysed against H4's original wording in Phase 6.

## Issues
- A test-coverage gap (regression baselines missing from Gate 3 parameter-count table) failed 4 tests; my command chain committed despite it because the pipe through `tail` hid pytest's exit code. Fixed before any sweep ran (85/85 pass), and gates now check exit codes directly. No results were affected.

## Next: Phase 6
Formal tests of H1–H4 and H11 (concealment ratio, flux-vs-continuity attribution, gradient-ratio AUC with bootstrap CIs), material-specific and gradient-trajectory figures, and the recorded selection of the two strongest formulations for 2D (rule: lowest total failed seeds -> currently F6 = 3, F2 = 8).
