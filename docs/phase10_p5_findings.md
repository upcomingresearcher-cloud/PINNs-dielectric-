# Phase 10: P5 Thin High-Permittivity Film — Results
Runs (complete, unique; no technical failures): main 165 (F1/F2/F6 × 11 contrasts × 5) + 45 reduced (F3–F5 × {1,80,2000}) +
thickness 60 (F2/F6 × t {0.1, 0.005} × {1,80,2000}) + sampling density 40 (F2/F6 × {128, 2048} film points × {80,2000}) + 60 regression.
Reference: converged graded P2-FEM (<0.1% band-field change; 1D three-slab reproduction ~1e-13). Self-tests 17/17 (reference-aware).

## Measured
- **Best 2D result:** F2 and F6 pass at kappa = 5 (0/5 failed) and mostly at kappa = 20 (2/5, 1/5); fail 5/5 from kappa = 80. Median band error at kappa = 80: F2 0.070, F6 0.069; F1 30, F3 1.6, F4 2.4, F5 2.8.
- All formulations fail all inverse contrasts.
- **F3 (0.95) and F5 (0.30) fail even at kappa = 1.** Likely cause (not tested): the frozen level set psi = (0.25-x^2)(t^2/4-y^2)/(0.25 t) has magnitude up to ~150 away from the film, so the |psi| input feature saturates tanh units in F3/F5 (F4 uses psi only inside the film). A design limitation of this benchmark, not of the formulations; reported, not rerun.
- **Sampling density has no measurable effect:** 128 / 512 / 2048 film points give band error 0.069 / 0.070 / 0.069 (F2, kappa 80) and 1.23 / 1.21 / 1.21 (kappa 2000).
- **Thickness:** band error at kappa = 80 (F2) 3.4% (t = 0.005, 2/5 failed), 7.0% (t = 0.02), 13% (t = 0.1); but the field error INSIDE the film is 13–16% at every thickness and ~2–3.4 (200–340%) at kappa = 2000.
- H2 (P5): median field/potential ratio 8.2; R > 1 in 97% of runs; **37% of failures concealed** (potential error < 5%) — the highest of any problem.
- H8 vs P1: tau = 0.47 (failure counts), 0.47 (median error).
- FEM (graded): kappa = 80 band error 2.9% (1577 DOFs, 0.04 s), 0.24% (5313, 0.13 s), 0.056% (21461, 0.54 s).

## Interpretation (not demonstrated)
Thinner films look easier only in the pooled band metric, because the band is dominated by the exterior field; the film-interior field error is thickness-independent. Density-independence suggests the high-kappa failure is not a sampling-resolution problem.
