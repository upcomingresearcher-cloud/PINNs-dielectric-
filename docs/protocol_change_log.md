# Protocol Change Log

Every change after the freeze records: date, version, what changed, reason, affected experiments, reruns required.

| Date | Version | Change | Reason | Affected | Reruns |
|---|---|---|---|---|---|
| 2026-10-16 | 1.0.0 | Protocol frozen | — | — | — |
| 2026-10-16 | 1.0.0 (pre-freeze) | Effective-contrast secondary criterion made symmetric in log space | One-sided form mis-stated "factor of 2" | none (no runs yet) | none |
| 2026-10-16 | 1.0.0 (pre-freeze) | F4-full buffer determinant corrected from -(a e2+(1-a)e1) to +(a e2+(1-a)e1) | Sign error in hand derivation caught by symbolic check; solvability conclusion unchanged | none | none |

Pending decisions already scheduled by the protocol (not changes):
- Pilot learning-rate choice (Sec. 6.4) - to be recorded here after the pilot.
- Selection of the two strongest formulations for 2D (Sec. 8.2) - to be recorded before the 2D sweep.
- Detailed configs for P2-P5 - added as protocol_v1.x before each phase, without altering P1 settings.

## Phase 3 entries
| Date | Version | Change | Reason | Affected | Reruns |
|---|---|---|---|---|---|
| 2026-10-16 | 1.0.1 | Evaluator gains optional `extra_breaks` (extra quadrature panel breakpoints). Used for FEM predictions (FEM nodes added) so piecewise-linear integrands are integrated exactly. Metric *definitions* unchanged. | Self-test 7 showed ~5% quadrature error in error metrics when FEM element kinks fall inside GL panels | FEM evaluations only; no PINN runs exist | FEM sweep regenerated |
| 2026-10-16 | 1.0.1 | Tolerances fixed: exact-solution metrics < 1e-12; conforming FEM (N <= 200) < 1e-10 | Round-off documented to grow with N (5e-10 at N = 20000) | self-tests | n/a |

## Phase 4 entries
| Date | Version | Change | Reason | Affected | Reruns |
|---|---|---|---|---|---|
| 2026-10-16 | 1.1.0 | **Pilot decision (Sec. 6.4): learning_rate.init stays 1e-3.** Total pilot failures: 12 (lr 1e-3) vs 12 (lr 3e-3) -> pre-registered tie-break -> 1e-3. lr 3e-3 had lower median errors for several formulations; NOT used as a criterion (would be post-hoc tuning). | Pre-registered rule | none (no primary runs yet) | none |
| 2026-10-16 | 1.1.0 | Budget kept at 3000 gradient evaluations. Pilot: all formulations pass at kappa=1 (benchmark valid); F3/F4/F5 fail at kappa=80. Bug check (pilot-labelled, seed 100, float64, 20000 steps): F3 -> 4.4e-3, F5 -> 1.9e-2 (pass), F4 -> 0.11 (still decreasing). Failures are budget-limited optimisation, not implementation errors (Gate 3 verified losses and constraints). | Pilot validity check | none | none |
| 2026-10-16 | 1.1.0 | **Robustness variation 8 added: training budget 12000 gradient evaluations** (all formulations, kappa in {1, 80, 2000}, 5 seeds). | Pilot shows failure outcomes depend on the budget; conclusions must be tested against it. Added BEFORE the primary sweep. | robustness set (+90 runs) | none |
| 2026-10-16 | 1.1.0 | Execution note: background processes are killed when a tool call returns in this sandbox; sweeps run in resumable foreground chunks (< 300 s). A run interrupted mid-training writes no row and is rerun from scratch. | Environment constraint discovered | none | none |
| 2026-10-16 | 1.1.0 | H6 thresholds rescaled for 8 variations: supported if tau >= 0.6 in >= 6/8 (was >= 5/7); refuted if tau < 0.6 in >= 3/8 (was >= 3/7). | Consequence of adding variation 8; proportion kept approximately equal (0.71 -> 0.75). Made before any primary-sweep data exist. | H6 | none |

## Phase 6 entries
| Date | Version | Change | Reason | Affected | Reruns |
|---|---|---|---|---|---|
| 2026-10-16 | 1.1.0 | **2D selection recorded (Sec. 8.2 rule, before any 2D run): strongest two = F6 (3 failed seeds), F2 (8).** Full contrast sweep in 2D: F1, F2, F6. Reduced grid {1, 80, 2000}: F3, F4, F5. | Pre-registered rule applied to primary P1 data | 2D sweeps | none |
| 2026-10-16 | 1.1.0 | H11 operational choice: gradient ratio uses weighted per-term norms (gradient of the actual training loss); unweighted reported as sensitivity. Fixed before computing either. | YAML did not specify weighted vs unweighted | H11 | none |
| 2026-10-16 | 1.1.0 | EXPLORATORY additions (not pre-registered, labelled): within-kappa stratified AUC for H11; field-supervised regression REGD_* (run_type regression_exploratory, 120 runs). | Investigate H11 confounding and H4 regression-baseline weakness | none | none |

## Phase 7 (P2) configuration — frozen before any P2 run
| Date | Version | Item | Value / rule |
|---|---|---|---|
| 2026-10-16 | 1.2.0 | Geometry | box [-1,1]^2, cylinder radius R = 0.25 at origin, E0 = 1 (x-directed); material 1 = matrix (eps = 1), material 2 = inclusion (eps = kappa); exact isolated-cylinder Dirichlet data on the box |
| 2026-10-16 | 1.2.0 | Points | 512 interior per material (uniform, seed-dependent), 128 interface points (equispaced in theta, seed-dependent rotation), 256 boundary points (64 per edge, uniform) |
| 2026-10-16 | 1.2.0 | Training | identical to P1: 3x32 tanh, Adam, cosine 1e-3 -> 1e-4, 3000 counted gradient evaluations, float32, seeds 0-4 |
| 2026-10-16 | 1.2.0 | Contrasts | F1, F2, F6: {1,5,20,80,400,2000} + inverse {0.2,0.05,0.0125,0.0025,0.0005}; F3, F4, F5: {1,80,2000} |
| 2026-10-16 | 1.2.0 | Interface band | |r - R| < 0.05 |
| 2026-10-16 | 1.2.0 | Hard BC (F4, F5) | phi = G + B*(...), G = Coons transfinite interpolation of the boundary data (uses boundary data only), B = (1-x^2)(1-y^2) |
| 2026-10-16 | 1.2.0 | F6 scales | L_c = 2 (box width), V_c = E0*L_c, D_s = eps_h*V_c/L_c = eps_h*E0; same divisors as P1 |
| 2026-10-16 | 1.2.0 | Metrics | vector L2 errors of E and D; [phi] relative to V_c; [D_n] as interface-L2 norm relative to ||D_n,exact||; tangential-E jump; energy error; kappa_eff = 2E0/mean(E_x, r < R-0.05) - 1; BC error |
| 2026-10-16 | 1.2.0 | Reference | exact; FEM (gmsh body-fitted, P2 elements, scikit-fem) as independent cross-check with convergence; FEM errors computed with FEM quadrature on the mesh's own material tags (polygonal interface; O(h^2) sliver difference documented) |
| 2026-10-16 | 1.2.0 | Baselines | phi-only regression (REG_F1..F4) and loss visibility (exact vs uniform-field solution) at kappa {1, 80, 2000} |
| 2026-10-16 | 1.2.0 | Database | 2D runs in results/raw/runs_2d.csv (2D metric schema), logs as in P1 |
| 2026-10-16 | 1.2.0 | CORRECTION of a stated property (rule unchanged): "F6 identical to F2 when eps1 == eps2" holds only when V_c = L_c = 1 (P1). In P2 the frozen divisors give weights (4, 4, 1/4, 1/4, 1) at kappa = 1. F6 is applied exactly as frozen; the property claim in protocol_v1.yaml/Sec. 5 is P1-specific. | Caught by the 2D Gate 3 test | interpretation of F6 vs F2 at kappa=1 in 2D | none |
| 2026-10-16 | 1.2.0 | P2 executed as frozen. Pilot-labelled bug check (F2, kappa=5, float64, 20000 steps) + loss-consistency test (BFGS on exact-containing family) recorded; no changes made to formulations or settings. | Verify P2 failures are not implementation errors | none | none |

## Phase 8 (P3) configuration — frozen before any P3 run
| Date | Version | Item | Value / rule |
|---|---|---|---|
| 2026-10-16 | 1.3.0 | Geometry | box [-1,1]^2, interface y = 0; material 1 (eps 1) y > 0 contains line charge lambda = 2*pi at (0, 0.25); material 2 (eps kappa) y < 0. 2D per-unit-length problem |
| 2026-10-16 | 1.3.0 | Singularity | subtract phi_s = -ln r1 everywhere; train w = phi - phi_s (smooth; continuous; flux jump Q = H(eps1-eps2)/(x^2+H^2), verified symbolically: results/validation/p3_decomposition_checks.txt). Evaluate physical phi = w + phi_s |
| 2026-10-16 | 1.3.0 | Regions | excluded disc r1 < 0.05; near-source ring 0.05 <= r1 < 0.15 and far field reported separately; band |y| < 0.05 |
| 2026-10-16 | 1.3.0 | Scales | V_c = lambda/(2 pi eps1) = 1, L_c = 2; F6 D_s = eps_h V_c/L_c |
| 2026-10-16 | 1.3.0 | Points, training, contrasts, sweep design | identical to P2 (512/material, 128 interface, 256 boundary; 3000 Adam; F1/F2/F6 full incl. inverse; F3/F4/F5 {1,80,2000}; regression at {1,80,2000}) |
| 2026-10-16 | 1.3.0 | FEM cross-check | body-fitted P2-FEM for w with interface load int_Gamma Q v ds; wrong reference = uniform-eps answer phi_s alone |
| 2026-10-16 | 1.3.0 | Pre-run finding: with global subtraction the physical field in material 2 (~2/(1+kappa)) is a cancellation of O(1) terms w and phi_s, amplifying absolute errors ~kappa/2 (FEM band error 0.53 at h=0.025, kappa=2000, order 1.87). DECISION: keep global subtraction as primary (uniform across formulations, common practice). ADD pre-registered sensitivity: F2 with material-wise subtraction (class P3MatSub, variant "matsub") at kappa {80, 2000, 0.0125}, 5 seeds. Continuity residual generalised to w1 - w2 + PI (PI = 0 except matsub). | Avoid a decomposition-driven confound going unexamined | P3 | none (no P3 runs yet) |
| 2026-10-16 | 1.3.0 | Self-test 8 informational for problems with an interface load (non-conforming mesh has no interface edges; load cannot be applied) | Explained discrepancy, not a pass/fail property | P3 self-tests | none |
| 2026-10-16 | 1.3.0 | BUG (technical): AOT-executable cache keyed on settings only; in P3 the per-material boundary-point counts vary by seed -> 60 runs raised a shape mismatch and were recorded status=failed_technical (records kept). Fix: cache key includes all data-array shapes. The 16 completed runs are valid (an executable refuses mismatched shapes). Failed runs re-executed; analyses use status == complete. | Shape-dependent compilation | P3 first chunk | 60 reruns |

## Phase 9 (P4) configuration — frozen before any P4 run
| Date | Version | Item | Value / rule |
|---|---|---|---|
| 2026-10-16 | 1.4.0 | Geometry | box [-1,1]^2; material 2 (eps kappa) = quadrant x>0, y>0 (90 deg), material 1 = remaining 270 deg; corner at origin |
| 2026-10-16 | 1.4.0 | Reference | exact singular solution r^lam Phi(theta) (bisector-symmetric mode), kappa tan(lam pi/4) + tan(3 lam pi/4) = 0; Dirichlet data from it; verified (results/validation/p4_eigen_checks.txt) |
| 2026-10-16 | 1.4.0 | Regions | corner disc r < 0.02 excluded from all metrics; band = (|y|<0.05, x>-0.05) U (|x|<0.05, y>-0.05) |
| 2026-10-16 | 1.4.0 | H10 metrics | signed relative |E| error at r in {0.02, 0.05, 0.1} on rays theta = pi/4 (mat 2), 5pi/4 and pi (mat 1); fitted coefficient ratio on r = 0.05. No pointwise peak compared |
| 2026-10-16 | 1.4.0 | Level set | psi = x*y (smooth; zero on both interface rays; > 0 in material 2) |
| 2026-10-16 | 1.4.0 | Scales | V_c = 1 (material-1 amplitude normalised to 1), L_c = 2 |
| 2026-10-16 | 1.4.0 | Points, training, contrasts, sweep design | identical to P2/P3; interface points 64 per ray, r in (0.02, 1) |
| 2026-10-16 | 1.4.0 | INCIDENT: two sweep processes ran concurrently for ~3 min (11:37-11:40), duplicating 5 P4 runs (F1, kappa 0.0025, seeds 0-4) with IDENTICAL results (determinism confirmed). Raw rows kept; analyses de-duplicate by run_id (first by timestamp). Wall-clock of the 8 unique runs finished in that window is contaminated (shared core): flagged and excluded from cost analyses. | Overlapping tool calls | timing of 8 P4 runs | none |
| 2026-10-16 | 1.4.0 | H10 operationalisation (stated before computing): applied to every kappa > 1 (field singular); supported if, for F1 and F2, the median signed |E| error is < -10% at r = 0.02 and 0.05 on both bisector rays at every kappa > 1; refuted if within +/-10% or positive at any of these; otherwise inconclusive. kappa <= 1 reported descriptively. | H10 did not specify the contrast set | H10 | none |

## Phase 10 (P5) configuration — frozen before any P5 PINN run
| Date | Version | Item | Value / rule |
|---|---|---|---|
| 2026-10-16 | 1.5.0 | Geometry | film |x| < 0.5, |y| < t/2 (eps kappa) in box [-1,1]^2 (eps 1); phi = -E0 y on the box (E0 = 1); main sweep t = 0.02 |
| 2026-10-16 | 1.5.0 | Reference | body-fitted P2-FEM, graded mesh h_in = t/8 within t of the film -> h_far = 0.05 beyond 0.3; cached in results/reference/. Validated: (i) exact 1D three-slab reproduction to ~1e-13 (results/validation/p5_slab_fem_check.txt); (ii) reference level vs finest (t/16) band-field difference 0.013% (kappa 80), 0.014% (2000), 0.070% (0.0005), 0.035% (t = 0.1, kappa 80) — all < 0.1% (results/validation/p5_reference_convergence.csv) |
| 2026-10-16 | 1.5.0 | Regions | corner squares half-width t/2 excluded; band = |x| < 0.55 & |y| < t/2 + 0.05; interface metrics on the long faces |
| 2026-10-16 | 1.5.0 | Level set | psi = (0.25 - x^2)(t^2/4 - y^2)/(0.25 t) |
| 2026-10-16 | 1.5.0 | Scales | V_c = E0 L_c = 2, L_c = 2 |
| 2026-10-16 | 1.5.0 | Main sweep | as P2-P4 (F1/F2/F6 full incl. inverse; F3-F5 {1,80,2000}; 5 seeds; regression at {1,80,2000}) |
| 2026-10-16 | 1.5.0 | Thickness study | F2, F6 x t in {0.1, 0.005} x kappa {1, 80, 2000} x 5 seeds (t = 0.02 from main sweep) |
| 2026-10-16 | 1.5.0 | Sampling-density study | F2, F6 x film points {128, 2048} (baseline 512) x kappa {80, 2000} x 5 seeds, t = 0.02 |
| 2026-10-16 | 1.5.0 | Gate 3 for P5 | exact-loss check only at kappa = 1 (closed form -E0 y); kappa != 1 has no closed form |
