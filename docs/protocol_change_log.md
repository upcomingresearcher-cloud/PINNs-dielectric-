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
