---
title: "When do physics-informed neural networks resolve dielectric interfaces? A pre-registered, seed-level evaluation against interface-conforming finite elements"
bibliography: ../literature/references.bib
---

# Abstract

Physics-informed neural networks (PINNs) are increasingly applied to electrostatic problems with piecewise-constant permittivity. Their accuracy is usually reported as a global solution error from a best-of-several run.

We report a pre-registered evaluation of six PINN formulation families on five electrostatic dielectric-interface problems:
- a one-dimensional layered capacitor (P1);
- a dielectric cylinder in a uniform field (P2);
- a line charge above a planar interface (P3);
- a two-material corner (P4);
- a thin high-permittivity film (P5).

**Design.** Contrast ranges from κ = 1 to 2000, with inverse contrasts in 2D. Every configuration has 5 matched seeds and a fixed budget of 3,000 gradient evaluations. The primary metric is the relative L2 error of the electric field E and displacement D in an interface band, with failure defined as more than 5%. The database holds 3,856 completed training runs, plus 60 rows recording technical failures, all released. The 3,856 include 285 rows that were superseded after bug fixes; they are kept for transparency and excluded from analysis, and their 285 replacement reruns are also counted.

**Results.**
- **1D:** a single network with a strong-form loss fails at every κ ≥ 5, because the flux condition is invisible to its loss. Two-network formulations with soft coupling survive up to κ = 400–2000.
- **2D:** at κ ≥ 80, every formulation fails in every seed on all four problems. Below that, only the thin-film problem has passing configurations (F2 and F6 at κ = 5 and 20).
- **Masking:** global potential error understates band field error by a median factor of 2.4 (P1) to 8.2 (P5). On P5, 37% of failures have a potential error below 5%.
- **Interventions** (matched seeds, one factor at a time):
  - Gradient-norm rebalancing helps one formulation at high contrast but harms it at low contrast.
  - Residual normalisation does not help.
  - Float64 has no effect under Adam once the initial weights are held identical; an implementation error we found and corrected had made it appear to.
  - L-BFGS at equal function-and-gradient budget removes many 1D failures but none on the curved-interface problem.
- **Robustness:** the formulation ranking survives eight one-factor robustness variations (Kendall τ ≥ 0.65).
- **Cost:** interface-conforming quadratic FEM reaches 1% band field error in 0.10–4.75 s on P2 and P5. No PINN configuration reaches 1% at any κ ≠ 1, including the L-BFGS and 20,000-step remedies.

We do not claim that PINNs cannot solve interface problems, nor that FEM is universally superior. P3 shows that conforming FEM also struggles near singular sources. The contribution is a reproducible, seed-level map of where these formulations fail, why, and at what cost relative to FEM, under a stated budget.

# 1. Introduction

Electrostatic fields in layered and composite dielectrics are governed by the variable-coefficient Poisson equation

$$-\nabla\cdot(\varepsilon\nabla\varphi)=\rho,$$

where ε jumps across material interfaces. The potential φ and the normal displacement D·n are continuous there, but the normal electric field E·n jumps by the permittivity ratio κ = ε₂/ε₁. Engineering quantities depend on the field near interfaces: breakdown margins, capacitance, field enhancement at corners and in thin films.

PINNs [@raissi2019pinn] represent φ by a neural network and minimise PDE and boundary residuals. Smooth networks represent a kink in φ poorly [@rahaman2019spectral]. A single network with a strong-form loss cannot even *see* the flux condition [@nohra2024pinn]. This has motivated many interface-aware variants:
- domain decomposition [@jagtap2020xpinn; @jagtap2020cpinn; @sarma2024ipinn; @roy2024adaipinn];
- cusp and discontinuity capturing [@tseng2023cusp; @hu2022dcsnn; @fan2025drpinn];
- hard constraints [@sukumar2022exact; @chung2026hard];
- variational forms [@e2018deepritz; @guo2022nitsche].

Most of these papers report global solution error at a few contrasts, often from the best of several initialisations [@chung2026hard].

This study asks a narrower, testable question. Under a fixed and stated budget, for which formulations, contrasts and geometries does a PINN resolve the field at a dielectric interface? Where it does not, what fails, and does any single change rescue it? We pre-registered eleven hypotheses (H1–H11), the metrics, the failure threshold, the seed counts and the analyses before running the primary experiments. Every later change is in a dated change log. The aim is evidence another group can rerun, not a new architecture.

**Contributions** (measured on the stated problems and budget):
1. Seed-level failure maps for six formulation families over contrast, on five geometries (Sections 7–8).
2. A quantification of how much global potential error masks interface-band field error (Section 8.3).
3. Matched-seed interventions that change one factor at a time: gradient weighting, residual scaling, precision, hard constraints, optimiser (Section 10).
4. Robustness of the conclusions to eight training and architecture settings (Section 11).
5. Time-to-accuracy and error-at-equal-cost comparisons against interface-conforming FEM (Section 12).
6. A reusable evaluator and database, with all code, raw data and an analysis pipeline that regenerates every number (reproduction guide).

# 2. Related work

**Interface-aware neural solvers.** Domain-decomposition PINNs couple subdomain networks through interface penalties [@jagtap2020xpinn; @jagtap2020cpinn; @zhang2022mpinn; @moseley2023fbpinn]. I-PINNs and AdaI-PINNs give each material its own activation or network [@sarma2024ipinn; @roy2024adaipinn]. Cusp-capturing and discontinuity-capturing networks add an interface-distance input [@tseng2023cusp; @hu2022dcsnn]. Discontinuity-removing PINNs [@fan2025drpinn], interfaced neural networks [@wu2022inn], and Dirichlet–Neumann learning schemes [@arxiv2301_07361] build interface structure into the representation or the iteration. Several of these report accuracy at contrasts of 10³–10⁵ on elliptic test problems [@tseng2023cusp; @hu2022dcsnn; @guo2022nitsche; @he2020piecewise]. "PINNs have not been tested at high contrast" is therefore not a gap we claim. Convergence theory exists for domain-decomposed PINNs on elliptic interface problems [@arxiv2203_03407].

**Hard constraints.** Distance-function constructions impose boundary conditions exactly [@sukumar2022exact]. Chung et al. extend this to interface conditions [@chung2026hard] and compare six methods at κ = 10 using solution-L2 error and the best of four initialisations. That is the closest study to ours. Our F4 and F4full follow their idea in 1D. We propose no new architecture as a main contribution.

**Failure analysis.** Gradient imbalance between loss terms [@wang2021gradient], ill-conditioned loss landscapes [@krishnapriyan2021failure], and loss-scale imbalance in parallel PINNs [@huang2021parallel] are each proposed as failure mechanisms. Nohra and Dufour show that a piecewise-constant coefficient makes the flux condition invisible to a single-network strong-form residual [@nohra2024pinn]. We quantify this; we did not discover it. Recent work argues that many PINN failures are precision artefacts [@xu2025fp64] and that optimiser and precision interact [@tarekegn2026optprec]. Our Intervention C tests this directly on interfaces.

**Electromagnetic applications.** PINNs for electric-field distributions and dielectric scattering smooth the permittivity or report global errors [@zhang2023efield; @arxiv2607_27349], and PINNacle benchmarks PINNs broadly [@hao2023pinnacle]. Classical numerics shows that gradients at interfaces are typically an order less accurate than the solution, even for FEM [@arxiv1703_00093].

**PINN versus FEM.** The systematic comparison of Grossmann et al. finds FEM faster and more accurate on the problems considered, which have no material interfaces [@grossmann2024fem]. A cost-to-target comparison on interface-band field accuracy, against interface-conforming FEM, was not found in our literature review (Supplement: `literature/research_gap_report.md`).

**How this work differs.** Relative to the closest studies, we:
1. report interface-band E and D error with a fixed failure threshold, beside global potential error;
2. sweep contrast over three orders of magnitude, including inverse contrast, with five seeds per configuration and no best-seed selection;
3. attribute failures with matched-seed interventions instead of single remedies;
4. compare cost-to-accuracy against interface-conforming FEM;
5. release the evaluator, database and change log.

# 3. Research questions and hypotheses

All eleven hypotheses, their support and refutation criteria and their analyses were frozen in `docs/experimental_protocol.md` before the primary sweep. Gaps found later (for example, which configuration counts for H5) were filled by dated change-log entries written before the relevant runs.

| H | Statement (abridged) | Verdict |
|---|---|---|
| H1 | Single-network strong form fails at κ > 1; error grows with κ for soft coupling | Supported (P1) |
| H2 | Global potential error understates band field error (median R > 3) | Inconclusive on P1 (R = 2.39); R > 3 on P2–P5 |
| H3 | Soft-coupled failures are flux-dominated | Refuted: they are continuity failures |
| H4 | High-κ F2 failures are optimisation, not representation, failures | Refuted by its criteria |
| H5 | Each intervention moves failure in its predicted direction | A mixed, B refuted, D refuted, E mixed; C's no-effect prediction holds |
| H6 | Rankings and masking are robust to settings | Inconclusive: ranking robust (8/8), R > 3 in 1/8 |
| H7 | Conforming FEM reaches 1% ≥ 10× faster and is more accurate at equal cost | Supported (excluding degenerate κ = 1) |
| H8 | P1 conclusions transfer to P2 | Refuted (τ = 0.30 on failure counts) |
| H9 | The evaluator is reusable | Supported, with a 1D → 2D qualification |
| H10 | Smooth networks underestimate the corner field | Refuted: the sign depends on formulation and material |
| H11 | Early gradient ratios predict failure | Supported (AUC 0.85), confounded with κ |

# 4. Problems and formulations

## 4.1 Problems

Each problem is defined with permittivity ε₁ = 1 and ε₂ = κ.

- **P1 — 1D layered capacitor.** On [0, 1] with interface at a = 0.5, φ(0) = 1 and φ(1) = 0. The exact solution is piecewise linear. Conforming FEM is exact on it, so P1 is degenerate for cost comparisons.
- **P2 — Dielectric cylinder.** Radius 0.25, in a uniform field E₀x̂, inside a box of side 2. The exact isolated-cylinder solution is the Dirichlet data.
- **P3 — Line charge.** λ = 2π at height 0.25 above a planar interface, in the material with ε₁. The solution is the exact image solution. Networks learn the regular part w = φ + ln r₁, which has a flux jump at the interface. The disc r₁ < 0.05 is excluded from the metrics.
- **P4 — Two-material corner.** Material 2 occupies one quadrant. The exact singular eigen-solution φ = r^λ Φ(θ) is used, so the field is singular at the corner for κ > 1. The disc r < 0.02 is excluded.
- **P5 — Thin film.** A film of thickness t = 0.02 and width 1, in a uniform field. There is no closed form. The reference is converged body-fitted quadratic FEM: the band field changes by less than 0.1% between the two finest levels, and a 1D three-slab check reproduces to about 10⁻¹³. Small squares around the film corners are excluded.

**Interface band and metrics.**
- **Band:** the set within 0.05 of the interface. In P5 the film interior is included.
- **Primary metric:** the larger of the band relative L2 errors of E and D.
- **Failure:** primary error above 5%.
- **Secondary metrics:** global, material-wise and band errors of φ, E and D; potential and displacement jumps across the interface; capacitance and energy errors; and an effective-contrast diagnostic.

## 4.2 Formulations

All formulations use tanh multilayer perceptrons, 3 hidden layers of width 32, so one network has 2,209 parameters in 1D. Training uses Adam with a cosine schedule from 10⁻³ to 10⁻⁴ over 3,000 steps, in float32, with stratified jittered collocation.
- **1D sampling:** 128 points per material.
- **2D sampling:** 512 interior points per material, 128 interface points and 64 per box edge.

| ID | Representation | Loss |
|---|---|---|
| F1 | one network φ(x) | strong-form residual εφ'' on each material, with BCs; no interface term |
| F2 | one network per material | residuals, BCs, soft continuity [φ] = 0 and soft flux [εφ'] = 0, unit weights |
| F3 | one network N(x, \|x − a\|) with a cusp input [@tseng2023cusp] | residuals, BCs, soft flux |
| F4 | hard BCs and hard continuity: φ = V(1 − x) + x(1 − x)[N_A + max(x − a, 0) N_B] | residuals and soft flux |
| F5 | as F3, with hard BCs | Ritz energy ∫ ½ε\|∇φ\|² estimated on fixed points [@e2018deepritz] |
| F6 | as F2 | as F2, with each residual divided by its physical scale (frozen, contrast-derived weights) |

In 2D, F3 and F5 use a level-set feature ψ instead of |x − a|. F1, F2 and F6 were run at all 11 contrasts. The 2D grid is {0.0005, 0.0025, 0.0125, 0.05, 0.2, 1, 5, 20, 80, 400, 2000}. F3, F4 and F5 were run at κ ∈ {1, 80, 2000}, a pre-registered reduction.

# 5. Experimental protocol

- **Seeds and statistics.** Every configuration has seeds 0–4 (0–9 in the intervention set). The seed fixes the initialisation and the collocation points. We report medians and interquartile ranges, failure counts with Clopper–Pearson 95% intervals, and no best-seed selection. Five seeds give coarse rates: 0/5 has a 95% upper bound of 52%.
- **Paired comparisons.** Interventions use per-seed paired comparisons, exact sign tests and a bootstrap CI of the median difference (10,000 resamples, seed 12345).
- **Budget and timing.**
  - The budget is 3,000 value-and-gradient evaluations.
  - Wall-clock is the sum of blocked update times after ahead-of-time compilation (`time.perf_counter`, single-threaded CPU). Compile time is recorded separately.
  - L-BFGS runs count every function-and-gradient evaluation, checked against an independent host-side counter. The budget is never exceeded: 2,981–3,000 evaluations were used.
- **Change control.** The protocol froze at version 1.0.0. The change log records every later decision with its date, reason, affected experiments and reruns, including six code bugs found during the study (Section 13).

# 6. Evaluator and reference validation

**Self-tests.** Before any PINN run, the evaluator passed:
- 98/98 self-tests in 1D (Gate 1): exact solution, controlled perturbations, interface jumps, mutation tests and refined-quadrature agreement;
- 17/17 per 2D problem.

**Formulation checks (Gate 3).** Each loss is zero, to round-off, at the exact solution and positive at constructed wrong solutions, except where invisibility is the point: F1 has zero loss at the uniform-ε solution. Gradients are finite in every formulation.

**FEM references.**
- P1: conforming FEM is exact to round-off for any mesh.
- P2–P4: interface-conforming quadratic FEM converges in the band. At κ = 80, P2 reaches 1.0% at 30,337 DOFs.
- Non-conforming FEM does not converge in the band at κ ≠ 1 (P2: band error 0.21–580 on the finest meshes and contrasts tested).
- P3 at κ = 2000: conforming FEM does not reach 1% even at 1.2 × 10⁵ DOFs (band error 0.12). That problem is hard for FEM too.

**Evaluator reusability (H9).** `evaluator1d.py` was unchanged after the P1 freeze. The 2D evaluator, written for P2, changed once afterwards: one additive hook for P4's corner metrics. No metric definition was rewritten. The 1D → 2D step needed a new 69-line module, so reusability holds within a dimension.

# 7. Results in 1D (P1)

Figures: `p1_failure_map.png`, `p1_median_band_field_error.png`, `final_convergence_curves.png`.

**Failure counts.** Failed seeds over the 25 runs with κ ∈ {5, …, 2000} per formulation:

| | F1 | F2 | F3 | F4 | F5 | F6 |
|---|---|---|---|---|---|---|
| Failed runs (of 25) | 25 | 8 | 19 | 20 | 20 | 3 |

- **Critical contrast** (smallest κ with ≥3/5 failing): F1 at 5; F3, F4 and F5 at 20; F2 and F6 at 2000. At the pre-registered extension κ = 10⁴, F2 and F6 fail 5/5.
- **H1 (supported).** F1 fails 5/5 at every κ ≥ 5: its loss is exactly zero for the uniform-permittivity solution, which it finds. F2's error grows monotonically with log κ (Spearman ρ = 0.96, CI 0.90–0.98).
- **H3 (refuted).** Failing F2 and F6 runs violate potential continuity (median jump 5.3% of V), while the flux jump is small (median 2.6 × 10⁻⁴). None of 11 failing runs is flux-dominated.
- **H4 (refuted).** Supervised fitting of φ with the F2 architecture is not more accurate in the field than PINN training where F2 fails, so the pre-registered "representation is adequate" criterion fails. Interventions (Section 10) add the optimiser evidence.
- **Interface-aware designs fail earlier.** F3–F5 fail from κ = 20, earlier than F2 and F6. Pilot runs showed F3 and F5 succeeding with 20,000 steps, and Phase 11 E shows L-BFGS rescuing F3 (Section 10). At this budget, building the interface into the representation made the optimisation harder, not easier.

# 8. Results in 2D (P2–P5)

Figures: `p2_failure_map.png`, `P3_failure_map.png`, `P4_failure_map.png`, `P5_failure_map.png`, `final_critical_contrast.png`, `final_cross_geometry.png`.

## 8.1 Failure maps

**At κ ∈ {80, 2000}, every formulation fails all 10 runs on P2, P3, P4 and P5.** On P1 the same cells give F2 6/10 and F6 3/10. Median band field errors at κ = 80:

| Problem | F1 | F2 | F3 | F4 | F5 | F6 |
|---|---|---|---|---|---|---|
| P2 cylinder | 30 | 0.63 | 1.0 | 0.84 | 1.2 | 0.63 |
| P3 line charge | 31 | 6.1 | 3.5 | 8.5 | 1.3 × 10⁴ | 11 |
| P4 corner | 14 | 0.69 | 1.0 | 17 | 1.7 | 0.70 |
| P5 thin film | 30 | 0.070 | 1.6 | 2.4 | 2.8 | 0.069 |

**The only 2D passes away from κ = 1:**
- **P5:** F2 and F6 at κ = 5 (0/5 failed) and κ = 20 (2/5 and 1/5).
- **P3:** F2 at mild inverse contrasts (0/5 failed at κ = 0.2 and 0.05).

The critical contrast of F2 and F6 drops from 2000 (P1) to 5 (P2–P4) and 80 (P5). **H8 (refuted):** Kendall τ between the P1 and P2 rankings by failure count is 0.30. On median error it is 0.73, so the order of errors partly transfers but the pass/fail picture does not.

## 8.2 Problem-specific findings

- **P2 — stationary point.** F2 and F6 plateau at about 0.63 band error. A float64, 20,000-step run stays at 0.486 from step 4,000 to 20,000 at κ = 5; per-term gradients are about 0.1 but cancel. Minimising the F2 loss over a three-parameter family containing the exact solution recovers it to six digits (loss about 10⁻³¹), so the loss is correct and the plateau is a property of the network training.
- **Fixed-point quadrature overfitting (F5).** F5 fails even at κ = 1 on P2 and P4. Its Monte Carlo energy on the fixed points (0.59–1.59 on P2) falls below the true minimum (2.0), while the true energy of its prediction is 4.2–75. The same mechanism reappears in 1D under L-BFGS (Section 10) and with non-uniform sampling (Section 13).
- **P3 — singular subtraction.** Subtracting the singular part per material (a pre-registered sensitivity analysis) lowers F2's κ = 80 error from 6.1 to 0.61. F2 still fails 5/5.
- **P4 — corner field (H10 refuted).** F2 underestimates the near-corner field on both bisectors (median signed |E| error −0.58 to −0.88). F1 overestimates it in the high-ε wedge (+5.7 at κ = 80, +18.6 at κ = 2000) and underestimates it in the low-ε wedge. The sign of the error depends on formulation and material.
- **P5 — thin film.**
  - Band error falls with thinner films (F2 at κ = 80: 13% at t = 0.1, 7.0% at t = 0.02, 3.4% at t = 0.005). But the field error *inside* the film is 13–16% at every thickness, so the pooled band metric flatters thin films.
  - Sampling density in the film has no measurable effect: 128, 512 and 2,048 film points give 0.069, 0.070 and 0.069.
  - F3 and F5 fail at κ = 1. The frozen level set reaches about 150 away from the film, which plausibly saturates their tanh inputs. This is a limitation of the benchmark design and was not tested further.

## 8.3 Masking: global potential error versus band field error (H2)

The ratio R = band field error / global potential error (over κ ≠ 1 runs) has medians of 2.39 (P1), 7.5 (P2), 3.95 (P3), 4.0 (P4) and 8.2 (P5). R exceeds 1 in 97–100% of runs.

Concealed failures are runs that fail on band field error while their global potential error is below 5%. They make up 23% (P1), 1.7% (P2), 2.4% (P3), 11.7% (P4) and 37% (P5) of failures. H2's pre-registered threshold R > 3 is not met on P1 (CI 1.73–3.18), so H2 is inconclusive as stated; the threshold is met on every 2D problem.

The practical consequence: reporting only a global potential error would have called 37% of the failing thin-film runs accurate.

# 9. Failure mechanisms

**Measured:**
1. **Loss invisibility.** F1's loss is exactly zero at a wrong solution [@nohra2024pinn]. Rescaling its residuals (Intervention B) cannot help: F1norm converges to the uniform-ε solution at every κ, with loss 2.147 × 10⁻⁶ at every κ (Section 10).
2. **Continuity, not flux.** Soft-coupled F2 and F6 fail with an unsatisfied potential continuity (P1 and P2).
3. **Gradient imbalance precedes failure but is confounded with κ (H11).** The median early (steps 0–300) log₁₀ ratio of the largest to smallest weighted per-term gradient norm predicts failure with AUC 0.85 (CI 0.77–0.92). It correlates with log κ (ρ = 0.81), and within a single κ the AUC is 0.42–0.88 (`final_gradratio_vs_error.png`).
4. **Fixed-point quadrature overfitting** in the Ritz formulation (Sections 8.2, 10, 13).

**Interpretation, not demonstrated.** At high contrast, the unit-weight flux penalty in F2, which scales as D², dominates the continuity penalty. The optimiser then satisfies flux balance first and is left at a stationary point with a potential jump. F6's physical rescaling is consistent with its lower 1D failure count. On curved, singular and thin geometries, the same stationary point is reached at much lower contrast, and neither a stronger optimiser nor a longer budget escapes it on P2 (Section 12).

# 10. Controlled interventions (P1; H4, H5)

Each intervention changes one factor, uses seeds 0–9 paired with the unchanged baseline, and was judged by criteria logged before its runs. 1,155 rows were recorded, 270 of them superseded (see C). Figures: `p1_interventions_{AB,C,D,E,paired}.png`.

| Intervention | Prediction | Verdict | Key numbers |
|---|---|---|---|
| A: gradient-norm rebalancing [@wang2021gradient] (L2-norm variant, every 10 steps) | lowers error for F2 and F3 | **mixed**: 3/12 supported | helps F3 at κ ≥ 80 (κ = 400 median 1.06 → 0.033); harms F2 at every κ (10/10 fail at κ = 1 and 5). F3 at κ = 80 does not survive Holm correction (p = 0.107) |
| B: residual normalisation r_i/ε_i | helps F2 at κ ≥ 80 | **refuted** (3/3) | 1–2 of 10 seeds improve; F1 "no change" holds (identical flags 10/10) |
| C: float64 | no material effect under Adam; larger under L-BFGS | **holds** | Adam: median fold change 1.0001 (max 1.07), no flag changes; L-BFGS: float64 better in 6/6 cells (median fold 3.4) |
| D: hard interface conditions | F4 fails less than F2 at κ ≥ 400; F4full < 1% | F4 **refuted**; F4full **holds** | F4 fails 10/10 from κ = 20; F4full median 1.2 × 10⁻⁶ to 9.9 × 10⁻⁴ (trivial on P1: the exact solution lies in its buffer space) |
| E: L-BFGS, Adam 2k → L-BFGS 1k (equal evaluations) | lower error for F3 and F5 | **mixed**: 9/12 supported | F3 κ = 80: 0.55 → 3.5 × 10⁻⁵; F5 refuted at κ = 1 (quadrature overfitting); F2 (descriptive) κ = 2000 failures 9/10 → 4/10 |

**H4's intervention clause is not met.** A and B leave F2 at 9/10 failures at κ = 2000.

**Two findings deserve emphasis.**

First, the pre-registered rebalancing rule inverts its intent near convergence. It sets λ_i ∝ ‖∇L_pde‖/‖∇L_i‖, so satisfied terms (small gradients) receive weights up to 5 × 10⁴, while the violated continuity term is pushed down to 0.05. This was verified against an independent implementation. It applies to the L2-norm variant tested, not to every rebalancing scheme.

Second, we initially measured precision effects of up to 20.7× in median error. Investigation showed that the float64 runs had drawn *different initial weights* from the float32 runs with the same seed. With identical initialisation the effect vanishes (fold change at most 1.07). On this problem, initialisation variability alone exceeds the genuine precision effect under Adam by more than an order of magnitude. We see this as direct support for seed-level reporting, and as a caution for precision studies that do not control the initial weights [@xu2025fp64].

# 11. Robustness (P1; H6)

The robustness runs vary one setting at a time: width 16 and 64; depth 2 and 5; 32 and 512 points per material; learning rate 3 × 10⁻³; and 12,000 steps. Each covers all six formulations, κ ∈ {1, 80, 2000} and 5 seeds (720 runs). Two descriptive extras were also run: a parameter-matched width-22 two-network variant, and 16 extra band points per side. Figures: `p1_robustness_failcounts.png`, `p1_robustness_errors.png`.

**Results.**
- **Ranking:** robust in 8/8 variations (Kendall τ-b ≥ 0.65; τ = 1 in 5 of them).
- **Masking:** R > 3 in only 1/8, but R is stable at 1.84–2.49 in 7/8, against 2.24 for the baseline on the same subset. The absolute threshold, not the effect, fails. R also shrinks with κ (3.4 at κ = 5–20, 2.1 at 2000).
- **Verdict:** H6 is inconclusive by its criteria.

**Notable effects.**
- 12,000 steps reduces F2 failures from 6 to 1 out of 15, but doubles the concealed failures (7 → 16) and raises R to 5.6. With more training, global potential error becomes a *worse* indicator of band field accuracy.
- Capacity affects F2 and F6 only.
- Matching parameters (width 22, fewer parameters than F1) does not remove their advantage: 6 and 4 failures, against 10 for every single-network formulation.

# 12. Comparison with finite elements (P2, P5; H7)

We compared the baseline PINNs and the two remedies that changed outcomes on P1 against interface-conforming quadratic FEM: L-BFGS for F2, F3 and F6 (90 runs), and 20,000-step Adam for F2 and F6 at κ = 80 (20 runs). FEM cost is mesh generation plus assembly plus a sparse direct solve. PINN time-to-target is the first logged evaluation (every 250 steps) meeting the target, scaled from the run's wall-clock. Figure: `p13_error_vs_cost.png`.

**Results.**
- **1% target:** no PINN configuration reaches 1% band field error at any κ ≠ 1 on P2 or P5, across 64 configurations and every seed. Conforming FEM reaches 1% in 0.65–4.75 s (P2) and 0.10–0.58 s (P5).
- **5% target:** only P5 F2 and F6 at κ = 80 reach it, with L-BFGS in about 5 s and 20,000-step Adam in about 47 s, against 0.042 s for FEM. That is about 115× and 1,100× slower.
- **Equal cost:** at the PINN's median wall-clock, FEM's error is 86× to 1.7 × 10⁵× lower (away from κ = 1).
- **Remedies:** on P2, neither L-BFGS nor 20,000 steps moves F2 or F6 from their plateau (0.634, 0.635, 0.635 at κ = 80). This contrasts with P1, where L-BFGS rescued F3 and halved F2's κ = 2000 failures. On P5, L-BFGS brings F2 and F6 just under 5% at κ = 80 and lowers κ = 2000 errors (F2: 1.21 → 0.072), still failing.

**H7: supported, excluding κ = 1.** At κ = 1, FEM is exact at every mesh. There, 10 PINN configurations reach 1% within the first 250-step logging interval, so the 10× margin cannot be resolved at our logging resolution. At the logged times the ratios are 21–139×.

**Caveats.** Timings are single-threaded CPU. P2 and P5 FEM ran on different CPU models, and GPU training would change absolute PINN times. Neither would change which configurations reach 1%. κ = 5 and 20, where PINNs do pass on P5, have no FEM refinement records, so the PINN's most favourable regime is not in the time-to-target comparison.

# 13. Limitations

- **Budget and architecture.** All conclusions are conditional on the fixed budget (3,000 evaluations; 12,000 and 20,000 in specific checks), the small tanh MLPs, and float32 except where stated. Larger networks, adaptive sampling, curriculum over κ, or other architectures [@roy2024adaipinn; @fan2025drpinn; @chung2026hard] were not tested at scale.
- **Statistical resolution.** Five seeds per configuration give coarse failure rates (0/5 has an upper bound of 52%).
- **Reduced grid.** F3–F5 were run at κ ∈ {1, 80, 2000} only in 2D, so their critical contrasts there are resolved only to that grid.
- **Pre-registration gaps.** Several gaps were filled during the study (H5's configuration sets, the 10-seed failure threshold, the L-BFGS budget rule). All were logged before the affected runs. One analysis addition, the H7 logging-resolution check, was added after a trial analysis but before the verdict, and is stricter than the original rule.
- **Bugs found and fixed:**
  - a gradient bug in the 2D FEM;
  - a shape bug in compiled 2D training (P3; 60 failed rows kept);
  - a derivative bug in the P4 verification code (no results affected);
  - float64 initialisation differing from float32 (270 runs rerun);
  - F5 energy quadrature with non-uniform points (15 runs rerun);
  - a missing problem filter in an analysis script (outputs regenerated).

  Each has a regression test, and the original rows remain in the database, marked superseded.
- **Benchmark design.** The P5 level-set scaling probably disadvantages F3 and F5. F4full's success on P1 is trivial. Interventions were run on P1 only, and remedies on P2 and P5 only.
- **Bibliography.** Two references could not be fully verified: author lists for the J. Phys. D article [@zhang2023efield] and the 2026 scattering preprint [@arxiv2607_27349]. They are marked as such in the bibliography.

# 14. Discussion

**What fails.** Within the tested budget, PINN accuracy at dielectric interfaces is governed less by contrast alone than by geometry:
- a planar 1D interface tolerates κ up to 400–2000 for soft-coupled two-network formulations;
- curved, singular and corner geometries fail from κ = 5;
- the thin film fails from κ = 80.

**How it fails.** The dominant failure is a stationary point with unsatisfied potential continuity. More training, a better optimiser or higher precision escapes it in 1D but not on the curved interface.

**How failure is hidden.** Global potential error hides much of this: up to 37% of failures look accurate by that measure. In practice, studies of PINNs for dielectric problems should report interface-band field error with a stated threshold, and seed-level failure rates rather than best runs.

**What did not hold up.** Several plausible remedies failed under matched seeds: rebalancing, residual normalisation, hard continuity. The one apparent precision effect was an artefact of the initialisation.

**What this does not show.** It does not show that PINNs cannot solve these problems. Interface-specialised architectures and larger budgets may succeed, and published work reports such successes, usually as best-of-several runs. It does not show that FEM is always preferable either: P3 near a singular source is hard for conforming FEM as well, and PINNs offer mesh-free inverse and parametric uses that we did not evaluate. What it shows is that for these forward problems, on this hardware, with these formulations and this budget, interface-conforming FEM reached 1% band field accuracy in seconds where no PINN configuration reached it at all.

# 15. Conclusion

We evaluated six PINN formulations on five dielectric-interface problems with pre-registered hypotheses, seed-level statistics, interface-band field metrics, matched-seed interventions, robustness variations and an FEM cost comparison:
- The 1D results transfer poorly to 2D.
- Global potential error systematically understates the interface field error.
- No single tested intervention removes high-contrast failure on curved interfaces.
- Interface-conforming FEM dominates on accuracy per unit cost for the forward problems tested.

All code, raw data, the change log and a single command that regenerates every table and figure are released (Reproduction Guide).

# Data and code availability

The project archive contains the source (`src/`), the raw databases (`results/raw/`), the processed outputs and figures, the protocol and change log (`docs/`), and the analysis pipeline (`scripts/make_all.py`). See `docs/REPRODUCTION_GUIDE.md`.

# References
