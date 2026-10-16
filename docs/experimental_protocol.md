# Experimental Protocol v1.0.0: PINNs Across Dielectric Interfaces

**Frozen:** 2026-10-16. The machine-readable copy is `configs/protocol_v1.yaml`; where the two differ, the YAML wins and the difference is a bug to log. The only setting the pilot may change is the shared Adam initial learning rate (Section 6.4). Every other change goes in `docs/protocol_change_log.md` with date, reason, affected experiments and required reruns.

**Precedence.** The project brief overrides the team's protocol PDF. Where they conflicted, the resolution is recorded in Section 15.

---

## 1. Research questions and hypotheses

Every hypothesis below was written before any training run. Thresholds are fixed. An unexpected result is reported against the original wording; the wording is not changed afterwards.

| ID | RQ | Proposed relationship / mechanism | Experiment | Supported if | Refuted if | Metrics |
|---|---|---|---|---|---|---|
| H1 | RQ1 | Contrast drives failure for formulations without interface-aware design (F1) and unit-weight soft coupling (F2). Interface-aware designs fail less. | P1 primary sweep | F1 fails ≥4/5 seeds at every κ≥5. Spearman ρ(log κ, primary error) for F2 > 0 with bootstrap CI excluding 0. Each of F3–F6 has fewer failed seeds than F1, summed over κ>1. | F1 passes ≥2/5 at any κ≥20; or the F2 ρ CI includes 0 or is negative; or any of F3–F6 fails as often as F1 | Primary band field error; failure counts |
| H2 | RQ2 | Global potential error understates interface-band field error. | All primary P1 runs with κ>1 | Median of R = e_field,band / e_φ,global > 3; R>1 in ≥80% of runs; and ≥20% of primary failures have e_φ,global < 5% ("concealed failures") | Median R ≤ 1 | R; concealed-failure fraction |
| H3 | RQ3 | Among soft-coupled formulations, failures come from violating flux continuity, not potential continuity. | F2, F6 primary runs | ≥80% of failing F2/F6 runs have \|[D_n]\|/D > 5% while \|[φ]\|/V < 1% | Fewer than 50% of failing runs show the flux-dominant pattern | Relative D jump; relative φ jump |
| H4 | RQ4 | F1 failures are loss-visibility failures: the loss cannot distinguish right from wrong. F2 failures at high κ are optimisation failures: representation and visibility are adequate. | Regression baseline; loss-visibility test; Interventions A/B | Regression baseline < 1% band field error for every architecture and κ. L_F1(wrong) ≤ L_F1(exact) at every κ. L(wrong)/L(exact) > 10 for F2–F6. A or B removes ≥3 F2 failures at some κ where the F2 baseline fails ≥4/5 (10-seed intervention set). | Regression fails (>5%) where F2 fails, which means a representation limit; or interventions change no F2 outcome | Regression error; loss ratio; matched failure counts |
| H5 | RQ5 | Each intervention moves failure in a stated direction (see Section 10). | Interventions A–E, matched seeds | Predicted sign holds in ≥9/10 seeds (sign test p≈0.02) for the pre-specified configuration | Predicted sign holds in ≤5/10 seeds, or the opposite sign holds in ≥9/10 | Per-seed paired differences |
| H6 | RQ6 | Formulation rankings and the H2 masking effect are not artefacts of one setting. | Robustness variations (Section 11) | Kendall τ ≥ 0.6 between baseline and varied formulation ranking (by failure count) in ≥5/7 variations; H2 median R>3 in ≥5/7 | τ < 0.6 in ≥3/7 variations | Kendall τ; R |
| H7 | RQ7 | Interface-conforming FEM reaches a given field accuracy at much lower cost than PINNs, and is more accurate at equal wall-clock. | Remedy comparison on P2 and P5 (P1 is degenerate, Section 9) | For every formulation, FEM time-to-1% band field error is ≥10× smaller than the PINN median, or the PINN never reaches 1%. FEM error at the PINN median wall-clock is lower. | Any PINN formulation's median time-to-target is within 10× of FEM | Wall-clock; time-to-target; error at equal cost |
| H8 | RQ8 | P1 conclusions transfer to curved and singular geometries. | P2–P5 | Kendall τ ≥ 0.6 between P1 and P2 formulation rankings on the shared κ grid; H2 median R > 3 on each of P2–P5 | τ < 0.6, or R ≤ 1 on any of P2–P5 | τ; R |
| H9 | RQ9 | The evaluator is reusable: new problems need only a problem module. | P2–P5 implementation | No edits to `src/evaluator/` core functions after the P1 freeze except additive, documented metrics | A core metric function has to be rewritten | Git diff of evaluator core |
| H10 | RQ8 | Smooth networks underestimate the near-junction field at a dielectric corner. Replaces the PDF's peak-field H4, which compared an undefined singular peak. | P4 | Median signed relative field error at r = 0.02 and 0.05 from the junction (fixed rays, each material) is below −10% for F1 and F2 | Within ±10%, or positive (overestimate) | Signed fixed-distance field error |
| H11 | RQ4 | Early-training gradient ratios predict failure. From the PDF's H2. | Primary P1 runs of soft formulations F2, F3, F4, F6 | AUC ≥ 0.75 with bootstrap CI lower bound > 0.5, using the median log gradient ratio over steps 0–300 | AUC CI includes 0.5 | ROC AUC |

A null result is reportable. Correlations (H11) are never interpreted as causation unless an intervention (H5) moves the outcome in the predicted direction.

---

## 2. Test problem P1: 1D layered capacitor

The domain is x ∈ [0, 1] with the interface at a = 0.5. Permittivity is ε = 1 for x < a and ε = κ for x > a. Boundary conditions are φ(0) = 1 and φ(1) = 0. The governing equation is (ε φ′)′ = 0, with E = −φ′ and D = εE.

Exact solution (verified symbolically; `results/validation/phase2_derivation_checks.txt`, 20/20 PASS):

- D = 1/[a + (1−a)/κ]
- φ₁ = 1 − D·x for x < a
- φ₂ = D·(1−x)/κ for x > a
- E₁ = D and E₂ = D/κ
- Capacitance per unit area: C = D (from flux) = 2W (from energy, with V = 1)

**Units.** Everything is nondimensional. Length is in units of the gap L. Potential is in units of the applied voltage V. Permittivity is in units of ε₁ε₀, so E is in V/L and D is in ε₁ε₀V/L.

**Wrong reference.** The uniform-permittivity solution φ_w = 1 − x. At κ = 80 it gives E₁ = 1 against the exact 1.9753, a 49.4% error, so evaluator self-test 3 must flag it.

**Inverse contrast.** For the PDE, the arrangement (κ, 1) is equivalent to (1, κ) under x → 1 − x, φ → 1 − φ and scaling by 1/κ. The PINN losses are not scale-invariant, so the two arrangements are not equivalent for training. Inverse contrasts are not run in P1; they are run in P2.

---

## 3. Evaluator metrics (frozen definitions)

**Evaluation grid.** Gauss–Legendre panels with breakpoints at {0, a−0.05, a, a+0.05, 1}. Panels are at most 0.05 long, with 16 nodes each, giving 320 points. No panel straddles the interface. One-sided quantities at x = a are taken as limits from the respective material's representation.

**Relative L2 error** over region Ω_R: e_R(u) = ‖u_pred − u_ex‖_{L2(Ω_R)} / ‖u_ex‖_{L2(Ω_R)}, computed by the quadrature above, for u ∈ {φ, E, D}. Regions are global, material 1, material 2, band |x−a|<0.05, and the two band halves.

**Interface diagnostics.**
- Potential jump [φ] = φ(a⁺) − φ(a⁻), reported relative to V (not to φ(a), which tends to 0 as κ grows).
- Displacement jump [D_n] = D(a⁺) − D(a⁻), reported relative to D_exact.

**Capacitance and energy.**
- Flux capacitance C_flux = ½[D(0) + D(1)]/V.
- Gauss imbalance |D(0) − D(1)|/D_exact.
- Energy capacitance C_W = 2W/V², where W = ½∫εE² is computed by quadrature.
- Each is reported as relative error against D_exact.

**Effective contrast.** κ_eff is the ratio of mean E over material 1 to mean E over material 2, both excluding the band. Report ln(κ_eff/κ).

**Boundary errors.** |φ(0) − 1| and |φ(1)|.

**Non-finite check.** Any NaN or Inf anywhere marks the run failed with the flag `nonfinite`.

**Cost.** Gradient evaluations (counted, not assumed), collocation-point evaluations, parameter count, wall-clock (training loop only, measured after JIT warm-up; compile time recorded separately), and peak RSS memory.

---

## 4. Failure definitions and why both a pooled and a strict criterion exist

The **primary criterion**, from the brief, is that a run fails if max(e_E,band, e_D,band) > 5%.

Pooling E across the band hides material 2 at high contrast. E jumps by a factor κ, so ‖E_ex‖ in the band is dominated by the material-1 side. D is continuous, so pooled D weights both sides equally. Within one material, the relative errors of E and D are identical because ε is constant there. The **strict secondary criterion** is therefore the larger of the two half-band field errors exceeding 5%. Pooled errors never exceed the larger side error, and pooled D is within √2 of it, so the strict criterion is never weaker than the primary one. Both are reported. Only the primary criterion is used for headline failure rates.

**Potential failure**, reported separately and never mixed with field failure, is e_φ,global > 5%.

**Secondary indicators**, stored but not used for the failure flag:
- |ln(κ_eff/κ)| > ln 2;
- |[D_n]|/D > 5%.

A **configuration** fails if at least 3 of its 5 seeds fail. The **critical contrast** is the smallest primary κ whose configuration fails. If any larger κ passes, the configuration is labelled non-monotone; the cautionary failure at equal coefficients in L22 shows this can happen.

---

## 5. Formulations

Every network is a 3×32 tanh MLP with Glorot-normal weights and zero biases. Losses are means over their point sets. Residual notation: r_i = ε_i φ_i″ in material i, since ε′ = 0 inside each material. The "loss visibility" column predicts whether the loss can distinguish φ_exact from φ_w, and is tested in Section 7.2.

| | Ansatz | Loss terms (unit weights unless stated) | Interface treatment | Boundary | Params |
|---|---|---|---|---|---|
| **F1** | φ = N(x) | mean r(x)² over interior points; (φ(0)−1)²; φ(1)² | None. No interface point. | Soft | 2209 |
| **F2** | φ = N₁(x) for x<a, N₂(x) for x>a | Σ_i mean r_i²; BC terms; (N₁(a)−N₂(a))²; (ε₁N₁′(a) − ε₂N₂′(a))² | Soft continuity and flux at x=a | Soft | 4418 |
| **F3** | φ = N(x, \|x−a\|) (cusp feature, Tseng et al.) | mean r²; BC terms; (ε₁φ′(a⁻) − ε₂φ′(a⁺))², with φ′(a±) = ∂₁N(a,0) ± ∂₂N(a,0) | Continuity built in (continuous inputs). Flux soft. | Soft | 2241 |
| **F4** | φ = (1−x) + x(1−x)[N_A(x) + max(x−a, 0)·N_B(x)] | Σ mean r²; flux penalty (ε₁φ′(a⁻) − ε₂φ′(a⁺))² | Continuity and BCs exact by construction. **Flux is NOT exact**; it is penalised. | Hard | 4418 |
| **F5** | φ = (1−x) + x(1−x)·N(x, \|x−a\|) | Energy J = Σ_i \|Ω_i\| · mean ½ε_i φ′² (quadrature on the same stratified points) | Flux continuity is the natural condition of the minimiser. Continuity is built in. | Hard | 2241 |
| **F6** | as F2 | as F2, each term divided by its physical scale (below) | Soft continuity and flux | Soft | 4418 |

**Why the cusp feature (F3, F5).** The exact φ is continuous with a kink at a. A tanh network of x alone is C^∞ and can only approximate a kink by steep curvature. A network of (x, |x−a|) represents a kink exactly, because |x−a| itself has the derivative jump.

**F4 note.** The max(x−a, 0) branch makes φ′ jump by x(1−x)N_B(a) at a while keeping φ continuous. Its distinction from F3 is hard BCs and a multiplicative branch rather than input augmentation. The brief forbids claiming more than is built in: only continuity and the Dirichlet BCs are exact.

**F4-full (Intervention D only, 1D only).** Two networks plus a linear buffer g_i = c_i0 + c_i1·x per material, after Chung et al. The four coefficients solve a 4×4 linear system that enforces both BCs, continuity and flux exactly. Its determinant is a·ε₂ + (1−a)·ε₁ > 0 (verified), so it is always solvable. **Caveat:** the exact P1 solution is piecewise linear, so it lies in the buffer space with N_i = 0. F4-full on P1 therefore tests optimisation from initialisation, not representation, and its P1 results must not be generalised.

**F5 derivation.** J[φ] = ½∫₀¹ ε φ′² dx over H¹ functions with the Dirichlet data. The first variation, using test functions v with v(0) = v(1) = 0, is ∫εφ′v′ = 0 for all v. Integrating by parts on each material gives (εφ′)′ = 0 in each material, plus [εφ′]·v(a) = 0 for all v(a), which forces flux continuity. The minimiser is therefore exactly the P1 solution. The minimum loss value is W = D/2, which is not zero, so the F5 loss value is not an error indicator.

**F6 weights (frozen).** Every term is divided by its natural physical scale before squaring:
- PDE residual r_i by D_s/L;
- BC and continuity residuals by V;
- flux residual by D_s;

where D_s = ε_h·V/L and ε_h = 2ε₁ε₂/(ε₁+ε₂) is the harmonic mean, as in contrast-robust Nitsche/DG weighting. Properties:
- F6 equals F2 when ε₁ = ε₂.
- For P1 with a = ½, D_s equals D_exact exactly (verified). This is a coincidence of the geometry, disclosed rather than hidden; in 2D, D_s is only an estimate.

**Known limitation shared by all.** A smooth network has spectral bias. Only F3, F4 and F5 contain an exact representation of the kink.

---

## 6. Training budget and sampling

### 6.1 Optimiser and budget
Adam (β = 0.9, 0.999, ε = 10⁻⁸), full batch. Learning rate follows a cosine decay from 10⁻³ to 10⁻⁴ over 3000 steps. Budget: exactly 3000 gradient evaluations per run, counted by an instrumented loss function. Precision: float32 (float64 is Intervention C).

### 6.2 Points
- Interior: 128 per material, stratified-jittered (one uniform draw in each of 128 equal strata), fixed per seed and not resampled during training.
- Boundary: the 2 endpoints.
- Interface: the single point x = a for every formulation except F1.

The same seed produces the same points for every formulation.

### 6.3 Parameter counts
These differ (2209–4418). The difference is reported, not hidden. A parameter-matched two-network variant (width 22, 2158 parameters) runs in the robustness set. The primary comparison stays at a fixed training budget.

### 6.4 Pilot
Seeds 100–102, κ ∈ {1, 80}, all formulations, learning rate ∈ {10⁻³, 3×10⁻³}: the same two trials for every formulation. One shared value is chosen, the one with the lower total pilot failure count (tie → 10⁻³). Pilot rows are stored with `run_type = pilot` and never enter reported statistics.

### 6.5 Logging
Per-term loss and gradient norms are logged every 25 steps (plus step 0 and the final step). Full evaluator metrics are logged every 250 steps. Fields are listed in the YAML.

---

## 7. Diagnostics for P1

**7.1 Regression-only baseline.** Each distinct architecture (single 1-input, cusp 2-input, two-network, F4 ansatz) is fit to φ_exact. The loss is MSE on φ at the same collocation points, with the same 3000 Adam steps and learning rate. Evaluation uses the same metrics. 4 architectures × 6 κ × 5 seeds = 120 runs, labelled `run_type = regression`. These are representation baselines, not PINN results.

**7.2 Loss visibility.** Each formulation's loss is evaluated with φ_exact and with φ_w substituted directly, for each κ. The loss functional is the same; the network is replaced by the analytic function. Expected results, to be confirmed or refuted:
- F1: both losses = 0 (invisible).
- F2–F4 and F6: L(φ_w) > 0 = L(φ_exact) through the flux term.
- F5: J(φ_w) − J(φ_exact) = ½[a + κ(1−a)] − ½D > 0.

Smoothing check: F1 with sigmoid-smoothed ε (widths 0.01 and 0.002) and 16 extra points inside the band, to test whether smoothing restores visibility.

**7.3 Gradient analysis.** Per-term gradient-norm ratio trajectories; Spearman ρ against band error by κ and formulation; H11 AUC. Sample counts and bootstrap intervals are always reported.

---

## 8. Two-dimensional benchmarks (pre-specified; detailed configs written before each phase starts)

### 8.1 Gate
No 2D work starts before the P1 evaluator passes all self-tests, the primary sweep is complete, the failure maps exist, and the pipeline regenerates them.

### 8.2 Choosing the two strongest formulations
Fixed now, before any results exist:
1. Lowest total failed seeds over the P1 primary grid.
2. Tie-break: lower median primary error at κ = 2000.
3. Then lower median wall-clock.

F1 is always fully swept. The two strongest are fully swept over the primary set plus the inverse set (P2 only). The remaining formulations use κ ∈ {1, 80, 2000}. The decision is recorded in the change log before the 2D sweep.

### 8.3 P2: dielectric cylinder in a uniform field
Radius R = 0.25, centred in [−1, 1]². κ = ε_in/ε_out. Dirichlet data on the box come from the exact isolated-cylinder solution, so the reference is exact in the box (verified):
- φ_in = −E₀·2ε_out/(ε_in+ε_out)·x
- φ_out = −E₀x + E₀·K·R²·x/r², with K = (ε_in−ε_out)/(ε_in+ε_out)

The effective contrast is recovered from the interior field: κ_eff = 2E₀/E_in − 1. An independent conforming FEM cross-check with convergence is required.

### 8.4 P3: 2D line charge near a planar interface
Line charge λ at (0, h) with h = 0.25, ε₁ for y > 0, ε₂ for y < 0, in a [−1, 1]² box with exact Dirichlet data. Image solution (verified):
- φ = −λ/(2πε₁)·[ln r₁ + (ε₁−ε₂)/(ε₁+ε₂)·ln r₂] for y > 0
- φ = −λ/(2π)·2/(ε₁+ε₂)·ln r₁ for y < 0

This is a 2D (per-unit-length) problem and is labelled as such. The singular part is subtracted: the PINN learns φ − φ_s, with φ_s = −λ/(2πε₁)·ln r₁ restricted to the upper half and continued harmonically. The exact form is finalised and verified before Phase 8. Evaluation regions: excluded disc r₁ < 0.05; near-source ring 0.05 ≤ r₁ < 0.15; far field; interface band |y| < 0.05.

### 8.5 P4: two-material corner
Material 2 occupies the quadrant x > 0, y > 0 (90°); material 1 occupies the remaining 270°. The reference is the exact singular solution r^λ·Φ(θ). λ comes from the transcendental interface-matching equation, and the solution is exact on the box when used as Dirichlet data; the derivation is verified before Phase 9. Field metrics are taken at fixed distances r ∈ {0.02, 0.05, 0.1} on fixed rays, together with the fitted coefficient of r^λ. No pointwise peak is compared (H10).

### 8.6 P5: thin high-permittivity film
A strip of length 1 and thickness t ∈ {0.1, 0.02, 0.005}, centred in [−1, 1]², with an applied field from Dirichlet data φ = −E₀y on the box. The reference is a body-fitted FEM with at least 4 elements across the film thickness and refinement until the band field changes by less than 0.1%. A 1D three-slab exact solution is used as a code check. The thickness × collocation-density sweep is specified before Phase 10.

---

## 9. FEM references

**P1 FEM.** Uniform P1 elements with N even, so there is a node at a. Each element uses its own coefficient. Dirichlet values are imposed by row replacement. The element field is E_e = −(φ_{j+1} − φ_j)/h and D_e = ε_e·E_e. Evaluation points are interior to elements, so no field is ever averaged across the interface.

**Degeneracy.** The exact P1 solution is piecewise linear, so conforming FEM reproduces it to round-off for any N ≥ 2. This makes FEM-versus-PINN cost on P1 uninformative: FEM wins trivially. It is still reported, but H7 is tested on P2 and P5.

**Convergence check.** To verify the FEM code, a manufactured variant P1-S with source f = 1 must show O(h²) convergence in φ and O(h) convergence in the field. Self-test 8 uses a non-conforming mesh (N odd). Its error is reported and explained, with no assumed magnitude.

---

## 10. Interventions (pre-registered; 10 matched seeds; one factor changed)

| | Change | Applied to | κ | Prediction (direction) |
|---|---|---|---|---|
| A | Gradient rebalancing. Every 10 steps λ̂_i = ‖∇L_pde‖/‖∇L_i‖ and λ_i ← 0.9λ_i + 0.1λ̂_i for the BC, continuity and flux terms. Uses L2 norms, a documented variant of Wang et al. | F2, F3 | primary set | Lowers primary error and raises F2's critical contrast by ≥1 grid step |
| B | Residual normalisation: r_i / ε_i | F1, F2 | primary set | Helps F2 at κ ≥ 80; no change for F1 (still invisible) |
| C | float64, same everything else. Full-grid replica, plus a 2×2 precision × optimiser design | all (replica); F2, F3 (2×2) | primary; {1, 80, 2000} | Under Adam, median error changes by ≤1.25× and the failure flag changes in ≤1 configuration. Under L-BFGS, larger effects (per L32). |
| D | Hard interface: F2 vs F4 (continuity hard) vs F4-full (all hard) | — | primary set | F4 fails less than F2 at κ ≥ 400. F4-full reaches < 1% error at all κ (expected trivially; see caveat in Section 5). |
| E | Optimiser: Adam (3000) vs L-BFGS (≤3000 function+gradient evaluations, history 50, zoom line search) vs Adam 2000 → L-BFGS 1000 | F2, F3, F5 | {1, 80, 2000} | L-BFGS variants give lower median error for F3 and F5 |

A successful intervention is evidence about a mechanism for these problems. It is not a universal explanation.

---

## 11. Robustness (P1; one factor at a time; baseline retained)

Seven variations, each run for all formulations, κ ∈ {1, 80, 2000}, 5 seeds:

1. Width 16
2. Width 64
3. Depth 2
4. Depth 5
5. Interior points 32 per material
6. Interior points 512 per material
7. Learning rate 3×10⁻³ with cosine decay

Also run: the parameter-matched two-network variant, and 16 extra band points per side (interface sampling). In 1D, boundary sampling is fixed at the two endpoints, so it cannot be varied and is reported as not applicable. Precision and optimiser are covered by Interventions C and E. Total: 630 runs plus the extras.

---

## 12. Cost comparisons

**Error at equal cost.** Error is reported at the PINN's measured budget (wall-clock and gradient evaluations) against FEM with the mesh whose wall-clock is closest.

**Cost at equal accuracy.** Time to reach 5% and 1% primary error is reported. For a PINN this is the first logged evaluation (every 250 steps) that meets the target; if none does, the result is "not reached". For FEM it is the smallest mesh in a refinement sequence that meets the target.

**Timing.** Timing uses `time.perf_counter`, single-threaded, after warm-up. JIT compile time is reported separately. Hardware is recorded per run. Equal gradient evaluations do not imply equal wall-clock, and the report says so.

---

## 13. Statistics

For every configuration, the report includes:
- seed count, failed count, failure rate with Clopper–Pearson 95% interval;
- median, interquartile range, minimum and maximum.

Paired interventions use per-seed differences, a sign test, and a bootstrap CI of the median difference (10,000 resamples, seed 12345).

Five seeds give coarse failure-rate estimates. For example, 0/5 has a 95% upper bound of 52%, and the report says so. No best-seed reporting. Valid failed runs are never removed. Technically excluded runs keep their records with a reason code.

---

## 14. Acceptance gates

**Gate 1 (before any PINN result is accepted).** The eight evaluator self-tests from the brief:
1. exact solution scores about 0;
2. conforming FEM scores about 0;
3. the uniform-ε answer at κ = 80 is flagged as failing;
4. a 4% perturbation passes and a 6% perturbation fails;
5. NaN and Inf are flagged;
6. known jumps are detected;
7. errors are stable under grid doubling;
8. a non-conforming mesh is compared and its discrepancy explained.

Results are saved to `results/validation/`.

**Gate 2.** Derivation checks pass. They already do: 20/20.

**Gate 3 (per formulation).** Each formulation passes a unit test: the loss evaluated with the exact solution matches the expected value from Section 7.2, and the gradient logger returns finite norms for every term.

**Gate 4.** Every table and figure regenerates from `results/` with one script.

---

## 15. Conflicts between the brief and the team PDF, and their resolution

| Item | Brief | PDF | Resolution |
|---|---|---|---|
| Contrasts | {1, 5, 20, 80, 400, 2000} | {1, 3, 10, …, 10⁴} | Brief's set is primary; 10⁴ is a pre-registered extension |
| Main precision | (Intervention C implies float32 baseline) | 64-bit main grid | float32 primary plus a full float64 replica. L31/L32 make precision a first-class factor rather than an afterthought. |
| Optimiser | Adam; L-BFGS as intervention | Adam then L-BFGS | Brief: Adam primary; Adam→L-BFGS in Intervention E |
| Peak junction field (PDF H4) | Forbidden as pointwise peak | Compare peaks | Fixed-distance metric (H10) |
| Critical contrast (PDF H2) | — | Assumes a threshold | Non-monotone label added (Section 4) |
| Configuration failure | — | "___ of its seeds" | ≥3/5 |
| P6 (charged interface, σ) | Not in brief | Listed | Not in scope; optional after P5 if resources remain |

---

## 16. Records and reproducibility

**Run ID.** `{problem}-{formulation}-k{kappa}-s{seed}-{run_type}-{variant}`.

**Outputs per run.** Each run writes:
- a JSON config;
- one CSV row in `results/raw/runs.csv` (schema in `docs/database_schema.md`, written in Phase 3);
- a gradient log `results/raw/gradlogs/{run_id}.csv`;
- a metric trajectory `results/raw/trajectories/{run_id}.csv`.

Every row stores the git commit, package versions and hardware.

**Resumability.** A run is skipped if its row exists with status `complete`. Partially completed runs are rerun from scratch. A P1 run takes about 10 s, so checkpointing within a run is unnecessary. 2D runs will checkpoint every 500 steps.
