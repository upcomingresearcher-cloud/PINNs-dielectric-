# Research Gap Report

Phase 1 deliverable D (16 October 2026). Detailed rows are in `research_gap_matrix.csv` (17 topics). This report states which candidate contributions survive the literature check, which do not, and the refined contribution statement that Phase 2 hypotheses will be built on.

## 1. Classification summary

| Category | Topics |
|---|---|
| 1. Investigated extensively | Interface-aware PINN-type solvers in general (G01); high contrast ≥10³ for interface-aware networks (G02); energy/Nitsche formulations (G13) |
| 2. Studied elsewhere, not established for the dielectric problem | Canonical electrostatic geometries with physical metrics (G04); per-term gradient statistics as failure predictors (G08) |
| 3. Evidence incomplete | Quantification of the single-network silent failure (G03); combined matched-seed intervention design (G09) |
| 4. Potential gap supported by the literature | Interface-band E/D versus global φ error (G05); seed-level failure rates (G06); fixed-budget multi-family contrast sweep (G07); cost-at-equal-accuracy versus FEM for interface problems (G11); reusable interface evaluator (G17) |
| 5. Novelty uncertain (not yet searched) | Precision effects (G10); contrast-derived closed-form weights (G12); line charge near interface (G14); wedge/triple junction (G15); thin film (G16) |

## 2. Candidate contributions that do not survive

**"PINNs have not been tested at high contrast."** This is false for the field. Interface-aware networks have been run at 10³ to 10⁵ (L09–L13). It survives only in the narrow form "the specific papers reviewed by the team used low contrast."

**"We identify why single-network PINNs fail at interfaces."** Nohra & Dufour (L02, Section 6) already show that a piecewise-constant coefficient makes the flux condition invisible to the strong-form residual. We can quantify this, but not claim to have discovered it.

**"Hard constraints remove the need for loss balancing."** This is Chung et al.'s claim (L08), not ours.

## 3. Gaps that survive, with the evidence that supports them

**A. Field-level, interface-local accuracy is not the reported quantity (G05).** Neural interface papers report global solution error, sometimes global H1/H2 error, and show interface-concentrated error only visually. Classical numerics shows gradients are typically an order less accurate than the solution at interfaces (L21). A thresholded interface-band E and D metric, reported beside φ error across formulations and contrasts, was not found.

**B. Robustness evidence is thin (G06, G07).** The closest study (L08) compares six methods at a single contrast with the best of four initialisations. Others use one contrast with five runs (L13) or three repeats without interface methods (L01). A fixed-budget, multi-family comparison over a contrast sweep, with per-configuration failure rates, was not found.

**C. Mechanism claims rest on correlation or single remedies (G08, G09).** Gradient imbalance (L18), NTK eigen-imbalance (L02) and loss-scale imbalance (L06) are each argued separately. A matched-seed design that changes one factor at a time (weights, residual scaling, precision, hard constraints, optimiser) on the same interface problem was not found.

**D. PINN versus FEM on interface problems (G11).** The systematic PINN–FEM comparison (L16) has no interfaces. Interface-PINN papers report cost only relative to other PINNs. Cost-to-target for 5% and 1% interface-band field error against interface-conforming FEM was not found.

**E. Dielectric-specific framing (G04).** This is weaker than A–D because the mathematics is generic. Its value is applied rather than methodological. Physical quantities (capacitance, effective contrast, interior field of a dielectric cylinder) are natural evaluation targets for engineers, and recent EM papers (L19, L20) still smooth the permittivity without quantifying the resulting field error.

## 4. Refined contribution statement (candidate)

> A reproducible, pre-registered evaluation of six PINN formulation families on electrostatic dielectric-interface problems. It reports interface-band electric-field and displacement error alongside potential error over a contrast sweep (κ = 1 to 2000) with at least five seeds per configuration. Failure attribution uses matched-seed interventions, and cost-at-equal-field-accuracy is compared against interface-conforming FEM.

What we would claim if the experiments succeed:

1. Failure maps and failure rates for these formulations, budgets and problems.
2. The measured relationship between potential error and interface-band field error.
3. Which interventions move the failure boundary for these problems.
4. Measured cost comparisons on this hardware.

What we would not claim: discovery of the interface-failure mechanism, the first high-contrast PINN study, the first hard-constrained interface PINN, or general superiority or inferiority of PINNs versus FEM.

How it differs from the closest work (Chung et al., L08): they propose architectures and show accuracy at κ = 10 with best-initialisation runs and solution-L2 metrics. We propose no new architecture as the main contribution. We evaluate existing families, including a hard-constrained one modelled on theirs, across a contrast sweep, with field metrics, seed statistics, interventions and FEM cost. F6 (physics-derived weights) is the one design element of ours. Its novelty is uncertain (G12) until the Phase 2 search.

## 5. Issues to resolve in Phase 2 before the protocol is frozen

1. **Contrast set.** The brief prescribes {1, 5, 20, 80, 400, 2000}. The supplied protocol PDF proposes {1, 3, 10, 30, 100, 300, 1000, 10⁴}. Recommendation: keep the brief's set as primary because it is mandatory, and add 10⁴ as a pre-registered extension so results can be compared with L09/L11.
2. **H2 "critical contrast".** It must handle non-monotone failure (L22): define the critical contrast as the smallest κ with a failing configuration, and report separately whether any larger κ passes.
3. **H4 peak field at a junction.** It must be restated with a fixed-distance or asymptotic-coefficient metric.
4. **Targeted searches** for G10, G12, G14–G16, plus inspection of L18, L03 and L30 originals.
5. **F2 interface loss** must use flux continuity (cPINN-style), not XPINN residual continuity.
6. **Best-initialisation reporting** in prior work means our seed medians will look worse than published numbers. The manuscript must explain this rather than tune toward published values.
