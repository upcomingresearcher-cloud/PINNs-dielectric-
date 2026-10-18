# Phase 6: P1 Analysis and Pre-registered Hypothesis Tests
Source: `src/analysis/p1_hypotheses.py` -> `results/processed/p1_hypothesis_tests.json`. All numbers regenerate from `results/raw/`.

| H | Verdict (pre-registered wording) | Key numbers |
|---|---|---|
| H1 | **Supported** | F1 fails 5/5 at every kappa >= 5; F2 Spearman rho(log kappa, error) = 0.96 [0.90, 0.98]; failed seeds (kappa > 1): F1 25, F2 8, F3 19, F4 20, F5 20, F6 3 |
| H2 | **Inconclusive** | median R = field/potential error = 2.39 [1.73, 3.18] (threshold 3); R > 1 in 97% of 150 runs; 23% (22/95) of field failures have potential error < 5% |
| H3 | **Refuted** | 0/11 failing F2/F6 runs are flux-dominant; failing runs violate potential continuity (median 5.3% of V) while flux jump is small (median 2.6e-4) |
| H4 | **Refuted by pre-registered criterion** (intervention clause pending Phase 11) | phi-only regression with F2 architecture > 5% where F2 fails; F5 loss ratio wrong/exact only 1.8 (kappa 5) and 5.5 (kappa 20) |
| H11 | **Supported**, strongly confounded with kappa | AUC 0.85 [0.77, 0.92]; ratio vs log kappa rho = 0.81; exploratory within-kappa AUC 0.42-0.88 (pooled 0.68) |

## Interpretation (separated from measurements)
1. Measured: the single network converges to the uniform-permittivity solution; its loss is identically zero for that wrong solution. Interpretation: F1 failure is a loss-visibility failure (known mechanism, now quantified across contrast).
2. Measured: soft-coupled networks fail through a potential jump at the interface, not a flux jump. Interpretation: at high contrast the unit-weight flux penalty (scale ~ D^2) dominates the continuity penalty (scale ~ V^2 jump), so the optimiser satisfies flux balance first; F6's rescaling (flux / D_s) reduces this, consistent with F6's lower failure count. This is a hypothesis for Intervention A/B, not a demonstrated mechanism.
3. Measured: phi-only regression is less accurate in the field than PINN training with the same architecture; field-supervised regression (exploratory) is better but still > 5% at kappa = 2000 for the two-network architecture (11%) and fails for F1/F4 architectures from kappa = 5. Interpretation: the highest-contrast failures of F2/F6 are partly a fitting limit at this budget/architecture; F1's are a representation limit (smooth net cannot represent the kink); F4's ansatz is hard to fit even by supervision.
4. Measured: interface-aware designs F3-F5 fail from kappa = 20 at 3000 steps while F2/F6 survive to 400. Pilot bug checks showed F3/F5 succeed with 20000 steps. Interpretation: building the interface into the representation did not make these formulations easier to optimise at this budget.

## 2D selection (recorded before any 2D run)
F6 and F2 (by total failed seeds, then median error at kappa = 2000).
