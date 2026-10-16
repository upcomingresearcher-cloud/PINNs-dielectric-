# Phase 2 Progress Report — Gaps, Hypotheses, Frozen Protocol
Date: 2026-10-16

## Implemented / executed
- 6 targeted searches (precision, closed-form interface weights, singular sources, thin layers/multi-material corners); 7 new records (L31–L37); gaps G10, G12, G14–G16 reclassified (none remain "unsearched").
- Symbolic verification of P1, P2, P3 reference solutions and F4-full/F6 algebra: `scripts/verify_reference_derivations.py` -> 20/20 PASS. One hand-derivation sign error (F4-full determinant) caught and corrected.
- Frozen protocol: `configs/protocol_v1.yaml` (parsed and validated) and `docs/experimental_protocol.md`; change log started.
- Installed optax 0.2.8 (L-BFGS for Intervention E).

## Key design decisions
- 11 hypotheses (H1–H11) with numeric support/refute thresholds; PDF's H4 replaced by a fixed-distance metric (H10); "critical contrast" allows non-monotone outcomes.
- Primary failure = brief's pooled band criterion; a strict per-side criterion is also reported because pooled E hides material 2 at high kappa (Sec. 4).
- float32 primary + full float64 replica; precision crossed with optimiser because L32 reports the precision effect is optimiser-dependent.
- F6 weight rule frozen (harmonic-mean flux scale); disclosed that it coincides with the exact flux in P1.
- Interventions use 10 seeds (5 seeds cannot give p<0.05 in a paired sign test: min p = 0.0625).
- Identified that P1 is degenerate for FEM (exact in FEM space) and for F4-full (exact solution in buffer space); H7 moved to P2/P5.

## Failed / unresolved
- Several bibliographic entries still marked VERIFY. Search depth remains modest (16 queries total).
- P2–P5 details (meshes, collocation budgets, singular subtraction for P3, corner exponent for P4) are specified only at design level; detailed configs are due before each phase.

## Next: Phase 3
Implement P1 analytical reference, conforming FEM (+ P1-S convergence), evaluator, database schema, and the 8 self-tests; Gate 1 must pass before any PINN code runs.
