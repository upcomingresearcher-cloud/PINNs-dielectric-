# Phase 3 Progress Report — P1 References, FEM, Evaluator, Self-tests
Date: 2026-10-16. Code commit for all recorded results: aabfd44.

## Implemented
- `src/problems/p1.py` — P1 exact solution, wrong uniform-eps solution, manufactured variant P1-S (FEM verification only); common prediction interface `pred(x, side) -> (phi, dphi)`.
- `src/fem/fem1d.py` — P1 FEM, exact element-averaged coefficient, row-replacement Dirichlet, side-aware piecewise-constant field reconstruction (never averages across the interface).
- `src/evaluator/evaluator1d.py` — all protocol Sec. 3–4 metrics and failure flags; float64; optional FEM-node quadrature breakpoints.
- `src/experiments/records.py`, `provenance.py` — schema (runs: 78 columns; FEM: 55), append-with-validation, resume, run/config IDs, git/software/hardware provenance.
- `docs/database_schema.md` — generated from code.
- `src/analysis/fig_fem_p1.py` — first database-driven figure.

## Executed and passed
- Gate 1 self-tests: **98/98 checks pass** (`results/validation/evaluator_selftests_p1.{csv,json}`), covering all 8 brief tests.
- Mutation tests: 6 deliberately injected bugs (wrong permittivity, flipped field sign, wrong threshold, interface side ignored, FEM single coefficient, NaN hidden) — **all caught**.
- pytest: 19/19 (incl. schema-vs-evaluator guard and CSV round-trip/resume tests).
- FEM reference sweep: 105 rows (7 contrasts × 15 meshes) with median-of-7 timings.

## Results
- Exact solution and conforming FEM score at round-off (≤ 5e-10 even at N = 20000; ≤ 1e-15 for N ≤ 20).
- FEM code verification on P1-S: φ order 2.000, E order 1.000 at κ = 1, 80, 2000.
- Wrong uniform-eps solution at κ = 80: primary field error 27.9 (2793%), material-1 E error 49.4% (matches the analytic 1 − 1/D), D jump 40×D — flagged.
- Non-conforming meshes (self-test 8): band field error converges at order ≈ 0.5 and φ at order 1 (degraded from 2). Explanation: the cut element cannot represent the kink, so an O(1) field error sits in one element of width h (L2 norm ∝ √h). At κ = 80, φ error < 5% from N = 21, but band field error stays > 5% until N ≈ 20001. This is a classical-method instance of the masking hypothesis H2, not a PINN result.

## Failed / fixed
- Self-test 7 (informational row) exposed ~5% quadrature error when evaluating FEM predictions with element kinks inside GL panels. Fixed by adding FEM nodes as quadrature breakpoints (logged as protocol 1.0.1; definitions unchanged; FEM sweep regenerated).
- Schema guard test initially failed on key order (not content); test corrected to set equality with explicit difference reporting.

## Remaining issues
- Peak-memory measurement and gradient-evaluation counting are implemented in Phase 4 with the training loop.
- Loss-visibility table schema exists; it is populated in Phase 5.

## Next: Phase 4
Implement F1–F6 (+ F4-full) in JAX, the instrumented training loop with per-term gradient logging, and Gate 3 tests (loss of exact solution matches expected value; finite gradient norms for every term); then run the pre-registered pilot.
