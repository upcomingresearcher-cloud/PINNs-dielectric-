# Phase 4 Progress Report — PINN Formulations, Training Loop, Pilot
Date: 2026-10-16. Code commit for pilot runs: 42b1a3c. Protocol now v1.1.0.

## Implemented
- `src/pinns/nets.py` (Glorot-normal tanh MLPs), `src/pinns/formulations.py` (F1–F6 + F4-full as representation + loss functional), `src/pinns/diagnostics.py` (analytic substitution into the training loss), `src/pinns/train.py` (Adam + cosine schedule; counted gradient evaluations; per-term gradient norms every 25 steps via jacrev; evaluator trajectory every 250 steps; AOT-compiled executables cached across kappa/seed), `scripts/run_sweep.py` (resumable; technical crashes recorded, never dropped).
- Parameter counts match protocol: F1 2209, F2 4418, F3 2241, F4 4418, F5 2241, F6 4418, F4-full 4418.

## Executed and passed
- Gate 3: 55 formulation tests — exact-solution loss (0, or D/2 for F5), wrong-solution loss (0 for F1 = invisible; > exact for F2–F6), hard constraints exact at random init (F3, F4, F5, F4-full), F6 ≡ F2 at kappa = 1, finite per-term gradient norms, float32/float64 dtype control, budget counting, seed-only collocation.
- Gate 3 mutation tests: 3 injected bugs (missing interface terms, broken hard continuity, missing energy quadrature weights) — all caught.
- Full suite: 77/77.
- Pilot: 72 runs (6 formulations × kappa {1, 80} × 3 pilot seeds × 2 learning rates), 0 non-finite, 0 technical failures. Plus 3 pilot-labelled bug-check runs.

## Pilot decision and observations (pilot results are NOT reported findings)
- Learning rate stays 1e-3 (12 vs 12 failures; pre-registered tie-break).
- All formulations pass at kappa = 1 -> benchmark is valid.
- kappa = 80: F2 and F6 pass 3/3; F1, F3, F4, F5 fail 3/3. F1's error (27.8) equals the uniform-eps solution's (27.9): silent convergence to the wrong answer, as predicted.
- Bug check (20000 steps, float64): F3 and F5 reach < 5%, F4 keeps decreasing (0.11) -> budget-limited optimisation, not implementation error. Response: a 12000-step budget variation was added to the robustness plan before the primary sweep.
- Training cost: 1–3.3 s per 3000-step run (median), plus first-compile 2–9 s per formulation.

## Problems found
- The compute environment did not support long-running background processes; the first pilot launch stopped after 17 runs. Database verified intact; sweeps now run in resumable foreground chunks.
- One Gate 3 test had a wrong expectation (F5 at kappa = 1), fixed in the test, not the code.

## Remaining issues
- Peak memory is process-level (ru_maxrss), not per-run; documented in the schema.

## Next: Phase 5
Primary P1 sweep (180 runs, seeds 0–4) + kappa = 1e4 extension (30) + regression-only baselines (120) + loss-visibility table; then failure maps.
