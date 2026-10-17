# HANDOVER — PINNs Across Dielectric Interfaces (resume at Phase 11)

**How to use this file.** Paste this whole file into a new chat and attach `pinn_dielectric_project_phase10.zip`. That is all the new chat needs. This file restates the parts of the original brief that still apply, the frozen designs for the remaining phases, the current state, all results so far, and the operating rules learned the hard way.

**Instruction to the assistant reading this:** you are continuing an existing research project, not starting one. Read this file completely. Then:

1. Unzip the attached package into `/home/claude/` (the directory inside is `pinn_dielectric_project/`).
2. Run the environment setup and verification in Section 4.
3. Continue with **Phase 13** (Phases 11-12 complete; read `docs/phase11_interventions_findings.md` and `docs/phase12_robustness_findings.md`).

Do not redo Phases 1–12. Do not change frozen settings without a change-log entry. Report progress after each phase.

---

## 1. Role and non-negotiable rules (from the original brief)

Act as a research scientist in PINNs, computational electromagnetics, numerical PDEs/FEM, scientific ML, optimisation, experimental design/statistics, reproducible software, and technical writing. The aim is a scientifically defensible investigation of where PINNs work, where and why they fail on dielectric-interface problems, and how their accuracy and cost compare with FEM.

- **Every numerical result must come from actual computation.** Never invent results, fabricate experiments, or claim code ran when it was only written.
- **Do not silently reduce** seeds, contrasts, formulations, training steps or experiments. If resources prevent completion, report exactly which experiments remain incomplete.
- **Do not change a hypothesis because a result is unexpected.** Investigate and report unexpected results.
- **Distinguish measured findings from interpretation.** Never infer causation from correlation alone.
- **No best-seed reporting.** Report the median and spread (IQR) across seeds. Never remove valid failed runs. If runs are excluded for a technical reason, keep their records and state the effect.
- **Change control.** Any change after the freeze goes in `docs/protocol_change_log.md` with date, reason, affected experiments and required reruns. Never silently change a metric definition.
- **If a bug is found:**
  1. Explain it.
  2. Fix it.
  3. Add a regression test.
  4. Determine which results were affected.
  5. Rerun them, keeping the original results distinguishable.
- **A phase is complete only** when its code has executed, its outputs exist, and its acceptance tests pass.
- **Progress report after each phase:** what was implemented, what was executed, what passed, what failed, results, files, remaining issues, next step. Concise but specific.
- **Do not ask the user** to perform the implementation. Make routine technical decisions independently, and document decisions that affect validity.

## 2. Project in one paragraph

We evaluate six PINN formulations on electrostatic dielectric-interface problems, scored on field and displacement error inside an interface band (primary failure criterion: relative E or D error above 5% in the band). We use matched seeds, a fixed training budget and pre-registered hypotheses, and compare against validated FEM. The formulations are:

| ID | Formulation |
|---|---|
| F1 | single network, piecewise ε (deliberately weak baseline) |
| F2 | two networks, soft continuity and flux penalties |
| F3 | cusp-capturing single network with a \|ψ\| input |
| F4 | hard boundary conditions and hard continuity, soft flux |
| F5 | Ritz energy with cusp feature and hard boundary conditions |
| F6 | F2 with frozen physics-derived weights |

The problems are P1 (1D layered capacitor), P2 (dielectric cylinder), P3 (2D line charge near a planar interface), P4 (two-material 90° corner) and P5 (thin high-ε film).

## 3. Current state (end of Phase 10; commit `59deb57`)

| Phase | Content | Status |
|---|---|---|
| 1–2 | Literature (37 records), gap analysis, 11 hypotheses, frozen protocol | complete |
| 3 | 1D evaluator, FEM, self-tests (98/98), mutation tests | complete |
| 4 | Formulations F1–F6 + F4-full, training loop, pilot | complete |
| 5 | Full P1 sweep (180 + 30 extension + 120 regression), loss visibility | complete |
| 6 | P1 hypothesis tests; 2D selection = F6, F2 | complete |
| 7 | P2 cylinder (210 + 60 regression) | complete |
| 8 | P3 line charge (210 + 15 sensitivity + 60 regression) | complete |
| 9 | P4 corner (210 + 60 regression), H10 | complete |
| 10 | P5 thin film (210 + 60 thickness + 40 density + 60 regression) | complete |
| 11 | Controlled interventions A–E (1,155 rows incl. 270 superseded; see docs/phase11_interventions_findings.md) | complete |
| 12 | Robustness (870 rows; see docs/phase12_robustness_findings.md) | complete |
| 13 | Remedy comparison vs FEM (cost) | not started |
| 14 | Final analyses, figures, paper, reproduction guide, educational resource | not started |

Authoritative records inside the zip:
- `docs/experimental_protocol.md` and `configs/protocol_v1.yaml` (frozen design; version 1.5.0 after amendments);
- `docs/protocol_change_log.md` (every amendment, bug and incident);
- `docs/phase6_p1_findings.md` through `docs/phase10_p5_findings.md`;
- `docs/cross_problem_hypotheses_after_P5.md`.

## 4. Environment setup and verification (do this first)

The sandbox is one CPU core (Xeon 2.8 GHz), about 4 GB RAM, no GPU. **PyTorch cannot be installed** (the wheel is too large for the 300 s command limit), so the project uses **JAX on CPU**.

```
cd /home/claude && unzip -q /mnt/user-data/uploads/pinn_dielectric_project_phase10.zip && cd pinn_dielectric_project
sh scripts/setup_environment.sh                 # jax 0.11.2, optax 0.2.8, scikit-fem 12.0.2, gmsh 4.15.2 (+ libGLU via apt), pytest 9.1.1
python scripts/rebuild_p5_references.py         # ~1 min; recreates results/reference/ (excluded from the zip)
timeout 280 python -m pytest tests -q -p no:cacheprovider > /tmp/pt.txt 2>&1; echo "exit=$?"; tail -1 /tmp/pt.txt
python scripts/run_evaluator_selftests.py       # 1D Gate 1: 98/98
for P in P2 P3 P4 P5; do timeout 280 python scripts/run_selftests_2d.py $P > /tmp/s.txt 2>/dev/null; echo "$P exit=$?"; done   # 17/17 each
git log --oneline | head -3                     # expect 59deb57 or later (a handover commit)
```

The full pytest suite takes about 2–3 min. If it exceeds the limit, run `tests/test_evaluator_gate1.py`, `tests/test_records.py`, `tests/test_formulations_gate3.py`, `tests/test_formulations2d_gate3.py` and `tests/test_fem2d.py` separately.

## 5. Operating rules for this sandbox (learned the hard way)

1. **Background processes are killed when a tool call returns.** Never use `nohup` or `&` for sweeps. Run sweeps in **foreground chunks** with `timeout 295` and the runner's `--max-seconds` (about 230–250). The runners are resumable: they skip run IDs already marked `complete`. An interrupted run writes no row and is simply rerun.
2. **Never start a new sweep command while a previous one might still be running.** Overlapping calls once duplicated 5 P4 runs and contaminated 8 timings. The results were identical, but timing was corrupted.
3. **Check real exit codes.** `cmd | tail` hides pytest's exit code. Use `cmd > file; echo "exit=$?"`. `${PIPESTATUS}` does not work, because the shell is `sh`.
4. **Commit and package at the start of every turn.** The sandbox can reset; unsaved runs would need rerunning. Package command:
   ```
   cd /home/claude && zip -qr /mnt/user-data/outputs/<name>.zip pinn_dielectric_project -x "*/__pycache__/*" "*/.pytest_cache/*" "*/results/reference/*"
   ```
   Commit with `git -c user.name="research-agent" -c user.email="agent@local" commit`.
5. **Throughput.** 1D runs take about 2–3 s (about 80–100 per chunk); 2D runs about 15–20 s (about 12–15 per chunk); F4 2D and dense-sampling runs are slower. Compiled executables are cached by formulation, settings and data shapes. Problems whose boundary crosses the interface (P3–P5) recompile per seed.
6. **Provenance.** Every database row stores the git commit (`-dirty` if src/configs/scripts had uncommitted changes), the software environment and the hardware. Commit code before running sweeps.

## 6. Code map (all paths relative to `pinn_dielectric_project/`)

**1D**
- `src/problems/p1.py`: P1 exact solution, wrong uniform-ε solution, manufactured P1-S.
- `src/fem/fem1d.py`: P1 FEM.
- `src/evaluator/evaluator1d.py`: metrics and failure flags.
- `src/evaluator/selftests.py`: Gate 1.
- `src/pinns/formulations.py`: F1–F6, F4full, REG_*, REGD_*.
- `src/pinns/train.py`: `train_run(fname, kappa, seed, cfg, run_type, variant, out_dir, overrides)`. Overrides: `lr_init, steps, width, depth, n_interior, precision`.
- `src/pinns/diagnostics.py`: loss visibility.
- `scripts/run_sweep.py`: 1D runner. Example: `--run-type intervention --formulations F2 --kappas 1,5 --seeds 0,1 --variant xyz --override precision=float64`.

**2D**
- `src/problems/p2d.py`: `P2Cylinder`, `P3LineCharge`, `P3MatSub`, `P4Corner`, `P5Film`, `P5FilmThick`, `P5FilmThin`.
- `src/fem/fem2d.py`: gmsh + scikit-fem P2, graded meshes, interface loads, side-aware evaluation.
- `src/evaluator/evaluator2d.py` and `src/evaluator/selftests2d.py`.
- `src/pinns/formulations2d.py` and `src/pinns/train2d.py`.
- `scripts/run_sweep2d.py`: `--problem P2 --run-type pinn --formulations F1 --kappas 1,80 --seeds 0,1 [--problem-class P5FilmThin --variant t0.005] [--override n_film=2048]`.

**Records**
- `src/experiments/records.py`: schemas, `append_row`, `completed_ids`, `run_id`.
- `src/experiments/provenance.py`.
- `docs/database_schema.md`: generated by `scripts/write_schema_doc.py`.

**Analysis**
- `src/analysis/p1_summary.py`, `src/analysis/p1_hypotheses.py`, `src/analysis/fig_fem_p1.py`.
- `src/analysis/p2_summary.py`.
- `src/analysis/summary2d.py <P3|P4|P5>`, with helpers `h10()` and `p5_variants()`.

**Data**
- `results/raw/runs.csv` (1D, 78 columns) and `results/raw/runs_2d.csv` (2D).
- `results/raw/fem_runs.csv` and `fem_runs_2d.csv`.
- `results/raw/gradlogs/{run_id}.csv` (per-term gradient norms every 25 steps) and `results/raw/trajectories/{run_id}.csv` (evaluator metrics every 250 steps).
- `results/raw/diagnostics_loss_visibility*.csv`, `results/processed/`, `results/figures/` (13 PNGs), `results/validation/`.

**Run-ID format:** `{problem}-{formulation}-k{kappa:g}-s{seed}-{run_type}-{variant}`. Analyses filter `status == "complete"` and de-duplicate by run_id (first by timestamp). Rows with `status = failed_technical` are retained records: 60 from P3's compile-cache shape bug, all rerun.

## 7. Frozen settings (do not change without a change-log entry)

**Training**
- 3 hidden layers × 32 units, tanh, Glorot-normal weights, zero biases.
- Adam (β = 0.9, 0.999, ε = 1e-8), cosine decay from 1e-3 to 1e-4 over 3000 steps.
- **3000 counted gradient evaluations**, float32 primary, full batch.

**Points**
- P1: 128 per material, stratified-jittered.
- 2D: 512 per material, 128 interface, 256 boundary.

**Seeds and contrasts**
- Primary seeds 0–4; intervention seeds **0–9**.
- P1 contrasts {1, 5, 20, 80, 400, 2000} plus extension 1e4.

**Failure criteria**
- **Primary:** max(relL2 E band, relL2 D band) > 0.05, or non-finite.
- **Strict:** per-side band error.
- **Potential failure:** reported separately.
- **Configuration fails** if at least 3 of its 5 seeds fail.

**Statistics**
- Clopper–Pearson 95% intervals for failure rates.
- Median and IQR for errors.
- Paired interventions: per-seed differences, a sign test, and a bootstrap CI of the median difference (10,000 resamples, seed 12345).

**F6 weights:** each residual is divided by its physical scale: PDE by D_s/L_c, BC and continuity by V_c, flux by D_s, with D_s = ε_h·V_c/L_c and ε_h the harmonic mean.

## 8. Results so far (measured; use these in the paper)

**Literature**
- High contrast is already tested by interface-aware networks: 10³ (DCSNN), 10⁴ (cusp PINN), 10⁵ (deep unfitted Nitsche). This corrects the claims in the supplied reviews.
- The F1 silent failure is prior art (Nohra & Dufour, Section 6).
- The closest prior work (Chung et al. 2026) tests κ = 10 only, with best-of-initialisation reporting.
- Precision papers: Xu et al. 2025 (FP64 under L-BFGS); a 2026 follow-up finds the precision effect is about 1.25× under Adam versus 17.7× under L-BFGS.

**FEM / evaluator (P1)**
- On non-conforming meshes, band field error converges at order ½ and φ at order 1. At κ = 80, φ < 5% from 21 elements, but the band field needs about 20,000 elements.

**P1 failed seeds out of 5** (κ = 1, 5, 20, 80, 400, 2000, 1e4)

| Formulation | Failed seeds |
|---|---|
| F1 | 0, 5, 5, 5, 5, 5, 5 |
| F2 | 0, 0, 0, 1, 2, 5, 5 |
| F3 | 0, 0, 4, 5, 5, 5, 5 |
| F4 | 0, 0, 5, 5, 5, 5, 5 |
| F5 | 0, 0, 5, 5, 5, 5, 5 |
| F6 | 0, 0, 0, 0, 0, 3, 5 |

- F1 converges to the uniform-ε solution, and its loss is identically 0 for both the exact and the wrong solution.
- Smoothing ε restores loss visibility, but the smoothed problem's solution has 36% band error at κ = 80 (φ error 1.7%).
- φ-only regression is less accurate than PINN training with the same architecture. Exploratory field-supervised regression still gives 11% at κ = 2000.

**Hypotheses (P1 data, pre-registered verdicts)**

| H | Verdict | Evidence |
|---|---|---|
| H1 | supported | F2 Spearman 0.96 [0.90, 0.98] |
| H2 | inconclusive | median R = field/potential error 2.39 [1.73, 3.18] < 3; R > 1 in 97% of runs; 23% of failures concealed |
| H3 | refuted | failing F2/F6 runs violate potential continuity (median 5.3% of V), not flux (2.6e-4) |
| H4 | refuted by criterion; intervention clause pending | regression baseline underestimates capacity |
| H11 | supported (AUC 0.85 [0.77, 0.92]), heavily confounded | ratio vs log κ ρ = 0.81; within-κ AUC 0.42–0.88 |

**P2 cylinder**
- All formulations fail at all κ ≠ 1, including inverse contrasts. F5 fails at κ = 1: its Monte Carlo energy (0.59–1.59) falls below the true minimum (2.0), meaning it overfits its fixed quadrature points.
- F2/F6 sit at a stationary point with continuity unsatisfied (20,000 steps in float64 stay at 0.486). Not a loss bug: a BFGS consistency test recovers the exact solution with loss about 1e-31.
- R = 7.5; concealed 1.7%. Body-fitted FEM: 3% band error at 7,869 DOFs (0.13 s).

**P3 line charge**
- All fail for κ > 1. For κ < 1, F2 passes at 0.2 and 0.05; F6 fails all inverse contrasts.
- Pre-registered decomposition sensitivity: material-wise subtraction cuts F2's error tenfold at κ = 80 (6.1 → 0.61), but it still fails.
- R = 3.95; concealed 2.4%. FEM: 2.1% at 29,929 DOFs (0.68 s).

**P4 corner**
- All fail at κ ≠ 1; F5 also fails at κ = 1.
- **H10 refuted:** F2 underestimates the near-corner field everywhere (−58% to −88%), but F1, F4 and F5 overshoot in the high-ε quadrant (F1 +570% at κ = 80).
- R = 4.0; concealed 11.7%. FEM: 1.3% at about 30k DOFs, 0.24% at 119k.

**P5 thin film (t = 0.02 main)**
- F2/F6 pass at κ = 5 and fail 2/5 and 1/5 at κ = 20, then fail from κ = 80 (median 0.070). All fail inverse contrasts.
- F3 (0.95) and F5 (0.30) fail at κ = 1. Likely cause: the level set reaches about 150 away from the film and saturates the |ψ| input (a frozen-design limitation, reported, not rerun).
- Film-point density 128/512/2048 has no effect.
- Thinner films improve the band metric (3.4% → 7.0% → 13% for t = 0.005 → 0.02 → 0.1 at κ = 80), but the film-interior field error is 13–16% at every thickness.
- R = 8.2; concealed 37%. FEM: 0.24% at 5,313 DOFs (0.13 s).

**H8 refuted:** τ(P1, P2) on failure counts = 0.30 < 0.6. τ on median error versus P1: P2 0.73, P3 0.07, P4 0.60, P5 0.47.

**Bugs found and fixed (all logged, with regression tests where applicable)**
1. FEM quadrature at element kinks.
2. A pytest exit code hidden by a pipe.
3. The F6 "equals F2 at κ = 1" claim, which is P1-only.
4. A 2D FEM gradient bug.
5. P3 quadrature sector misalignment.
6. P3 material-wise-subtraction variant coefficient and sign errors.
7. Compile-cache key missing data shapes.
8. P4 clip-derivative bug in verification code.
9. Background-process kill and concurrent-run duplicates (sandbox incidents, not code bugs).

## 9a. PHASE 11 RESULTS (complete)

- H5: A mixed (3/12 supported: F3 at κ ≥ 80; F2 worse everywhere, fails 10/10 at κ = 1, 5); B refuted (3/3); D refuted (F4 fails 10/10 at κ ≥ 20); E mixed (9/12 supported; F3 all 6 at 10/10 seeds; F5 refuted at κ = 1 via quadrature overfitting).
- C holds under Adam (median fold 1.0001, 0 flag changes); L-BFGS float64 lower in 6/6 cells (median fold 3.4).
- H4 intervention clause NOT met (A, B leave F2 at κ = 2000 at 9/10 failures) → H4 refuted by criterion. L-BFGS (descriptive): F2 κ = 2000 failures 9 → 4 (f32), → 1 (f64).
- BUG fixed: float64 initialisation differed from float32 (nets.init_mlp); all C rerun as C_f64v2*; originals kept as superseded. Init variability alone gave fold changes up to 20.7×.
- Recommended remedies for Phase 13: L-BFGS (and Adam→L-BFGS) for F2/F3; F5 needs resampled quadrature (not in the frozen design).
- `scripts/run_sweep.py` now has `--max-seconds` and string overrides (`--override optimiser=lbfgs`, `rebalance=1`, `adam_steps=2000`).

## 9. PHASE 11 — Controlled interventions (frozen design; protocol Sec. 10) — DONE, kept for reference

Run all of these on P1, with **10 matched seeds (0–9)**, changing one factor at a time. **The primary sweep has only seeds 0–4**, so first run the unchanged baseline for seeds 5–9: `run_type intervention`, `variant base`, for every (formulation, κ) used below. Matched comparisons then use seeds 0–9 of the identical baseline. Seeds 0–4 can come from the primary rows: same code path and settings, and the determinism is confirmed.

| ID | Change | Applied to | κ | Pre-registered prediction |
|---|---|---|---|---|
| A | Gradient rebalancing: every 10 steps λ̂_i = ‖∇L_pde‖/‖∇L_i‖, λ_i ← 0.9λ_i + 0.1λ̂_i for BC, continuity and flux (L2 norms; a variant of Wang–Teng–Perdikaris) | F2, F3 | primary set | lowers primary error and raises F2's critical contrast by ≥1 grid step |
| B | Residual normalisation r_i/ε_i | F1, F2 | primary set | helps F2 at κ ≥ 80; no change for F1 (still invisible) |
| C | float64 with everything else identical: full-grid replica (all F × primary κ × 5 seeds), plus a 2×2 precision × optimiser design for F2, F3 at {1, 80, 2000}, 10 seeds | as stated | as stated | under Adam, median error changes ≤1.25× and the failure flag changes in ≤1 configuration; larger effects under L-BFGS |
| D | Hard interface: F2 vs F4 (continuity hard) vs F4full (all conditions hard; caveat: the exact P1 solution lies in its buffer space) | F2, F4, F4full | primary set | F4 fails less than F2 at κ ≥ 400; F4full < 1% at all κ |
| E | Optimiser: Adam (3000) vs L-BFGS (≤3000 function+gradient evaluations, history 50, zoom line search) vs Adam 2000 → L-BFGS 1000 | F2, F3, F5 | {1, 80, 2000} | L-BFGS variants give lower median error for F3, F5 |

**H5** is supported if the predicted sign holds in ≥9/10 seeds (sign test p ≈ 0.02) for the pre-specified configuration. It is refuted if the predicted sign holds in ≤5/10, or the opposite sign holds in ≥9/10. **H4's intervention clause:** A or B removes ≥3 F2 failures at some κ where the F2 baseline fails ≥4/5 (10-seed set).

**Implementation notes (decisions to log before running)**
- **A:** `train.py` already passes weights `w` as an array argument to the compiled update, and has a compiled per-term gradient function (`term_grads`). Rebalancing can therefore update `w` in the Python loop. Its per-term gradients are part of the method: **count them as additional gradient evaluations and report them**. Keep 3000 optimiser steps, and state that fairness caveat.
- **B:** add a formulation variant dividing material-i residuals by ε_i (e.g. `F1norm`, `F2norm`), with Gate-3-style tests: loss 0 at the exact solution, finite gradients.
- **C:** use the runner override `precision=float64`.
- **E:** use `optax.lbfgs` with a zoom line search. **Count every function+gradient evaluation, including line-search evaluations**, and stop at 3000. Record function evaluations and gradient evaluations.
- Add `run_type intervention` with variants such as `A_rebal`, `B_resnorm`, `C_f64`, `E_lbfgs`, `E_adam2k_lbfgs1k`.
- Write `src/analysis/interventions.py`: per-seed paired differences, sign tests, bootstrap CIs, failure-count changes, critical-contrast shifts, and figures.

**Estimated size:** about 1,100 1D runs, 2–3 turns.

## 10a. PHASE 12 RESULTS (complete)

- H6 inconclusive: ranking criterion met 8/8 (tau-b >= 0.65); masking criterion R > 3 met 1/8 (12,000 steps, R = 5.6). Baseline R on the subset is 2.24 (H2 was already inconclusive, R = 2.39); R is stable at 1.84-2.49 in 7/8 settings.
- 12,000 steps: F2 failures 6 -> 1, concealed failures 7 -> 16. Parameter-matched width 22 does not remove F2/F6's advantage.
- BUG fixed: F5 energy with non-uniform points (extra band points) now uses importance weights; 15 F5 rows superseded (variant band16v2).
- Driver: `sh scripts/run_robustness.sh BUDGET_SECONDS` (resumable).

## 10. PHASE 12 — Robustness (frozen; protocol Sec. 11, YAML `robustness_p1`) — DONE, kept for reference

P1, all six formulations, κ ∈ {1, 80, 2000}, seeds 0–4, one factor at a time. The eight variations:

1. width 16
2. width 64
3. depth 2
4. depth 5
5. n_interior 32
6. n_interior 512
7. lr_init 3e-3
8. steps 12000

The first seven take about 630 runs and the 12,000-step variation 90 more, so about 720 in total. Extras:
- **Parameter-matched two-network variant:** width 22 for F2, F4, F6.
- **Extra band points:** 16 per side inside the band, `n_extra_band`. This needs a small code addition to `collocation` / `make_data` and the row's `n_extra_band` field.

Boundary sampling is not applicable in 1D (fixed endpoints); report it as such. Precision and optimiser are covered by Intervention C/E.

**H6:** supported if Kendall τ ≥ 0.6 between the baseline and varied formulation ranking (by failure count) in ≥6/8 variations, and H2's median R > 3 in ≥6/8. Refuted if τ < 0.6 in ≥3/8. Retain the baseline as the reference for every comparison. About 2 turns.

## 11. PHASE 13 — Remedy comparison against FEM (brief Sec. 20; protocol Sec. 12)

Test **H7** on **P2 and P5**. P1 is degenerate: its exact piecewise-linear solution lies in the FEM space.

**H7:** for every formulation, FEM time-to-1% band field error is ≥10× smaller than the PINN median, or the PINN never reaches 1%; and FEM error at the PINN's median wall-clock is lower. Refuted if any PINN formulation's median time-to-target is within 10× of FEM.

**Report two distinct comparisons**
- **(A) Error at equal cost:** PINN median wall-clock and gradient evaluations versus the FEM mesh with the closest wall-clock.
- **(B) Cost to reach 5% and 1% band field error:**
  - PINN: the first logged step in `results/raw/trajectories/` that meets the target, converted to time as `wall_clock_s × step/3000` (document this approximation).
  - FEM: the smallest mesh in `results/raw/fem_runs_2d.csv` that meets the target.

**Rules**
- Report "not reached" honestly. Never invent a time-to-target.
- Include the most promising remedies from Phases 11–12.
- Optionally pre-register extended-budget PINN runs (e.g. 20,000 steps) for the best formulations at κ = 5 and 80 to measure time-to-target. Log the design before running.
- Record hardware, software, precision, timing method, DOFs and FEM reference accuracy.
- State that PINNs and FEM have different computational structures, and that equal gradient evaluations do not imply equal wall-clock time.
- Exclude the 8 contaminated P4 timings, which aren't needed for P2/P5 anyway.
- In P5 FEM records, the finest level is the reference itself (error 0): exclude it from cost comparisons.

## 12. PHASE 14 — Final analyses, figures, paper, guides (brief Secs. 21–24)

**Single analysis pipeline.** Write `scripts/make_all.py`. It must read the databases and regenerate every table and figure. No manual editing of numerical data.

**Required figure types** (existing ones in `results/figures/`; 10 of the 13 have not yet been visually checked, so inspect all of them):
1. failure-rate heatmaps;
2. median band-error heatmaps;
3. global-versus-local errors;
4. material-specific errors;
5. potential and displacement jumps;
6. capacitance/energy errors;
7. gradient ratio versus error;
8. per-term gradient trajectories;
9. training convergence curves;
10. intervention comparisons;
11. failure contrast versus formulation;
12. robustness;
13. error versus cost;
14. FEM convergence;
15. cross-geometry comparison.

Every figure needs a title, labelled axes with units, a legible legend, defined quantities, and seed variability where appropriate.

**Research paper** (Deliverable J): abstract, introduction, related work, research questions and hypotheses, mathematical formulation, protocol, evaluator validation, 1D results, multidimensional results, failure-mechanism analysis, interventions, robustness, FEM comparison, limitations, discussion, conclusion, references (`literature/references.bib`; resolve entries marked VERIFY). Every claim must be supported by data or a citation. Do not claim universal PINN failure or universal FEM superiority. Explain exactly how the work differs from the closest studies (`literature/research_gap_report.md`).

**Reproduction guide** (Deliverable K): install, tests, reproducing experiments, regenerating figures, inspecting results.

**Educational resource** (Deliverable L): a beginner-friendly notebook or tutorial built from *actual* project results. It should explain:
- the dielectric-interface problem;
- PINN training;
- why discontinuous coefficients are hard;
- how the evaluator works;
- how to read the CSV;
- how to read failure maps;
- why FEM serves as a reference;
- what the experiments do and do not establish.

**Assess H9** (evaluator reusability). P2–P5 needed only problem modules plus additive hooks in `evaluator2d.py`: extra and corner metrics, physical conversion, and reference-aware self-test tolerances. No metric definition was rewritten.

**Final acceptance checklist:** the literature review and gaps are documented; the protocol is explicit and reproducible; the evaluator passes validation; the required grids are executed; the database is complete; failure maps and intervention comparisons are reproducible; FEM references are validated; cost comparisons are documented; robustness conclusions rest on executed experiments; the paper reflects the completed work; limitations and incomplete work are identified; and another researcher can reproduce the project.

## 13. Known limitations and open items to carry into the paper

- **Fixed budget:** a single 3000-step budget is the primary design. Budget sensitivity comes only from the 12,000-step robustness variation (pending).
- **Seeds:** five seeds per configuration give coarse failure-rate estimates (0/5 has a 95% upper bound of 52%).
- **F5 fixed points:** F5 uses fixed, non-resampled quadrature points, which drives its 2D failures (quadrature overfitting). Resampling was not tried, because it is outside the frozen design.
- **P5 level-set scaling:** likely saturates F3/F5 inputs.
- **P3 decomposition:** global singular subtraction amplifies errors in the high-ε material by about κ/2 (quantified by the sensitivity variant).
- **Regression baseline:** the φ-only baseline is a weak representation test.
- **Memory:** peak memory is process-level (`ru_maxrss`), not per run.
- **Timing:** single-core CPU only; timings are hardware-specific.
- **Bibliography:** several entries are marked VERIFY.
- **Literature search:** 16 queries, so "not found" means not found in those queries.
