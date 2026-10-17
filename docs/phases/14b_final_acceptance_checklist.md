# Final acceptance checklist (Phase 14)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Literature review and gaps documented | met | `literature/literature_review.md`, `research_gap_report.md`, `research_gap_matrix.csv`; 32 entries, 30 cited in the paper. 2 entries UNRESOLVED (author lists) and 9 VERIFY notes (publication details) remain, listed in `references.bib` |
| 2 | Protocol explicit and reproducible | met | `docs/experimental_protocol.md`, `configs/*.yaml`, `docs/protocol_change_log.md` (every decision dated, made before the affected runs) |
| 3 | Evaluator passes validation | met | 1D Gate 1 98/98; 2D self-tests 17/17 for each of P2–P5; Gate 3 loss tests; H9 assessment `docs/phases/14a_evaluator_reusability_h9.md` |
| 4 | Required grids executed | met | P1: 6 formulations × 6 κ × 5 seeds + 10⁴ extension; P2–P5: F1/F2/F6 × 11 κ, F3–F5 × 3 κ (pre-registered reduced grid) × 5 seeds; interventions, robustness, remedies. 3,856 completed runs, 0 missing |
| 5 | Database complete | met | `results/raw/runs.csv`, `runs_2d.csv`, `fem_runs*.csv`, `trajectories/`, `gradlogs/`; schema `docs/database_schema.md`; superseded and technical-failure rows kept and labelled |
| 6 | Failure maps and intervention comparisons reproducible | met | `python scripts/make_all.py` regenerates all tables/figures from raw data; regenerated outputs match committed ones (Phase 14 change log) |
| 7 | FEM references validated | met | P1 exact to round-off; P2–P4 against exact solutions (convergence in `final_fem_convergence.png`); P5 graded-mesh reference converged to <0.1% band change plus a 1D three-slab check (~1e-13) |
| 8 | Cost comparisons documented | met | `docs/phases/13_remedies_vs_fem.md` (timing method, hardware, compile time, resolution limit, caveats) |
| 9 | Robustness conclusions rest on executed experiments | met | 870 robustness rows; `docs/phases/12_robustness.md` |
| 10 | Paper reflects the completed work | met | `manuscript/paper.md`; headline numbers re-derived from raw data in Phase 14 (κ = 80 medians, P1 failure counts, R and concealed fractions for P1–P5, H1 ρ); citation keys and figure references checked |
| 11 | Limitations and incomplete work identified | met | paper Sec. 13; per-phase findings; unresolved references listed |
| 12 | Another researcher can reproduce the project | met | `docs/REPRODUCTION_GUIDE.md`, `scripts/setup_environment.sh`, pinned `requirements.txt`, determinism regression tests (1D and 2D) |

## Incomplete or not done (stated, not hidden)

- Not tested: interventions on 2D problems (A–E were P1 only, by design); κ = 5/20 FEM refinement records (the PINN's best 2D regime is absent from the time-to-target comparison); resampled quadrature for F5; larger architectures.
- Bibliography: 2 UNRESOLVED author lists and 9 VERIFY notes on publication details (journal versions, pages, first names); none affects a cited claim.
- The manuscript is in Markdown with pandoc citations; it has not been typeset to a journal template.
