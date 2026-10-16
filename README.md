# PINNs Across Dielectric Interfaces: Failure Mechanisms, Robustness, and Computational Cost

Research project status: **Phase 4 complete (6 formulations verified; pilot run, 72 pilot runs). Primary sweep not yet run.**

| Phase | Status |
|---|---|
| 1. Literature database and review | complete |
| 2. Gaps, hypotheses, frozen protocol | complete (configs/protocol_v1.yaml, docs/experimental_protocol.md) |
| 3. P1 references, FEM, evaluator, self-tests | complete (results/validation/, results/raw/fem_runs.csv) |
| 4. PINN formulations, training loop, pilot | complete (src/pinns/, pilot rows in results/raw/runs.csv) |
| 5–14 | not started |

## Layout
- `literature/` – literature database, review, gap matrix/report, BibTeX, and the scripts that build the CSVs
- `docs/` – progress reports and design notes
- `src/` – (empty) problems, references, pinns, fem, evaluator, experiments, analysis
- `results/logs/` – environment records and timing benchmarks
- `scripts/` – utilities (`record_environment.py`, JAX step benchmark)

## Regenerating Phase 1 outputs
```
python literature/build_literature_database.py
python literature/build_research_gap_matrix.py
python scripts/record_environment.py
python scripts/verify_reference_derivations.py   # 20 symbolic checks
python scripts/run_evaluator_selftests.py        # Gate 1 (98 checks)
python -m pytest tests -q                        # unit + mutation tests
python scripts/run_fem_references_p1.py          # FEM reference sweep -> results/raw/fem_runs.csv
python src/analysis/fig_fem_p1.py                # figure from the database
python scripts/run_sweep.py --run-type pilot --formulations F1,F2,F3,F4,F5,F6 --kappas 1,80 --seeds 100,101,102 --variant lr1e-3 --override lr_init=1e-3
```
Sweeps are resumable: rerun the same command to continue after an interruption.
