# PINNs Across Dielectric Interfaces: Failure Mechanisms, Robustness, and Computational Cost

Research project status: **Phase 3 complete (evaluator validated: 98/98 self-test checks, 19/19 pytest). No PINN training runs have been executed yet.**

| Phase | Status |
|---|---|
| 1. Literature database and review | complete |
| 2. Gaps, hypotheses, frozen protocol | complete (configs/protocol_v1.yaml, docs/experimental_protocol.md) |
| 3. P1 references, FEM, evaluator, self-tests | complete (results/validation/, results/raw/fem_runs.csv) |
| 4–14 | not started |

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
```
