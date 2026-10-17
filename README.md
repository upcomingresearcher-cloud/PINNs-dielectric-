# PINNs Across Dielectric Interfaces: Failure Mechanisms, Robustness, and Computational Cost

Research project status: **all 14 phases complete.** Paper: `manuscript/paper.md`. Reproduction: `docs/REPRODUCTION_GUIDE.md`. Tutorial: `educational_resource/tutorial.md`.

| Phase | Status |
|---|---|
| 1–5. Literature, protocol, evaluator, formulations, P1 sweep | complete |
| 6. P1 analysis (docs/phase6_p1_findings.md) | complete |
| 7–10. P2 cylinder, P3 line charge, P4 corner, P5 thin film (docs/phase7…10_*.md) | complete |
| 11. Interventions A–E (docs/phase11_interventions_findings.md) | complete |
| 12. Robustness (docs/phase12_robustness_findings.md) | complete |
| 13. Remedies vs FEM (docs/phase13_remedy_vs_fem_findings.md) | complete |
| 14. Final analyses, paper, guides (manuscript/, docs/REPRODUCTION_GUIDE.md, educational_resource/) | complete |

Quick start: `sh scripts/setup_environment.sh && python scripts/make_all.py` regenerates every table and figure from `results/raw/`.

## Layout
- `literature/` – literature database, review, gap matrix/report, BibTeX, and the scripts that build the CSVs
- `docs/` – progress reports and design notes
- `src/` – problems, references, pinns (formulations, training), fem, evaluator, experiments (records), analysis
- `results/` – raw databases and logs, processed tables, figures, validation outputs
- `manuscript/` – paper; `educational_resource/` – beginner tutorial
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
Analysis: `python src/analysis/p1_summary.py` (tables + failure maps from the database).

Sweeps are resumable: rerun the same command to continue after an interruption.
