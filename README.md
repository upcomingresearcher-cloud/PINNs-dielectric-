# PINNs Across Dielectric Interfaces: Failure Mechanisms, Robustness, and Computational Cost

Research project status: **Phase 2 complete (protocol v1.0.0 frozen). No training runs have been executed yet.**

| Phase | Status |
|---|---|
| 1. Literature database and review | complete |
| 2. Gaps, hypotheses, frozen protocol | complete (configs/protocol_v1.yaml, docs/experimental_protocol.md) |
| 3–14 | not started |

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
```
