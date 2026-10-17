# Reproduction Guide (Deliverable K)

This guide lets another researcher reinstall the environment, verify the code, regenerate every table and figure from the released raw data, and rerun any experiment.

All results were produced on a single-threaded CPU (Intel Xeon, 2.10 and 2.80 GHz) under Ubuntu 24 with Python 3.12. No GPU was used.

## 1. Install

```sh
unzip pinn_dielectric_project_*.zip && cd pinn_dielectric_project
sh scripts/setup_environment.sh          # pinned pip packages + gmsh system libraries; prints versions
python scripts/rebuild_p5_references.py  # ~3 min: rebuilds the P5 FEM reference .npz files (excluded from the archive for size)
```

**Expected setup output:** `jax 0.11.2 optax 0.2.8 skfem 12.0.2 gmsh 4.15.2`.

**Other operating systems:** install from `requirements.txt` plus the gmsh system libraries listed there.

**Git checkouts:** the P5 references are tracked in git, so in a git checkout `git checkout -- results/reference` restores them exactly.

## 2. Verify the code

```sh
python -m pytest tests -q -p no:cacheprovider        # 169 tests, ~9 min on one core
python scripts/run_evaluator_selftests.py            # 1D evaluator Gate 1: 98/98
for P in P2 P3 P4 P5; do python scripts/run_selftests_2d.py $P; done   # 2D evaluator: 17/17 each
python scripts/verify_reference_derivations.py      # symbolic checks of the exact references
```

**Tests that guard reproducibility:**
- `tests/test_interventions.py::test_baseline_path_reproduces_primary_rows` retrains two stored primary configurations and requires agreement to within 10⁻¹³.
- `tests/test_remedies2d.py::test_2d_baseline_path_reproduces_primary_row` does the same in 2D.

## 3. Regenerate every table and figure from the raw data

```sh
python scripts/make_all.py          # ~70 s; reads results/raw only; writes results/processed and results/figures
python scripts/make_all.py --list   # the 11 steps and what each produces
```

The pipeline reads only `results/raw/`:
- `runs.csv` (1D);
- `runs_2d.csv` (2D);
- `fem_runs*.csv`;
- `trajectories/`;
- `gradlogs/`.

No number is edited by hand anywhere. A log of steps and exit codes is written to `results/processed/make_all_log.json`. Regenerated outputs match the committed ones, apart from last-digit float formatting (change log, Phase 14).

## 4. Rerun experiments

All runners are resumable: they skip run IDs already marked `complete`. An interrupted run writes no row.

**Run sweeps one at a time, in the foreground.** Concurrent sweeps duplicate rows and corrupt timings.

| Experiment | Command (example) | Approx. time |
|---|---|---|
| P1 primary sweep | `python scripts/run_sweep.py --run-type pinn --formulations F1,F2,F3,F4,F5,F6 --kappas 1,5,20,80,400,2000 --seeds 0,1,2,3,4` | ~10 min |
| P1 interventions (Phase 11) | `python scripts/run_sweep.py --run-type intervention --variant A_rebal --override rebalance=1 ...` (all variant/override pairs: `docs/protocol_change_log.md`, Phase 11) | ~2 h total |
| P1 robustness (Phase 12) | `sh scripts/run_robustness.sh 3600` | ~1.5 h |
| 2D sweeps (P2–P5) | `python scripts/run_sweep2d.py --problem P2 --run-type pinn --formulations F1,F2,F6 --kappas 1,80,2000 --seeds 0,1,2,3,4` | ~10–15 s per run |
| Remedies vs FEM (Phase 13) | `sh scripts/run_remedies.sh 7200` | ~1.5 h (20k-step runs ~150 s each) |
| FEM references | `python scripts/run_fem_references_p1.py`; 2D FEM records are created by `src/analysis/summary2d.py` / `p2_summary.py` when absent | seconds to minutes |

**Determinism.** Determinism holds on the same software stack and CPU type. Different hardware or library versions can change results at round-off level, and through training dynamics this can grow into larger differences. Compare medians and failure counts rather than single runs.

## 5. Inspect results

- **Database schema:** `docs/database_schema.md`. There is one row per training run, with the full configuration, all metrics, cost, provenance, software environment and git commit.
- **Statuses:**
  - `complete` for finished runs;
  - `failed_technical` for runs that broke for a technical reason. These are kept and excluded from statistics; the reason is in `notes`.
- **Superseded rows:** variants `C_f64`, `C_f64_lbfgs` and the F5 rows of `band16` were superseded after bug fixes. They are kept and excluded by the analysis modules. See the change log.
- **Per-run histories:**
  - `results/raw/trajectories/<run_id>.csv`: metrics every 250 steps;
  - `results/raw/gradlogs/<run_id>.csv`: per-term loss values and gradient norms every 25 steps.
- **Findings by phase:** `docs/phase6_p1_findings.md` … `docs/phase13_remedy_vs_fem_findings.md`, `docs/h9_evaluator_reusability.md`.
- **Protocol and every later decision:** `docs/experimental_protocol.md`, `docs/protocol_change_log.md`.
- **Paper:** `manuscript/paper.md` (references in `literature/references.bib`).

**Example: failure counts for P2 at κ = 80:**

```python
import pandas as pd
d = pd.read_csv("results/raw/runs_2d.csv", low_memory=False)
d = d[(d.status == "complete") & (d.run_type == "pinn") & (d.variant == "base")].drop_duplicates("run_id")
print(d[(d.problem_id == "P2") & (d.kappa == 80)].groupby("formulation").fail_primary.sum())
```

## 6. Known caveats

- Wall-clock timings depend on the CPU and on other load. Compare within one machine.
- The P2 and P5 FEM records were timed on different CPU models.
- 60 P3 rows are technical failures from a fixed shape bug, and 5 duplicated P4 rows come from a concurrency incident. Both are documented, and the analysis de-duplicates by `run_id`.
- Two bibliography entries are marked UNRESOLVED (author lists not verified).
