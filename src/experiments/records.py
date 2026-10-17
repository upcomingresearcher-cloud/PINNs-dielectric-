"""Results database (brief Sec. 14). One row per run; separate tables per record type.

Tables (all under results/raw/):
  runs.csv                       trained models: run_type in {pinn, regression, pilot,
                                 intervention, robustness, extension, refinement}
  fem_runs.csv                   FEM reference/cost evaluations
  diagnostics_loss_visibility.csv  loss of exact vs wrong solutions (no training)
  gradlogs/{run_id}.csv          per-term gradient logs
  trajectories/{run_id}.csv      evaluator metrics during training
Units: nondimensional (protocol Sec. 2); times in seconds; memory in MB.
"""
import csv, hashlib, json, os, pathlib

# ---- evaluator metric columns (frozen; tests assert evaluator output == this list)
METRIC_COLUMNS = (
    [f"relL2_{q}_{r}" for q in ("phi", "E", "D") for r in ("global", "mat1", "mat2", "band", "band1", "band2")]
    + ["phi_jump", "phi_jump_rel", "D_jump", "D_jump_rel", "C_flux", "C_flux_relerr", "gauss_imbalance_rel",
       "energy", "energy_relerr", "C_energy_relerr", "kappa_eff", "log_kappa_eff_ratio", "bc_err_left",
       "bc_err_right", "n_eval_points", "nonfinite", "primary_field_error", "strict_field_error",
       "fail_primary", "fail_strict", "fail_potential", "sec_fail_effective_contrast", "sec_fail_D_jump"])

RUN_ID_COLUMNS = ["run_id", "run_type", "status", "problem_id", "geometry", "formulation", "variant", "kappa", "seed"]
RUN_SETTING_COLUMNS = ["architecture", "hidden_layers", "width", "activation", "n_params", "optimiser",
                       "lr_schedule", "lr_init", "precision", "n_interior_per_material", "n_boundary",
                       "n_interface", "n_extra_band", "step_budget", "grad_evals", "collocation_evals"]
RUN_OUTCOME_COLUMNS = ["final_loss", "final_loss_terms_json"]
RUN_COST_COLUMNS = ["wall_clock_s", "compile_s", "peak_rss_mb"]
RUN_PROV_COLUMNS = ["config_id", "protocol_version", "software_env", "hardware", "git_commit", "timestamp", "notes"]
RUNS_COLUMNS = (RUN_ID_COLUMNS + RUN_SETTING_COLUMNS + RUN_OUTCOME_COLUMNS + METRIC_COLUMNS
                + RUN_COST_COLUMNS + RUN_PROV_COLUMNS)

FEM_COLUMNS = (["fem_id", "problem_id", "kappa", "n_elements", "dofs", "conforming", "element_eps_rule",
                "quadrature", "solve_time_s", "timing_repeats"] + METRIC_COLUMNS
               + ["software_env", "hardware", "git_commit", "timestamp"])

LOSS_VIS_COLUMNS = ["diag_id", "problem_id", "formulation", "variant", "kappa", "candidate", "loss_total",
                    "loss_terms_json", "smoothing_width", "n_extra_band", "git_commit", "timestamp"]

GRADLOG_COLUMNS = ["step", "term", "loss_value", "grad_norm_unweighted", "grad_norm_weighted", "weight",
                   "learning_rate", "precision", "nonfinite", "effectively_zero"]

STATUS_COMPLETE = "complete"


def config_id(cfg: dict) -> str:
    """Short stable hash of a configuration dictionary."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def append_row(path, row: dict, columns):
    """Append one row to a CSV database, refusing rows with missing or unknown columns."""
    missing = [c for c in columns if c not in row]
    extra = [c for c in row if c not in columns]
    if missing or extra:
        raise ValueError(f"schema mismatch for {path}: missing={missing} extra={extra}")
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if new: w.writeheader()
        w.writerow(row); f.flush(); os.fsync(f.fileno())


def completed_ids(path, id_col="run_id"):
    """Set of run identifiers already marked complete in a database file."""
    path = pathlib.Path(path)
    if not path.exists(): return set()
    with path.open() as f:
        return {r[id_col] for r in csv.DictReader(f) if r.get("status", STATUS_COMPLETE) == STATUS_COMPLETE}


def run_id(problem, formulation, kappa, seed, run_type, variant="base"):
    """Canonical run identifier: problem-formulation-k<kappa>-s<seed>-<run_type>-<variant>."""
    return f"{problem}-{formulation}-k{kappa:g}-s{seed}-{run_type}-{variant}"
