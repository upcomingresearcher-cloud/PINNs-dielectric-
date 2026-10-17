"""Phase 13: 2D L-BFGS remedy path; the default 2D path must still reproduce stored primary rows."""
from conftest import reproduction_tolerance
import pathlib
import numpy as np, pandas as pd, pytest, yaml
from src.pinns.train2d import train_run
from src.problems import p2d

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/p2.yaml"))
PROB = {getattr(p2d, n).problem_id: getattr(p2d, n) for n in dir(p2d) if hasattr(getattr(p2d, n), "problem_id")}


def test_2d_baseline_path_reproduces_primary_row():
    db = pd.read_csv(ROOT / "results/raw/runs_2d.csv", low_memory=False)
    ref = db[db.run_id == "P2-F2-k80-s0-pinn-base"].iloc[0]
    tol = reproduction_tolerance(ref.hardware)
    row, _ = train_run("F2", PROB["P2"], 80.0, 0, CFG)
    for c in ("primary_field_error", "final_loss", "relL2_phi_global"):
        assert abs(float(row[c]) - float(ref[c])) <= tol * max(1.0, abs(float(ref[c]))), c


def test_2d_lbfgs_budget_and_labels():
    row, _ = train_run("F2", PROB["P2"], 80.0, 0, CFG, overrides=dict(steps=120, optimiser="lbfgs"))
    assert 120 - 21 <= row["grad_evals"] <= 120 and row["optimiser"].startswith("lbfgs")
    assert f"fn_evals={row['grad_evals']}" in row["notes"] and np.isfinite(row["final_loss"])
