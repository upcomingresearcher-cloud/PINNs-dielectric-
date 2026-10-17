"""Tests of the run-database writer and identifiers."""
import csv, pytest
from src.experiments import records as R
from src.evaluator.evaluator1d import evaluate
from src.problems.p1 import P1Problem


def test_metric_columns_match_evaluator_exactly():
    p = P1Problem(80.0)
    keys = set(evaluate(p, p.exact).keys()); cols = set(R.METRIC_COLUMNS)
    assert len(R.METRIC_COLUMNS) == len(cols), "duplicate schema columns"
    assert keys == cols, f"metric definitions changed (update schema + change log): only in evaluator={keys - cols}, only in schema={cols - keys}"


def _row():
    r = {c: "" for c in R.RUNS_COLUMNS}; r.update(run_id="x1", status=R.STATUS_COMPLETE); return r


def test_append_and_resume(tmp_path):
    f = tmp_path / "runs.csv"
    R.append_row(f, _row(), R.RUNS_COLUMNS)
    r2 = _row(); r2.update(run_id="x2", status="failed_technical"); R.append_row(f, r2, R.RUNS_COLUMNS)
    assert R.completed_ids(f) == {"x1"}
    rows = list(csv.DictReader(open(f))); assert [r["run_id"] for r in rows] == ["x1", "x2"]
    assert list(rows[0].keys()) == R.RUNS_COLUMNS


def test_schema_mismatch_rejected(tmp_path):
    r = _row(); r.pop("seed")
    with pytest.raises(ValueError): R.append_row(tmp_path / "a.csv", r, R.RUNS_COLUMNS)
    r = _row(); r["unexpected"] = 1
    with pytest.raises(ValueError): R.append_row(tmp_path / "b.csv", r, R.RUNS_COLUMNS)


def test_run_id_and_config_id_stable():
    assert R.run_id("P1", "F2", 80.0, 3, "pinn") == "P1-F2-k80-s3-pinn-base"
    assert R.config_id({"a": 1, "b": [1, 2]}) == R.config_id({"b": [1, 2], "a": 1})


def test_fem_runs_file_matches_schema():
    import pathlib
    f = pathlib.Path(__file__).resolve().parents[1] / "results/raw/fem_runs.csv"
    if f.exists():
        assert next(csv.reader(open(f))) == R.FEM_COLUMNS
