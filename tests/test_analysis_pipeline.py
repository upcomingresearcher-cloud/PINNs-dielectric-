"""Phase 14: analysis-pipeline regression tests."""
from src.analysis import p2_summary


def test_p2_fem_records_only_p2():
    """Regression (bug 2026-10-16): fem_runs_2d.csv holds P2-P5; the P2 summary must use P2 rows only."""
    f = p2_summary.fem_records()
    assert set(f.problem_id) == {"P2"} and len(f[(f.kappa == 80) & f.conforming]) == 5
