"""Phase 12 robustness code paths: extra band points (protocol Sec. 11 extra)."""
import pathlib
import numpy as np, yaml
from src.pinns import train as T
from src.problems.p1 import P1Problem

CFG = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "configs/protocol_v1.yaml"))


def test_extra_band_points_inside_band_and_primary_unchanged():
    p = P1Problem(80.0)
    for seed in range(5):
        b1, b2 = T.extra_band_points(seed, p, 16)
        assert len(b1) == len(b2) == 16
        assert np.all((b1 >= p.a - p.band_halfwidth) & (b1 < p.a)) and np.all((b2 > p.a) & (b2 <= p.a + p.band_halfwidth))
        x1, x2 = T.collocation(seed, p, 128); y1, y2 = T.collocation(seed, p, 128)
        assert np.array_equal(x1, y1) and np.array_equal(x2, y2)       # separate stream: primary points untouched
    assert not np.array_equal(T.extra_band_points(0, p, 16)[0], T.extra_band_points(1, p, 16)[0])


def test_extra_band_run_records_counts():
    row, _ = T.train_run("F2", 80.0, 0, CFG, overrides=dict(steps=20, n_extra_band=16))
    assert row["n_extra_band"] == 32
    base, _ = T.train_run("F2", 80.0, 0, CFG, overrides=dict(steps=20))
    assert base["n_extra_band"] == 0 and row["collocation_evals"] == base["collocation_evals"] + 20 * 32
