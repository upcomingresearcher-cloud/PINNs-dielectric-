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


def test_band_quadrature_weights_sum_to_lengths():
    p = P1Problem(80.0); x1, x2 = T.collocation(0, p, 128); b1, b2 = T.extra_band_points(0, p, 16)
    x1, x2 = np.sort(np.concatenate([x1, b1])), np.sort(np.concatenate([x2, b2]))
    q1, q2 = T.band_quadrature_weights(p, x1, x2, 128, 16)
    assert np.isclose(q1.sum(), p.a, rtol=0.02) and np.isclose(q2.sum(), p.L - p.a, rtol=0.02)


def test_f5_energy_with_band_points_is_unbiased():
    """Regression (bug 2026-10-16): the weighted energy of a fixed smooth field must match its exact integral; the
    old equal-weight mean over non-uniform points does not."""
    import jax.numpy as jnp
    from src.pinns.formulations import energy_term
    p = P1Problem(1.0); x1, x2 = T.collocation(0, p, 128); b1, b2 = T.extra_band_points(0, p, 16)
    x1, x2 = np.sort(np.concatenate([x1, b1])), np.sort(np.concatenate([x2, b2]))
    d = T.make_data(p, x1, x2, jnp.float64)
    phi = lambda x, s: jnp.sin(3.0 * x) * 0.3 + x**3          # phi'(x) = 0.9 cos 3x + 3x^2
    xs = np.linspace(0, 1, 200001); g = 0.9 * np.cos(3 * xs) + 3 * xs**2
    exact = np.trapezoid(0.5 * g**2, xs)                       # eps1 = eps2 = 1
    biased = float(energy_term(phi, d))
    q1, q2 = T.band_quadrature_weights(p, x1, x2, 128, 16); d["q1"], d["q2"] = jnp.asarray(q1), jnp.asarray(q2)
    fixed = float(energy_term(phi, d))
    assert abs(fixed - exact) / exact < 5e-3 and abs(biased - exact) > 5 * abs(fixed - exact)


def test_baseline_energy_path_unchanged_without_extra_points():
    import jax.numpy as jnp
    from src.pinns.formulations import energy_term
    p = P1Problem(80.0); x1, x2 = T.collocation(3, p, 128); d = T.make_data(p, x1, x2, jnp.float32)
    phi = lambda x, s: jnp.sin(3.0 * x)
    ref = p.a * jnp.mean(0.5 * d["eps1"] * (3 * jnp.cos(3 * d["x1"]))**2) + (p.L - p.a) * jnp.mean(0.5 * d["eps2"] * (3 * jnp.cos(3 * d["x2"]))**2)
    assert "q1" not in d and np.isclose(float(energy_term(phi, d)), float(ref), rtol=1e-6)
