"""Gate 3 must catch injected formulation bugs."""
import jax.numpy as jnp, pytest
import tests.test_formulations_gate3 as G
from src.pinns import formulations as Fm


def test_catches_missing_interface_terms(monkeypatch):
    F2 = Fm.REGISTRY["F2"]; orig = Fm.F2.terms
    def broken(self, f, d):
        t = orig(self, f, d); t["cont"] = 0 * t["cont"]; t["flux"] = 0 * t["flux"]; return t
    monkeypatch.setattr(Fm.F2, "terms", broken)
    with pytest.raises(AssertionError):
        G.test_loss_at_wrong_solution("F2", 80.0)


def test_catches_broken_hard_continuity(monkeypatch):
    orig = Fm.F4.rep
    monkeypatch.setattr(Fm.F4, "rep", lambda self, p, x, side, d: orig(self, p, x, side, d) + (0.0 if side == 1 else 1e-3))
    with pytest.raises(AssertionError):
        G.test_hard_constraints_at_random_init(80.0)


def test_catches_wrong_energy_weighting(monkeypatch):
    orig = Fm.F5.terms
    import jax
    def broken(self, f, d):  # forget the |Omega_i| quadrature weights
        g1 = jax.vmap(Fm._d1(f, 1))(d["x1"]); g2 = jax.vmap(Fm._d1(f, 2))(d["x2"])
        return dict(energy=jnp.mean(0.5 * d["eps1"] * g1**2) + jnp.mean(0.5 * d["eps2"] * g2**2))
    monkeypatch.setattr(Fm.F5, "terms", broken)
    with pytest.raises(AssertionError):
        G.test_loss_at_exact_solution("F5", 80.0)
