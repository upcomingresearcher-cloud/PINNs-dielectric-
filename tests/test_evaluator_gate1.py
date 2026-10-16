"""Gate 1: all evaluator self-tests must pass; and each injected bug must be caught."""
import numpy as np, pytest
import src.evaluator.evaluator1d as ev
import src.evaluator.selftests as st
from src.problems.p1 import P1Problem
from src.fem import fem1d


@pytest.mark.parametrize("test", st.ALL, ids=lambda t: t.__name__)
def test_selftest_passes(test):
    bad = [r for r in test() if not r["passed"]]
    assert not bad, bad


def _caught(test):
    return any(not r["passed"] for r in test())


def test_mutation_wrong_permittivity_in_evaluator(monkeypatch):
    monkeypatch.setattr(P1Problem, "eps_side", lambda self, side: np.ones(np.shape(side)))
    assert _caught(st.test1_exact)          # C_flux vs D_exact must expose it


def test_mutation_field_sign(monkeypatch):
    orig = ev.evaluate
    def flipped(problem, pred, **kw):
        return orig(problem, lambda x, s: (pred(x, s)[0], -pred(x, s)[1]), **kw)
    monkeypatch.setattr(st, "evaluate", flipped)
    assert _caught(st.test1_exact)


def test_mutation_threshold(monkeypatch):
    monkeypatch.setattr(ev, "FAIL_THRESHOLD", 0.07)
    assert _caught(st.test4_threshold)


def test_mutation_interface_side_ignored(monkeypatch):
    orig = ev.evaluate
    def no_side(problem, pred, **kw):
        return orig(problem, lambda x, s: pred(x, np.where(np.isclose(x, problem.a), 2, s)), **kw)
    monkeypatch.setattr(st, "evaluate", no_side)
    assert _caught(st.test6_interface)


def test_mutation_fem_single_coefficient(monkeypatch):
    monkeypatch.setattr(fem1d, "element_eps", lambda nodes, a, e1, e2: np.full(nodes.size - 1, e1))
    assert _caught(st.test2_fem)


def test_mutation_nonfinite_not_flagged(monkeypatch):
    orig = ev.evaluate
    def hide(problem, pred, **kw):
        return orig(problem, lambda x, s: tuple(np.nan_to_num(v) for v in pred(x, s)), **kw)
    monkeypatch.setattr(st, "evaluate", hide)
    assert _caught(st.test5_nonfinite)
