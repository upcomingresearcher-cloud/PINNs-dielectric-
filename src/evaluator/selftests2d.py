"""Gate 1 for 2D problems (same 8 tests as P1, adapted). problem_factory(kappa) -> Problem2D."""
import numpy as np
from src.evaluator.evaluator2d import evaluate
from src.fem import fem2d


def _c(t, c, v, crit, ok): return dict(test=t, check=c, value=float(v) if not isinstance(v, bool) else v, criterion=crit, passed=bool(ok))


def _pert(p, delta):
    def pred(X, side):
        f, g = p.exact(X, side); x, y = X[:, 0], X[:, 1]; b = (1 - x**2) * (1 - y**2)
        return f + delta * b * x, g + delta * np.c_[b + x * (-2 * x) * (1 - y**2), x * (1 - x**2) * (-2 * y)]
    return pred


def run(factory, kappas, geometry, h_list=(0.2, 0.1, 0.05, 0.025), wrong_kappa=80.0):
    """Tolerances: analytic references -> round-off (1e-12; quadrature 1e-8). FEM references (P5) carry their own
    round-off and element kinks inside quadrature cells -> 1e-7 and 1e-4 (both far below the 5% threshold).
    T6 expectations are computed from the reference itself (exact for analytic references)."""
    out = []
    fem_ref = getattr(factory, "reference_type", "exact") == "fem"
    tol1, tol7 = (1e-7, 1e-4) if fem_ref else (1e-12, 1e-8)
    for k in kappas:
        p = factory(k); m = evaluate(p, p.exact)
        worst = max(v for kk, v in m.items() if kk.startswith("relL2") or kk.endswith("jump_rel") or kk in ("energy_relerr", "bc_err_max_rel"))
        out.append(_c("T1_exact", f"kappa={k:g}: max error metric", worst, f"< {tol1:g}", worst < tol1 and not m["fail_primary"]))
    p = factory(wrong_kappa); m = evaluate(p, p.wrong_uniform)
    out.append(_c("T3_wrong_uniform", f"kappa={wrong_kappa:g} primary", m["primary_field_error"], "flagged", m["fail_primary"]))
    for k in (kappas[0], kappas[-1]):
        p = factory(k); unit = evaluate(p, _pert(p, 1.0))["primary_field_error"]
        for tgt, fail in ((0.04, False), (0.06, True)):
            m = evaluate(p, _pert(p, tgt / unit))
            out.append(_c("T4_threshold", f"kappa={k:g}: {tgt:.0%} perturbation", m["primary_field_error"], f"fail=={fail}",
                          m["fail_primary"] == fail and abs(m["primary_field_error"] - tgt) < 1e-9))
    p = factory(kappas[-1])
    def nanpred(X, side):
        f, g = p.exact(X, side); f = f.copy(); f[len(f) // 2] = np.nan; return f, g
    out.append(_c("T5_nonfinite", "NaN flagged", evaluate(p, nanpred)["nonfinite"], "True", evaluate(p, nanpred)["fail_primary"]))
    q = p.quadrature(); XI, WI, NI = q["XI"], q["WI"], q["NI"]; o1, o2 = np.ones(len(XI), int), np.full(len(XI), 2)
    e1, g1 = p.exact(XI, o1); e2, g2 = p.exact(XI, o2); je = e2 - e1
    Dn1 = -p.eps1 * np.sum(g1 * NI, 1); Dn2 = -p.eps2 * np.sum(g2 * NI, 1)
    for c in (1e-3, 0.02):
        def jp(X, side, c=c):
            f, g = p.exact(X, side); return f + c * (np.asarray(side) == 2), g
        m = evaluate(p, jp); exp = np.sqrt(np.sum(WI * (je + c)**2) / np.sum(WI)) / p.Vc
        out.append(_c("T6_interface", f"[phi]={c}", m["phi_jump_rel"], f"== {exp:.6g}", abs(m["phi_jump_rel"] - exp) < 1e-12 and m["D_jump_rel"] < 1e-12))
    for s in (0.01, 0.1):
        def jd(X, side, s=s):
            f, g = p.exact(X, side); return f, g * np.where((np.asarray(side) == 2)[:, None], 1 + s, 1.0)
        m = evaluate(p, jd); exp = s * np.sqrt(np.sum(WI * Dn2**2) / np.sum(WI * Dn1**2))
        out.append(_c("T6_interface", f"[D_n] scale {s}", m["D_jump_rel"], f"== {exp:.6g}; flag=={s > 0.05}",
                      abs(m["D_jump_rel"] - exp) < 1e-10 and m["sec_fail_D_jump"] == (s > 0.05)))
    ref = evaluate(p, _pert(p, 0.1), quad=p.quadrature(**p.refined_quadrature)); m = evaluate(p, _pert(p, 0.1))
    d = max(abs(m[k] - ref[k]) / abs(ref[k]) for k in ("primary_field_error", "relL2_phi_global", "relL2_E_band2", "energy_relerr"))
    out.append(_c("T7_quadrature", "default vs refined quadrature (smooth perturbation)", d, f"< {tol7:g}", d < tol7))
    errs = []
    for h in h_list:
        s = fem2d.solve(p, h, geometry=geometry); errs.append(evaluate(p, s)["primary_field_error"])
    order = -np.polyfit(np.log(h_list), np.log(errs), 1)[0] * -1
    out.append(_c("T2_fem_conforming", f"band field errors {['%.2e' % e for e in errs]}; order", order, "decreasing, order > 0.8",
                  all(np.diff(errs) < 0) and order > 0.8))
    nerrs = []
    for h in h_list:
        s = fem2d.solve(p, h, geometry=geometry, conforming=False); nerrs.append(evaluate(p, s)["primary_field_error"])
    norder = np.polyfit(np.log(h_list), np.log(nerrs), 1)[0]
    if hasattr(p, "interface_load"):
        out.append(_c("T8_fem_nonconforming", f"band field errors {['%.2e' % e for e in nerrs]} (INFORMATIONAL: no interface edges -> interface load cannot be applied -> converges to a different problem)", norder, "reported", True))
    else:
        out.append(_c("T8_fem_nonconforming", f"band field errors {['%.2e' % e for e in nerrs]}; order", norder, "reported; must decrease", nerrs[-1] < nerrs[0]))
    return out
