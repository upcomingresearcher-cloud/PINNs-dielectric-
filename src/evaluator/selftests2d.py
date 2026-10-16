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
    out = []
    for k in kappas:
        p = factory(k); m = evaluate(p, p.exact)
        worst = max(v for kk, v in m.items() if kk.startswith("relL2") or kk.endswith("jump_rel") or kk in ("energy_relerr", "bc_err_max_rel"))
        out.append(_c("T1_exact", f"kappa={k:g}: max error metric", worst, "< 1e-12", worst < 1e-12 and not m["fail_primary"]))
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
    for c in (1e-3, 0.02):
        def jp(X, side, c=c):
            f, g = p.exact(X, side); return f + c * (np.asarray(side) == 2), g
        m = evaluate(p, jp)
        out.append(_c("T6_interface", f"[phi]={c}", m["phi_jump_rel"], f"== {c}/Vc", abs(m["phi_jump_rel"] - c / p.Vc) < 1e-12 and m["D_jump_rel"] < 1e-12))
    for s in (0.01, 0.1):
        def jd(X, side, s=s):
            f, g = p.exact(X, side); return f, g * np.where((np.asarray(side) == 2)[:, None], 1 + s, 1.0)
        m = evaluate(p, jd)
        out.append(_c("T6_interface", f"[D_n] scale {s}", m["D_jump_rel"], f"== {s}; flag=={s > 0.05}",
                      abs(m["D_jump_rel"] - s) < 1e-10 and m["sec_fail_D_jump"] == (s > 0.05)))
    ref = evaluate(p, _pert(p, 0.1), quad=p.quadrature(**p.refined_quadrature)); m = evaluate(p, _pert(p, 0.1))
    d = max(abs(m[k] - ref[k]) / abs(ref[k]) for k in ("primary_field_error", "relL2_phi_global", "relL2_E_band2", "energy_relerr"))
    out.append(_c("T7_quadrature", "default vs doubled quadrature (smooth perturbation)", d, "< 1e-8", d < 1e-8))
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
