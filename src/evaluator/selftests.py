"""Gate 1: evaluator self-tests (brief Sec. 12; protocol Sec. 14).
Each test returns a list of check dicts {test, check, value, criterion, passed}."""
import numpy as np
from src.problems.p1 import P1Problem, P1SProblem
from src.fem import fem1d
from src.evaluator.evaluator1d import evaluate

KAPPAS = [1, 5, 20, 80, 400, 2000]
TOL_EXACT = 1e-12      # exact solution: round-off only
TOL_FEM = 1e-10        # conforming FEM: round-off of a sparse solve at kappa up to 2000


def _c(test, check, value, criterion, passed):
    return dict(test=test, check=check, value=value, criterion=criterion, passed=bool(passed))


def _err_keys(m):
    return [k for k in m if k.startswith("relL2_")] + ["phi_jump_rel", "D_jump_rel", "C_flux_relerr",
            "gauss_imbalance_rel", "energy_relerr", "bc_err_left", "bc_err_right"]


def test1_exact():
    """T1: the exact solution must give zero error (to round-off) at every contrast."""
    out = []
    for k in KAPPAS:
        p = P1Problem(k); m = evaluate(p, p.exact)
        worst = max(m[q] for q in _err_keys(m))
        out.append(_c("T1_exact", f"kappa={k}: max error metric", worst, f"< {TOL_EXACT}", worst < TOL_EXACT))
        out.append(_c("T1_exact", f"kappa={k}: |ln(kappa_eff/kappa)|", abs(m["log_kappa_eff_ratio"]), "< 1e-10",
                      abs(m["log_kappa_eff_ratio"]) < 1e-10))
        out.append(_c("T1_exact", f"kappa={k}: not flagged", m["fail_primary"] or m["fail_strict"], "False",
                      not (m["fail_primary"] or m["fail_strict"] or m["nonfinite"])))
    return out


def test2_fem():
    """T2: conforming FEM solutions must be exact and pass the evaluator."""
    out = []
    for k in KAPPAS:
        for n in (2, 4, 20, 200):
            p = P1Problem(k); m = evaluate(p, fem1d.solve(p, n))
            worst = max(m[q] for q in _err_keys(m))
            out.append(_c("T2_fem_conforming", f"kappa={k}, N={n}: max error metric", worst, f"< {TOL_FEM}", worst < TOL_FEM))
    # convergence on the manufactured variant P1-S (FEM code verification)
    for k in (1, 80, 2000):
        p = P1SProblem(k); Ns = [10, 20, 40, 80, 160]
        ephi, eE = [], []
        for n in Ns:
            fe = fem1d.solve(p, n, f=p.f); m = evaluate(p, fe, extra_breaks=fe.nodes)
            ephi.append(m["relL2_phi_global"]); eE.append(m["relL2_E_global"])
        sphi = -np.polyfit(np.log(Ns), np.log(ephi), 1)[0]; sE = -np.polyfit(np.log(Ns), np.log(eE), 1)[0]
        out.append(_c("T2_fem_convergence_P1S", f"kappa={k}: phi L2 order (errors {['%.2e' % e for e in ephi]})", sphi, "in [1.9, 2.1]", 1.9 <= sphi <= 2.1))
        out.append(_c("T2_fem_convergence_P1S", f"kappa={k}: E L2 order (errors {['%.2e' % e for e in eE]})", sE, "in [0.9, 1.1]", 0.9 <= sE <= 1.1))
    return out


def test3_wrong_uniform():
    """T3: the uniform-permittivity (wrong) solution must be flagged as a failure."""
    p = P1Problem(80); m = evaluate(p, p.wrong_uniform)
    return [_c("T3_wrong_uniform_k80", "primary field error", m["primary_field_error"], "> 0.05 and flagged", m["fail_primary"]),
            _c("T3_wrong_uniform_k80", "relL2_E_mat1 (expected 1 - 1/D = 0.4938)", m["relL2_E_mat1"], "approx 0.4938",
               abs(m["relL2_E_mat1"] - (1 - 1 / p.D_exact)) < 1e-12),
            _c("T3_wrong_uniform_k80", "potential failure flag", m["fail_potential"], "True", m["fail_potential"]),
            _c("T3_wrong_uniform_k80", "D jump detected (rel)", m["D_jump_rel"], "> 0.05", m["sec_fail_D_jump"])]


def _perturbed(p, delta):
    def pred(x, side):
        """Exact solution plus a smooth perturbation of size delta that vanishes at the plates."""
        phi, d = p.exact(x, side)
        return phi + delta * x * (1 - x), d + delta * (1 - 2 * x)
    return pred


def test4_threshold():
    """T4: controlled perturbations just below / above the threshold must pass / fail."""
    out = []
    for k in (1, 80, 2000):
        p = P1Problem(k)
        unit = evaluate(p, _perturbed(p, 1.0))["primary_field_error"]   # error is linear in delta
        for target, should_fail in ((0.04, False), (0.06, True)):
            m = evaluate(p, _perturbed(p, target / unit))
            out.append(_c("T4_threshold", f"kappa={k}: perturbation giving {target:.0%}", m["primary_field_error"],
                          f"fail_primary == {should_fail}", m["fail_primary"] == should_fail and abs(m["primary_field_error"] - target) < 1e-9))
    return out


def test5_nonfinite():
    """T5: non-finite predictions must be detected and flagged."""
    p = P1Problem(80); out = []
    for label, bad in (("NaN in phi", ("phi", np.nan)), ("+Inf in dphi", ("d", np.inf)), ("-Inf in phi at interface", ("phia", -np.inf))):
        def pred(x, side, bad=bad):
            """Exact solution with one non-finite value injected into phi or its derivative."""
            phi, d = p.exact(x, side); phi = phi.copy(); d = d.copy()
            if bad[0] == "phi": phi[len(phi) // 3] = bad[1]
            elif bad[0] == "d": d[len(d) // 2] = bad[1]
            elif bad[0] == "phia" and np.any(np.isclose(x, p.a)): phi[np.isclose(x, p.a)] = bad[1]
            return phi, d
        m = evaluate(p, pred)
        out.append(_c("T5_nonfinite", label, m["nonfinite"], "nonfinite and fail_primary both True", m["nonfinite"] and m["fail_primary"]))
    return out


def test6_interface():
    """T6: imposed potential and displacement jumps must be reported with the correct size."""
    out = []
    for k in (1, 80, 2000):
        p = P1Problem(k)
        for c in (1e-3, 0.02):
            def pred(x, side, c=c):
                """Exact solution with a constant potential jump c added in material 2."""
                phi, d = p.exact(x, side); return phi + c * (np.asarray(side) == 2), d
            m = evaluate(p, pred)
            out.append(_c("T6_interface", f"kappa={k}: imposed [phi]={c}", m["phi_jump"], f"== {c} (1e-12), [D]==0",
                          abs(m["phi_jump"] - c) < 1e-12 and abs(m["D_jump"]) < 1e-12))
        for s in (0.01, 0.1):
            def pred(x, side, s=s):
                """Exact solution with the material-2 displacement scaled by (1 + s)."""
                phi, d = p.exact(x, side); return phi, d * np.where(np.asarray(side) == 2, 1 + s, 1.0)
            m = evaluate(p, pred)
            exp = s * p.D_exact
            out.append(_c("T6_interface", f"kappa={k}: imposed [D]={s}*D", m["D_jump"], f"== {exp:.6g}; D-jump flag == {s > 0.05}",
                          abs(m["D_jump"] - exp) < 1e-12 * max(1, exp) and m["sec_fail_D_jump"] == (s > 0.05) and abs(m["phi_jump"]) < 1e-12))
    return out


def test7_grid():
    """T7: metrics must be insensitive to doubling the evaluation grid."""
    out = []
    keys = ["relL2_phi_global", "relL2_E_band", "relL2_D_band", "relL2_E_mat2", "energy_relerr", "primary_field_error"]
    for k in (5, 2000):
        p = P1Problem(k)
        cases = {"wrong_uniform": p.wrong_uniform, "perturbed": _perturbed(p, 0.1)}
        for name, pred in cases.items():
            ref = evaluate(p, pred, pts_per_panel=64, panel_max=0.0125)
            for pts, pm in ((16, 0.05), (32, 0.05), (16, 0.025)):
                m = evaluate(p, pred, pts_per_panel=pts, panel_max=pm)
                d = max(abs(m[q] - ref[q]) / max(abs(ref[q]), 1e-300) for q in keys)
                out.append(_c("T7_grid", f"kappa={k} {name}: {pts} pts/panel, panel {pm} vs fine", d, "< 1e-10 (polynomial integrands)", d < 1e-10))
        # informational: non-smooth (FEM, kinks inside panels) - not a pass criterion beyond convergence
        fe = fem1d.solve(p, 21)
        ref = evaluate(p, fe, pts_per_panel=64, panel_max=0.0125, extra_breaks=fe.nodes)
        m = evaluate(p, fe)
        d = max(abs(m[q] - ref[q]) / max(abs(ref[q]), 1e-300) for q in keys)
        out.append(_c("T7_grid_info", f"kappa={k} non-conforming FEM N=21: default grid (kinks inside panels) vs node-aligned fine grid", d, "reported; motivates extra_breaks for FEM", True))
        m2 = evaluate(p, fe, extra_breaks=fe.nodes)
        d2 = max(abs(m2[q] - ref[q]) / max(abs(ref[q]), 1e-300) for q in keys)
        out.append(_c("T7_grid", f"kappa={k} non-conforming FEM N=21: node-aligned default vs node-aligned fine", d2, "< 1e-10", d2 < 1e-10))
    return out


def test8_mesh_alignment():
    """T8: misaligned FEM meshes must show the expected interface-band error behaviour."""
    out = []
    for k in (5, 80, 2000):
        p = P1Problem(k)
        conf = evaluate(p, fem1d.solve(p, 20))
        out.append(_c("T8_mesh", f"kappa={k}: conforming N=20 primary", conf["primary_field_error"], f"< {TOL_FEM}", conf["primary_field_error"] < TOL_FEM))
        Ns = [21, 41, 81, 161, 321, 641]; prim, phis = [], []
        for n in Ns:
            fe = fem1d.solve(p, n); m = evaluate(p, fe, extra_breaks=fe.nodes)
            prim.append(m["primary_field_error"]); phis.append(m["relL2_phi_global"])
        sp_ = -np.polyfit(np.log(Ns), np.log(prim), 1)[0]; sf = -np.polyfit(np.log(Ns), np.log(phis), 1)[0]
        out.append(_c("T8_mesh", f"kappa={k}: non-conforming band field errors {['%.3g' % e for e in prim]}", sp_, "reported order; must decrease with N", all(np.diff(prim) < 0)))
        out.append(_c("T8_mesh", f"kappa={k}: non-conforming global phi errors {['%.3g' % e for e in phis]}", sf, "reported order; must decrease with N", all(np.diff(phis) < 0)))
    return out


ALL = [test1_exact, test2_fem, test3_wrong_uniform, test4_threshold, test5_nonfinite, test6_interface, test7_grid, test8_mesh_alignment]
