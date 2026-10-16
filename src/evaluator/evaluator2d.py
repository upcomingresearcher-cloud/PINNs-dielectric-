"""Generic 2D evaluator (protocol Secs. 3-4 generalised; change log v1.2.0).
Needs from the problem: quadrature() dict, exact(X, side), eps_side, Vc, h (boundary data),
optional extra_metrics(X, W, side, grad_pred). Relative L2 errors use vector norms for E and D."""
import json
import numpy as np

FAIL_THRESHOLD = 0.05


def _rel(err2, ref2, w):
    den = np.sqrt(np.sum(w * ref2))
    return float(np.sqrt(np.sum(w * err2)) / den) if den > 1e-300 else float("nan")


def evaluate(problem, pred, quad=None):
    q = quad or problem.quadrature()
    X, W, side, band = q["X"], q["W"], q["side"], q["band"]
    eps = problem.eps_side(side)
    phi_p, g_p = (np.asarray(v, np.float64) for v in pred(X, side))
    phi_e, g_e = problem.exact(X, side)
    XI, WI, NI = q["XI"], q["WI"], q["NI"]; n_i = len(XI)
    o1, o2 = np.ones(n_i, int), np.full(n_i, 2)
    f1, gg1 = (np.asarray(v, np.float64) for v in pred(XI, o1)); f2, gg2 = (np.asarray(v, np.float64) for v in pred(XI, o2))
    e1, ge1 = problem.exact(XI, o1); e2, ge2 = problem.exact(XI, o2)
    pb, _ = (np.asarray(v, np.float64) for v in pred(q["XB"], q["sideB"]))
    hb = np.asarray(problem.h_np(q["XB"]))
    m = {}
    m["nonfinite"] = bool(not all(np.all(np.isfinite(v)) for v in (phi_p, g_p, f1, f2, gg1, gg2, pb)))
    if m["nonfinite"]:
        phi_p, g_p, f1, f2, gg1, gg2, pb = (np.nan_to_num(v, nan=0., posinf=0., neginf=0.) for v in (phi_p, g_p, f1, f2, gg1, gg2, pb))
    E_p, E_e = -g_p, -g_e; D_p, D_e = eps[:, None] * E_p, eps[:, None] * E_e
    regions = {"global": np.ones(len(X), bool), "mat1": side == 1, "mat2": side == 2, "band": band,
               "band1": band & (side == 1), "band2": band & (side == 2)}
    for name, (p, e) in {"phi": (phi_p[:, None], phi_e[:, None]), "E": (E_p, E_e), "D": (D_p, D_e)}.items():
        for r, mk in regions.items():
            m[f"relL2_{name}_{r}"] = _rel(np.sum((p[mk] - e[mk])**2, 1), np.sum(e[mk]**2, 1), W[mk])
    Vc = problem.Vc; L_gamma = np.sum(WI)
    jphi = f2 - f1
    m["phi_jump_rel"] = float(np.sqrt(np.sum(WI * jphi**2) / L_gamma) / Vc)
    Dn1 = -problem.eps1 * np.sum(gg1 * NI, 1); Dn2 = -problem.eps2 * np.sum(gg2 * NI, 1)
    Dn1e = -problem.eps1 * np.sum(ge1 * NI, 1); Dn2e = -problem.eps2 * np.sum(ge2 * NI, 1)
    jD = (Dn2 - Dn1) - (Dn2e - Dn1e)                         # deviation from the exact jump (0 unless surface charge)
    m["D_jump_rel"] = _rel(jD**2, Dn1e**2, WI)
    T = np.c_[-NI[:, 1], NI[:, 0]]
    jEt = -np.sum((gg2 - gg1) * T, 1) + np.sum((ge2 - ge1) * T, 1)
    m["Et_jump_rel"] = _rel(jEt**2, np.sum(ge1 * T, 1)**2, WI)
    Wp = 0.5 * np.sum(W * eps * np.sum(E_p**2, 1)); We = 0.5 * np.sum(W * eps * np.sum(E_e**2, 1))
    m["energy"] = float(Wp); m["energy_relerr"] = float(abs(Wp - We) / We)
    m["bc_err_max_rel"] = float(np.max(np.abs(pb - hb)) / Vc)
    extra = problem.extra_metrics(X, W, side, g_p) if hasattr(problem, "extra_metrics") else {}
    if hasattr(problem, "corner_metrics"): extra.update(problem.corner_metrics(pred))
    m["log_kappa_eff_ratio"] = extra.get("log_kappa_eff_ratio", float("nan"))
    m["extra_json"] = json.dumps(extra)
    m["n_eval_points"] = int(len(X))
    primary = max(m["relL2_E_band"], m["relL2_D_band"]); strict = max(m["relL2_E_band1"], m["relL2_E_band2"])
    m["primary_field_error"] = primary; m["strict_field_error"] = strict
    m["fail_primary"] = bool(m["nonfinite"] or not (primary <= FAIL_THRESHOLD))
    m["fail_strict"] = bool(m["nonfinite"] or not (strict <= FAIL_THRESHOLD))
    m["fail_potential"] = bool(m["nonfinite"] or not (m["relL2_phi_global"] <= FAIL_THRESHOLD))
    lk = m["log_kappa_eff_ratio"]
    m["sec_fail_effective_contrast"] = bool(np.isfinite(lk) and abs(lk) > np.log(2)) if "log_kappa_eff_ratio" in extra else False
    m["sec_fail_D_jump"] = bool(not (m["D_jump_rel"] <= FAIL_THRESHOLD))
    return m


METRIC2D_COLUMNS = ([f"relL2_{q}_{r}" for q in ("phi", "E", "D") for r in ("global", "mat1", "mat2", "band", "band1", "band2")]
                    + ["nonfinite", "phi_jump_rel", "D_jump_rel", "Et_jump_rel", "energy", "energy_relerr", "bc_err_max_rel",
                       "log_kappa_eff_ratio", "extra_json", "n_eval_points", "primary_field_error", "strict_field_error",
                       "fail_primary", "fail_strict", "fail_potential", "sec_fail_effective_contrast", "sec_fail_D_jump"])
