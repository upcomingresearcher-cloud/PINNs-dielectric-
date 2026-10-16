"""Evaluator for 1D interface problems (protocol Secs. 3-4). Frozen definitions.

All computation is float64; predictions from float32 models are upcast.
Relative L2 over region R:  ||u_pred - u_ex||_{L2(R)} / ||u_ex||_{L2(R)}  by
Gauss-Legendre quadrature on panels that never straddle the interface or band edges.
"""
import numpy as np

FAIL_THRESHOLD = 0.05


def gl_grid(breaks, panel_max=0.05, pts=16):
    """Composite Gauss-Legendre nodes/weights between consecutive breakpoints."""
    t, w = np.polynomial.legendre.leggauss(pts)
    xs, ws = [], []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        n_pan = max(1, int(np.ceil((hi - lo) / panel_max - 1e-12)))
        edges = np.linspace(lo, hi, n_pan + 1)
        for l, r in zip(edges[:-1], edges[1:]):
            xs.append(0.5 * (r - l) * t + 0.5 * (r + l)); ws.append(0.5 * (r - l) * w)
    return np.concatenate(xs), np.concatenate(ws)


def _rel(pred, ex, w):
    den = np.sqrt(np.sum(w * ex**2))
    return float(np.sqrt(np.sum(w * (pred - ex)**2)) / den) if den > 1e-300 else float("nan")


def evaluate(problem, pred, pts_per_panel=16, panel_max=0.05, extra_breaks=None):
    """extra_breaks: additional panel breakpoints, e.g. FEM nodes, so that piecewise
    predictions are integrated exactly (kinks never fall inside a panel)."""
    a, hb, L, V = problem.a, problem.band_halfwidth, problem.L, problem.V
    breaks = [0.0, a - hb, a, a + hb, L]
    if extra_breaks is not None:
        breaks = np.unique(np.concatenate([breaks, np.asarray(extra_breaks, float)]))
        breaks = breaks[np.concatenate([[True], np.diff(breaks) > 1e-13])]
    x, w = gl_grid(breaks, panel_max, pts_per_panel)
    side = problem.side_of(x)
    eps = problem.eps_side(side)

    phi_p, dphi_p = (np.asarray(v, dtype=np.float64) for v in pred(x, side))
    phi_e, dphi_e = problem.exact(x, side)
    E_p, E_e = -dphi_p, -dphi_e
    D_p, D_e = eps * E_p, eps * E_e

    # one-sided interface values and boundary values
    xa = np.array([a, a]); sa = np.array([1, 2])
    phia, dphia = (np.asarray(v, dtype=np.float64) for v in pred(xa, sa))
    xb = np.array([0.0, L]); sb = np.array([1, 2])
    phib, dphib = (np.asarray(v, dtype=np.float64) for v in pred(xb, sb))
    epsa = problem.eps_side(sa); epsb = problem.eps_side(sb)

    m = {}
    arrays = [phi_p, dphi_p, phia, dphia, phib, dphib]
    m["nonfinite"] = bool(not all(np.all(np.isfinite(v)) for v in arrays))
    if m["nonfinite"]:
        # sanitise so remaining metrics are computable but flagged
        phi_p, dphi_p, phia, dphia, phib, dphib = (np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in arrays)
        E_p, D_p = -dphi_p, eps * (-dphi_p)

    band = np.abs(x - a) < hb
    regions = {"global": np.ones_like(band), "mat1": side == 1, "mat2": side == 2,
               "band": band, "band1": band & (side == 1), "band2": band & (side == 2)}
    for q, (p, e) in {"phi": (phi_p, phi_e), "E": (E_p, E_e), "D": (D_p, D_e)}.items():
        for r, mask in regions.items():
            m[f"relL2_{q}_{r}"] = _rel(p[mask], e[mask], w[mask])

    D_ex = problem.D_exact
    m["phi_jump"] = float(phia[1] - phia[0])                    # [phi] = phi(a+) - phi(a-)
    m["phi_jump_rel"] = abs(m["phi_jump"]) / V
    Da = -epsa * dphia
    m["D_jump"] = float(Da[1] - Da[0])                           # [D_n]
    m["D_jump_rel"] = abs(m["D_jump"]) / abs(D_ex)
    Db = -epsb * dphib
    m["C_flux"] = float(0.5 * (Db[0] + Db[1]) / V)
    m["C_flux_relerr"] = abs(m["C_flux"] - D_ex / V) / abs(D_ex / V)
    m["gauss_imbalance_rel"] = float(abs(Db[0] - Db[1]) / abs(D_ex))
    W = 0.5 * np.sum(w * eps * E_p**2)
    W_ex = 0.5 * np.sum(w * eps * E_e**2)
    m["energy"] = float(W); m["energy_relerr"] = float(abs(W - W_ex) / W_ex)
    m["C_energy_relerr"] = m["energy_relerr"]                    # C_W = 2W/V^2, same relative error
    out1, out2 = (side == 1) & ~band, (side == 2) & ~band
    mE1 = np.sum(w[out1] * E_p[out1]) / np.sum(w[out1]); mE2 = np.sum(w[out2] * E_p[out2]) / np.sum(w[out2])
    kappa_eff = mE1 / mE2 if (mE2 > 0 and mE1 > 0) else float("nan")
    m["kappa_eff"] = float(kappa_eff)
    m["log_kappa_eff_ratio"] = float(np.log(kappa_eff / problem.kappa)) if np.isfinite(kappa_eff) else float("nan")
    m["bc_err_left"] = float(abs(phib[0] - V)); m["bc_err_right"] = float(abs(phib[1]))
    m["n_eval_points"] = int(x.size)

    # ---- failure flags (protocol Sec. 4)
    primary = max(m["relL2_E_band"], m["relL2_D_band"])
    strict = max(m["relL2_E_band1"], m["relL2_E_band2"])           # E and D side errors coincide
    m["primary_field_error"] = primary; m["strict_field_error"] = strict
    m["fail_primary"] = bool(m["nonfinite"] or not (primary <= FAIL_THRESHOLD))
    m["fail_strict"] = bool(m["nonfinite"] or not (strict <= FAIL_THRESHOLD))
    m["fail_potential"] = bool(m["nonfinite"] or not (m["relL2_phi_global"] <= FAIL_THRESHOLD))
    lk = m["log_kappa_eff_ratio"]
    m["sec_fail_effective_contrast"] = bool(not np.isfinite(lk) or abs(lk) > np.log(2))
    m["sec_fail_D_jump"] = bool(not (m["D_jump_rel"] <= FAIL_THRESHOLD))
    return m
