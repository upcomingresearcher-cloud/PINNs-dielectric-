"""P5 reference validation: FEM levels vs the finest level (band field), per kappa. Writes results/validation/p5_reference_convergence.csv."""
import sys, pathlib, time; root = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
import numpy as np, pandas as pd
from src.problems import p2d
from src.fem import fem2d
from src.evaluator.evaluator2d import evaluate
LEVELS = [dict(h_far=0.2, frac_in=1 / 2), dict(h_far=0.1, frac_in=1 / 4), dict(h_far=0.05, frac_in=1 / 8), dict(h_far=0.025, frac_in=1 / 16)]
cls = getattr(p2d, sys.argv[1]); kappas = [float(k) for k in sys.argv[2].split(",")]
out = root / "results/validation/p5_reference_convergence.csv"; rows = []
for k in kappas:
    fine = cls(k); fine.REF_LEVEL = LEVELS[-1]; t0 = time.perf_counter(); fs = fine.build_reference(LEVELS[-1]); tf = time.perf_counter() - t0
    for lv in LEVELS[:-1]:
        p = cls(k); s = fem2d.solve(p, lv["h_far"], geometry=p.geometry(), grade=(p.t * lv["frac_in"], p.t, 0.3))
        m = evaluate(fine, s)
        rows.append(dict(cls=sys.argv[1], t=p.t, kappa=k, h_far=lv["h_far"], h_in=p.t * lv["frac_in"], dofs=s.dofs, finest_dofs=fs.dofs,
                         band_field_diff_vs_finest=m["primary_field_error"], film_field_diff=__import__("json").loads(m["extra_json"])["relL2_E_film"],
                         phi_diff=m["relL2_phi_global"], solve_s=s.t_solve, finest_build_s=tf))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows); df = pd.concat([pd.read_csv(out), df]) if out.exists() else df; df.to_csv(out, index=False)
