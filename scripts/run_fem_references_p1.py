"""Compute and record P1 FEM references (conforming and non-conforming) with timing.
Writes results/raw/fem_runs.csv (skips fem_ids already present)."""
import sys, pathlib, time
root = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
import numpy as np, yaml
from src.problems.p1 import P1Problem
from src.fem import fem1d
from src.evaluator.evaluator1d import evaluate
from src.experiments import records as R, provenance as P

cfg = yaml.safe_load(open(root / "configs/protocol_v1.yaml"))
out = root / "results/raw/fem_runs.csv"; done = R.completed_ids(out, "fem_id")
kappas = cfg["contrasts"]["primary"] + cfg["contrasts"]["extension_preregistered"]
meshes = [(n, True) for n in (2, 4, 20, 200, 2000, 20000)] + [(n, False) for n in (21, 41, 81, 161, 321, 641, 1281, 2561, 20001)]
REPEATS = 7
for k in kappas:
    p = P1Problem(float(k))
    for n, conf in meshes:
        fid = f"P1-FEM-k{k:g}-N{n}"
        if fid in done: continue
        fem1d.solve(p, n)                                   # warm-up
        ts = []
        for _ in range(REPEATS):
            t0 = time.perf_counter(); sol = fem1d.solve(p, n); ts.append(time.perf_counter() - t0)
        assert sol.conforming == conf
        m = evaluate(p, sol, extra_breaks=sol.nodes)
        row = dict(fem_id=fid, problem_id="P1", kappa=k, n_elements=n, dofs=sol.dofs, conforming=conf,
                   element_eps_rule="exact element average", quadrature="GL16 panels + FEM nodes",
                   solve_time_s=float(np.median(ts)), timing_repeats=REPEATS, **m,
                   software_env=P.software_env(), hardware=P.hardware(), git_commit=P.git_commit(), timestamp=P.now())
        R.append_row(out, row, R.FEM_COLUMNS)
print("rows:", sum(1 for _ in open(out)) - 1)
