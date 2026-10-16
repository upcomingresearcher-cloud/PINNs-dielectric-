"""Resumable 2D sweep runner -> results/raw/runs_2d.csv.
python scripts/run_sweep2d.py --problem P2 --run-type pinn --formulations F1 --kappas 1,80 --seeds 0,1"""
import argparse, sys, pathlib, traceback, time
root = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
import yaml
from src.pinns.train2d import train_run, RUNS2D_COLUMNS
from src.experiments import records as R
from src.problems import p2d
ap = argparse.ArgumentParser()
for a_ in ("--problem", "--run-type", "--formulations", "--kappas", "--seeds"): ap.add_argument(a_, required=True)
ap.add_argument("--variant", default="base"); ap.add_argument("--override", action="append", default=[])
ap.add_argument("--max-seconds", type=float, default=255.0); ap.add_argument("--problem-class", default=None)
a = ap.parse_args()
PROBS = {"P2": p2d.P2Cylinder}
for nm in ("P3LineCharge", "P4Corner", "P5Film"):
    if hasattr(p2d, nm): PROBS[getattr(p2d, nm).problem_id] = getattr(p2d, nm)
cls = getattr(p2d, a.problem_class) if a.problem_class else PROBS[a.problem]; cfg = yaml.safe_load(open(root / f"configs/{a.problem.lower()}.yaml"))
ov = {}
for s in a.override:
    k, v = s.split("="); ov[k] = v if k == "precision" else float(v)
out = root / "results/raw/runs_2d.csv"; done = R.completed_ids(out); t0 = time.time(); n = 0
for f in a.formulations.split(","):
    for k in a.kappas.split(","):
        for s in a.seeds.split(","):
            rid = R.run_id(a.problem, f, float(k), int(s), a.run_type, a.variant)
            if rid in done: continue
            if time.time() - t0 > a.max_seconds: print(f"time budget reached; {n} new runs", flush=True); sys.exit(0)
            try:
                row, _ = train_run(f, cls, float(k), int(s), cfg, run_type=a.run_type, variant=a.variant, out_dir=root / "results/raw", overrides=ov)
            except Exception as e:
                row = {c: "" for c in RUNS2D_COLUMNS}
                row.update(run_id=rid, run_type=a.run_type, status="failed_technical", problem_id=a.problem, formulation=f, kappa=float(k),
                           seed=int(s), variant=a.variant, notes="".join(traceback.format_exception_only(e)).strip()[:500])
            R.append_row(out, row, RUNS2D_COLUMNS); n += 1
            print(f"[{time.time()-t0:6.1f}s] {rid} {row['status']} primary={row.get('primary_field_error')}", flush=True)
print(f"ALL DONE ({n} new runs)", flush=True)
