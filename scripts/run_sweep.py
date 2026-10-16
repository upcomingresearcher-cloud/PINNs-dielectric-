"""Resumable sweep runner. Usage:
  python scripts/run_sweep.py --run-type pilot --formulations F1,F2 --kappas 1,80 --seeds 100,101 \
      --variant lr3e-3 --override lr_init=3e-3
Skips run_ids already marked complete in results/raw/runs.csv. Technical crashes are recorded
with status=failed_technical (never silently dropped)."""
import argparse, sys, pathlib, traceback, time
root = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
import yaml
from src.pinns.train import train_run
from src.experiments import records as R

ap = argparse.ArgumentParser()
ap.add_argument("--run-type", required=True); ap.add_argument("--formulations", required=True)
ap.add_argument("--kappas", required=True); ap.add_argument("--seeds", required=True)
ap.add_argument("--variant", default="base"); ap.add_argument("--override", action="append", default=[])
ap.add_argument("--max-seconds", type=float, default=None, help="stop before starting a run after this many seconds")
a = ap.parse_args()
cfg = yaml.safe_load(open(root / "configs/protocol_v1.yaml"))
ov = {}
for s in a.override:
    k, v = s.split("=")
    try: ov[k] = float(v)
    except ValueError: ov[k] = v            # string overrides: precision, optimiser
out = root / "results/raw/runs.csv"; done = R.completed_ids(out)
todo = [(f, float(k), int(s)) for f in a.formulations.split(",") for k in a.kappas.split(",") for s in a.seeds.split(",")]
t0 = time.time(); n = 0
for f, k, s in todo:
    rid = R.run_id("P1", f, k, s, a.run_type, a.variant)
    if rid in done: continue
    if a.max_seconds is not None and time.time() - t0 > a.max_seconds:
        print(f"max-seconds reached; {sum(1 for f2, k2, s2 in todo if R.run_id('P1', f2, k2, s2, a.run_type, a.variant) not in done) - n} runs left", flush=True)
        break
    try:
        row, _ = train_run(f, k, s, cfg, run_type=a.run_type, variant=a.variant, out_dir=root / "results/raw", overrides=ov)
    except Exception as e:
        row = {c: "" for c in R.RUNS_COLUMNS}
        row.update(run_id=rid, run_type=a.run_type, status="failed_technical", problem_id="P1", formulation=f, kappa=k, seed=s,
                   variant=a.variant, notes=("".join(traceback.format_exception_only(e))).strip()[:500])
    R.append_row(out, row, R.RUNS_COLUMNS); n += 1
    print(f"[{time.time()-t0:7.1f}s] {rid} status={row['status']} primary={row.get('primary_field_error')}", flush=True)
print(f"done: {n} new runs in {time.time()-t0:.1f}s", flush=True)
