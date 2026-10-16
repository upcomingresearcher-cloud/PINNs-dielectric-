"""Run Gate 1 self-tests; write results/validation/evaluator_selftests_p1.{csv,json}. Exit 1 on any failure."""
import sys, json, csv, datetime, subprocess, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.evaluator.selftests import ALL
rows = [r for t in ALL for r in t()]
root = pathlib.Path(__file__).resolve().parents[1] / "results/validation"; root.mkdir(parents=True, exist_ok=True)
try: commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception: commit = "none"
for r in rows: r["value"] = r["value"] if isinstance(r["value"], (bool, int)) else float(r["value"])
with open(root / "evaluator_selftests_p1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["test", "check", "value", "criterion", "passed"]); w.writeheader(); w.writerows(rows)
summary = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
           "n_checks": len(rows), "n_passed": sum(r["passed"] for r in rows),
           "by_test": {t: {"checks": sum(r["test"] == t for r in rows), "passed": sum(r["passed"] for r in rows if r["test"] == t)}
                       for t in dict.fromkeys(r["test"] for r in rows)}}
json.dump(summary, open(root / "evaluator_selftests_p1.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
for r in rows:
    if not r["passed"]: print("FAILED:", r)
sys.exit(0 if summary["n_passed"] == summary["n_checks"] else 1)
