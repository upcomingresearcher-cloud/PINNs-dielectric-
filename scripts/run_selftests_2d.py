import sys, pathlib, csv, json
root = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from src.evaluator import selftests2d
from src.problems import p2d
name = sys.argv[1]
if name == "P2":
    rows = selftests2d.run(p2d.P2Cylinder, [1.0, 0.0005, 80.0, 2000.0], lambda occ: [occ.addDisk(0, 0, 0, 0.25, 0.25)])
out = root / f"results/validation/evaluator_selftests_{name}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(("PASS " if r["passed"] else "FAIL ") + r["test"], "|", r["check"][:90], "|", f"{r['value']:.4g}" if not isinstance(r["value"], bool) else r["value"])
sys.exit(0 if all(r["passed"] for r in rows) else 1)
