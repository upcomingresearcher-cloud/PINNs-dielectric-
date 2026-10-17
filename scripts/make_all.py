"""Regenerate every processed table and figure from the raw databases (Deliverable: single analysis pipeline).

Usage:  python scripts/make_all.py            # all steps
        python scripts/make_all.py --list     # show steps
        python scripts/make_all.py p1 p13     # selected steps (by key)

Every step only READS results/raw/* (runs.csv, runs_2d.csv, fem_runs*.csv, trajectories/, gradlogs/) and writes
results/processed/* and results/figures/*. No numerical value is edited by hand anywhere in the pipeline.
FEM refinement records are read from results/raw/fem_runs_2d.csv when present (they are; see summary2d.fem_rows).
A log with per-step wall time and exit status is written to results/processed/make_all_log.json.
"""
import json, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
STEPS = [
    ("p1",        "P1 primary sweep: failure/median-error maps, critical contrast (Phase 6)",   ["-m", "src.analysis.p1_summary"]),
    ("p1hyp",     "P1 hypotheses H1-H4, H11; global-vs-local, material/jump, gradient figures", ["-m", "src.analysis.p1_hypotheses"]),
    ("femp1",     "P1 FEM mesh-alignment figure",                                              ["-m", "src.analysis.fig_fem_p1"]),
    ("p2",        "P2 summary, maps, FEM convergence (Phase 7)",                                ["-m", "src.analysis.p2_summary"]),
    ("p3",        "P3 summary and maps (Phase 8)",                                              ["-m", "src.analysis.summary2d", "P3"]),
    ("p4",        "P4 summary and maps (Phase 9)",                                              ["-m", "src.analysis.summary2d", "P4"]),
    ("p5",        "P5 summary and maps (Phase 10)",                                             ["-m", "src.analysis.summary2d", "P5"]),
    ("p11",       "Interventions A-E, H4 clause, H5 (Phase 11)",                                ["-m", "src.analysis.interventions"]),
    ("p12",       "Robustness, H6 (Phase 12)",                                                  ["-m", "src.analysis.robustness"]),
    ("p13",       "Remedies vs FEM, H7 (Phase 13)",                                             ["-m", "src.analysis.remedy_vs_fem"]),
    ("final",     "Cross-phase figures: energy, gradient ratio, convergence, critical contrast, FEM, cross-geometry",
                  ["-m", "src.analysis.final_figures"]),
]


def main(argv):
    """Run the selected pipeline steps (all by default), print a status line per step and write a JSON log."""
    if "--list" in argv:
        for k, d, _ in STEPS: print(f"{k:8s} {d}")
        return 0
    sel = [s for s in STEPS if not argv or s[0] in argv]
    log, bad = [], 0
    for k, d, cmd in sel:
        t0 = time.time()
        r = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
        dt = time.time() - t0; ok = r.returncode == 0; bad += not ok
        print(f"[{'ok' if ok else 'FAIL'}] {k:8s} {dt:6.1f}s  {d}", flush=True)
        if not ok: print(r.stderr[-2000:])
        log.append(dict(step=k, description=d, command=" ".join(cmd), exit=r.returncode, seconds=round(dt, 1)))
    (ROOT / "results/processed").mkdir(parents=True, exist_ok=True)
    json.dump(log, open(ROOT / "results/processed/make_all_log.json", "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
