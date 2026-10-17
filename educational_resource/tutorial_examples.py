"""Companion to tutorial.md: every number quoted in the tutorial is printed by this script from results/raw/.
Run from the project root:  python educational_resource/tutorial_examples.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from src.problems.p1 import P1Problem

# 1. The exact P1 solution and the interface conditions
p = P1Problem(80.0)
x = np.array([0.25, 0.75]); phi, E = p.exact(x, np.array([1, 2]))[0], None
D = p.D_exact
print(f"[1] P1, kappa=80: D = {D:.4f}; E1 = D/eps1 = {D/1:.4f}, E2 = D/eps2 = {D/80:.5f} (E jumps by 80, D is continuous)")
print(f"    potential at x=0.25: {phi[0]:.4f}; at x=0.75: {phi[1]:.5f}")

runs = pd.read_csv("results/raw/runs.csv", low_memory=False)
P = runs[(runs.run_type == "pinn") & (runs.variant == "base") & (runs.status == "complete")]

# 2. One row of the database
r = P[(P.formulation == "F2") & (P.kappa == 80) & (P.seed == 0)].iloc[0]
print(f"[2] run {r.run_id}: primary={r.primary_field_error:.4f} phi_global={r.relL2_phi_global:.2e} "
      f"phi_jump={r.phi_jump_rel:.2e} D_jump={r.D_jump_rel:.2e} fail={bool(r.fail_primary)} wall={r.wall_clock_s:.2f}s evals={int(r.grad_evals)}")

# 3. F1 converges to the wrong (uniform-permittivity) solution
g = P[(P.formulation == "F1") & (P.kappa == 80)]
print(f"[3] F1 kappa=80: median primary error {g.primary_field_error.median():.1f}, median final loss {g.final_loss.median():.2e}, "
      f"median effective-contrast log ratio {g.log_kappa_eff_ratio.median():.2f} (0 = correct contrast)")

# 4. Failure map numbers (P1)
fm = P[P.kappa <= 2000].pivot_table(index="formulation", columns="kappa", values="fail_primary", aggfunc="sum")
print("[4] P1 failed seeds of 5:\n" + fm.astype(int).to_string())

# 5. Masking: field error vs potential error
Q = P[P.kappa > 1].assign(R=lambda d: d.primary_field_error / d.relL2_phi_global)
fl = Q[Q.fail_primary.astype(bool)]
print(f"[5] P1 kappa>1: median R = {Q.R.median():.2f}; concealed failures {int((fl.relL2_phi_global < .05).sum())}/{len(fl)}")
two = pd.read_csv("results/raw/runs_2d.csv", low_memory=False)
T = two[(two.run_type == "pinn") & (two.variant == "base") & (two.status == "complete")].drop_duplicates("run_id")
Q5 = T[(T.problem_id == "P5") & (T.kappa != 1)]; f5 = Q5[Q5.fail_primary.astype(bool)]
print(f"    P5 kappa!=1: median R = {(Q5.primary_field_error / Q5.relL2_phi_global).median():.2f}; concealed {int((f5.relL2_phi_global < .05).sum())}/{len(f5)}")

# 6. Seed variability
g = P[(P.formulation == "F2") & (P.kappa == 400)].sort_values("seed")
print("[6] F2 kappa=400 per seed:", ", ".join(f"s{int(s)}={e:.3f}" for s, e in zip(g.seed, g.primary_field_error)))

# 7. FEM vs PINN on P2 at kappa=80
fem = pd.read_csv("results/raw/fem_runs_2d.csv")
f = fem[(fem.problem_id == "P2") & (fem.kappa == 80) & fem.conforming].sort_values("dofs")
print("[7] P2 kappa=80 conforming FEM: " + "; ".join(f"{int(d)} DOFs {e:.3f} in {t:.2f}s" for d, e, t in zip(f.dofs, f.primary_field_error, f.mesh_plus_solve_s)))
g = T[(T.problem_id == "P2") & (T.formulation == "F2") & (T.kappa == 80)]
print(f"    PINN F2: median error {g.primary_field_error.median():.3f} after {g.wall_clock_s.median():.1f}s training")
