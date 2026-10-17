# Tutorial: neural-network solvers at dielectric interfaces, in plain terms

This tutorial explains what the project did and how to read its results. It is meant for a reader with basic calculus and Python. Every number quoted below is printed by `educational_resource/tutorial_examples.py` from the project database. Its output is saved in `tutorial_examples_output.txt`, and the bracketed tags such as [1] point to the relevant line. Run it yourself from the project root:

```sh
python educational_resource/tutorial_examples.py
```

## 1. The physical problem

A capacitor filled with two layers of different insulating materials ("dielectrics") is the simplest case. The electric potential φ satisfies

$$\frac{d}{dx}\Big(\varepsilon(x)\,\frac{d\varphi}{dx}\Big) = 0,$$

where the permittivity ε jumps from ε₁ to ε₂ at the interface. Let the contrast be κ = ε₂/ε₁. Two things must hold at the interface:
1. **φ is continuous.** There is no jump in voltage.
2. **The displacement D = εE is continuous,** where E = −dφ/dx. Because ε jumps, the field E must jump by a factor κ.

**Example [1].** In our 1D problem P1, with κ = 80, D = 1.9753 on both sides. The field is E₁ = 1.9753 in the low-permittivity layer and E₂ = 0.0247 in the high one. Almost all of the voltage drops across the low-permittivity layer: φ falls from 1 to 0.506 by x = 0.25. The potential has a sharp kink at the interface.

Engineers care about E near interfaces, because that is where insulation breaks down. So the project measures accuracy of E and D *in a thin band around the interface*, not only overall.

## 2. How a PINN is trained

A physics-informed neural network (PINN) represents φ(x) by a small neural network: here 3 layers of 32 tanh units. Training adjusts its weights to make a *loss* small. The loss is the sum of squared violations of:
- the differential equation, at sample points inside each material;
- the boundary values (φ = 1 at the left plate, 0 at the right);
- in some formulations, the interface conditions, as extra penalty terms.

Derivatives come from automatic differentiation. The optimiser (Adam) takes 3,000 steps. No solution data is used, only the equations.

The project compared six ways of setting this up ("formulations"):
- **F1:** one network, equation only.
- **F2:** one network per material, plus penalties for the potential jump and the flux jump.
- **F3:** one network that also receives the distance to the interface.
- **F4:** builds continuity into the network.
- **F5:** minimises the electrostatic energy instead of the equation residual.
- **F6:** F2 with each penalty scaled by its physical size.

## 3. Why discontinuous coefficients are hard

**A smooth function struggles with a kink.** A tanh network is smooth, and a smooth function needs many units to bend sharply. Networks also learn smooth components first ("spectral bias").

**The loss can be blind to the interface.** This is the stronger reason. Inside each material ε is constant, so F1's equation reduces to φ'' = 0 on each side. Nothing in F1's loss says how the two sides must match. The *wrong* answer, a straight line as if both layers had the same permittivity, satisfies every term exactly.

**Example [3].** At κ = 80, F1's median band field error is 27.8, i.e. 2,780%. Its effective-contrast diagnostic is −4.39, which is ln(1/80): the network behaves as if κ were 1. It found the wrong solution, not a nearly-right one.

Formulations that add interface penalties (F2, F6) do much better in 1D, but they face a balancing problem. When κ is large, the flux penalty is numerically huge compared with the continuity penalty. The optimiser satisfies one and leaves the potential slightly discontinuous.

**Example [2].** The F2 run at κ = 80 with seed 0 ends with a potential jump of 1.6% of the voltage and a flux jump of only 0.005%. Its band field error is 5.25%, just over the failure threshold.

## 4. How the evaluator works

After training, the evaluator compares the network with an exact or reference solution on a fine grid:
- **Global potential error:** relative L2 error of φ over the whole domain.
- **Band field error (the primary metric):** relative L2 error of E and of D in the band within 0.05 of the interface. The larger of the two counts.
- **Failure:** a run *fails* if the band field error exceeds 5%.
- **Interface diagnostics:** the potential jump and displacement jump across the interface, which say *which* condition is violated.
- **Physical checks:** capacitance and energy errors.

The evaluator was itself tested before use: 98 checks in 1D and 17 per 2D problem. The tests feed it known solutions and deliberately broken ones and check that it reports what it should.

## 5. How to read the CSV database

Every training run is one row of `results/raw/runs.csv` (1D) or `results/raw/runs_2d.csv` (2D). The run ID encodes problem, formulation, contrast, seed, run type and variant, for example `P1-F2-k80-s0-pinn-base`.

| Column | Meaning |
|---|---|
| `primary_field_error` | band field error (the primary metric) |
| `fail_primary` | True if above 5% |
| `relL2_phi_global` | global potential error |
| `phi_jump_rel`, `D_jump_rel` | interface violations |
| `wall_clock_s` | training time (compile time excluded) |
| `grad_evals` | gradient evaluations used |
| `status` | `complete`, or `failed_technical` (kept but not analysed) |

Per-run training histories are in `results/raw/trajectories/` (metrics every 250 steps) and `results/raw/gradlogs/` (per-term gradients every 25 steps). The full schema is in `docs/database_schema.md`.

```python
import pandas as pd
d = pd.read_csv("results/raw/runs.csv", low_memory=False)
p = d[(d.run_type == "pinn") & (d.variant == "base") & (d.status == "complete")]
print(p.pivot_table(index="formulation", columns="kappa", values="fail_primary", aggfunc="sum"))
```

## 6. How to read a failure map

A failure map (`results/figures/p1_failure_map.png`) is a table of formulations by contrast. Each cell shows how many of the 5 random seeds failed.

**Example [4].** For P1:
- F1 fails 5/5 at every κ ≥ 5.
- F3, F4 and F5 fail from κ = 20.
- F2 fails 1/5 at κ = 80, 2/5 at 400 and 5/5 at 2000.
- F6 fails only at κ = 2000 (3/5).

**Why five seeds matter [6].** F2 at κ = 400 gives band errors of 0.212, 0.082, 0.005, 0.011 and 0.048 for seeds 0–4: two failures and three passes. A paper that showed only the best seed would report 0.5% error. The honest summary is "fails in 2 of 5 runs". Even five seeds give coarse estimates: 0/5 failures still allows a true failure rate of up to about 50%.

## 7. Why a small global error can hide a failure

**Example [5].** Over all P1 runs with κ > 1, the band field error is a median 2.39 times the global potential error. 22 of 95 failing runs have a global potential error below 5%, so they "look fine". On the thin-film problem P5 the ratio is 8.25, and 60 of 163 failures are concealed this way.

**Lesson:** check the quantity you care about, where you care about it.

## 8. Why FEM serves as the reference

The finite element method (FEM) solves the same equations on a mesh. When the mesh follows the interface (*interface-conforming*), the kink is built into the approximation and the error shrinks predictably as the mesh is refined. In P1 it is exact. In 2D it converges.

**Example [7].** For the dielectric cylinder P2 at κ = 80:

| Conforming FEM mesh | Band field error | Time |
|---|---|---|
| 565 DOFs | 17.6% | 0.01 s |
| 7,869 DOFs | 3.0% | 0.13 s |
| 30,337 DOFs | 1.0% | 0.67 s |
| 119,697 DOFs | 0.4% | 4.4 s |

The F2 PINN reaches a median 63% after 10 s of training.

FEM is the reference for two reasons. It provides accurate solutions where no formula exists (P5). And it shows what accuracy is achievable at a given cost. A mesh that *ignores* the interface does not converge in the band, which is the same difficulty the PINN faces.

## 9. What the experiments establish, and what they do not

**Established** (for these problems, these formulations and a 3,000-step budget):
- The single-network strong-form PINN (F1) cannot represent a dielectric interface. Its loss is blind to it.
- Two-network formulations work in 1D up to high contrast, but on curved, singular, corner and thin-film geometries every formulation failed at κ ≥ 80.
- Reporting only global potential error hides a substantial fraction of failures.
- Several popular fixes did not help when tested fairly, one factor at a time with matched seeds: gradient rebalancing, residual normalisation, hard continuity. An apparent effect of float64 precision was traced to a bug: the two precisions had used different random starting weights.
- Interface-conforming FEM reached 1% band accuracy in seconds where no PINN configuration reached it.

**Not established:**
- That PINNs cannot solve interface problems. Larger networks, other architectures, adaptive sampling or much longer training were not tested at scale.
- That FEM is always better. FEM also struggles near singular sources (P3), and PINNs have uses we did not test, such as inverse problems.
- Anything about 3D, time-dependent or nonlinear problems.

**Further reading:** the paper (`manuscript/paper.md`); per-phase findings (`docs/phase*_findings.md`); and the protocol and change log, which record every decision and every bug found and fixed (`docs/`).
