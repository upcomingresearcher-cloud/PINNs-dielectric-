# H9 — Evaluator reusability (final assessment)

**Pre-registered criterion** (protocol, H9):
- **Supported:** no edits to `src/evaluator/` core functions after the P1 freeze, except additive, documented metrics.
- **Refuted:** a core metric function has to be rewritten.
- **Measure:** git diff of the evaluator core.

## Evidence (from `git log` / `git show` on `src/evaluator/`)

| File | Commits touching it after the P1 freeze (`aabfd44`, Phase 3) | Nature of change |
|---|---|---|
| `evaluator1d.py` (P1 metrics) | none | unchanged |
| `evaluator2d.py` (69 lines) | `e7e4ae9` (Phase 7, P2): created; `3d0100c` (Phase 9, P4): +1 line | new module for 2D; then one additive hook (below) |
| `selftests2d.py` | `e7e4ae9` created; `8561d55` (P3): 5+/3−; `c220ffc` (P5): 15+/7− | test-harness generalisation (below) |

**Changes to the 2D evaluator after it was created:**
- The P4 corner metrics enter through one added line: `if hasattr(problem, "corner_metrics"): extra.update(problem.corner_metrics(pred))`. The metrics themselves live in the problem module.
- The P3 line-charge problem, the P5 thin film and the P5 FEM-based reference needed **no change** to `evaluator2d.py`.

**Physical-unit conversion** (non-dimensional training variables back to physical fields) is a problem method, `problem.to_physical`, called from the training code. It is not an evaluator change.

**Self-test harness changes.** These are tests, not metric definitions:
- **P3:** the refined-quadrature resolution became a problem attribute. The non-conforming-FEM convergence check became informational for problems with an interface charge load, because those have no interface edges to conform to.
- **P5:** tolerances became aware of the reference type. Analytic references are held to round-off, 10⁻¹²; FEM references, which carry their own round-off and element kinks, are held to 10⁻⁷ and 10⁻⁴, both far below the 5% failure threshold. The interface-jump test now computes its expected value from the reference instead of assuming an exactly continuous one.

**No metric definition was rewritten.** The primary band field error, strict error, global/material/band L2 errors, jump metrics, energy and capacitance errors are computed by the same code for P2–P5. P1 is unchanged.

## Verdict: **supported, with one qualification**

The literal criterion is met. No core function was rewritten after the P1 freeze, and the only post-creation change to the 2D evaluator is one additive hook. P3, P4 and P5 each needed only a problem module, plus that hook for P4 and test-harness generalisations.

**Qualification:** the step from 1D to 2D needed a new evaluator module. `evaluator1d.py` is written for a 1D grid, so `evaluator2d.py` re-implements the same metric definitions on 2D quadrature. It was validated by the same Gate 1-style self-tests: 17/17 per problem, covering exact reference, perturbations, interface jumps, FEM convergence, and refined-quadrature agreement. The claim "new problems need only a problem module" therefore holds *within a dimension*, not across the 1D → 2D step. A dimension-generic evaluator from the start would have avoided this.

## What this does and does not establish

- **Establishes:** the metric layer is stable under four different 2D geometries: a curved interface, a point singularity, a corner singularity, and a thin film with a FEM reference. Results across P2–P5 are computed by identical metric code.
- **Does not establish:** reusability for 3D problems, for time-dependent or nonlinear problems, or for problems whose reference has no computable band. These were not tested.
