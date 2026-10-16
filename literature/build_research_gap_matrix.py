"""Build literature/research_gap_matrix.csv (Phase 1 deliverable C).

category codes (from the project brief):
 1 = already investigated extensively
 2 = studied in other physical settings, not established for the dielectric problem
 3 = existing evidence incomplete
 4 = potential research gap supported by the reviewed literature
 5 = gap whose novelty remains uncertain (search not exhaustive)
novelty_confidence describes how confident we are that the item is NOT already
published: low / medium / high. It is never 'high' for anything we searched only briefly.
"""
import csv, pathlib
COLS = ["gap_id","topic","category","category_label","evidence_lit_ids","what_is_known","what_is_missing",
        "novelty_confidence","linked_rq","planned_test","claim_we_may_make"]
LBL = {1:"investigated extensively",2:"studied elsewhere, not for dielectric problem",3:"evidence incomplete",
       4:"potential gap supported by literature",5:"novelty uncertain"}
rows = [
 ("G01","PINN-type solvers for discontinuous-coefficient elliptic interface problems (in general)",1,
  "L02;L03;L08;L09;L10;L11;L12;L13;L14;L15;L25","Many architectures (piecewise/multi-network, cusp/discontinuity capturing, hard constraints, energy/Nitsche) with accurate results on manufactured problems","Nothing at the level of 'does it work at all'","n/a (not a gap)","RQ1","Baseline context only","None - must not be claimed"),
 ("G02","High contrast (>=1e3) for interface-aware networks",1,
  "L09;L10;L11;L12;L13;L22","Contrasts 1e3 (L10,L12,L13,L22), 1e4 (L09), 1e5 (L11) solved accurately by interface-aware networks","Systematic sweeps; seed statistics; field errors (see G05-G07)","n/a (not a gap)","RQ1","Contrast grid spans these values so results are comparable","None. Several supplied reviews state the opposite; corrected in review audit"),
 ("G03","Silent failure of a single network with piecewise coefficient (flux condition invisible to strong-form residual)",3,
  "L02;L01","Mechanism shown qualitatively in 1D by Nohra & Dufour Sec. 6 (constant solution satisfies residual); PINNacle shows piecewise-coefficient Poisson unsolved","Quantification versus contrast and seeds; loss-visibility test comparing exact vs wrong solutions; effect of coefficient smoothing width","low","RQ3;RQ4","F1 failure map; loss-visibility diagnostic (Sec. 13B)","Quantitative characterisation of a known mechanism - NOT discovery of the mechanism"),
 ("G04","Electrostatic / dielectric-specific benchmarks (dielectric cylinder in uniform field, line charge near planar interface, dielectric wedge, thin high-eps film) solved with PINNs and scored on physical quantities",2,
  "L02;L19;L20;L07","Generic elliptic circles/spheres solved (L09-L13); magnetostatic/resistive interfaces with contrast <=3 (L02); electrostatic PINNs with smoothed eps (L19)","Permittivity-contrast sweeps on canonical electrostatic geometries with closed-form references and physical metrics (capacitance, effective contrast, interior field)","medium","RQ8","P1-P5 ladder","'We did not find' wording only; mathematically these are elliptic interface problems already studied in generic form"),
 ("G05","Interface-band field (E) and displacement (D) error versus global potential error",4,
  "L21;L05;L13;L12","Classical numerics: gradients ~1 order less accurate at interfaces (L21); NN papers report global H1/H2 (L05,L13) or show interface-concentrated error visually (L12)","An explicit, thresholded interface-band E/D metric reported alongside phi error across contrast and formulations","medium","RQ2","Evaluator metrics + H1 test","Measured gap between metrics; novelty limited to the systematic reporting"),
 ("G06","Failure-rate statistics across seeds for interface PINN formulations",4,
  "L08;L13;L01;L17","Single runs or best-of-initialisations (L08), 5 runs at one contrast (L13), 3 repeats general benchmark (L01), >=10 seeds for non-interface coefficients (L17)","Failure rate per (formulation, contrast) with a pre-registered threshold","medium","RQ1;RQ6","Primary sweep, 5 seeds minimum","Failure maps with uncertainty; must state 5 seeds give coarse estimates"),
 ("G07","Fixed-budget comparison of several interface formulation families across a contrast sweep",4,
  "L08;L01;L03","L08 compares 6 methods at one contrast (10), best initialisation; L01 fixed-budget but no interface families","Same budget, same seeds, multiple contrasts, multiple families","medium","RQ1;RQ7","F1-F6 x 6 contrasts x 5 seeds","Comparison valid only for the tested budgets/architectures"),
 ("G08","Per-loss-term gradient statistics as predictors of interface failure versus contrast",2,
  "L18;L06;L02","Gradient pathologies documented generally (L18); loss-scale imbalance in layered heat transfer (L06); NTK eigen-imbalance for smoothed interface (L02)","Whether gradient ratios predict field failure across contrast for dielectric interfaces","low","RQ4;RQ5","Gradient logging + correlation analysis with uncertainty","Correlational evidence only unless interventions confirm"),
 ("G09","Matched-seed controlled interventions to separate representation, loss design, optimisation and precision",3,
  "L17;L08;L02;L18","Individual remedies studied separately (curriculum L17; hard vs soft L08; formulation order L02; rebalancing L18)","A combined, pre-registered intervention design on the same interface problem","medium","RQ4;RQ5","Interventions A-E","Mechanism evidence for these problems only"),
 ("G10","Effect of float32 vs float64 at high permittivity contrast",2,
  "L31;L32","Precision drives PINN failure modes under L-BFGS (L31); effect small under Adam, large under L-BFGS (L32)","Whether contrast-induced interface failure is precision-limited","medium","RQ4","Intervention C crossed with optimiser (2x2)","Interface-specific precision evidence only"),
 ("G11","PINN vs FEM cost-at-equal-accuracy for interface problems with field-accuracy targets",4,
  "L16;L01;L03","PINN vs FEM for smooth problems (L16); runtime-only reporting (L01); relative cost among PINNs (L03)","Error-at-equal-cost and cost-to-target (5%,1% interface-band field error) vs interface-conforming FEM","medium","RQ7","Remedy comparison (Sec. 20)","Scoped to tested problems and hardware"),
 ("G12","Physics-derived (contrast-scaled) closed-form loss weights for interface penalties",4,
  "L06;L03;L08","Hand-tuned weights (L03); scale-derived balancing in heat transfer (L06); hard constraints avoid weights (L08)","Whether a pre-specified kappa-derived weight rule works for dielectric contrasts; Phase 2 search found empirical/adaptive weights and harmonic weighting in classical Nitsche/DG only","low-medium (one targeted search)","RQ5","F6, formula frozen before sweep","Only as an evaluated design, not as a novel method, until searched"),
 ("G13","Energy/Ritz formulation with interface feature",1,
  "L11;L30","Energy and Nitsche formulations reach contrasts 1e5 (L11)","Field-level/seed-level evaluation under our protocol","n/a","RQ1","F5 as a baseline","None"),
 ("G14","Singular sources (line charge) near a dielectric interface for PINNs",2,
  "L34;L35","Singularity enrichment / singular-part subtraction for point sources and corners in single-material problems (L34,L35)","Image-charge-referenced 2D benchmark with a dielectric interface and regional error reporting","medium","RQ8","P3","Pending search"),
 ("G15","Dielectric wedge / triple-junction local field behaviour for PINNs",2,
  "L08;L34;L36","Corner singularities (L34), multi-material corners with variational PINN (L36), corner stiffness of hard ansatz (L08)","Fixed-distance or asymptotic-coefficient field metrics versus an exact singular reference across contrast","medium","RQ8","P4","Pending search"),
 ("G16","Thin high-permittivity film (small length scale) for PINNs",2,"L33;L37","Thin insulating layers with jump conditions (L33); high-contrast inner layers (L37)","Resolved (non-jump) thin high-eps film with thickness/sampling sweep and field metrics","medium","RQ8","P5","Pending search"),
 ("G17","Reusable interface-aware evaluator for PINN outputs",4,
  "L01","General PINN benchmark toolbox (L01) without interface metrics","Problem-agnostic evaluator for interface-band/material/jump/capacitance metrics with self-tests","medium","RQ9","Evaluator + reuse on P2-P5","Software contribution; value depends on actual reuse"),
]
out = pathlib.Path(__file__).with_name("research_gap_matrix.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(COLS)
    for r in rows:
        gid,topic,cat,ev,known,missing,nov,rq,test,claim = r
        w.writerow([gid,topic,cat,LBL[cat],ev,known,missing,nov,rq,test,claim])
print(f"wrote {len(rows)} rows to {out}")
