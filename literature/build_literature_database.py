"""Build literature/literature_database.csv from the structured records below.

Phase 1 deliverable A. Each record distinguishes:
  *_author   : what the original authors state
  *_team     : what our team infers (marked as inference)
verification_level values:
  full_text_read      - the primary source text was read by the authors
  partial_text_read   - abstract plus substantial passages read
  abstract_snippets   - abstract/snippets only; details may be incomplete
  supplied_review_only- only the supplied team review was available
  cited_only          - bibliographic details taken from another paper's reference list
Run:  python literature/build_literature_database.py
"""
import csv, pathlib

COLS = ["id","title","authors","year","doi_or_url","source","supplied_review_files",
        "verification_level","governing_equation","physical_application","dimensionality",
        "interface_geometry","material_contrast","pinn_architecture","interface_treatment",
        "optimisation","seeds_reported","evaluation_metrics","reference_solution",
        "fem_or_classical_comparison","cost_reporting","main_findings_author",
        "limitations_author","limitations_team_inferred","relevance_to_project",
        "review_audit_notes"]

R = []
def add(**k):
    """Append one literature record, filling every missing column with 'not reported'."""
    for c in COLS: k.setdefault(c, "not reported")
    R.append(k)

# ---------------- Supplied papers ----------------
add(id="L01", title="PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs",
    authors="Z. Hao, J. Yao, C. Su, H. Su, Z. Wang, F. Lu, Z. Xia, Y. Zhang, S. Liu, L. Lu, J. Zhu", year="2023",
    doi_or_url="arXiv:2306.08827v2", source="supplied (PDF, no team review)", supplied_review_files="2306_08827_compressed__1___1_.pdf",
    verification_level="full_text_read (pdftotext of all 50 pages; relevant sections read closely)",
    governing_equation="20+ PDEs. Relevant: Poisson3d-CG -mu_i Lap u + k_i^2 u = f (two domains); Poisson2d-MS -div(a grad u)=f with piecewise-linear a on many squares; Heat2d-VC with GRF coefficient; Wave2d-CG heterogeneous medium",
    physical_application="General benchmark (heat, fluids, EM, geophysics)", dimensionality="1D-5D+ incl. time",
    interface_geometry="Poisson3d-CG: planar split z=0.5 with spherical holes; Poisson2d-MS: square subdomains",
    material_contrast="Poisson3d-CG: mu1=mu2=1 (diffusion NOT discontinuous), k1=8,k2=10; Poisson2d-MS contrast values stored in external file (not in text)",
    pinn_architecture="MLP 5 layers x 100; ~10 variants (PINN-w, LBFGS, LRA, NTK, RAR, MultiAdam, gPINN, hp-VPINN, LAAF, GAAF, FBPINN)",
    interface_treatment="None specific to interfaces", optimisation="Adam lr 1e-3, 20,000 epochs; 8192 collocation / 2048 boundary points (2D)",
    seeds_reported="3 repeats, mean and std", evaluation_metrics="L2RE, L1RE, MSE, max error, Fourier-band errors, runtime",
    reference_solution="Numerical solvers (FEM etc.)", fem_or_classical_comparison="FEM used only as reference, not as a cost competitor",
    cost_reporting="Runtime per method (20k epochs, 3-run average)",
    main_findings_author="Using 10% L2RE as success threshold, vanilla PINN solves 10/22 tasks; complex geometry and multi-scale problems hardest; Poisson2d-MS unsolved by all methods (best ~59% L2RE)",
    limitations_author="Benchmark breadth over depth; vanilla MLP backbone",
    limitations_team_inferred="No interface-local metrics, no field/flux errors, no contrast sweep, 3 repeats only; the Poisson3d-CG 'two-domain' case has no diffusion-coefficient jump",
    relevance_to_project="Precedent for fixed-budget multi-method benchmarking and a fixed success threshold; evidence piecewise-coefficient Poisson is hard for PINNs",
    review_audit_notes="No supplied review. Team observation: Poisson3d-CG parameters (mu1=mu2=1) mean it is not a coefficient-contrast interface test.")

add(id="L02", title="Physics-Informed Neural Networks for the Numerical Modeling of Steady-State and Transient Electromagnetic Problems with Discontinuous Media",
    authors="M. Nohra, S. Dufour", year="2024", doi_or_url="arXiv:2406.04380v1",
    source="supplied (team reviews)", supplied_review_files="P003_Tolga_and_Yasir_docx.pdf; P003_Tolga_and_Yasir_docx_1.pdf (identical); P006_YASIR.pdf (same paper)",
    verification_level="full_text_read (primary arXiv text fetched)",
    governing_equation="First-order low-frequency Maxwell (steady and quasi-static transient); electrostatic div(eps grad V)=rho stated but not tested",
    physical_application="Magnetostatics / conduction with material discontinuities", dimensionality="1D NTK study; 2D disc; 3D sphere, concentric cylinders; transient",
    interface_geometry="Circle/sphere/cylinder; single interface",
    material_contrast="Permeability 3:1 (2D disc); mu_out 0.5-1.5 vs 1 (3D); resistivity 0.5-1.5 vs 1. NO permittivity contrast tested",
    pinn_architecture="Common block feeding two branches f1 + H_hat*f2, strong BC via d0(x) and lift V(x)",
    interface_treatment="Material smoothed with sigmoid Heaviside S(alpha*phi), alpha=100-500; no interface loss terms",
    optimisation="Adam (3000-7000 it) then L-BFGS (3500-4000) in some cases", seeds_reported="not reported",
    evaluation_metrics="Max relative error only for NON-interface cases (0.4% Ampere continuous media; 2e-3 transient manufactured); interface cases compared to FEM visually",
    reference_solution="Analytical (no-interface) and FEM (interface cases)", fem_or_classical_comparison="Qualitative/visual for interface cases",
    cost_reporting="not reported",
    main_findings_author="NTK analysis: first-order formulation less stiff than second order for smoothed interface; smoothed-Heaviside architecture removes interface terms; Sec. 6 shows that with piecewise-constant mu a single network can satisfy the residual at every collocation point with the wrong (constant) solution",
    limitations_author="Network size grows with number of interfaces; sharper interfaces increase difficulty (frequency bias)",
    limitations_team_inferred="Contrast at most 3; no seeds; no quantitative interface error; smoothing width is a modelling error not separated from training error",
    relevance_to_project="Direct evidence for the F1 silent-failure mechanism (prior art, not our novelty); source for a smoothed-coefficient variant and NTK framing",
    review_audit_notes="CORRECTIONS to supplied reviews: (1) the 0.2-0.4%/2e-3 errors are for problems WITHOUT an interface; (2) no permittivity interface was tested; (3) P003 and P006 give different reproducibility scores (2 vs 3) - a reviewer judgement, unresolved; (4) neither review mentions the Sec. 6 silent-failure example.")

add(id="L03", title="Adaptive Interface-PINNs (AdaI-PINNs): An Efficient Physics-informed Neural Networks Framework for Interface Problems",
    authors="S. Roy, C. Annavarapu, P. Roy, A. K. Sarma", year="2024", doi_or_url="arXiv:2406.04626v2",
    source="supplied (team review)", supplied_review_files="P009_Tolga_1.pdf",
    verification_level="supplied_review_only (existence and baseline role corroborated by Chung et al. 2026 citing it)",
    governing_equation="div(kappa_m grad u_m) = -f_m; 1D, 2D, 3D Poisson with jumps", physical_application="Generic heterogeneous diffusion (manufactured solutions)",
    dimensionality="1D (5 layers), 2D (letter-shaped inclusions), 3D (8 spheres)", interface_geometry="Planar (1D), curved closed curves, spheres",
    material_contrast="~10 (1D), ~3.5 (2D), ~9 (3D) per review", pinn_architecture="Shared weights, per-subdomain trainable activation slope a_m",
    interface_treatment="Soft penalties on jump in u and in flux", optimisation="Adam lr 5e-3; 20k-60k iterations (JAX)",
    seeds_reported="not reported (per review)", evaluation_metrics="RMSE vs analytical; relative training cost",
    reference_solution="Analytical", fem_or_classical_comparison="none (per review)", cost_reporting="Relative cost vs I-PINN; no hardware",
    main_findings_author="Faster convergence and lower RMSE than I-PINNs at equal iterations",
    limitations_author="Future work: transient, coupled, moving interfaces (per review)",
    limitations_team_inferred="Moderate contrasts; no seeds; hand-tuned weights. Independent evidence: Chung et al. (2026) report AdaI-PINN >80% error on their 2D slanted-interface problem",
    relevance_to_project="Candidate F2-family baseline idea (adaptive activation); weight-sensitivity evidence",
    review_audit_notes="Original not read by the authors; numbers are as reported in P009 and must be checked before being cited in the paper.")

add(id="L04", title="Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear PDEs",
    authors="A. D. Jagtap, G. E. Karniadakis", year="2020 (journal); 2021 workshop version",
    doi_or_url="Commun. Comput. Phys. 28(5) (2020); workshop version CEUR-WS Vol-2964 article_60",
    source="supplied (team review)", supplied_review_files="P008_Tolga_1.pdf", verification_level="supplied_review_only; journal details from Chung et al. reference list",
    governing_equation="Burgers u_t + u u_x = nu u_xx (test case)", physical_application="Nonlinear convection-diffusion", dimensionality="1D space-time",
    interface_geometry="One irregular space-time interface", material_contrast="none (no coefficient jump)",
    pinn_architecture="Separate sub-networks per subdomain (different depth/width/activation)", interface_treatment="Average-solution continuity + residual continuity penalties",
    optimisation="Adam lr 8e-4, 100k iterations", seeds_reported="not reported", evaluation_metrics="Relative L2 (global and on interface)",
    reference_solution="Reference solution", fem_or_classical_comparison="none", cost_reporting="none quantitative",
    main_findings_author="Arbitrary space-time decomposition works; enables parallelisation",
    limitations_author="Strongly non-convex loss; delayed convergence of subdomains without boundary data",
    limitations_team_inferred="No flux-continuity condition in the original XPINN interface loss; not tested on coefficient jumps",
    relevance_to_project="Background for F2/F6 two-network design; our F2 must use a flux condition instead of residual continuity",
    review_audit_notes="P008 cites the CEUR workshop version; the archival version is CiCP 2020. The code URL in P008 (github.com/Ameya.Jagtap/XPINNs) looks malformed - verify before citing.")

add(id="L05", title="Extended Interface Physics-Informed Neural Networks Method for Moving Interface Problems",
    authors="R. Bi, W. Deng, Y. Zhu", year="2025 (arXiv first submission 2508); review lists 2026", doi_or_url="arXiv:2508.01463",
    source="supplied (team review)", supplied_review_files="P004_Tolga.pdf", verification_level="abstract_snippets (primary abstract fetched) + supplied review",
    governing_equation="Parabolic u_t - div(beta grad u) = f with jumps; Oseen with moving interface",
    physical_application="Moving-interface diffusion/flow", dimensionality="2D and 3D + time", interface_geometry="Moving ellipsoids etc.",
    material_contrast="beta 10/0.1 (=100); Oseen viscosity 1e-3 vs 1 (fluid, not diffusion)",
    pinn_architecture="Single network with extended-variable technique + level-set (learned or given)", interface_treatment="Penalties on solution and flux jumps",
    optimisation="Levenberg-Marquardt, 2000 epochs", seeds_reported="not reported", evaluation_metrics="Relative L2(0,T;L2) and L2(0,T;H1)",
    reference_solution="Analytical", fem_or_classical_comparison="none", cost_reporting="Training time on V100 (59-2080 s)",
    main_findings_author="XI-PINN beats X-PINN for continuous solutions / small networks",
    limitations_author="Prescribed interface velocity only",
    limitations_team_inferred="Reports H1 (gradient) error globally but not interface-local field error",
    relevance_to_project="Precedent for reporting gradient-type error; single-network extended-variable idea relates to F3",
    review_audit_notes="Review gives year 2026; arXiv id indicates Aug 2025 first submission. The '1:1000' contrast is a fluid viscosity ratio, not a diffusion/permittivity contrast.")

add(id="L06", title="Parallel Physics-Informed Neural Networks with Bidirectional Balance",
    authors="Y. Huang", year="2021", doi_or_url="arXiv:2111.05641", source="supplied (team review, scanned)", supplied_review_files="P007_YASIR.pdf (OCR)",
    verification_level="abstract_snippets (primary abstract fetched) + supplied review",
    governing_equation="1D transient heat conduction in 3 fabric layers", physical_application="Thermal protective clothing", dimensionality="1D + time",
    interface_geometry="2 planar interfaces", material_contrast="Conductivity 0.045-0.37 W/(m K) (~8x)",
    pinn_architecture="Parallel MLPs per layer (4x10, tanh)", interface_treatment="Penalties on temperature and heat-flux continuity",
    optimisation="Adam lr 1e-3, 20k epochs", seeds_reported="not reported", evaluation_metrics="MSE vs FDM",
    reference_solution="Finite differences", fem_or_classical_comparison="FDM as reference only", cost_reporting="Hardware listed (RTX 2080Ti)",
    main_findings_author="Forward (unit) balance + backward (loss-scale) balance both needed for convergence",
    limitations_author="Only fully connected nets; 1D only",
    limitations_team_inferred="Loss-scale coefficients found by line search; no seed statistics",
    relevance_to_project="Precedent for scale-derived loss weights (relevant to F6 and Intervention A/B)",
    review_audit_notes="OCR of scanned review; equations partly garbled; check originals before quoting numbers.")

add(id="L07", title="Physics-Informed Neural Networks for the Time-Domain Maxwell Equations with Split-Field Perfectly Matched Layers",
    authors="X. Liu, L. Li, G. Moss, S. Langdon", year="2026", doi_or_url="arXiv:2606.25323 (submitted to IEEE AP-S 2026)",
    source="supplied (team review)", supplied_review_files="P001_Daksh.pdf", verification_level="abstract_snippets (primary abstract fetched) + supplied review",
    governing_equation="1D/2D TMz split-field PML Maxwell", physical_application="Open-domain time-domain wave propagation", dimensionality="1D, 2D + time",
    interface_geometry="Planar material interface; PML boundaries", material_contrast="eps 1:4",
    pinn_architecture="Vanilla MLP (4 layers, 40-80 wide)", interface_treatment="None (same equations across regions)",
    optimisation="Adam then L-BFGS (DeepXDE)", seeds_reported="not reported", evaluation_metrics="Qualitative comparison only",
    reference_solution="Analytical and FDTD", fem_or_classical_comparison="FDTD (qualitative)", cost_reporting="not reported",
    main_findings_author="Split-field PML integrates naturally into PINN loss",
    limitations_author="Future: higher dimensions, multiple networks",
    limitations_team_inferred="No quantitative errors; low contrast; wave rather than electrostatic problem",
    relevance_to_project="Low; context for PINNs in computational EM only",
    review_audit_notes="P001 marks 'Direct comparison: Yes (parallel plates)'; team disagrees - time-domain wave propagation is not an electrostatic layered-capacitor problem.")

# ---------------- Additional literature ----------------
add(id="L08", title="Hard-constrained Physics-informed Neural Networks for Interface Problems",
    authors="S. W. Chung, S. T. Castonguay, S. Roy, M. S. Penwarden, Y. Fu, P. Roy", year="2026", doi_or_url="arXiv:2604.08453 (v2, 15 May 2026)",
    source="additional (search)", supplied_review_files="", verification_level="full_text_read",
    governing_equation="-div(kappa grad u) = f, piecewise-constant kappa", physical_application="Generic diffusion interface benchmarks",
    dimensionality="1D (1 and 3 interfaces), 2D slanted interface", interface_geometry="Planar / slanted line",
    material_contrast="kappa 0.1 vs 1 (=10) in all reported cases",
    pinn_architecture="Windowing ansatz (polynomial windows) and buffer ansatz (additive polynomial/RBF correction); baselines phi-PINN, I-PINN, AdaI-PINN, M-PINN",
    interface_treatment="Hard: continuity and flux balance embedded in ansatz (buffer exact in 1D, discrete in 2D)",
    optimisation="SOAP optimiser lr 5e-3, 1e4-3e4 iterations; 20-60 collocation points (1D)",
    seeds_reported="Single run per method; best of 4 initialisations reported (ablation in App. C)",
    evaluation_metrics="Relative L2 of u (and of LHS[u])", reference_solution="Analytical (1D), MFEM fine mesh (2D)",
    fem_or_classical_comparison="FEM as reference only", cost_reporting="not reported",
    main_findings_author="Hard constraints improve 1D accuracy by orders of magnitude; buffer approach more robust in 2D; windowing limited by window-derivative shape",
    limitations_author="Expect larger advantage at much larger contrasts but not tested; windowing struggles at corners",
    limitations_team_inferred="Contrast fixed at 10; best-initialisation reporting inflates method performance; no field/flux error, no seeds, no cost vs FEM",
    relevance_to_project="Closest prior work and the F4 source. Our contrast sweep, seed statistics and field metrics address exactly what they leave open",
    review_audit_notes="This is the 'Chung et al. (2026)' named in the supplied protocol for F4.")

add(id="L09", title="A cusp-capturing PINN for elliptic interface problems",
    authors="Y.-H. Tseng, T.-S. Lin, W.-F. Hu, M.-C. Lai", year="2023", doi_or_url="J. Comput. Phys. 491 (2023) 112359; arXiv:2210.08424",
    source="additional (search)", supplied_review_files="", verification_level="abstract_snippets",
    governing_equation="Variable-coefficient elliptic interface problem, continuous solution with derivative jump",
    physical_application="Generic elliptic", dimensionality="2D, 3D, irregular domains", interface_geometry="Curved (level set)",
    material_contrast="Includes high-contrast ratios eta = 1e-4 and 1e4",
    pinn_architecture="Shallow network (1 hidden layer, 40-60 neurons) with |phi| cusp feature input",
    interface_treatment="Cusp-enforced level-set feature; flux jump enforced as penalty",
    optimisation="Levenberg-Marquardt (to verify)", seeds_reported="to verify", evaluation_metrics="Relative L2 (abstract: 1e-5 to 1e-6)",
    reference_solution="Analytical", fem_or_classical_comparison="Compared to grid-based methods from literature",
    cost_reporting="to verify",
    main_findings_author="Cusp feature yields high accuracy, outperforming several NN and grid methods",
    limitations_author="Designed for continuous solutions with derivative jumps",
    limitations_team_inferred="Contrast is not swept systematically; field-error reporting to verify",
    relevance_to_project="F3 source; shows contrast 1e4 is attainable for an interface-aware network",
    review_audit_notes="Contradicts supplied-review claims that contrasts >1:1000 are untested for PINN-type methods.")

add(id="L10", title="A discontinuity capturing shallow neural network for elliptic interface problems",
    authors="W.-F. Hu, T.-S. Lin, M.-C. Lai", year="2022", doi_or_url="J. Comput. Phys. 469 (2022) 111576; arXiv:2106.05587",
    source="additional (search)", supplied_review_files="", verification_level="abstract_snippets",
    governing_equation="Elliptic interface problems with jumps", physical_application="Generic", dimensionality="2D, 3D, 6D",
    interface_geometry="Curved", material_contrast="Default 1e3 (beta-/beta+ = 1e3) in nearly all examples",
    pinn_architecture="Shallow network with augmented categorical input", interface_treatment="Penalties (alpha=1)",
    optimisation="to verify", seeds_reported="to verify", evaluation_metrics="Relative errors", reference_solution="Analytical",
    fem_or_classical_comparison="to verify", cost_reporting="to verify",
    main_findings_author="Sharp jump capture with a single hidden layer, up to 6D",
    limitations_author="to verify", limitations_team_inferred="Contrast fixed, not swept",
    relevance_to_project="Shows high-contrast accuracy is achievable; template for F3 family", review_audit_notes="")

add(id="L11", title="Deep unfitted Nitsche method for elliptic interface problems",
    authors="H. Guo, X. Yang", year="2022 (journal; arXiv 2021)", doi_or_url="arXiv:2107.05325; Commun. Comput. Phys. (volume to verify)",
    source="additional (search)", supplied_review_files="", verification_level="abstract_snippets",
    governing_equation="Elliptic interface problem in energy (Nitsche) form", physical_application="Generic", dimensionality="2D up to 20D",
    interface_geometry="Circular / high-dimensional spheres", material_contrast="1000/1, 1/1000, 1e5/1, 1/1e5",
    pinn_architecture="Two networks, energy functional with Nitsche terms", interface_treatment="Weak (Nitsche) interface coupling",
    optimisation="to verify", seeds_reported="to verify", evaluation_metrics="Relative L2", reference_solution="Analytical",
    fem_or_classical_comparison="to verify", cost_reporting="to verify",
    main_findings_author="Energy-based unfitted method handles high and huge contrasts",
    limitations_author="to verify", limitations_team_inferred="Field-level and interface-local errors not reported (to verify)",
    relevance_to_project="F5 (energy/Ritz) family precedent at contrasts up to 1e5", review_audit_notes="")

add(id="L12", title="A Mesh-free Method Using Piecewise Deep Neural Network for Elliptic Interface Problems",
    authors="C. He, X. Hu, L. Mu", year="2022 (journal; arXiv 2020)", doi_or_url="arXiv:2005.04847 (journal to verify)",
    source="additional (search)", supplied_review_files="", verification_level="abstract_snippets",
    governing_equation="Elliptic interface", physical_application="Generic", dimensionality="2D", interface_geometry="Circle",
    material_contrast="beta 1000 vs 1 and 1 vs 1000", pinn_architecture="Piecewise DNN (one per subdomain)",
    interface_treatment="Penalties", optimisation="to verify", seeds_reported="to verify", evaluation_metrics="Error profiles",
    reference_solution="Analytical", fem_or_classical_comparison="to verify", cost_reporting="to verify",
    main_findings_author="Errors dominate near the interface in the high-contrast circle case",
    limitations_author="to verify", limitations_team_inferred="Interface-local error observed visually, not quantified as a metric",
    relevance_to_project="Qualitative precedent for interface-concentrated error (RQ2)", review_audit_notes="")

add(id="L13", title="Convergence of physics-informed neural networks applied to linear second-order elliptic interface problems",
    authors="authors to verify (arXiv:2203.03407)", year="2022/2023", doi_or_url="arXiv:2203.03407",
    source="additional (search)", supplied_review_files="", verification_level="abstract_snippets",
    governing_equation="Linear second-order elliptic interface", physical_application="Generic", dimensionality="2D",
    interface_geometry="Circle r0=0.5", material_contrast="1000 vs 1",
    pinn_architecture="Domain-decomposition PINN with Lipschitz regularisation and gradient-enhanced terms (LIPR-GE)",
    interface_treatment="Penalties + gradient enhancement", optimisation="to verify",
    seeds_reported="Five runs (different training data and initialisation), one-std bands",
    evaluation_metrics="L2 and H2 errors; pointwise error", reference_solution="Analytical",
    fem_or_classical_comparison="none", cost_reporting="to verify",
    main_findings_author="Convergence theory; gradient-enhanced terms reduce pointwise error at interface and boundary",
    limitations_author="to verify", limitations_team_inferred="Single contrast; no failure-rate statistic",
    relevance_to_project="Precedent for multi-run statistics and higher-order (derivative) error norms", review_audit_notes="Author list must be verified before citation.")

add(id="L14", title="INN: Interfaced neural networks as an accessible meshless approach for solving interface PDE problems",
    authors="S. Wu, B. Lu", year="2022", doi_or_url="J. Comput. Phys. 470 (2022) 111588", source="additional (cited in L02/L09)",
    verification_level="cited_only", relevance_to_project="Multi-network interface method; high-contrast/high-dimensional reference", review_audit_notes="Details not inspected.")

add(id="L15", title="Interface PINNs (I-PINNs): A physics-informed neural networks framework for interface problems",
    authors="A. K. Sarma, S. Roy, C. Annavarapu, P. Roy, S. Jagannathan", year="2024", doi_or_url="Comput. Methods Appl. Mech. Engrg. 429 (2024) 117135",
    source="additional (cited in L08)", verification_level="cited_only", relevance_to_project="Baseline family for F2", review_audit_notes="Details not inspected.")

add(id="L16", title="Can physics-informed neural networks beat the finite element method?",
    authors="T. G. Grossmann, U. J. Komorowska, J. Latz, C.-B. Schoenlieb", year="2024", doi_or_url="doi:10.1093/imamat/hxae011; arXiv:2302.04107",
    source="additional (search)", verification_level="abstract_snippets",
    governing_equation="Poisson 1D/2D/3D, Allen-Cahn, semilinear Schroedinger", physical_application="Generic",
    material_contrast="none (no interfaces)", fem_or_classical_comparison="Systematic PINN vs FEM", cost_reporting="Solution time and accuracy",
    main_findings_author="PINNs did not outperform FEM in solution time and accuracy; sometimes faster to evaluate",
    limitations_team_inferred="No discontinuous-coefficient or interface problems",
    relevance_to_project="Methodological precedent for RQ7; we extend the comparison to interface problems and field metrics", review_audit_notes="")

add(id="L17", title="Characterizing possible failure modes in physics-informed neural networks",
    authors="A. S. Krishnapriyan, A. Gholami, S. Zhe, R. M. Kirby, M. W. Mahoney", year="2021", doi_or_url="NeurIPS 2021",
    source="additional (search)", verification_level="abstract_snippets", seeds_reported="At least 10 seeds per case",
    main_findings_author="PINN failures driven by optimisation difficulty as PDE coefficients grow; curriculum and seq2seq reduce error",
    limitations_team_inferred="Convection/reaction/diffusion coefficients, not interface contrasts",
    relevance_to_project="Methodological template for coefficient sweeps with seeds and for distinguishing optimisation from representation failure (RQ4)", review_audit_notes="")

add(id="L18", title="Understanding and mitigating gradient flow pathologies in physics-informed neural networks",
    authors="S. Wang, Y. Teng, P. Perdikaris", year="2021", doi_or_url="SIAM J. Sci. Comput. 43(5) (2021) A3055-A3081",
    source="additional (cited in L08)", verification_level="cited_only",
    relevance_to_project="Basis for per-term gradient logging and Intervention A (gradient rebalancing)", review_audit_notes="Details to inspect in Phase 2.")

add(id="L19", title="Physics informed neural networks for electric field distribution characteristics analysis",
    authors="Zhang et al. (full list to verify)", year="2023", doi_or_url="J. Phys. D: Appl. Phys. 56 (2023) 165202 (ADS 2023JPhD...56p5202Z)",
    source="additional (search)", verification_level="abstract_snippets",
    governing_equation="Electrostatics in 2D/3D", physical_application="Electric field distribution with dielectrics", interface_treatment="Sigmoid-smoothed permittivity",
    fem_or_classical_comparison="Compared with FEM", relevance_to_project="Electrostatic PINN with dielectric interface; contrast and metrics to verify", review_audit_notes="")

add(id="L20", title="Physics-Informed Neural Networks for 2D Plane Wave Scattering in Arbitrary Dielectric Structures",
    authors="to verify", year="2026", doi_or_url="arXiv:2607.27349", source="additional (search)", verification_level="abstract_snippets",
    physical_application="Frequency-domain EM scattering", interface_treatment="tanh smoothing of 1/eps for TE polarisation",
    main_findings_author="Discontinuity of 1/eps_r induces singular behaviour near interfaces in TE case; smoothing improves agreement with FDTD",
    relevance_to_project="Recent EM evidence that permittivity discontinuities degrade PINNs near interfaces", review_audit_notes="")

add(id="L21", title="Accurate gradient computations at interfaces using finite element methods",
    authors="to verify (arXiv:1703.00093)", year="2017", doi_or_url="arXiv:1703.00093", source="additional (search)", verification_level="abstract_snippets",
    main_findings_author="Structured-mesh interface methods are typically one order less accurate in the gradient than in the solution",
    relevance_to_project="Classical evidence that potential accuracy overstates field accuracy at interfaces (RQ2 prior)", review_audit_notes="")

add(id="L22", title="Dirichlet-Neumann learning algorithm for solving elliptic interface problems",
    authors="to verify (arXiv:2301.07361)", year="2023", doi_or_url="arXiv:2301.07361", source="additional (search)", verification_level="abstract_snippets",
    material_contrast="(1, 1e3) and (1, 1)", main_findings_author="DeepDDM solves the 1e3 case but fails when c1=c2 because of erroneous interface flux",
    relevance_to_project="A failure that occurs at LOW contrast: failure need not be monotone in contrast; our analysis must not assume monotonicity", review_audit_notes="")

for i,(t,a,y,d,rel) in enumerate([
    ("Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks","N. Sukumar, A. Srivastava","2022","Comput. Methods Appl. Mech. Engrg. 389 (2022) 114333","Hard-constraint construction (F4 background)"),
    ("Finite basis physics-informed neural networks (FBPINNs)","B. Moseley, A. Markham, T. Nissen-Meyer","2023","Adv. Comput. Math. 49(4) (2023) 62","Windowed domain decomposition"),
    ("Novel and general discontinuity-removing PINNs for elliptic interface problems","H. Fan, Z. Tan","2025","J. Comput. Phys. 529 (2025) 113861","Recent interface-PINN variant"),
    ("Conservative physics-informed neural networks on discrete domains for conservation laws","A. D. Jagtap, E. Kharazmi, G. E. Karniadakis","2020","Comput. Methods Appl. Mech. Engrg. 365 (2020) 113028","cPINN flux-continuity coupling (F2 background)"),
    ("Multi-domain physics-informed neural network for steady-state heat conduction in multilayer media","B. Zhang, G. Wu, Y. Gu, X. Wang, F. Wang","2022","Phys. Fluids 34(11) (2022)","M-PINN layered-media baseline"),
    ("Physics-informed neural networks: a deep learning framework for forward and inverse problems","M. Raissi, P. Perdikaris, G. E. Karniadakis","2019","J. Comput. Phys. 378 (2019) 686-707","Original PINN"),
    ("On the spectral bias of neural networks","N. Rahaman et al.","2019","ICML 2019, PMLR 97","Spectral-bias mechanism"),
    ]):
    add(id=f"L{23+i}", title=t, authors=a, year=y, doi_or_url=d, source="additional (cited in L08/L02)",
        verification_level="cited_only", relevance_to_project=rel, review_audit_notes="Bibliographic data from a reference list; not inspected.")

add(id="L30", title="The Deep Ritz Method: a deep learning-based numerical algorithm for solving variational problems",
    authors="W. E, B. Yu", year="2018", doi_or_url="Commun. Math. Stat. 6 (2018) 1-12",
    source="additional (background knowledge)", verification_level="not verified - to be checked before citation",
    relevance_to_project="F5 energy formulation", review_audit_notes="Included because F5 depends on it; bibliographic details must be verified.")

# ---------------- Phase 2 targeted-search additions (2026-10-16) ----------------
add(id="L31", title="FP64 is All You Need: Rethinking Failure Modes in Physics-Informed Neural Networks",
    authors="C. Xu, D. Liu, A. Nassereldine, J. Xiong", year="2025", doi_or_url="arXiv:2505.10949; NeurIPS 2025 poster",
    source="additional (Phase 2 search)", verification_level="abstract_snippets",
    optimisation="L-BFGS (failure attributed to FP32 premature convergence test)",
    main_findings_author="With FP32, L-BFGS satisfies its convergence test prematurely and stalls in a failure phase; FP64 removes known failure modes for vanilla PINNs",
    limitations_team_inferred="Effect demonstrated under L-BFGS; interface contrasts not studied",
    relevance_to_project="Motivates Intervention C and requires precision to be crossed with optimiser")
add(id="L32", title="Optimizer or Precision? Understanding Optimizer-Precision Interactions in PINNs",
    authors="M. Tarekegn, C. Wu, T. Sharma, S. Garg", year="2026", doi_or_url="NeurIPS 2026 (virtual poster 161445)",
    source="additional (Phase 2 search)", verification_level="abstract_snippets", seeds_reported="32 runs over 8 configurations, 4 benchmarks",
    main_findings_author="Doubling precision changes Adam error by at most 1.25x but up to 17.7x under L-BFGS; low-error runs end in L-BFGS",
    limitations_team_inferred="No interface problems",
    relevance_to_project="Pre-registered prediction for Intervention C under Adam (small effect) vs L-BFGS (large); 2x2 design")
add(id="L33", title="A deep learning method for multi-material diffusion problems based on physics-informed neural networks",
    authors="Y. Yao, J. Guo, T. Gu", year="2023", doi_or_url="Comput. Methods Appl. Mech. Engrg. 417 (2023) 116395; arXiv:2305.09125",
    source="additional (Phase 2 search)", verification_level="abstract_snippets", seeds_reported="mean +/- spread reported",
    main_findings_author="Standard PINN fails (error ~1.33) on thin-insulating-layer jump problem; domain-scaling DS-PINN/nDS-PINN succeed; interface loss normalised by known jump functions",
    relevance_to_project="Thin-layer (P5) precedent and residual-normalisation precedent (Intervention B)")
add(id="L34", title="Solving Poisson Problems in Polygonal Domains with Singularity Enriched Physics Informed Neural Networks",
    authors="T. Hu, B. Jin, Z. Zhou", year="2023", doi_or_url="arXiv:2308.16429", source="additional (Phase 2 search)",
    verification_level="abstract_snippets", main_findings_author="PINN accuracy degrades near point sources and corner singularities; singular-function enrichment restores it",
    relevance_to_project="P3/P4 singularity treatment precedent (single material)")
add(id="L35", title="Physics informed neural network for charged particles surrounded by conductive boundaries",
    authors="F. Hafezianzade, M. Biagooi, S. Nedaaee Oskoee", year="2023", doi_or_url="Sci. Rep. (2023) doi:10.1038/s41598-023-40477-y; arXiv:2301.02191",
    source="additional (Phase 2 search)", verification_level="abstract_snippets",
    main_findings_author="Potential split into singular and smooth parts; PINN learns smooth part; faster than FEM at evaluation",
    limitations_team_inferred="Conducting walls, no dielectric interface",
    relevance_to_project="Singular-part subtraction precedent for P3")
add(id="L36", title="INI-VPINN: A Variational PINN with Implicit Neumann and Interface Handling for Multi-Material Domains with Geometric Singularities",
    authors="to verify", year="2026", doi_or_url="arXiv:2606.18032", source="additional (Phase 2 search)", verification_level="abstract_snippets",
    main_findings_author="Variational PINN handles material discontinuities and corners; compared with vanilla PINN and cPINN",
    relevance_to_project="Multi-material corner (P4) precedent")
add(id="L37", title="Physics-Informed Tailored Finite Point Operator Network for Parametric Interface Problems",
    authors="to verify", year="2024", doi_or_url="arXiv:2409.10284", source="additional (Phase 2 search)", verification_level="abstract_snippets",
    material_contrast="1D high-contrast case with 0<a(x)<<1 in one region",
    main_findings_author="High contrast produces inner layers; operator network captures thin layers at coarse resolution",
    relevance_to_project="Thin-layer / high-contrast layer precedent (P5)")

out = pathlib.Path(__file__).with_name("literature_database.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(R)
print(f"wrote {len(R)} records to {out}")
