# Literature Review: PINNs Across Dielectric Interfaces

Phase 1 deliverable B. Date of searches: 16 October 2026. Structured data for every paper is in `literature_database.csv` (30 records); this document gives the narrative, the search log, and an audit of the supplied team reviews.

Throughout, **[author]** marks a statement made by a paper's authors and **[team]** marks our inference.

## 1. Sources and how they were read

Ten files were supplied. Two were not in the conversation context and were read from disk: the PINNacle paper (50 pages; full text extracted with `pdftotext`) and P007 (a scanned review with no text layer, read by OCR; the OCR text is kept in `source_extracts/`). The other eight were team reviews of six papers.

Duplicate reviews were consolidated. `P003_Tolga_and_Yasir_docx.pdf` and `P003_Tolga_and_Yasir_docx_1.pdf` are identical, and `P006_YASİR.pdf` reviews the same paper (Nohra & Dufour, 2024). All three map to record L02.

Where a supplied review summarised a paper whose original we could reach, we read the original. That changed several conclusions (Section 5). Records carry a `verification_level` field so that nobody later cites a number we only saw second-hand.

## 2. Search log

Inclusion criteria: primary publications (journal, conference or arXiv) that solve an elliptic or Maxwell problem with a material interface or discontinuous coefficient using a neural-network solver, or that benchmark PINNs against classical solvers or characterise PINN failure. Reviews and blog posts were excluded. Search engine: web search tool (top-10 results per query) plus direct fetches of arXiv abstract and PDF pages.

| # | Query | Purpose | Outcome |
|---|---|---|---|
| 1 | arXiv 2606.25323 PINN time-domain Maxwell split-field PML | verify P001 | verified (L07) |
| 2 | Extended Interface PINN moving interface Bi Deng Zhu 2508.01463 | verify P004 | verified (L05) |
| 3 | cusp-capturing PINN elliptic interface Tseng Lin Hu Lai | F3 source | L09, L10 |
| 4 | hard-constrained PINN interface conditions high contrast 2025 2026 | F4 source; closest prior work | L08 (full text read) |
| 5 | PINN electrostatics dielectric interface permittivity contrast | dielectric-specific prior work | L19, L20; L02 confirmed |
| 6 | Grossmann "Can PINNs beat the FEM" | PINN vs FEM cost | L16 |
| 7 | neural network elliptic interface high contrast 1000 PINN failure flux jump | high-contrast prior work | L11, L12, L13, L22 |
| 8 | systematic study PINN interface contrast sweep seeds failure rate | does our core design already exist? | none found; L17 and adjacent benchmarks |
| 9 | PINN interface gradient error H1 near interface flux recovery | RQ2 prior | L21 (classical FEM evidence) |
| 10 | arXiv 2111.05641 parallel PINN bidirectional balance | verify P007 | verified (L06) |

Not yet searched (scheduled for Phase 2 before the protocol is frozen): float32/float64 effects in PINNs (G10), closed-form contrast-derived loss weights (G12), singular sources near interfaces (G14), wedge/triple-junction singularities (G15), thin films (G16), Deep Ritz with interfaces, cPINN/I-PINN originals, and the Wang–Teng–Perdikaris gradient-pathology paper itself. Ten queries is a modest search; a "we did not find" statement in this project means "not found in these queries", nothing stronger.

## 3. What the literature establishes

**Interface-aware networks work on manufactured elliptic interface problems, including at very high contrast.** [author] Discontinuity-capturing shallow networks were run with coefficient ratio 10³ as the default (L10). The cusp-capturing PINN includes ratios of 10⁻⁴ and 10⁴ (L09). The deep unfitted Nitsche method reports ratios up to 10⁵ in both directions (L11). Piecewise deep networks were tested at 1000:1 and 1:1000 (L12). A convergence study used 1000:1 with five runs (L13). [team] High contrast on its own is therefore not an open question, and the project must not present it as one.

**The single-network failure mechanism is documented.** [author] Nohra and Dufour (L02, Section 6) show for a 1D problem with piecewise-constant coefficient that a constant (wrong) solution satisfies the residual at every collocation point, because the coefficient's derivative is zero there; the interface condition is invisible unless points are placed at the interface or the coefficient is smoothed. This is exactly the F1 baseline's weakness. [team] Our contribution for F1 can only be quantitative (how loss visibility, error and failure rate change with contrast and smoothing), not the discovery of the mechanism.

**Hard constraints help, but evidence comes from one contrast.** [author] Chung et al. (L08) embed continuity and flux balance in the ansatz (windowing and buffer constructions) and report order-of-magnitude error reductions over soft-constrained baselines in 1D and modest gains in 2D. All reported cases use κ = 0.1 versus 1. Results are from single runs with the best of four initialisations, and the authors say they expect larger advantages at much larger contrasts but did not test them. [team] This is the closest prior work. It defines the gap our contrast sweep, seed statistics, and field metrics fill. Their 2D finding that AdaI-PINN exceeded 80% error is also independent evidence about L03.

**PINNs did not beat FEM on smooth problems.** [author] Grossmann et al. (L16) found PINNs slower and less accurate than FEM for Poisson, Allen–Cahn and Schrödinger problems. None had interfaces. PINNacle (L01) treated FEM as the reference only, reported runtime, used 3 repeats and a 10% L2 threshold, and found a piecewise-coefficient Poisson problem (Poisson2d-MS) unsolved by all eleven methods.

**Failure-mode methodology.** [author] Krishnapriyan et al. (L17) swept PDE coefficients with at least ten seeds and attributed failures to optimisation rather than representation. Wang, Teng and Perdikaris (L18; not yet inspected) are the standard reference for per-term gradient imbalance. Huang (L06) needed both unit and loss-scale balancing for layered heat transfer.

**Field accuracy at interfaces.** [author] In classical numerics, structured-mesh interface methods are typically one order less accurate in the gradient than in the solution (L21). Neural interface papers report global solution L2 error and occasionally global H1/H2 (L05, L13). Interface-concentrated error is shown visually in L12. [team] We did not find an explicit, thresholded interface-band E/D metric applied across formulations and contrasts.

**Electrostatics specifically.** [author] Electrostatic PINNs with sigmoid-smoothed permittivity were compared with FEM (L19; abstract only). A 2026 scattering paper reports that the permittivity discontinuity degrades TE-polarisation predictions near interfaces and uses tanh smoothing (L20). [team] We did not find a permittivity-contrast sweep on the canonical electrostatic geometries (dielectric cylinder in a uniform field, line charge near a planar interface, wedge, thin film) scored on physical quantities such as capacitance or interior field. Mathematically these are elliptic interface problems already studied in generic form, so the novelty claim is modest.

**A cautionary result.** [author] The Dirichlet–Neumann learning paper (L22) shows DeepDDM failing when the two coefficients are *equal*, not when they differ. [team] Failure need not be monotone in contrast. Our analysis must not assume a single "critical contrast" exists, and the κ = 1 control in the primary grid is essential.

## 4. Implications for the experimental design

The primary contrast set must reach the range where prior work already reports success (10³ and above) so our results can be compared against it. The prescribed set {1, 5, 20, 80, 400, 2000} does this. The supplied protocol PDF suggests a different set {1, 3, 10, 30, 100, 300, 1000, 10⁴}; this is resolved in Phase 2.

Chung et al. report best-of-initialisation results. Our median-and-spread reporting across five seeds will make some methods look worse than in their papers, and the manuscript must explain why.

The "critical contrast" in supplied-protocol hypothesis H2 presumes monotone failure. In light of L22, H2 must define what happens if failure is non-monotone.

Supplied-protocol hypothesis H4 compares a peak field at a triple junction. Our brief forbids treating an undefined singular peak as finite, so H4 will be rewritten with a fixed-distance or asymptotic-coefficient metric.

## 5. Audit of the supplied reviews

These corrections are recorded per paper in the `review_audit_notes` column.

1. **Nohra & Dufour (P003, P006):** The quoted errors (0.4%, 2×10⁻³, "0.2–0.4%") come from problems *without* an interface. All interface cases were compared to FEM only visually. No permittivity interface was tested; the interfaces are permeability (3:1) and resistivity (0.5–1.5). The two reviews give different reproducibility scores (2 and 3). Neither mentions the Section 6 silent-failure example, the most relevant passage for this project.
2. **"High contrast is untested" (P003, P004, P006, P001, P009 alignment rows):** This is contradicted by L09–L13 and L22. The claim holds only for the specific reviewed papers, not for the field.
3. **Bi, Deng, Zhu (P004):** The year is listed as 2026, but the arXiv identifier indicates first submission in August 2025. The "1:1000" contrast is an Oseen viscosity ratio, not a diffusion or permittivity coefficient; the diffusion example has contrast 100.
4. **XPINN (P008):** The CEUR workshop version is cited; the archival version is *Commun. Comput. Phys.* 28(5), 2020. The code URL given appears malformed. XPINN's interface loss uses residual continuity, not flux continuity, so it is background for F2 rather than an F2 implementation.
5. **Liu et al. (P001):** The "Direct comparison: Yes" entry overstates relevance. The paper is about time-domain wave propagation with PML, not electrostatics, and reports no quantitative error.
6. **PINNacle (no review):** Its "two-domain" Poisson3d-CG problem has equal diffusion coefficients (μ₁ = μ₂ = 1) and is not a coefficient-contrast test.
7. **AdaI-PINN (P009) and Huang (P007):** The originals were not read this session. Numbers in the database come from the reviews and are flagged for verification before citation.
