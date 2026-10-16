# Phase 1 Progress Report: Literature

Date: 2026-10-16. Commit at report time: 54b5046.

## Implemented and executed
The project directory follows the structure in the brief. All ten uploads were read. PINNacle was extracted in full with pdftotext, the scanned P007 was read by OCR, and the duplicates (P003 ×2 plus P006) were consolidated. Seven supplied papers were verified against primary arXiv pages; the Nohra & Dufour and Chung et al. full texts were read. Ten documented web searches were run. Database and gap-matrix builder scripts were executed.

## Outputs
| File | Content |
|---|---|
| literature/literature_database.csv | 30 records × 26 fields, with a verification level per record |
| literature/literature_review.md | Narrative, search log, audit of the supplied reviews |
| literature/research_gap_matrix.csv | 17 topics classified into the brief's five categories |
| literature/research_gap_report.md | Surviving and rejected gap claims; refined contribution statement |
| literature/references.bib | 30 entries, each with a verification note |
| results/logs/environment_2026-10-16.json | Hardware/software record |
| results/logs/env_bench_jax_1d_step.txt | Timing benchmark |

## Key findings
1. High contrast (10³–10⁵) is already handled by interface-aware networks (L09–L13). Several supplied reviews state otherwise, and this has been corrected.
2. The F1 silent-failure mechanism is already documented (Nohra & Dufour, Section 6).
3. The closest prior work (Chung et al., 2026) tests only contrast 10, reports best-of-initialisation runs, and uses solution-L2 metrics. That leaves the contrast sweep, seed statistics, field metrics and FEM cost comparison open (gaps G05–G07, G11).

## Environment and resource limits (affects later phases)
| Item | Value |
|---|---|
| CPU | 1 core (Intel Xeon 2.8 GHz) |
| RAM | ~4 GB |
| GPU | none |
| Free disk | ~4.7 GB |
| PyTorch | not available: CPU index unreachable, and the PyPI wheel exceeded the 300 s command limit |
| JAX | 0.11.2 CPU, installed |

Selected framework: JAX (CPU), which supports float64 for Intervention C.

Measured cost: about 1.7–1.8 ms per gradient evaluation (float32, 3×32 tanh, 256 points, second derivative). At that rate one 3000-step P1 run is roughly 5–10 s with logging, and the 180-run primary P1 sweep is roughly 30–40 min. The 2D problems and interventions will be substantially more expensive; estimates will be made per phase before running.

**Persistence limit:** this sandbox's file system resets between sessions. Each phase ends with a zip of the full project. To continue, re-upload the latest zip at the start of the next session. Without it the work cannot resume from the saved state.

## Not done or open
- Ten queries is a modest search. Topics G10, G12, G14–G16 are unsearched.
- The AdaI-PINN, Huang, and Wang–Teng–Perdikaris originals were not read.
- Several bibliographic entries are marked VERIFY.
- Contrast-set and hypothesis conflicts between the brief and the supplied protocol PDF are listed in research_gap_report.md, Section 5.

## Next step: Phase 2
1. Run the targeted searches.
2. Write hypotheses H1–Hn with support and refute criteria.
3. Resolve the contrast set and H2/H4 wording.
4. Write the F6 weighting formula.
5. Freeze the protocol (YAML config plus document).
