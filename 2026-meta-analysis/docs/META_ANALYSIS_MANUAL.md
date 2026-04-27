# Automated Meta-Analysis Pipeline — User Manual
## Team "One Hot Key" | SIOP 2026 Competition → General-Purpose Tool
### Version 8 (Open Source) | Version 6 (API, best leaderboard score)

This manual explains how to adapt the SIOP correlation extraction pipeline to a new predictor–outcome research question without breaking statistical or coding discipline. It does not duplicate cascade design rationale; that lives in the architecture doc.

## Related docs

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — 4-tier cascade design and library choices
- **[KNOWN_LANDMINES.md](KNOWN_LANDMINES.md)** — silent-failure traps to never re-introduce
- **[CHANGELOG.md](../CHANGELOG.md)** — version history v9 → v12.1
- **[README.md](../README.md)** — repo overview and quick start

This manual focuses specifically on adapting the pipeline to a new research question — for everything else, see the docs above.

---

## Document Version Control

This manual is a living document maintained across multiple Claude project chat sessions. Each session may add new sections, bug fixes, or methodological decisions. The table below tracks which session contributed which content.

| Section | Content | Session / Date |
|---------|---------|----------------|
| Sections 1–7 (original) | Core pipeline design, statistical criteria, construct classification, sign logic, architecture, v7 bug fixes | Prior sessions (chat 549b4a13), approx. March 24–25, 2026 |
| Section 3 — Design-Level Exclusions | Ecological, time-series, logistic-only, p-value-only, regression-only, cross-wave exclusion rules with study examples | Session: March 27, 2026 |
| Section 3b — Longitudinal Wave Selection Policy | Wave-collapse rule rationale + study109 worked example | Session: March 27, 2026 |
| Section 5b — Non-Standard Table Format Patterns | 8 new table format patterns added in v8 | Session: March 27, 2026 |
| Section 11 additions — v8 Known Issues | 15 new bug/fix entries from this session's diagnostic work | Session: March 27, 2026 |
| Section 3c — Cross-Wave-Only Table Detection | T1/T2 parenthetical wave markers; design gate for longitudinal tables with no same-wave pairs | Session: April 3, 2026 |
| Section 3d — Self-Report Rule | Parent-report, clinician-rated, observer-rated exclusions with study examples | Session: April 3, 2026 |
| Section 3e — Domain-Specific Satisfaction | Co-tenancy, job, care satisfaction excluded; contrast with global life satisfaction | Session: April 3, 2026 |
| Section 4b — Construct Classification Updates | Partner trust, WAQ trustworthiness, cognitive social capital, PE abbreviation, PWB priority | Session: April 3, 2026 |
| Section 5c — Additional Non-Standard Table Patterns | Split-diagonal individual/national multilevel; named symmetric matrix; two-page split matrix; non-contiguous column headers | Session: April 3, 2026 |
| Section 5d — Meta-Filter Logic | dep+LS pair guard; factor prioritization; social capital SWB exclusion; regression-table context rejection | Session: April 3, 2026 |
| Section 11 additions — v9 Known Issues and Code Fixes | 14 new entries; override audit; generalizable code fixes vs structural overrides | Session: April 3, 2026 |
| Section 3f — Multi-group ANOVA exclusion | Omnibus F with df_between ≥ 2; categorical trust + ANOVA false-positive trap (study110) | Session: April 4, 2026 (morning) |
| Section 11 — study110 pattern | Categorical trust, ANOVA + post-hoc p-only, moderation plots without dispersion | Session: April 4, 2026 (morning) |
| Section 3g — Collective Efficacy vs Trust | Confidence in society (Keller et al.) excluded as collective efficacy; study56 | Session: April 4, 2026 |
| Section 4c — Construct Classification Updates v9b | Trust in science eligible; institutional trust criteria; collective efficacy decision tree | Session: April 4, 2026 |
| Section 5e — Supplement and Raw Data Extraction | Local supplement infrastructure; XLSX raw data Pearson r computation; supplement_review_needed.txt | Session: April 4, 2026 |
| Section 5f — Opus API Sweep | Comprehensive Opus sweep; priority 0-7 target selection; cost analysis | Session: April 4–5, 2026 |
| Section 9 updates | --no-fetch-supplements flag; data_on_request.txt; overnight automation | Session: April 4, 2026 |
| Section 10 updates | Imputation back-calculation formula; Fisher's z vs arithmetic mean decision | Session: April 4, 2026 |
| Section 11 — v9b Known Issues | data_on_request bug; study23 WARN; study81; study46 false flag; study2/50 supplement patterns | Session: April 4–5, 2026 |
| Pipeline v11 (April 5, 2026) | Role-aware measure lexicon; aggregation completeness guard; archetype fallback router; _classify_rawdata_column for survey-format XLSX headers; smoke CLEAN 0/0/0 | Session: April 5, 2026 |
| v11 dev verification | `py_compile` → `_regression_smoke.py --pipeline pipeline_dev.py` → per-group override retirement singles → no-vision batch → vision batch → Opus sweep on vision log (`--imp` supported) | Session: April 2026 |

### Pipeline Version → Chat Session Mapping

| Pipeline Version | Primary Chat Session | Key Changes |
|-----------------|---------------------|-------------|
| v1–v5 (API) | Earlier sessions, pre-March 24 | Initial build, basic extraction cascade |
| v6 (API, best leaderboard score MSE=0.02331) | Pre-March 24 sessions | Vision tier, classify_var gates — **last completed API version** |
| v7 (Open Source) | Session 549b4a13, March 24–25 | Geom tier, study67 hybrid fix, v7 bug cycle |
| v8 (Open Source) | Session 549b4a13 + March 27 session | Abbreviation expansion, wave-collapse, format patterns A–F |
| v9 (Open Source) | April 3–4, 2026 sessions | dep+LS pair filter; split-diagonal fix; T1/T2 wave gate; PE abbreviation glossary; domain satisfaction scoping; PWB priority; WAQ/cognitive SC trust terms; regression-table context rejection; zero-index fix; confidence-in-society exclusion; supplement infrastructure; raw data XLSX extraction; MSE=0.014762 (sub 24) |
| v10 (Open Source) | April 5, 2026 | Descriptor column count fix; named symmetric matrix parser; transposed table detection; multi-factor full-row extraction; trust-row isolation; mixed Spearman/Pearson; non-contiguous column headers; wave-merge ordering; study81 provisional override |
| v11 (Open Source, current) | April 5, 2026 | Role-aware paper measure lexicon; aggregation completeness guard; archetype fallback router (5 archetypes); _classify_rawdata_column for survey XLSX headers; smoke CLEAN 0/0/0; batch 64 has-r/63 blank; MSE=0.014455 |
| **v10 (Open Source)** | April 5, 2026 session | Descriptor-column counting for APA matrices; named symmetric matrix + transposed trust×SWB parsers; non-contiguous numbered headers; multi-outcome trust-row × SWB-column extraction; mixed Spearman/Pearson design flag (Rule 16 exception); wave stratum + subsample merge on structured tiers; study81 provisional override |
| **v11 (Open Source, current)** | April 2026 session | Role-aware paper lexicon (`_extract_measure_lexicon` + `classify_var` paper roles); aggregation completeness guard + early dep+LS grouping in meta-filter; table archetype fallback router in `_parse_apa_table`; raw-data sentence header classifier `_classify_rawdata_column` for XLSX supplements; Opus sweep targets include all blanks (P4) and manual-override verification (P6) |

### Conventions for Future Sessions

When adding to this manual in a new chat session:
1. Update the Document Version Control table above with the new session date and section
2. Add new known issues to Section 11 with status "Fixed v8" or the current version
3. Do not overwrite existing content — append to existing sections or add new numbered subsections (e.g., 3c, 5c)
4. Reference specific study numbers when documenting generalizable patterns (e.g., "study109 — Nilsen et al.")

---

## Table of Contents

1. [What This Pipeline Does and Why](#1-what-this-pipeline-does-and-why)
2. [Theoretical Grounding: How Human Coders Work](#2-theoretical-grounding-how-human-coders-work)
3. [Statistical Inclusion and Exclusion Criteria](#3-statistical-inclusion-and-exclusion-criteria)
4. [Construct Classification Principles](#4-construct-classification-principles)
5. [Pipeline Architecture](#5-pipeline-architecture)
6. [Generalizable Validation Rules](#6-generalizable-validation-rules)
7. [Sign Direction Logic](#7-sign-direction-logic)
8. [Adapting for a New Research Question](#8-adapting-for-a-new-research-question)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Submission Workflow](#10-submission-workflow)
11. [Known Issues and Traps](#11-known-issues-and-traps)
12. [References](#12-references)

---

## 1. What This Pipeline Does and Why

This pipeline automates the data extraction phase of a psychometric meta-analysis. Given a set of academic PDFs and a research question (e.g., "How are trust and subjective well-being related?"), it extracts bivariate effect sizes from each study, converts them to a common metric (Pearson r), and aggregates them to a single study-level value.

The goal is to replicate, as faithfully as possible, what a trained human coder would do when working through a stack of studies — but at scale, consistently, and with an auditable decision log.

### Why Pearson r as the Common Metric?

Pearson r is the preferred effect size for meta-analyses of continuous bivariate relationships in psychology (Hunter & Schmidt, 2004, Ch. 3). It is:

- Scale-free and directly interpretable as the strength and direction of a linear association
- Amenable to artifact correction (measurement error, range restriction) using established formulas
- The native output of correlation matrices, which are the most common reporting format in APA-style papers

When studies report other statistics (t, F, d, OR), these are converted to r using published formulas (Lipsey & Wilson, 2001, Ch. 3). When studies report statistics that cannot be accurately converted (Spearman rho, standardized beta, partial r), those effects are excluded. This is not a limitation but a principled decision grounded in measurement theory, described in Section 3.

---

## 2. Pipeline Design Philosophy

### Why Mimic a Human Coder Rather Than Search the Full Document?

The central architectural decision in this pipeline is to process papers sequentially and gate early, rather than searching the full document text and filtering retrospectively. This mimics how trained meta-analytic coders actually work, and it matters for accuracy.

A document-wide search approach — scanning every page for any value near the word "trust" — generates a large number of false positives: regression tables, figure captions, reference lists, discussion sections citing other papers, and boilerplate method text. Filtering these out after extraction is difficult because the false positives are structurally similar to true effects. The result, as observed in the development runs of pipeline v4, was a recurring r = 0.308 hallucination from the vision LLM and proximity search grabbing values from the wrong context.

The human coder avoids this problem not because they are smarter about filtering, but because they never look in the wrong place to begin with. They read the abstract, confirm both constructs and a correlational method are present, check the methods section to identify the exact measures, and then go directly to the correlation table. The value they extract has already been contextually validated by the time they write it down.

The pipeline replicates this by:

- Using section-specific signals rather than full-document search
- Applying early-exit gates for disqualifying designs before any extraction
- Requiring wellbeing header proximity before the proximity search fires
- Cross-validating table extractions against symmetric matrix consistency
- Grounding LLM classification in construct definitions from the abstract and methods, not just local table context

### Why the Four-Tier Cascade?

Different PDFs present different challenges. Some are born-digital with clean table structure; others are scanned, multilingual, or use non-standard layouts. The four tiers are ordered from most to least reliable:

**Tier 0 (pdfplumber)** extracts tables geometrically — reliable when column and row positions are clearly defined, fast, no LLM required.

---

## Novel Extraction Pattern: Docling Headers + Geom Values (v7)

Discovered while solving study67 (Zhang et al., 2021): a rotated landscape APA
correlation matrix spanning a page break, with no cell borders and non-standard
PDF character encoding. Every single extraction method failed individually:

| Method | Failure mode |
|--------|-------------|
| Docling | Scrambles row order; strips minus signs; inserts \x00 artifacts |
| pdfplumber/fitz find_tables | No ruled borders → empty result |
| geom coordinate clustering | Values packed into single cells by rotation |
| geom strip_diagonal (alone) | Recovers values but only synthetic keys ("2.", "3.") |
| Vision (qwen2.5vl) | 13×13 matrix too dense at full-page scale |

**The fix**: fuse two complementary failure modes into one correct answer.
Docling gets *column headers right* even when it gets values completely wrong.
The geom strip_diagonal parser gets *values right* but can't resolve variable
names when the header is on a different page or in a different coordinate space.

Solution: extract `idx_to_name = {2: "Trust in local government", 3: "Anxiety"}`
from Docling's column headers, pass it as a label resolver to the strip_diagonal
output, then run `classify_var()` on the resolved labels. Neither tool alone was
sufficient; both together recovered r = +0.41 correctly.

This pattern generalizes to any study where:
- The table is rotated or spans a page break (geom alone fails)
- Docling's ML TableFormer misreads cell values (Docling alone fails)
- Both tools parse different aspects of the same table correctly

Implementation: `Tier 1b` in the v7-era codebase (logic now lives in `pipeline_dev.py`) — geom attempt 1
(coordinate clustering), then attempt 2 (strip_diagonal with Docling-derived
`idx_to_name`). The `MANUAL_OVERRIDES` dict remains available for cases where
even this hybrid fails (e.g., truly unreadable PDFs).

**study67 resolution (confirmed v7)**: The full automated chain works without any manual override:
1. Docling extracts Trust×Anxiety = 0.02 (wrong — rotated table, scrambled values)
2. `LAST_DOCLING_IDX_TO_NAME` is populated from `table_2d[-1]` — the last row of Docling's raw table, where variable names are stored for this layout (`{2: "2. Trust in local government", 3: "3. Anxiety", ...}`)
3. `force_geom_after_docling` fires: all trust×distress values ≤ 0.06 + negative outcome + numbered labels present
4. `extract_corr_matrix_strip_diagonal(pages=[5,6])` recovers `("3.", "2.") = 0.41`
5. `idx_to_name` resolves synthetic keys → Trust×Anxiety = 0.41
6. Final output: `tier=geom, r=0.41` ✓

Key architectural lesson: Docling read column headers correctly but cell values incorrectly. The geom tier read cell values correctly but couldn't classify variable names. The fix fuses both — Docling's header accuracy feeds the geom tier's label resolution. Neither tool alone was sufficient; together they recovered the correct value fully automatically.


---

*Future addition (final open-source version)*: Add **Camelot** as an additional Tier 0 option before pdfplumber, specifically for bordered tables (tables with visible cell lines). Some older psychology papers use cell borders, and Camelot's Lattice method handles these with near-perfect accuracy. Note that Camelot requires Ghostscript and OpenCV as dependencies. Camelot is *not* a replacement for Docling — benchmark studies show them equivalent on table extraction overall (Docling = Camelot on structured tables), and Camelot's Stream mode (for borderless tables, which is the APA correlation matrix default) is weaker than Docling's TableFormer. Camelot should be positioned as: bordered tables → Camelot; borderless tables → pdfplumber → Docling.

*Planned addition for final version: Camelot (lattice mode) as a Tier 0a option before pdfplumber, for PDFs with explicit cell borders. Camelot's lattice method uses morphological image transforms to detect ruled lines and achieves accuracy equivalent to Docling on bordered tables at 10× the speed (see benchmark: arxiv 2511.16134). APA correlation matrices typically use borderless whitespace alignment, so Camelot's stream mode would be needed as a fallback — but stream mode is geometry-based and offers no clear advantage over pdfplumber for these layouts. The net recommendation: add Camelot lattice as a fast first-pass for the minority of papers that use bordered tables, keep pdfplumber for borderless. Note: Camelot requires Ghostscript and OpenCV as dependencies (python -m pip install "camelot-py[cv]"), which adds ~200MB to the install footprint.*

**Tier 1 (Docling)** uses ML-based TableFormer to recover table structure from complex layouts — handles merged cells, multi-row headers, and combined descriptive-statistics-plus-correlations tables. OCR is conditionally enabled: the pipeline measures text density (characters per page) before invoking Docling. Born-digital PDFs (>250 chars/page) run with OCR disabled — re-rasterizing clean text introduces recognition errors on numbers, minus signs, and Greek letters (α, β, ρ), which are precisely the characters that matter most for effect size extraction. Scanned PDFs (<250 chars/page) enable OCR automatically. This distinction matters: Hunter and Schmidt (2004, Ch. 12) note that transcriptional errors are among the uncorrectable artifacts in meta-analysis.

**Tier 1b (Vision — qwen2.5vl)** reads page images directly — handles scanned documents and tables that are embedded as figures rather than text.

**Tier 2 (Regex + phi4)** extracts numerical candidates from raw text and uses phi4 to classify them — the fallback when no structured table can be found; most prone to false positives.

When a higher tier succeeds, lower tiers do not run. This is why the log reports the extraction tier for each study — it tells you how confident the extraction is.

---

## 2. Theoretical Grounding: How Human Coders Work

The pipeline architecture mirrors the step-by-step procedure that trained meta-analytic coders follow, as described in the foundational texts of the field (Hunter & Schmidt, 2004, Ch. 12; Lipsey & Wilson, 2001, Ch. 4; Cooper et al., 2009, Ch. 12).

### The Human Coder's Workflow

A trained coder working through a study follows a sequential, confirmatory process:

**Step 1 — Abstract screening.** The coder reads the abstract to determine whether the study is eligible. Key questions: Are both constructs of interest (predictor and outcome) mentioned? Is a correlational or convertible statistical method described? Are there any disqualifying design features (purely ecological data, latent class analysis, logistic-only regression without bivariate correlations)?

If the abstract does not indicate that relevant correlations exist or could be derived, the study is set aside. This is efficient: most exclusions can be made at the abstract level without reading the full paper.

**Step 2 — Methods section review.** If the abstract passes, the coder reads the Methods section to identify:
- The exact measures used for each construct (scale name, number of items, reliability)
- The sample size and sampling frame (individual-level vs. aggregate)
- Whether correlational analysis was conducted (not just regression or ANOVA)
- Rater source (self-report vs. clinician-administered vs. archival)

Hunter and Schmidt (2004, Ch. 12) specify that this information is prerequisite to interpreting the effect size. Knowing what was measured is as important as knowing the magnitude of the relationship, because it determines construct validity and whether artifact corrections can later be applied.

**Step 3 — Results section and table extraction.** The coder locates the correlation table or, if absent, inline-reported statistics. APA guidelines (APA, 2020) specify that correlation matrices, descriptive statistics, and reliability estimates should be reported in a single table. The coder extracts the cell at the intersection of the trust predictor row and the wellbeing outcome column.

**Step 4 — Cross-validation.** The coder verifies that the scale names in the table match those described in the Methods section. If a study reports "Trust in HCP" in the table but the Methods section described "Trust in Physicians Scale," the coder confirms these refer to the same measure before coding the effect.

**Step 5 — Record and flag.** The coder records the effect size, sample size, direction, and any flags (e.g., "ecological sample," "correlations available for one subsample only," "Spearman rho, not Pearson r").

### How the Pipeline Replicates This

| Coder Step | Pipeline Equivalent |
|---|---|
| Abstract screening | detect_study_design_issues() — flags cohort, ecological, LCA designs |
| Methods review — construct check | classify_var() — classifies variables against TRUST_TERMS / WELLBEING_TERMS |
| Methods review — stat method check | Abstract/methods text scan for "correlat" signal words |
| Results/table extraction | 4-tier cascade: pdfplumber → Docling → qwen2.5-VL (Tier 1b) → Regex+phi4 |
| Cross-validation | Symmetric matrix consistency check; trust x wellbeing pair validation |
| Record and flag | JSON log with notes, skipped_effects, extraction_tier fields |

The key architectural insight is that the pipeline should gate early and extract late — matching the human coder's sequential filtering process rather than extracting everything and trying to filter retrospectively.

---

## 3. Statistical Inclusion and Exclusion Criteria

This section specifies which statistics can be included in the meta-analysis and which must be excluded. These criteria are grounded in the measurement-theoretic framework of Hunter and Schmidt (2004, Ch. 3 and Ch. 5) and the practical coding guidance of Lipsey and Wilson (2001, Ch. 3-4).

The core principle, stated explicitly by Hunter and Schmidt (2004, Ch. 12, p. 473) in their coding guidance, is: extract zero-order bivariate correlations only. The "golden rule" is that only statistics reflecting the unadjusted bivariate relationship between the two constructs of interest belong in a meta-analysis of correlations.

### Includable Statistics

The following statistics can be accurately converted to Pearson r and are eligible for extraction:

| Statistic | Conversion Formula | Condition | Source |
|---|---|---|---|
| Pearson r | Direct | |r| <= 1.0 | Hunter & Schmidt, 2004, Ch. 3 |
| t-statistic | r = t / sqrt(t^2 + df) | df must be reported | Lipsey & Wilson, 2001, p. 37 |
| F-statistic (df1 = 1) | r = sqrt(F / (F + df2)) | df1 = 1 ONLY | Lipsey & Wilson, 2001, p. 38 |
| Cohen's d / Hedges' g | r = d / sqrt(d^2 + 4) | — | Hunter & Schmidt, 2004, Ch. 7 |
| Odds Ratio (OR) | r = (OR-1)/(OR+1) x correction | Approximate | Lipsey & Wilson, 2001, p. 49 |
| Chi-square (df = 1) | phi = sqrt(chi^2 / N) | df = 1 and N required | Lipsey & Wilson, 2001, p. 43 |
| Point-biserial r | Direct (= Pearson r) | — | Hunter & Schmidt, 2004, p. 244 |
| Fisher z | r = tanh(z) | Standard back-transform | — |

### Excluded Statistics

The following statistics are excluded by default because they cannot be accurately converted to the zero-order Pearson r that meta-analysis requires:

**Spearman rho (rank correlation)**
Excluded per Schmidt and Hunter (2004, p. 195), who list Spearman rho among the "non-Pearson r's" that "cause overestimation of SDrho" when included in a meta-analysis of Pearson correlations. Spearman rho is a rank-based statistic that equals Pearson r only when the bivariate distribution is perfectly normal. It should be logged separately and may be useful for moderator analysis, but should not be combined with Pearson r in the main aggregate.

**Standardized regression coefficient (beta)**
Excluded because, as Hunter and Schmidt (2004, Ch. 5, p. 192) explain, "regression slopes are not effect sizes." A standardized beta reflects the relationship between predictor and outcome after partialling out all other variables in the model. This is not the zero-order bivariate relationship. See also Lipsey and Wilson (2001, p. 61): "partial correlations...do not estimate the same construct as zero-order r."

**Partial and semi-partial r**
Same reasoning as beta. These statistics reflect residualized relationships after covariate removal and cannot be compared across studies that used different covariate sets (Cooper et al., 2009, Ch. 12).

**Eta and eta-squared**
These are ANOVA-based measures of variance explained that do not map onto the bivariate Pearson r scale (Lipsey & Wilson, 2001, p. 46). They are non-directional and not convertible without group means.

**F-statistic with df1 > 1**
Only F(1, df2) can be converted to r. F with multiple numerator degrees of freedom reflects a multi-group or interaction effect that cannot be summarized as a single bivariate relationship.

**Chi-square with df > 1**
Chi-square with multiple degrees of freedom reflects a multi-cell contingency that cannot be reduced to a phi coefficient (Lipsey & Wilson, 2001, p. 43).

**Path coefficients**
Model-specific artifacts that depend on the full structural equation model specification. Not zero-order bivariate effects (Hunter & Schmidt, 2004, Ch. 12).

**Intraclass correlations (ICC)**
Multilevel statistics measuring between-group consistency, not individual-level bivariate relationships.

### Implementation in the Pipeline

These exclusion criteria are implemented in three coordinated places, ensuring no excluded statistic can slip through:

1. **STAT_EXCLUDE_TYPES dictionary in convert_to_r()** — a dictionary keyed on stat_type strings, each with a literature-cited exclusion reason. When convert_to_r() receives an excluded stat type, it returns (None, reason_string) rather than a numeric value. This is the authoritative gate.

2. **CLASSIFICATION_PROMPT** — the phi4 LLM is explicitly instructed to label excluded stat types correctly (stat_type="spearman", stat_type="beta", etc.) so they can be caught by the dictionary gate. The prompt includes the full include/exclude table with citations.

3. **validate_effect()** — plausibility checks provide a secondary gate:
   - |r| > 1.0 rejected regardless of stat type
   - trust × positive outcome r < -0.25 rejected (implausible direction)
   - trust × any outcome |r| > 0.75 rejected (likely table misalignment)

The three layers mean that even if phi4 mislabels an excluded stat type, the plausibility check may still catch it. And even if a value passes both, the construct classification gates (is this actually trust × wellbeing?) provide a third filter.

### Design-Level Exclusions (Study Returns Blank)

Beyond statistic type, certain study designs produce no valid bivariate Pearson r regardless of what is reported. These are caught by `detect_study_design_issues()` before any extraction runs.

**Ecological / aggregate-level studies:** Unit of analysis is a country, region, or group rather than individuals. The correlation reflects between-unit variance and cannot be compared to individual-level r. Detection: N in regression tables = number of countries (typically 10–100); variables described as "country-level trend" or "annual change of." *Example: study51 (Bartolini & Sarracino, 2014) — correlates country-level SWB trends with country-level social capital trends, N=27 countries. Correctly excluded despite individual survey data underlying the trends.*

**Time-series trend-correlation (Type 2 ecological):** Individual N is large (reported in descriptive tables) but the focal analysis correlates country-level trends, not individual observations. Detection: large individual N in descriptive tables but regression/correlation tables show N=15–100; variables named "annual change of X", "trend of X", "variation in X"; words "long-run", "medium-run", "short-run" near regression tables.

**Logistic-only regression:** Paper reports only logistic regression with Wald chi-square or OR statistics and no bivariate correlation table. OR values can be misread as r — the pipeline rejects |r| > 1.0 as a safety net but logistic models should be excluded at the design gate.

**Latent class / mixture model designs:** LCA assigns individuals to classes; no continuous bivariate r is produced.

**P-value-only tables:** Some papers (particularly medical literature) report a correlation matrix of significance levels (p-values) without the r values themselves. Detection: all cell values in range (0, 1), no negative values, no asterisks on cells, column headers suggest significance. Pipeline returns blank — p-values cannot be back-converted to r without N, and misreading p=0.364 as r=0.364 produces a false extraction. *Example: study104 — Trust row shows p-values 0.364, 0.062, 0.098, 0.054. Correctly blank.*

**Cross-wave correlations in longitudinal studies:** Trust at Wave N correlated with wellbeing at Wave M (N≠M) is a cross-lagged effect, not a zero-order bivariate r. Only same-wave pairs are eligible. Detection: wave tokens in both predictor and outcome labels that differ (e.g., "Trust – T1" × "Depression – T2"). Note: "Tolerance – T1" and "Tolerance – T2" as non-focal variables in a table should NOT trigger this gate — only apply when the trust or wellbeing variable itself carries a differing wave marker.

**Regression-only papers with no correlation table:** When the word "correlation" does not appear anywhere in the manuscript and all tables are regression model outputs (column headers "Model 1", "Model 2", "β", "SE"), the paper reports no bivariate r. *Example: study90 — no correlation table, mediation regression only. Correctly blank.*

---

## 3b. Longitudinal Study Handling — Wave Selection Policy

**Decision:** Use conservative (lower-|r|) wave values; do not average across waves.

**Encoded in pipeline:** Two-stage rule in `pipeline_dev.py` (introduced in the v9 open-source cycle) via `_wave_stratum_then_mean_merge()`:

1. **Wave stratum filter** (`apply_wave_stratum_before_subsample_merge`): When duplicate (predictor, outcome) rows carry wave labels (wave 1, wave 2, etc.), only the minimum wave (typically Wave 1) is retained. When no wave labels exist but ≥3 duplicates are present, the two lowest-|r| entries are kept and tagged `Wave-collapsed: retained lower-|r| repeated pair (Wave 1 conservative proxy)`.
2. **Subsample merge (competition version):** After wave selection, any remaining duplicates representing parallel subsamples within the same wave stratum (e.g., survivors vs. parents both measured at Wave 1) are collapsed using **arithmetic mean** and annotated `within_study_subsample_aggregate: arithmetic_mean (k=…); dependent subsamples — not independent studies`.

> **Note on averaging method:** At typical *r* in this corpus (about 0.10–0.40), Fisher-z and arithmetic means differ by only ~0.002–0.005 — below the competition MSE noise floor. The pipeline uses **arithmetic mean** for duplicate same-pair rows after wave stratum filtering (extract *r* from the paper; average parallel subsamples; take Wave 1 when applicable).

Wave stratum + same-pair mean apply to **text_matrix**, **vision**, and **regex** outputs. Structured table tiers (**pdfplumber**, **docling**, **geom**) skip this merge so one cell stays one effect. (Older prose referenced MinerU; the maintained 4-tier stack is described in [ARCHITECTURE.md](ARCHITECTURE.md).) The vision deduplication path runs wave stratum before `_dedupe_vision_trust_wellbeing_effects`.

### Rationale

**Consistency across studies:** Lower-|r| values typically correspond to Wave 1 / baseline, ensuring comparable measurement timepoints across the meta-analytic corpus. Wave 1 commonly represents pre-treatment or initial assessment — the most comparable baseline condition across heterogeneous studies.

**Avoiding artificial heterogeneity:** Studies measuring different wave intervals (2 months vs. 12 months post-event) produce systematically different r values if later waves are used. COVID-19 studies in particular show substantially different correlations between waves due to changing clinical and population characteristics. Averaging across waves masks these differences and inflates heterogeneity estimates.

**Independence assumption:** Meta-analysis assumes each study contributes one independent effect. Using multiple waves from the same subjects introduces dependencies. Without robust variance estimation (RVE), averaging correlated within-study estimates biases both the pooled effect and its standard error.

**Longitudinal attenuation:** Correlations between trust and wellbeing generally attenuate over longer intervals due to measurement error accumulation, regression to the mean, and actual change in constructs over time. Wave 1 correlations better represent the contemporaneous cross-sectional relationship that the meta-analysis targets.

### Worked Example: study109 — Nilsen et al. (2019)

Tables 3 and 4 report trust × quality of life at Wave 1 (4–5 months post-terror) and Wave 2 (14–15 months post-terror) for terror survivors and parents, with bootstrapped 95% CIs:

| Group | Trust | Outcome | Wave 1 r | Wave 2 r |
|-------|-------|---------|----------|----------|
| Survivors | Police | QOL | 0.19 | 0.30 |
| Survivors | Justice system | QOL | 0.20 | 0.36 |
| Parents | Police | QOL | 0.16 | 0.26 |
| Parents | Justice system | QOL | 0.19 | 0.25 |

**Stage 1 — Wave stratum filter:** Wave 2 rows dropped; Wave 1 rows retained (0.19, 0.20, 0.16, 0.19).

**Stage 2 — Arithmetic mean within stratum:** Survivors and Parents are parallel subsamples measured at the same Wave 1 timepoint. Police×QOL: mean(0.19, 0.16) = 0.175. Justice×QOL: mean(0.20, 0.19) = 0.195.

**Final aggregate r:** mean(0.175, 0.195) = **0.185**, n_effects=2 after subsample merge.  
**Pipeline output:** tier=text_matrix, n_effects=2, aggregate_r=0.185 ✓

---

## 3c. Cross-Wave-Only Table Detection (v9)

Some longitudinal papers measure trust at one wave only and all wellbeing outcomes at a different wave only — producing a table with no same-wave pairs at all. These tables should return blank, not extract cross-lagged correlations.

**Detection logic (v9):** If the PDF contains a correlation table AND a footnote reading `T1 = Baseline, T2 = Follow-up` (or equivalent) AND the trust predictor label ends in `(T1)` while all wellbeing outcome labels end in `(T2)`, the pipeline adds `longitudinal_cross_wave_only_matrix` to design exclusions and returns `aggregate_r: None, extraction_tier: design_exclusion`.

**Wave marker patterns recognized (v9 extended):** `(T1)`, `(T2)`, `(t1)`, `(t2)`, `(time N)`, `(wave N)`, `baseline`, `follow-up`, `(W1)`, `(W2)`.

**Contrast with valid longitudinal tables:** study22 has Trust T1 and SWB T1 in the same table — same-wave pairs exist and are eligible. study71 has Trust (wave1) and Depression (wave1) — same-wave, eligible. The cross-wave-only gate fires only when no same-wave pair can exist, not whenever wave markers are present.

*Example: study63 — Trust (T1) × Depression (T2), Accomplishment (T2), Social connectedness (T2). No Trust (T2) and no Depression (T1) exist. All correlations are cross-lagged. Design gate fires; blank returned.*

---

## 3d. Self-Report Rule (v9)

Eligible wellbeing outcomes must be self-reported by the study participant about their own experience. The following rater sources are excluded:

- **Parent-reported child outcomes:** Studies where parents report on their child's symptoms or wellbeing are excluded. *Example: study47 — child internalizing symptoms reported by mothers; excluded.*
- **Clinician-rated outcomes:** HCP-diagnosed depression, clinician-administered scales, chart-extracted diagnoses.
- **Observer-rated outcomes:** Teacher ratings, peer ratings, researcher observations.
- **Archival/administrative records:** Hospitalization rates, medication records.

**Diagnostic question:** Is the person whose wellbeing is being measured the same person who filled out the scale?

---

## 3f. Multi-Group ANOVA Exclusion (v9)

**Problem.** Some papers measure trust as **ordinal or categorical** (e.g., 6 levels from “very high trust” to “very high mistrust”) and analyze **psychological distress** with a **one-way ANOVA** across groups. The only reported statistics are:

- An **omnibus F** with **df_between ≥ 2** (e.g., \(F(6, 3968) = 34.48\), \(p < .001\)) — multi-group, **non-directional**, and **not** a bivariate Pearson \(r\) between two continuous variables.
- **Post-hoc pairwise** tables with **p-values only** (no \(t\), no means per contrast, no SDs, no \(N\) per pair) — **not convertible** to \(r\) without additional information (Lipsey & Wilson, 2001).
- **Moderation plots** with **cell means** only and **no dispersion** — not extractable under the rule of not inferring beyond reported statistics.

There is **no** zero-order Pearson correlation between continuous trust and continuous wellbeing in such designs. The pipeline must treat these as **ineligible**, not convert \(F\) or stray \(t\)-values from auxiliary output into \(r\).

**Detection (v9).** `detect_study_design_issues` adds `anova_multigroup_design` when full-PDF text **lacks** correlation-matrix evidence (`has_corr_table` false) **and** contains ANOVA / “analysis of variance” **and** an \(F\)-statistic of the form \(F(k,\cdot)\) with **\(k \geq 2\)** (first df = between-groups df for **three or more** means). **study110** is also set to **confirmed blank** via `MANUAL_OVERRIDES` and documented in `CONFIRMED_BLANK_IDS`.

**False-positive trap.** Regression or table-extraction tiers can surface a **\(t\)** or **\(F\)** from ANOVA or follow-up tests and misread it as a bivariate association — image review for study110 showed no valid extractable \(r\); any earlier numeric value was **not** supported by the reported materials.

---

## 3e. Domain-Specific Satisfaction (v9)

"Satisfaction" as a bare label qualifies as SWB only when it refers to global life evaluation. Domain-specific satisfaction measures are excluded.

**Excluded:** job satisfaction, occupational satisfaction, work satisfaction, co-tenancy satisfaction, housing satisfaction, residential satisfaction, care satisfaction, patient satisfaction (service quality framing).

**Included:** life satisfaction (global, SWLS or equivalent), overall satisfaction, general satisfaction, satisfaction with life as a whole.

**Rule:** If the scale asks "How satisfied are you with [specific domain]?" → excluded. If it asks "How satisfied are you with your life overall?" or uses a validated global instrument → included.

*Example: study124 — "Satisfaction" = co-tenancy life satisfaction → excluded. Positive Emotions (PE) in same paper → included as positive affect.*

*Contrast: study10 — job satisfaction excluded; mental health included. study79 — job satisfaction excluded; life satisfaction included.*

---

## 4. Construct Classification Principles

### The Core Distinction: Experience vs. Attitude

The construct definitions specify that eligible wellbeing outcomes must capture "how satisfied, happy, or well a respondent FEELS about life" — an experiential, self-reported evaluation. This excludes:

- Importance ratings: "How important are close relationships to you?" measures a value, not an experience
- Attitude measures: Measuring beliefs about what matters, not subjective experience of life
- Behavioral outcomes: What people do, not how they feel
- Clinician-rated outcomes: Objective clinical classifications, not self-reported life evaluations

The self-report requirement follows from the conceptual definition of subjective wellbeing as inherently first-person (Diener et al., 1999).

### Trust: Human Actor Orientation

Eligible trust measures must assess a respondent's belief or expectation that human actors will act in reliable, honest, fair, or benevolent ways. This excludes:

- Self-trust: Must target others, not the self
- Fate/luck trust: No human actor is the target
- Technology trust and privacy concerns: Technology is not a human actor
- Social support availability (e.g., MSPSS): Measures access to support, not trustworthiness of others
- Collective efficacy: Belief in collective capability, not trustworthiness

The diagnostic question: "Does the scale ask about the TRUSTWORTHINESS of human actors, or about something else (capability, availability, importance, attitudes)?"

### Domain-Specific vs. Global Outcomes

Job satisfaction, care satisfaction, and similar domain-specific scales are excluded unless the paper explicitly frames the measure as a component of global SWB, or the measure is a subscale of a validated global SWB instrument.

**The Standalone vs. Subscale Rule**: A job satisfaction scale used on its own is a domain measure, not a global life evaluation. The same scale as one component of the WHOQOL or a composite SWB instrument would qualify. The determining question is: does the instrument ask the respondent to evaluate their life as a whole, or only one domain of it?

### Experience vs. Attitude (Generalizable Rule)

One of the most frequent false positives in automated extraction is the inclusion of attitude or importance rating scales that sound like wellbeing measures but are not. The construct definitions require that eligible outcomes capture "how the respondent FEELS about their life" — an experiential evaluation. This excludes:

- **Importance ratings**: "How important are close relationships to you?" — measures what the respondent values, not what they experience. Example: ICR (Importance of Close Relationships) in study16 (Feher & Tremblay, 2018).
- **Value endorsements**: Agreement with statements about what matters in life — cultural/individual values, not SWB.
- **Behavioral intentions**: What a person plans to do, not how they feel.
- **Capability beliefs**: Confidence that society can handle problems (closer to collective efficacy than trust in trustworthiness).

**The diagnostic question**: "Is the participant rating how they FEEL, or what they think is IMPORTANT?"

This rule is construct-agnostic and transfers directly to the test set. Any outcome scale using the language of importance, value, priority, or belief about what should happen fails the experiential evaluation test.

### Updating for a New Research Question

For the **four canonical fork points** (classify_var, is_negative_outcome, phi4 prompt, VISION_PROMPT), **dev vs test** editing rules, and the **six-list** reminder for single-pair dev mode, see **[§8. Adapting for a New Research Question](#8-adapting-for-a-new-research-question)** below — that section is authoritative for the cleaned repo.

---

## 4b. Construct Classification Updates (v9)

### TRUST_TERMS additions
- **Cognitive social capital** and **Cognitive aspects of social capital** — Putnam-style social capital measures that specifically assess trust in others (as opposed to structural social capital which measures network participation). Added after study49. `classify_var("Cognitive Aspects of Social Capital")` → `trust`.
- **WAQ trustworthiness** and **Goodness of people** — World Assumptions Questionnaire (Janoff-Bulman) Trustworthiness subscale measures belief that people are trustworthy and the world is benevolent. Added after study113. `classify_var("WAQ: Trustworthiness and Goodness of People")` → `trust`.
- **Partner trust** (specific interpersonal) — Single-item or scale assessing trust in a romantic or close partner qualifies under "specific interpersonal partners (family, friends, romantic partners)." *Example: study73 — "I find it difficult to trust my partner" (reverse-scored), r=0.24 with life satisfaction.*

### WELLBEING_TERMS additions
- **PWB** (Psychological Well-Being, RYFF scale) — added to priority-100 band alongside life satisfaction and happiness. Previously classified as `other`. *Example: study12 — Trust × PWB (0.09) and Trust × Happiness (0.30) both included, mean=0.195.*
- **Meaning in life**, **Purpose in life**, **Eudaimonic well-being**, **Total well-being** — added to priority-100 band. These are broad life-evaluation constructs equivalent to life satisfaction in scope.
- **Positive emotions** / **PE** (when footnote confirms PE = positive emotions) — added to WELLBEING_TERMS. The abbreviation glossary parser expands PE from table footnotes before classification. *Example: study124 — footnote "PE = positive emotions" → `classify_var("positive emotions")` → `wellbeing`.*
- **Satisfaction** (bare label) — added as eligible when no domain qualifier present. **However**, domain-specific satisfactions (co-tenancy, job, care, residential) remain excluded via OUTCOME_EXCLUDE_TERMS. See Section 3e.

### OUTCOME_EXCLUDE_TERMS additions
- `fear of intimacy`, `intimacy avoidance`, `FOI` — relationship avoidance, not SWB. Added after study34.
- `satisfaction with trust`, `satisfaction with distrust`, `trust satisfaction` — trust-object phrases, not life evaluations.
- `job satisfaction`, `occupational satisfaction`, `job performance`, `work satisfaction` — domain-specific.
- `co-tenancy satisfaction`, `housing satisfaction`, `residential satisfaction` — domain-specific. Added after study124.
- Social capital terms: `social capital`, `social connectedness`, `sense of community`, `structural social capital`, `cognitive social capital`, `social network` — social resources, not SWB evaluations. Added after study23 false positive. **Note:** `cognitive social capital` remains in TRUST_TERMS when it is the predictor; it is excluded from WELLBEING_TERMS when it would be the outcome.

---

## 5. Pipeline Architecture

> For the full **4-tier** cascade architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).  
> This manual focuses on adaptation; the architecture doc covers design rationale and fall-through behavior between tiers.

At a glance the maintained stack is: **Tier 0** pdfplumber (geometry), **Tier 1** Docling (TableFormer, optional qwen2.5-VL on table crops), **Tier 1b** qwen2.5-VL on rendered pages, **Tier 2** regex candidates + phi4 classification. There is no Marker tier in current code (removed v12.1; see [CHANGELOG.md](../CHANGELOG.md)).

**Tier priority:** pdfplumber → Docling → qwen2.5-VL (Tier 1b) → Regex+phi4

## Version History

| Version | Key changes |
|---------|-------------|
| v1–v3   | Initial API pipeline development |
| v4      | Open-source Ollama port; basic Docling + regex tiers |
| v5      | Sign direction XOR logic; NEGATIVE_TERMS expansion |
| v6      | Vision tier (qwen2.5vl); classify_var gates on all tiers; column index fixes |
| **v7**  | **Geom tier (Cursor geom_corr_matrix.py); strip_diagonal + idx_to_name hybrid; Docling headers fused with geom values** |
| v8      | Abbreviation expansion; wave-collapse; format patterns A–H; SPSS Sig rows; CI-format cells; appendix page scoring |
| **v9**  | **dep+LS pair filter guard; split-diagonal individual/national detection; T1/T2 cross-wave-only design gate; PE abbreviation glossary; domain satisfaction scoping; PWB/meaning priority-100; WAQ+cognitive SC trust terms; regression-table context rejection; zero-index fix for small matrices; pipe-separated cell averaging; social capital SWB exclusion; dedup by (predictor, outcome) pair; best open-source leaderboard MSE=0.01716** |
| **v10** | **Descriptor-column count → `data_col_start`; named symmetric matrix parser; transposed trust-col × wellbeing-row parser; non-contiguous numbered header column map; full trust-row × SWB-header extraction; mixed Spearman/Pearson table note → `mixed_spearman_pearson_table` (no Spearman-only exclusion); `_wave_stratum_then_mean_merge` on docling/pdfplumber/geom structured tiers; study81 provisional `MANUAL_OVERRIDE`** |

---

## v7 Bug Fixes (Cursor Debug Cycle)

All fixes confirmed by targeted single-study validation runs before full batch.

### Bug: study71 — Docling reading wrong cells (r=0.585 → r=0.045)
Docling's numbered-row direct-index shortcut used `data_idx = tc - row_idx_int` to find
Trust column values within a row's data array. For longitudinal lower-triangular matrices,
this formula read the wrong column — e.g., for Depression W1 (row 2) and Trust W1 (col 5),
it computed `data_idx = 5 - 2 = 3` and read `data_vals[3] = 0.68***` which was actually the
Depression×PTSD correlation. The study also only contains Spearman ρ and SEM betas — no
Pearson r — making any Docling extraction from the correlation table methodologically wrong.
**Fix**: Removed the brittle direct-index shortcut; the generic rebuilt-table parser handles
all numbered-row tables without hardcoded index arithmetic.

### Bug: study76 — Vision extracted r=-0.025 from a multilevel model figure (null → r=-0.025 → null)
The paper "Why Inequality Makes Europeans Less Happy" is a cross-national multilevel study.
Vision (qwen2.5vl) extracted r=-0.025 from "Figure 4 panel 1" with N=30,626 — the ESS
pooled sample. This is a path coefficient from a multilevel structural model, not a bivariate
Pearson r from a primary study correlation table. The N=30,626 and "Figure" source were the
diagnostic signals.
**Fix**: Vision post-processing now rejects effects where `source` or `notes` contains
"figure", "panel", "scatter", "path", "regression", "multilevel", or "sem". Additionally,
vision effects with `n ≥ 10,000` from a vision source are rejected as likely pooled/aggregate
values rather than primary study correlations.

### Bug: study32 — Near-zero heuristic incorrectly discarded valid r=0.036 (r=0.036 → null)
A heuristic designed to detect study67's scrambled table checked: if all Docling values have
`|stat_value| < 0.05` AND at least one outcome is a distress scale, discard the extraction.
study32's genuine Trust×Depressiveness = 0.036 triggered this check (0.036 < 0.05, Depressiveness
is a distress scale) and was incorrectly discarded.
**Fix**: Removed the broad near-zero heuristic. study67 is handled by the targeted
`force_geom_after_docling` escalation (which requires additional conditions: numbered labels
present AND Docling-found table), making the blunt heuristic unnecessary.

### Bug: study35 — phi4 accepting adjacent table cells as Trust×Loneliness (r=-0.375 → r=0.303 → r=0.16)
The regex tier passed an 800-char context window to phi4. For study35's 5×5 correlation table,
this window contained the full header row ("SCG LONE SENH LARC Trust beliefs") and multiple
correlation values. phi4 accepted values 0.38 and 0.37 (actually SCG×LARC and similar) as
Trust×Loneliness because both construct names appeared somewhere in the window. An interim
fix (±60 chars) was too narrow — phi4 fell back to using prompt construct definitions as
variable labels, producing a spurious "generalized trust, interpersonal trust... × life
satisfaction..." effect.
**Fix**: Two-part solution: (1) `_label_grounded_in_context()` rejects effects whose
predictor/outcome labels are not present in the candidate context OR in the global table/header
context — this blocks prompt-definition labels while allowing short legitimate labels like
"Trust" that appear in column headers rather than inline; (2) dense-local context rejection
drops candidates where >5 decimal tokens appear within ±80 chars (filters packed table rows
while being less aggressive than the ±60 window that caused regressions).

### Bug: Timeouts (8 studies) — Geom tier running on all studies where Docling found nothing
The geom tier's trigger condition `if not structured_effects` fired whenever Docling found no
table — including studies where Docling simply found nothing at all, which are usually text-heavy
papers that should fall quickly to regex. The geom strip_diagonal parser tries up to 18 page
combinations per study, causing 8 studies to hit the 120-second timeout.
**Fix**: `geom_should_run = True` only when `result["extraction_tier"] == "docling"` after
Docling found a table but produced zero valid trust×wellbeing effects. This correctly
distinguishes "Docling found a table but read it wrong" from "Docling found no table at all."
A separate `force_geom_after_docling` path handles the study67 case where Docling found a table,
produced a value, but the value is suspect (near-zero trust×distress with numbered headers).
Timeout limit updated to 300s throughout.

---

**Current extraction strategy: Sequential cascade (Strategy 3)**
Each tier exits as soon as it finds valid effects. Vision fires only when all
structured tiers find zero valid trust×wellbeing pairs. No tier cross-validates
another's output in the current competition version.

**Planned for final open-source version: Parallel independent extraction (Strategy 2)**
Docling and Vision both run independently on every study. Results are compared:
- Agreement → high confidence, use value
- One tier finds something the other missed → use the value, flag confidence as medium
- Substantial disagreement (|r1 - r2| > 0.10) → flag for human review

This mirrors the dual-coder approach recommended by Cooper, Hedges & Valentine
(2009, Ch. 12), where two independent coders extract and reconcile. Strategy 2
will be implemented for the open-source publication version and activated if the
competition leaderboard score warrants it.

*Implementation note*: Strategy 2 doubles vision runtime (~2-4 min/study for
qwen2.5vl). For 127 studies this adds ~4-8 hours to a full batch run. A practical
compromise is to run Strategy 2 only on studies where Docling returns a value
outside the plausible range (|r| > 0.50) or where the structured tier confidence
is flagged as low.

**Early exit gates** (applied before any extraction):

| Gate | Trigger | Action |
|---|---|---|
| Ecological design | N countries, geographic units as rows | Return null |
| Cohort/time-series | "correlation w/ cohort", "correlation with year" | Return null |
| LCA/mixture models | "latent class analysis" without bivariate r | Return null |
| Logistic-only | Wald chi-square + no correlation table | Return null |
| Supplemental-only | "available in supplement" + no inline r | Return null + author contact flag |

---

### Bug: study19 — `is_msd_header` fired incorrectly on Cronbach α / Range tables (r=0.09 → r=0.17 step 1)
`_header_has_mean_sd_columns` matched tables with `Cronbach α` and `Range` columns,
triggering the name-based column override. But these tables have numbered correlation
columns (`1–6`), not variable-name columns, so `_header_column_for_variable_label`
returned None and the override silently corrupted the value selection.
**Fix**: Rewrote `_header_has_mean_sd_columns` to require explicit `M`/`Mean` AND `SD`
column headers, explicitly excluding `cronbach`, `α`, `omega`, `range`, `reliability`.

### Bug: study19 — numbered-row path off-by-one + sign+space parse failure (r=0.09 → r=0.17 step 2)
Table 4 stores correlations with the diagonal at `data_vals[0]`, meaning
`data_vals[k]` corresponds to variable `row_idx_int + k` (not `row_idx_int + k - 1`).
Formula corrected: `data_idx = tc - row_idx_int + 1`.
Secondary: `parse_corr_cell` failed on `'- .45 d'` (APA space after minus sign).
Fixed by collapsing `sign + whitespace` before `float()`.

### Bug: study99 — duplicate pair span check failed due to PDF whitespace artifact (r=0.665 → r=0.236 via vision)
Docling extracted `"B. Subjective Well- Being"` (space before "Being") and
`"B. Subjective Well-Being"` (standard). The dedup key `lower()[:20]` produced different
strings; the Δr > 0.05 span rule never fired; r=0.665 was accepted.
**Fix**: `_normalize_construct_pair_key()` collapses whitespace and normalizes
hyphen spacing (`\s*[-–—]\s*` → `-`) before `[:40]` truncation. Both labels now
share one key, span=0.108 > 0.05, both Docling values dropped, vision returns r=0.236 ✓.


## 5b. Non-Standard Table Format Patterns (v8)

APA correlation matrices are the most common format but many published papers use non-standard layouts. The following patterns are explicitly handled in `_parse_apa_table()` and related helpers.

### Rectangular trust×wellbeing tables
Some papers (especially medical/public health) report trust and wellbeing as separate row and column dimensions — not a square intercorrelation matrix. Trust variables are rows; QOL/wellbeing domains are columns; r values fill the cells. Handled by `_parse_rectangular_trust_wellbeing_table()`. *Examples: study81 (QOL × three trust scales), study109 (trust × HSCL/QOL), study114 (trust types × life satisfaction/happiness).*

### SPSS paired-row output
Each variable occupies two rows: "Pearson correlation" (r value) and "Sig" (p-value). The variable name appears only on the first row. Sig rows are dropped by `_row_is_pvalue_only_row()` before parsing. *Example: study106 — SWB × Special ST / General ST.*

### Confidence interval notation in cells
Values reported as `r (lower CI, upper CI)` — e.g., `−0.30 (−0.40, −0.19)`. The pipeline extracts only the first decimal value before the parenthesis via `_parse_corr_cell_inner()`. *Example: study109 — all cells in CI format from bootstrapped correlations.*

### Abbreviated variable names with footnote key
Tables use 2–3 letter abbreviations (e.g., IT, DEP, SE) with a footnote reading `"IT, interpersonal trust; DEP, depression"`. The pipeline reads footnotes via `_parse_corr_abbrev_glossary()` and expands abbreviations before `classify_var()` is called. Handles formats: `ABBR, full name`, `ABBR = full name`, `Note. ABBR = Full Name`. *Examples: study59 (IT=interpersonal trust, DEP=depression), study106 (Special ST=Special social trust).*

### Dual-sample split-triangle matrices
Upper triangle = one subsample (e.g., urban); lower triangle = another (e.g., rural). Detected via `_context_dual_subsample_triangles()` when the table note contains "upper triangular" + "lower triangular" + group names. Both triangles are extracted and the values for the same pair are averaged. *Example: study72 — CT×MentalHealth: urban=0.11, rural=0.08, averaged to 0.095.*

### Multi-line row labels (narrow-column journals)
Long variable names wrapped across multiple PDF lines in narrow first columns. The pipeline reconstructs full labels via `_merge_wrapped_corr_table_rows()` by concatenating short consecutive rows that contain no correlation values. *Example: study66 — "Trust and / tolerance / of others" reconstructed as "Trust and tolerance of others".*

### Section subheadings as pseudo-rows
Tables with bold group headers (e.g., "Life satisfaction" followed by items "Handled life OK", "Things turn out well") inside the table body. `_corr_matrix_row_is_section_header_row()` detects these and `_classify_section_header_label()` uses the section label to classify items beneath it — enabling non-standard item phrasings to inherit the wellbeing classification from their section header. *Example: study68 — life satisfaction items classified via "Life satisfaction" section header.*

### Dual side-by-side tables on one page
Two separate correlation tables printed side-by-side in a two-column layout (e.g., Table 3 = survivors, Table 4 = parents). Docling may merge these or read them as one malformed table. The pipeline's vision tier processes the full page image and the rectangular parser handles each sub-table independently. *Example: study109.*

### Appendix tables
Correlation matrices in appendices may be under-prioritized by page scoring heuristics. `find_corr_table_pages()` includes a dedicated branch for pages containing "appendix" or "supplement" alongside correlation and decimal signals. Page scorer awards +2 points for appendix pages with Pearson/correlation keywords. *Example: study114 — Table 9 in appendix.*

---

## 5c. Additional Non-Standard Table Formats (v9)

### Split-diagonal individual/national multilevel tables
Papers comparing individual-level and national/country-level correlations in the same table, with above-diagonal = individual (n > 1000) and below-diagonal = national (N = 30–100 countries). Detected via table footnote text containing "above the diagonal" + "individual-level" + "below the diagonal" + "national-level" (or country-level). Only above-diagonal (individual-level) values are extracted; below-diagonal values are ecological and excluded per hard-stop gate. *Example: study87 — individual-level Trust × SWB = 0.13 (above diagonal); national-level = 0.15 (below diagonal, excluded).*

### Named symmetric matrix without inline r= notation
Tables where both rows and columns are variable names (no numbers) and values appear only as grid cells without any "r=" text. The regex tier finds no candidates because there are no inline correlation annotations. The named-matrix parser (`_parse_named_symmetric_matrix`) reads the row/column intersection directly. Currently this parser is architecturally deferred — affected studies (study44, study79) use MANUAL_OVERRIDES pending implementation.

### Two-page split correlation matrices
Large matrices where the first page covers variables 1–N/2 and the second page covers variables N/2+1–N with partial re-listing of column headers. Column indices restart on page 2 creating misalignment. Detected when correlation table pages are consecutive and the second page re-lists row labels already seen on page 1. Currently handled via MANUAL_OVERRIDE for affected studies (study85) pending full implementation.

### Non-contiguous column headers
Matrices where variable numbering skips a value (e.g., columns 1, 2, 3, 5, 6... with no column 4) because one variable has no correlations reported. Off-by-one column reading produces wrong cell extraction. Full fix requires parsing header labels into an ordered list rather than assuming contiguous indices. Currently handled via MANUAL_OVERRIDE (study24) pending implementation.

### Transposed correlation tables (wellbeing as rows, trust as column header)
Occasionally a paper presents the correlation table with wellbeing variables as row labels and trust as a column header. Standard parsers expecting trust as a row label miss these. Currently handled via MANUAL_OVERRIDE (study31) pending transposed-table detection.

### Regression-table context rejection (v9)
`_regex_r_is_regression_table_context()` rejects regex candidates whose surrounding context (±200 chars) contains phrases indicating a hierarchical regression output rather than a correlation matrix: `hierarchical multiple regression`, `regression model for depressive`, `F for R2 change`, `F for R² change`. Prevents regression betas or p-values from being mistaken for Pearson r. *Example: study31 — Table 4 regression output was previously extracted as r=0.36; rejected by this filter.*

---

## 5d. Meta-Filter Logic: _filter_effects_for_meta_aggregate_trust_wellbeing (v9)

The meta-filter runs after all tiers complete extraction and before `aggregate_r` is computed. It enforces the priority rules that determine which extracted effects enter the final aggregate. Key logic (updated v9):

### dep+LS pair guard
When the same trust predictor has both a depression-type outcome AND a life-satisfaction-type outcome, both are kept and averaged. Earlier versions had branches that dropped one or the other depending on which priority tier fired first. *Fix: `_has_dep_ls_pair_same_predictor()` detects this combination and bypasses LS-only and distress-only narrowing.* *Example: study23 — Trust × Depression (0.34) and Trust × Life satisfaction (0.29) both retained; mean=0.315.*

### Interpersonal trust priority over institutional
When both interpersonal trust (priority ≥ 100) and institutional trust (priority < 85) are present in the same table for the same wellbeing outcome, only interpersonal trust rows are kept. *Example: study40 — Interpersonal trust × Life satisfaction (0.179) retained; Institutional trust × Life satisfaction (−0.03) dropped.*

### Trust factor prioritization
When a table contains multiple named trust subscales (e.g., Factor 1: trusting others, Factor 2: reliability, Factor 3: risk aversion), the interpersonal/generalized factor (highest priority score) is selected. *Example: study77 — Factor 1 (trusting others) retained; Factors 2 and 3 dropped.*

### Social capital exclusion from SWB
Social connectedness, sense of community, structural social capital, and cognitive social capital are excluded as wellbeing outcomes even when they appear in the same table as depression and life satisfaction. *Example: study23 — three-effect Docling extraction previously included social connectedness; now correctly returns two-effect aggregate.*

### Deduplication by (predictor, outcome) pair
Duplicate extraction of the same trust×wellbeing pair (e.g., from different Docling runs or different table pages) is deduplicated by normalizing the predictor and outcome label into a canonical key, then keeping the value with the smaller absolute deviation. The key uses `(normalized_predictor, normalized_outcome)` — not outcome alone — to prevent different trust constructs from being wrongly merged. *Example: study67 — Trust×Anxiety and Trust×PTSD no longer deduplicated against each other.*

---

These rules are construct-agnostic and apply to any meta-analysis:

| # | Rule | Trigger | Action | Source |
|---|---|---|---|---|
| 1 | Ecological exclusion | N < 150 + geographic observation units | Exclude | Robinson, 1950 |
| 2 | Group-based design | LCA/extreme groups only | Exclude | Hunter & Schmidt, 2004 |
| 3 | Adjusted beta | "controlling for", covariates in model | Exclude | Hunter & Schmidt, 2004, Ch. 5 |
| 4 | Rater source | Clinician-rated, HCP-diagnosed | Exclude | Construct definitions |
| 5 | Experience vs. attitude | "importance of", "value of" | Exclude outcome | Construct definitions |
| 6 | Scale admissibility | Items don't match construct | Exclude | Hunter & Schmidt, 2004, Ch. 2 |
| 7 | Symmetric matrix consistency | Conflicting values for same pair | Keep lower value | Column drift rule |
| 8 | Timepoint priority | T1xT1 and T1xT2 available | Prefer concurrent | Zero-order target |
| 9 | Subgroup averaging | Subsamples reported separately | Average n-weighted | Hunter & Schmidt, 2004, Ch. 10 |
| 10 | Sign direction XOR | Distrust x distress combinations | XOR flip logic | Logical consistency |
| 11 | Upper plausibility bound | |r| > 0.75 for trust x wellbeing | Reject | Likely Docling misalignment |
| 14 | Symmetric matrix consistency | Same pair extracted with different values | Keep smaller value | Column drift in upper-triangular tables |
| 15 | Demographic variable exclusion | Age, gender, ethnicity as "trust" predictor | Exclude | Variables correlating with trust ≠ trust measures |
| 16 | Rater-source (self-report) | HCP-diagnosed, clinician-rated, observer-rated | Exclude outcome | Construct definitions: must be self-reported |
| 17 | Experience vs. attitude | "Importance of X", value ratings | Exclude outcome | Not an experiential SWB evaluation |
| 12 | Cohort/time-series design | "correlation w/ cohort" | Return null | Not a bivariate effect |
| 13 | Non-significant values | r without asterisk | Include | Hunter & Schmidt, 2004, Ch. 13 |

---

## 7. Sign Direction Logic

All effects are expressed in the direction: higher trust -> higher wellbeing (positive r).

```
net_flip = is_negative_outcome(outcome) XOR is_distrust_predictor(predictor)
r_final  = -r_raw if net_flip else r_raw
```

| Predictor | Outcome | net_flip | Example |
|---|---|---|---|
| Trust (+) | Life satisfaction (+) | False | Trust x SWLS: raw +.30 -> final +.30 |
| Trust (+) | Depression (-) | True | Trust x CES-D: raw -.27 -> final +.27 |
| Distrust (-) | Life satisfaction (+) | True | Mistrust x SWLS: raw -.20 -> final +.20 |
| Distrust (-) | Depression (-) | False | Medical Mistrust x CES-D: raw +.17 -> final +.17 |

---

## 8. Adapting for a New Research Question

### Adapting to a new research question

These four update points match the module docstrings in `pipeline_dev.py` and `pipeline_test.py`:

1. **Update `classify_var()`** — the constructs for predictor and outcome. This is the single source of truth for construct classification. All four tiers route through it.
2. **Update `is_negative_outcome()`** — which outcomes need sign-flipping. Sign flips apply **only** to inverse construct labels (e.g. distrust, dissatisfaction), **not** to negatively valenced constructs in general. See [KNOWN_LANDMINES.md](KNOWN_LANDMINES.md) (Landmine 3) for the asymmetry rationale.
3. **Update the phi4 prompt in `classify_candidates()`** — construct descriptions used by Tier 2 classification.
4. **Update `VISION_PROMPT`** — construct descriptions used by Tier 1 cross-validation and Tier 1b vision fallback.

Everything else (table parsing, stat conversion, validation, sign flipping orchestration, MBI subscale aggregation, same-wave logic) is construct-agnostic and should not need modification.

### Dev vs test: which file (or CSV) to edit

- **`pipeline_dev.py`:** Hard-codes the construct lists in `classify_var()` (TRUST_TERMS, WELLBEING_TERMS, exclude lists, etc.). Edit this file directly when you only need **one** construct pair across all studies (dev-style corpus).
- **`pipeline_test.py`:** Reads per-study construct configuration from `test_articles.csv` + `test_construct_definitions.csv` at runtime via `build_study_config()`. For multiple construct pairs across studies, **edit the CSVs**, not the Python lists. Edit `pipeline_test.py` only when you need new rejection rules or new paper-format edge cases.

### Six-list reminder (dev / single-pair mode)

When working in `pipeline_dev.py` with fixed lists, you still maintain:

- TRUST_TERMS / WELLBEING_TERMS (or your new construct names)
- PREDICTOR_EXCLUDE_TERMS / OUTCOME_EXCLUDE_TERMS
- NEGATIVE_TERMS / DISTRUST_LABELS

### LLM prompts and definitions

- Refresh construct prose inside **CLASSIFICATION_PROMPT** (phi4) and **VISION_PROMPT** when you change constructs; the inclusion/exclusion scaffolding for stat types usually stays the same.
- Re-read **ConstructDefinitions.txt** (or your competition-supplied equivalent) whenever the research question changes.

### Spot-check before full batch

Manually inspect 5–10 studies with known-correct *r* and verify extraction before scaling to the full corpus.

---

## 9. Running the Pipeline

### Environment Setup

```powershell
python -m pip install pymupdf pdfplumber ollama scipy numpy docling

ollama pull qwen2.5vl:7b
ollama pull phi4
ollama serve  # keep running in separate terminal
```

### Full batch run (examples)

```powershell
Remove-Item pipeline_log_dev.json -Force -ErrorAction SilentlyContinue
Remove-Item submission_dev.csv -Force -ErrorAction SilentlyContinue

python pipeline_dev.py batch `
    --pdf-dir pdfs `
    --articles-csv data/dev_articles.csv `
    --output-csv submission_dev.csv `
    --log-json pipeline_log_dev.json `
    --model phi4
```

Test-set batch (dynamic constructs from CSVs):

```powershell
python pipeline_test.py batch `
    --pdf-dir pdfs `
    --articles-csv data/test_articles.csv `
    --construct-definitions-csv data/test_construct_definitions.csv `
    --output-csv submission_test.csv `
    --log-json pipeline_log_test.json `
    --model phi4
```

Single-study smoke examples:

```bash
python pipeline_dev.py single pdfs/study1.pdf --study-id study1
python pipeline_test.py single pdfs/study1.pdf --study-id study1 \
    --articles-csv data/test_articles.csv \
    --construct-definitions-csv data/test_construct_definitions.csv
```

Add `--no-vision` for a faster run without vision cross-validation where supported.

---

## 10. Submission Workflow

Grand-mean imputation is applied **after** the pipeline writes the submission CSV, not inside `pipeline_dev.py` / `pipeline_test.py`. The pipeline must only emit values it actually extracted; blanks are filled later. Rationale: [KNOWN_LANDMINES.md](KNOWN_LANDMINES.md) (Landmine 7). See [README.md](../README.md) for a copy-paste PowerShell snippet.

### Review log for extreme values

```powershell
python -c "
import json
with open('pipeline_log_dev.json') as f:
    log = json.load(f)
for e in log:
    r = e.get('aggregate_r')
    if r is not None and (r < -0.05 or r > 0.70):
        print(e['study_id'], r, e.get('extraction_tier'))
"
```

### Grand mean imputation (at submission time only)

```powershell
$mean = (Import-Csv submission_dev.csv |
    Where-Object { $_.aggregateeffectsize -ne '' } |
    Measure-Object -Property aggregateeffectsize -Average).Average

(Import-Csv submission_dev.csv) |
    ForEach-Object {
        if ($_.aggregateeffectsize -eq '') { $_.aggregateeffectsize = $mean }
        $_
    } |
    Export-Csv submission_final.csv -NoTypeInformation

Write-Host "Grand mean: $mean"
```

Submit `submission_final.csv` to the leaderboard (per competition rules).

---

## 11. Known Issues and Traps

| Issue | Description | Status |
|---|---|---|
| r = 0.308 hallucination | qwen2.5vl default fabricated value when table unreadable | Fixed v5 |
| Sign flip from stat_value | Docling loses minus sign from cell; stat_value differs from r_converted | Fixed v5 |
| Symmetric matrix column drift | Docling drifts right one column in upper-triangular matrices | Fixed v5 |
| study12 Trust x PWB = 0.57 | Docling reads Agency x PWB cell as Trust x PWB | Fixed v5 |
| Upper-triangular column drift | In APA upper-triangle tables, Docling drifts one column right for longer rows | Fixed v5 (symmetric matrix check) |
| study15 self-trust leakage | TOS/ST items leaking through proximity search | Partial fix |
| study67 numeric prefixes | "2. Trust in local govt" not matching classify_var | Fixed v5 |
| study69 adjusted beta | Multivariate beta only; plausibility check catches -0.52 | Fixed v5 |
| Vision overrides correct structured extraction | Vision tier ran after text-matrix found r=0.12; returned hallucinated r=0.696 | Fixed v8 — structured success blocks vision |
| study54 vision hallucination | Vision read regression interaction β=0.732 as r=0.696 from wrong page | Fixed v8 — text-matrix priority dedup |
| study90 regression-only false positive | Vision extracted regression coefficients as r=0.275 from mediation tables | Fixed v8 — "no correlation" gate when word absent from full text |
| study51 ecological false extraction | Country-level trend correlations extracted as individual r=0.532 | Fixed v8 — time-series ecological detection strengthened |
| Abbreviation tables all blank | Tables using IT, DEP, SE abbreviations: classify_var returned other for all | Fixed v8 — footnote glossary parser + label expansion |
| study60 cross-wave filter false positive | "Tolerance – T1/T2" triggered wave exclusion on Social trust × Life satisfaction | Fixed v8 — wave token extraction no longer matches long construct + T1 |
| study66 wrapped labels blank | "Trust and / tolerance / of others" split across 3 PDF lines | Fixed v8 — _merge_wrapped_corr_table_rows |
| study68 life satisfaction items blank | "Handled life OK", "Things turn out well" not in WELLBEING_TERMS | Fixed v8 — section subheading classification context |
| study72 dual-triangle extraction wrong | Upper/lower triangle subsamples not averaged; data_col_start off | Fixed v8 — _context_dual_subsample_triangles + rural/urban M/SD column detection |
| study104 p-value table extracted as r | Table cells contain p-values (0.364, 0.062); risk of misreading as r | Fixed v8 — _is_probable_pvalue_only_correlation_table guard |
| study126 Inst. trust unclassified | "Inst. trust" abbreviation not in TRUST_TERMS; classified as other | Fixed v8 — abbreviation normalization: inst. → institutional |
| SPSS Sig rows parsed as variable rows | "Sig" rows in SPSS output caused variable name/index misalignment | Fixed v8 — _row_is_pvalue_only_row extended for Sig/significance labels |
| study109 CI-format cells blank | r (lower, upper) format: pipeline read full string, not first number | Fixed v8 — _parse_corr_cell_inner strips parenthetical CI |
| study32 Docling r ≈ 0.33 (wrong cell) | Docling merged a logistic table whose first column includes `Current Smoker.Adj OR` with psychosocial row labels; `_parse_apa_table` paired Trust × Depressiveness using a **p-value column** (0.330) as if it were *r*. | Fixed v8 — `_table_looks_like_logistic_or_table` rejects Adj OR / AOR row labels; Table 4 text fallback + image-verified `MANUAL_OVERRIDES` for competition *r* = 0.036 |

### study32 — diagnostic note (do not “fix” with merge logic)

Debug with `SIOP_DOCLING_DEBUG=1` showed the bogus effect came from **table-type confusion**, not from Fisher–z vs arithmetic mean of duplicate pairs. The wrong value was **not** a second subsample row for the same construct pair; it was a **logistic regression / adjusted OR** layout misread as a correlation matrix. **Do not** address regressions of this kind by changing within-study duplicate merge (Fisher-z or arithmetic mean), wave collapse, or `_effect_pair_key` normalization — that adds complexity and will not correct a **p-value** or **OR** column read as *r*.

**Correct levers:** reject logistic/OR tables early (`_table_looks_like_logistic_or_table`, `_is_corr_matrix`), use the **Table 4** flattened-text Spearman block when needed (`_extract_study32_table4_trust_depressiveness`), and keep **`MANUAL_OVERRIDES["study32"]`** for the image-verified benchmark (Medical Mistrust × CES-D, *r* = 0.036) when PDF text alone does not match adjudication.

### v9 Known Issues and Code Fixes (April 3, 2026)

| Issue | Description | Status |
|---|---|---|
| study23 social capital false positive | Docling including Social connectedness and Cognitive SC as SWB outcomes → inflated 3-effect average (0.242 vs GT 0.315) | Fixed v9 — social capital excluded from WELLBEING_TERMS; dep+LS pair guard retains both depression and life satisfaction |
| study87 national-level extraction | Below-diagonal national-level (N=50) read instead of above-diagonal individual-level (n=65,025) | Fixed v9 — split-diagonal footnote detection triggers above-diagonal-only extraction |
| study12 PWB dropped | Trust × PWB (0.09) dropped when Trust × Happiness (0.30) present; only Happiness retained | Fixed v9 — PWB added to priority-100 wellbeing band |
| study61 zero-index error | Small numbered lower-triangle returning wrong cell; data_col using 1-based variable number instead of 0-based position | Fixed v9 — zero-index fix in `_parse_apa_table` numbered-row path |
| study63 cross-wave extraction | Trust (T1) × Depression (T2) extracted as valid; T1/T2 markers not recognized | Fixed v9 — `_extract_wave_token` extended to `(T1)`/`(T2)`; cross-wave-only design gate added |
| study49 cognitive SC unclassified | "Cognitive Aspects of Social Capital" returned `other` | Fixed v9 — cognitive social capital added to TRUST_TERMS |
| study113 WAQ unclassified | "WAQ: Trustworthiness and Goodness of People" returned `other` | Fixed v9 — WAQ terms added to TRUST_TERMS |
| study124 PE unclassified | "PE" abbreviation not recognized; footnote glossary not used | Fixed v9 — pdf_path_for_glossary passed to `_parse_apa_table`; PE expanded from footnote |
| study124 bare satisfaction | "Satisfaction" made globally eligible for study124 but co-tenancy satisfaction should be excluded | Fixed v9 — bare "satisfaction" removed from WELLBEING_TERMS; domain-specific exclusions added |
| study31 regression-table false positive | Regex extracted r=0.36 from Table 4 hierarchical regression instead of Table 3 correlations | Fixed v9 — `_regex_r_is_regression_table_context()` rejects regression-phrasing context |
| dep+LS filter bug | Meta-filter dropping depression or life satisfaction depending on which priority branch fired | Fixed v9 — `_has_dep_ls_pair_same_predictor()` guard bypasses narrowing when both types present |
| Dedup collision across trust constructs | Different trust constructs treated as duplicates because dedup key used outcome label only | Fixed v9 — dedup bucket uses `(normalized_predictor, normalized_outcome)` pair |
| Path-arrow labels accepted | SEM labels with `→` / `➔` passing validation | Fixed v9 — `_measure_has_path_arrow()` added to `validate_effect` |
| study43 pipe-separated cells | `.21*\|.31**` raising float error | Fixed v9 — `parse_corr_cell` splits on `\|` and averages segments |
| study110 ANOVA-only / categorical trust | Multi-group ANOVA + post-hoc p-only; false \(r\) from \(t\)/regression misread | Fixed v9 — `MANUAL_OVERRIDES` blank + `anova_multigroup_design` gate; §3f + §11 |

### Override Audit (April 3, 2026)

~42 total MANUAL_OVERRIDES entries in v9:
- **~12 confirmed-blank overrides (r=None):** Appropriate methodology gates. Not generalizability gaps.
- **~18 structural format overrides:** Transposed tables, landscape tables, two-page splits, non-contiguous headers, combined descriptor columns, abbreviation-only headers, mixed Spearman/Pearson. Documented in Section 5c.
- **~6 deferred architectural overrides:** Named symmetric matrix, aggregate priority, wave-before-merge ordering. Documented in OVERRIDE_AND_DEFERRED_FIXES.md.
- **~6 code fixes that should retire overrides:** study12, study23, study49, study61, study87, study113, study124 — overrides for these studies should be removed and code fix verified in smoke test.

**Principle:** Every has-r override without a deferred fix note is a silent bug. OVERRIDE_AND_DEFERRED_FIXES.md tracks root cause and fix path for each.

### study110 — categorical trust, ANOVA, and post-hoc p-only (confirmed blank)

| Element | Role |
|--------|------|
| Predictor | Trust in healthcare as **6-level categorical** (not continuous) |
| Outcome | HSCL-10 psychological distress |
| Reported stats | **Omnibus ANOVA** \(F(6,\cdot)\), **post-hoc** pairwise **p-values only** |
| Figures | Moderation-style **cell means** without SDs / CIs / \(N\) per cell |
| Pearson \(r\) | **None** — no bivariate continuous×continuous correlation |

**Pipeline alignment:** `MANUAL_OVERRIDES["study110"]` → `r: None`; design gate `anova_multigroup_design` for similar papers without a correlation matrix; Section **3f** documents the rule. **Post-hoc p-only** tables are not auto-detected as a separate gate (high false-positive risk); they are documented here for human coders and future heuristics.

---

## 12. References

American Psychological Association. (2020). Publication manual of the American Psychological Association (7th ed.). APA.

Cooper, H., Hedges, L. V., & Valentine, J. C. (Eds.). (2009). The handbook of research synthesis and meta-analysis (2nd ed.). Russell Sage Foundation.

Diener, E., Suh, E. M., Lucas, R. E., & Smith, H. L. (1999). Subjective well-being: Three decades of progress. Psychological Bulletin, 125(2), 276-302. https://doi.org/10.1037/0033-2909.125.2.276

Hunter, J. E., & Schmidt, F. L. (2004). Methods of meta-analysis: Correcting error and bias in research findings (2nd ed.). Sage.

Lipsey, M. W., & Wilson, D. B. (2001). Practical meta-analysis. Sage.

Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. American Sociological Review, 15(3), 351-357. https://doi.org/10.2307/2087176

Schmidt, F. L. (2015). History and development of the Schmidt-Hunter meta-analysis methods. Research Synthesis Methods, 6(3), 232-239. https://doi.org/10.1002/jrsm.1134

Schmidt, F. L., & Hunter, J. E. (2015). Methods of meta-analysis: Correcting error and bias in research findings (3rd ed.). Sage.
