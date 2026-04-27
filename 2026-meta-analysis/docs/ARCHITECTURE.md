# Pipeline architecture — 4-tier extraction cascade

This document explains the four-tier extraction stack, why each tier exists, and what falls through to the next when a tier comes up empty. Read it before diving into [`pipeline_dev.py`](../pipeline_dev.py) or [`pipeline_test.py`](../pipeline_test.py). For silent-failure traps and the guards that prevent them, see [`KNOWN_LANDMINES.md`](KNOWN_LANDMINES.md).

---

### 1. Overview

The pipeline extracts zero-order bivariate Pearson *r* values from academic PDFs in industrial-organizational psychology. Each PDF passes through a cascading **4-tier extraction stack**: earlier tiers are cheap and high-precision; later tiers are expensive and recover what slips through. A high-confidence value at any tier short-circuits the rest. Branching details live in `process_study()` and related helpers in [`pipeline_dev.py`](../pipeline_dev.py) and [`pipeline_test.py`](../pipeline_test.py).

---

### 2. The Cascade

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  PDF input                                                      │
    │    │                                                            │
    │    ▼                                                            │
    │  ┌──────────────────────────────────────────┐                   │
    │  │ TIER 0   pdfplumber                       │  geometric tables │
    │  └──────────────────────────────────────────┘                   │
    │    │  no high-confidence value                                   │
    │    ▼                                                            │
    │  ┌──────────────────────────────────────────┐                   │
    │  │ TIER 1   Docling + qwen2.5-VL crop CV     │  ML TableFormer   │
    │  └──────────────────────────────────────────┘                   │
    │    │  no high-confidence value                                   │
    │    ▼                                                            │
    │  ┌──────────────────────────────────────────┐                   │
    │  │ TIER 1b  qwen2.5-VL on rendered page      │  vision fallback  │
    │  └──────────────────────────────────────────┘                   │
    │    │  no high-confidence value                                   │
    │    ▼                                                            │
    │  ┌──────────────────────────────────────────┐                   │
    │  │ TIER 2   Regex candidates + phi4 classify │  text-layer pass  │
    │  └──────────────────────────────────────────┘                   │
    │    │                                                            │
    │    ▼                                                            │
    │  Aggregation → validation → submission row                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

PyMuPDF (fitz) sits underneath every tier as the page-text streamer, page rasterizer, and region cropper.

---

### 3. Per-tier deep-dive

#### Tier 0 — pdfplumber: geometric table detection

**What it is**  
Coordinate-aware PDF table parsing (pdfplumber): cell bounding boxes reconstruct rows and columns; no ML.

**When it fires**  
Every PDF, first.

**What it returns**  
List of cell-grid candidates with cell adjacency preserved; confidence semantics follow geometric and consistency checks in the pipeline.

**When it falls through**  
Multi-column papers, rotated pages, image-only tables, or PDFs with no text layer.

**Why this tier exists**  
Most academic correlation tables have regular bordered geometry. When that is true, geometric extraction is exact, fast, and free. There is no reason to wake an ML model.

#### Tier 1 — Docling: ML TableFormer (with optional qwen2.5-VL cross-validation)

**What it is**  
IBM's Docling library uses TableFormer (a transformer trained on academic table layouts) to parse complex or combined-cell tables that pdfplumber's geometric model cannot handle.

**When it fires**  
After Tier 0 yields nothing high-confidence.

**What it returns**  
Structured cell grid with header inference; confidence optionally cross-checked by passing the table crop through qwen2.5-VL.

**When it falls through**  
Scanned PDFs without an extractable text layer; tables embedded as raster images.

**Why this tier exists**  
Combined headers, rotated multi-row labels, and the APA-style intercorrelations matrix archetype defeat geometric extraction. TableFormer was trained on that distribution.

#### Tier 1b — qwen2.5-VL via Ollama: visual page parsing

**What it is**  
A local vision-language model (Alibaba Qwen2.5-VL 7B, served by Ollama) that reads rendered page images and returns structured JSON cells.

**When it fires**  
When neither Tier 0 nor Tier 1 returned a candidate (see `process_study()`).

**What it returns**  
Same structured cells as upstream tiers, with the source flagged as vision so downstream sign-flip and same-wave logic can apply.

**When it falls through**  
When the model returns malformed JSON, refuses the task, or returns no cells.

**Why this tier exists**  
Scanned PDFs and image-embedded tables have no text layer for upstream tiers. Vision is the only fallback that works on these.

#### Tier 2 — Regex candidates + phi4 classification

**What it is**  
Two-stage filter: construct-aware regex walks PyMuPDF-extracted text and table cells to gather every plausible *r* candidate with surrounding context; phi4 (Microsoft 14B, Ollama) reads each candidate against construct definitions and decides which row × column pair the value belongs to.

**When it fires**  
When upstream tiers returned no high-confidence candidates, or as a parallel sweep to recover values that were missed (depending on invocation context — see `process_study()`).

**What it returns**  
Classified cell tuples with confidence labels.

**When it falls through**  
When regex finds no candidates, or when phi4 rejects every candidate as off-target.

**Why this tier exists**  
Some correlations are reported in prose, not tables. Regex finds them; phi4 disambiguates which construct pair they describe. Regex's job is recall; phi4's job is precision — a role no rule-based pattern can do reliably alone.

---

### 4. The PyMuPDF substrate

PyMuPDF (imported as `fitz`) is not a tier; it is the underlying PDF reader every tier depends on.

Tier 0: pdfplumber wraps PyMuPDF for text streaming. Tier 1: Docling consumes PyMuPDF page objects. Tier 1b: PyMuPDF rasterizes pages at the resolution qwen2.5-VL expects. Tier 2: regex runs on PyMuPDF-extracted text and table-region cells.

If PyMuPDF fails to open a PDF, the entire cascade fails for that paper. That is intentional. There is no pre-Tier-0 fallback because every other tier needs structured access to the document.

---

### 5. Design principles

- **Blank entries are penalized as heavily as wrong values.** The competition metric is MSE against a hidden ground truth. A blank submission row gets the grand-mean-imputation penalty, which is roughly equivalent to a wrong-by-0.15 prediction. The cascade errs toward inclusion: extract if you can; only blank if every tier fails.

- **`classify_var()` is the single source of truth for construct classification.** All four tiers route their candidate values through `classify_var()`. This is the only place that decides whether a row labeled X is actually the trust variable (or the active study's predictor/outcome). It must remain centralized; any divergence between tiers becomes a silent bug. See [`KNOWN_LANDMINES.md`](KNOWN_LANDMINES.md) (Landmines 2 and 6).

- **Grand mean imputation belongs at submission time, not in the pipeline.** The pipeline must never write a non-blank value it did not actually extract. The competition's MSE U-curve is flat at the optimum (~0.152), so imputation is not a pipeline lever; it is a post-processing step applied to the final CSV. Mixing imputation into pipeline logic creates un-debuggable false positives. See [`KNOWN_LANDMINES.md`](KNOWN_LANDMINES.md) (Landmine 7).

---

### 6. Dev vs test pipeline difference

**`pipeline_dev.py`** targets the dev set: 127 studies, all measuring trust × subjective wellbeing. Construct classification uses hard-coded TRUST and SWB term lists in `classify_var()`. MSE **0.013641**, dev rank **6/10**, submission label **submission_v11_study59**.

**`pipeline_test.py`** targets the test set: 66 papers covering 23 different construct pairs. Construct classification uses a thread-local study config built from `test_articles.csv` and `test_construct_definitions.csv` at runtime — `classify_var()` consults the active study's config first, falling back to the dev term lists only if no config is present. MSE **0.0351** (best as of 2026-04-11).

Dynamic mode only fires when `build_study_config()` returns non-`None`, which only happens when both `Construct1` and `Construct2` are non-empty in the CSV row. If either is missing, dynamic mode silently stays False — see [`KNOWN_LANDMINES.md`](KNOWN_LANDMINES.md) (Landmine 2).

---

### 7. Library choice rationale

- **pdfplumber over Camelot / Tabula.** pdfplumber exposes per-cell bounding boxes cleanly; the pipeline uses that geometry for proximity-aware reasoning. Camelot stream/lattice modes are less flexible when **near** neighbors matter as much as strict grid adjacency.

- **Docling over pdfminer / AdvancedPDF.** TableFormer is the only locally runnable academic-table model in this stack. A hypothetical 32B+ layout tier does not fit **12 GB** VRAM.

- **qwen2.5-VL:7B over GPT-4V / Claude vision.** Tier 1b runs in every batch; remote APIs would violate the local-only constraint and explode cost. Qwen2.5-VL 7B is the strongest open-weights vision model that still fits the 12 GB budget.

- **phi4 over qwen2.5:72b for classification.** qwen2.5:72b at Q4 needs ~40 GB VRAM, over 3× the workstation's 12 GB. phi4 (~14B, ~9 GB Q4) is the largest model that stays fully GPU-resident; the 72B class would spill to CPU and lose ~10× throughput. See [`CHANGELOG.md`](../CHANGELOG.md) for version context.

- **Marker (initially considered, abandoned).** Marker is scaffolded in the codebase but never executes — `MARKER_AVAILABLE = False` is hardcoded because Marker requires Python ≤3.13 and the workstation runs 3.14. The dead `extract_via_marker()` function was removed in the v12.1 cleanup pass. See [`CHANGELOG.md`](../CHANGELOG.md) (v12.1).

---

### 8. What is NOT in this pipeline

- **Effect-size meta-analytic synthesis.** Only zero-order bivariate r extraction lives here. Random-effects pooling, publication-bias diagnostics, etc. belong downstream of the submission CSV.

- **Numeric confidence intervals on r.** Confidence is a **categorical** tier/source flag (high / medium / low), not a frequentist interval on the extracted coefficient.

- **Cross-paper reconciliation.** Each PDF is independent. No pass deduplicates "Study 2" across manuscripts.

- **PDF repair or re-authoring.** Inputs are read-only. Layout pathologies are handled by the cascade tiers or yield blanks — the source files are never rewritten.


