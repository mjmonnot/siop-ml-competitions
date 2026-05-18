# Postmortem — One Hot Key @ SIOP 2026 ML Competition

> A retrospective on the `2026-meta-analysis` pipeline: what it scored, how the
> first-place solution scored ~6× better, why the gap existed, and what is worth
> keeping versus rebuilding.

| | One Hot Key | goforit (1st place) |
|---|---|---|
| Team | Solo developer + agents | Nga Do & Michael Hazboun (U. Minnesota) |
| Automated test MSE | ~0.0589 | **0.009233** |
| Manual-coded submission | 0.0351 | — |
| Core approach | 4-tier local extraction cascade | 3-call frontier-model prompt pipeline |
| Models | pdfplumber · Docling · qwen2.5-VL · phi4 | `claude-opus-4-6` only |
| Runs offline | Yes | No (paid API) |

The fair comparison is automated-vs-automated: **0.0589 vs. 0.009233**, a factor
of roughly 6.4. The 0.0351 figure was a hand-coded submission and was not a
comparable fully-automated entry under the competition rules. This document does
not soften that. The point of a postmortem is to find out why.

---

## 1. The task, briefly

The competition asked for a fully automated, single-file pipeline that takes a
research article (PDF) plus a predictor/outcome construct definition and returns
one aggregate effect size — Pearson's *r* — per study. The held-out ground truth
for each study was defined as the **average of the relevant observed
correlations** in that paper. Submissions were scored by mean squared error
against the hidden test set: 66 papers spanning 23 construct pairs in I-O
psychology (the dev set was 127 trust × subjective-wellbeing studies).

Two design constraints shaped everything: the pipeline had to live in a single
file, and it had to use publicly available models or APIs — paid APIs explicitly
allowed.

---

## 2. What One Hot Key built

A **local extraction cascade** with four tiers, designed to run entirely on a
single workstation (RTX PRO 3000 Blackwell, 12 GB VRAM):

- **Tier 0 — pdfplumber.** Direct text and table extraction.
- **Tier 1 — Docling.** Layout-aware document parsing for papers Tier 0 mangled.
- **Tier 1b — qwen2.5-VL (via Ollama).** Vision fallback for tables that survived
  neither prior tier.
- **Tier 2 — Regex + phi4.** Pattern matching plus a small local LLM to identify
  zero-order correlations and apply sign logic.

The design philosophy was deterministic, auditable, methodologically strict, and
fully offline. It enforced real discipline: no study-specific hardcodes, unit
tests passing after every change, Unicode normalization before parsing, XOR sign
logic for inverse-labeled constructs, and a hard exclusion of anything that was
not a true zero-order correlation — partial correlations, regression betas,
ANCOVA tables, factor loadings, cross-lagged wave pairs.

It scored ~0.0589 fully automated.

---

## 3. What goforit built

No PDF parsing at all. No local models. No cascade.

goforit base64-encodes the **entire PDF** and sends it directly to the Claude API
(`claude-opus-4-6`) as a document block. The whole ~1,630-line program is three
things: prompt templates, orchestration, and one genuinely substantial piece of
deterministic code. The flow is three model calls per study:

1. **Screening.** Claude reads the Methods section and returns structured JSON:
   every measured variable, which ones match the predictor/outcome definitions,
   and a `negative_pole` flag judged from the *final scored variable* direction
   (after any reverse-scoring) — not from item content.
2. **Extraction.** A second call extracts statistics only for the matched
   variables, following a strict priority hierarchy: Level 1 bivariate
   correlations, Level 2 derivable statistics (contingency tables, group
   means/SDs, ordinal groups, t/F/χ², p-only), Level 3 adjusted regression
   coefficients as a last resort. It stops at the first level that yields
   anything.
3. **Quality check** (optional). A third call re-reads the PDF, scores the
   extraction for completeness/accuracy/reverse-coding, and can add missed pairs.

The only "real" computation is `convert_to_r`: deterministic meta-analytic
conversions from t, F, d/g, β, b/AME, OR/log-OR, χ², η², group means/SDs (with a
Feldt extreme-groups correction), ordinal groups, 2×2 contingency, and p-only
values. Aggregation is a plain mean of the per-study *r* values. A fallback
variant imputes the construct-pair mean when a study yields nothing.

It scored 0.009233 fully automated — first place. The top four solutions clustered
tightly between 0.0092 and 0.0129, which suggests this general shape converged
across strong teams.

---

## 4. Side-by-side

| Dimension | One Hot Key | goforit |
|---|---|---|
| PDF ingestion | 4-tier parse → transcribe → degrade | Native PDF to frontier model |
| Reasoning model | phi4 + qwen2.5-VL (local, small) | Opus (frontier) |
| Variable matching | Heuristic / regex | LLM screening pass over Methods |
| Effect-size scope | Zero-order only; everything else excluded | Graceful fall-through, t/F/OR/etc. converted |
| Empty studies | Frequent → imputation at submission | Rarer; pair-mean fallback in-pipeline |
| Self-correction | None | Dedicated QC call |
| Cost | ~Free after hardware | 3 Opus calls × full PDF per study |
| Determinism | High | Low (no ensembling) |
| Offline / private | Yes | No |
| Auditability of *why* | Inspectable intermediate tiers | Logged rationale only |

---

## 5. Diagnosis — why the gap was 6×, not 6%

**The lossy front-end was the single biggest cost.** Every tier in the cascade
transcribes the document before any reasoning happens. pdfplumber mis-joins
matrix cells; Docling caps pages under VRAM pressure and loses content; a 7B-class
VLM mis-reads cell intersections. By the time phi4 does extraction, it is
reasoning over already-degraded text. goforit never degrades the document — the
native PDF, layout and tables intact, goes straight to a model strong enough to
read it. The hardest sub-task in this whole problem, "which cell is the
intersection of row 7 and column 3 in a dense lower-triangular matrix," is
sidestepped rather than solved.

**Model quality decided the judgment calls.** Zero-order vs. partial, which row
× column, reverse-coded or not — these are reasoning calls, and goforit routed
them to Opus while One Hot Key routed them to small local models. This is also
the likely explanation for the One Hot Key 0.0351-vs-0.0589 gap: a frontier model
closes most of the distance between hand-coding and the automated pipeline.

**Hard exclusion hurt more than it helped.** Zero-order purity is the
methodologically correct stance for a meta-analyst. But MSE does not reward
methodological purity — it rewards proximity to a number. A paper that reports
only a β, converted approximately to *r*, is closer to ground truth than an empty
cell backfilled by a grand mean. goforit's Level 2/3 fall-through harvests a
usable estimate from papers One Hot Key left blank.

**In-pipeline imputation was pragmatic, not impure.** One Hot Key's golden rule
was that grand-mean imputation belongs at submission time only. goforit folds a
construct-pair mean fallback directly into the automated pipeline. Given that the
rules required one fully-automated file, making the imputation automated and
generalizable was the correct call for this task.

---

## 6. What One Hot Key got right

This is not a story of doing everything wrong. Several decisions hold up well and
should be carried forward:

- **Engineering discipline.** No hardcodes, unit tests gating every change, a
  documented landmine registry, a changelog. goforit's submission is a single
  file and a short README; the One Hot Key repo is a far better *teaching*
  artifact, and that was always a stated goal of this project.
- **Methodological seriousness.** The Unicode-normalization requirement, the XOR
  sign logic for inverse labels, the explicit zero-order definition — these are
  correct, and they would still be correct in any rebuild. The mistake was the
  *hard exclusion*, not the *understanding*.
- **Local-first as a deliberate constraint.** Offline, free, private, deterministic
  — these are real virtues in many production settings, even though they were not
  what this particular MSE-scored competition rewarded.
- **The handoff discipline.** The overnight iteration log, the division of labor,
  the "diagnose once, write one comprehensive prompt" rule — process held up even
  when the architecture did not.

---

## 7. What we would do differently

1. **Stop parsing. Send the PDF.** The parsing layer was buying less than it cost.
   A frontier model reading the raw document outperforms a cascade of local
   parsers feeding small models — not marginally, by an order of magnitude.
2. **Decompose into prompt stages, not parser tiers.** Screen → extract → verify.
   Each stage becomes a testable prompt with structured JSON output, and the dev
   set plus `score.py` becomes the prompt-iteration harness.
3. **Convert, do not exclude.** Keep zero-order as the *preferred* level, but add
   a deterministic fall-through (t/F/d/OR/contingency/p-only → *r*). An
   approximate number beats an empty cell under MSE.
4. **Ground every extraction.** Require the model to quote the verbatim source
   cell, then run a cheap deterministic check that the quoted number appears in
   the page text. This is the main defense against confident hallucination — the
   worst failure mode for a squared-error metric.
5. **Add self-consistency.** Frontier extraction is non-deterministic; goforit had
   no ensembling and still won, but running extraction N times and taking the
   median is cheap insurance and a likely further MSE gain.
6. **Keep the local pipeline as the documented contrast.** It is still the better
   teaching object. The repo's value is showing *both* paths and why one won.

---

## 8. The bigger lesson

An earlier design question in this project was whether the problem should have
been approached through prompt engineering rather than an ML pipeline. The
competition answered it. goforit *is* the prompt-engineering approach — three
prompts, one deterministic conversion function, no extraction infrastructure at
all — and it won decisively.

The transferable principle: for document-extraction tasks where the input is
heterogeneous and the real bottleneck is *judgment*, the quality of the model
doing the reading dominates the sophistication of the pipeline feeding it. One
Hot Key spent its effort building robust infrastructure to compensate for weak
local models. The winning move was to remove the infrastructure and use a strong
model directly.

The local-first pipeline was good engineering and excellent pedagogy. It was the
right answer to a different question than the one the competition was asking.

---

## 9. Credits & links

- **First place — goforit:** Nga Do and Michael Hazboun, University of Minnesota.
  Solution code:
  <https://github.com/izk8/2026_SIOP_Machine_Learning_Competition/tree/main/01_goforit>
- **Competition organizers:** Ivan Hernandez (Virginia Tech), Isaac Thompson and
  Egyn Zhu (Amazon).
- **Full results and decks:**
  <https://github.com/izk8/2026_SIOP_Machine_Learning_Competition>

*One Hot Key — `[0,0,1,0,0]`.*
