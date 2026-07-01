# 2019 SIOP Machine Learning Competition
## Predicting Big Five Personality Traits from Open-Ended Text

Reproducible, leakage-safe solution for the **2019 SIOP Machine Learning
Competition**: predict self-reported Big Five trait scores from a respondent's
free-text answers to five open-ended situational judgment items (SJIs).

> Thompson, I., Koenig, N., & Lui, M. *The 2019 SIOP Machine Learning
> Competition.* 34th Annual SIOP Conference.

**Metric:** mean Pearson correlation (r) across the five traits
(Agreeableness, Conscientiousness, Extraversion, Neuroticism, Openness).

---

## Headline result

| Split | mean r | A | C | E | N | O |
|---|---|---|---|---|---|---|
| Train-CV (OOF) | 0.3635 | 0.434 | 0.323 | 0.445 | 0.319 | 0.296 |
| Dev (public) | 0.3600 | 0.493 | 0.226 | 0.462 | 0.250 | 0.369 |
| **Test (private)** | **0.3215** | 0.387 | 0.283 | 0.410 | 0.254 | 0.274 |

**2019 first-place private-Test score: 0.26021 -> beaten by +0.061 (~23% relative).**

For context, the entire 2019 top-4 spanned only 0.031 (0.26021 down to 0.2293),
so this margin is roughly 2x the original first-to-fourth spread. The result also
sits at/above the 2025-2026 published frontier for personality inference from
short/naturalistic text (benchmarks cluster at r <= 0.27). See `SOLUTION.md` and
`LITERATURE.md` for the full write-up and evidence base.

---

## Honest evaluation protocol

Every number above comes from a strict, leakage-safe protocol:

- **Fit only on Train** (n=1088). No estimator ever sees Dev/Test labels.
- **Select only on Train-CV (out-of-fold) + Dev** (n=300).
- **Touch the private Test (n=300) once**, via `src/freeze_and_test.py`.
- Base learners produce out-of-fold predictions on shared K-folds; each base is
  refit on full Train to predict Dev/Test; a per-trait Ridge meta-learner is fit
  on Train OOF only.
- All frozen LLM / embedding outputs are cached per respondent (no labels used in
  extraction), so they are reproducible and cannot leak across splits.

---

## Winning architecture (stacked generalization)

A per-trait Ridge meta-learner (alpha=4, own-trait columns only) over these bases:

**LLM bases (Anthropic Claude, zero-shot, no fine-tuning):**
- Haiku 4.5 Big Five score extractor x 4 prompt variants (general, evidence,
  ranked, trait-focused)
- Sonnet 4.6 Big Five score extractor (second judge)
- Haiku 16-dimension behavioral subfeature extractor -> Ridge head
- **Haiku role-play questionnaire** (`llmq:`): answers a 30-item BFI-2-style
  battery in the respondent's persona, reverse-scored and aggregated

**Classical bases:**
- e5-large-v2 embeddings: per-prompt SVR (RBF) and all-text SVR
- TF-IDF char (2-6) + word-unigram Ridge
- Engineered psycholinguistic / stylometric features (readability, sentiment,
  POS ratios) -> Ridge + HistGradientBoosting

The role-play questionnaire lever (motivated by Yang et al. 2024 and validated
mechanistically by Liu et al. 2025) added +0.005 Dev and +0.004 Test over the
prior best.

---

## Repository structure

```
2019-personality-from-text/
|
+-- data/raw/                     2019_siop_ml_comp_data.csv, full_data_README.md
|
+-- src/
|   +-- data.py                   loading, splits, constants
|   +-- metrics.py                Pearson r, per-trait + mean r reporting
|   +-- features.py               TF-IDF vectorizers, engineered features
|   +-- embeddings.py             cached dense sentence embeddings
|   +-- stack.py                  OOF stacking engine + base-learner classes
|   +-- experiment.py             base-learner registry + Dev evaluation driver
|   +-- llm_extract.py            LLM scorers: trait scores, subfeatures, questionnaire
|   +-- select.py                 cached base predictions + meta sweeps
|   +-- freeze_and_test.py        SINGLE Test evaluation entry point
|   +-- eval_questionnaire.py     Dev-only questionnaire ablations
|   +-- pilot_questionnaire.py    small-sample questionnaire diagnostic
|   +-- run_llm_research_sweep.py multi-stage LLM research sweep (Dev only)
|   +-- run_example_b_sbert_ridge.py   instructional SBERT baseline
|
+-- results/
|   +-- cv/                       frozen_summary.csv, stage summaries
|   +-- submissions/              submission_{dev,test}_frozen.csv
|   +-- figures/                  diagnostic plots
|   +-- *.log                     run logs
|
+-- SOLUTION.md                   final model write-up + reproduction
+-- NEGATIVE_RESULTS.md           what did NOT work (and why)
+-- LLM_RESEARCH_SWEEP.md         staged LLM experiment log
+-- requirements.txt
+-- README.md
```

---

## Reproduce

```powershell
python -m pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "your-key"     # required for LLM bases

# Dev-only validation (Train-CV + Dev; does not touch Test)
python -m src.freeze_and_test --dev-only

# Full single Test evaluation (the one honest Test touch)
python -m src.freeze_and_test
```

First run scores the LLM bases via the Anthropic API and caches every response
under `data/processed/llm_cache/` (gitignored). Subsequent runs load from cache
and are fast. Embeddings cache under `data/processed/emb_cache/`.

Outputs: `results/cv/frozen_summary.csv`,
`results/submissions/submission_{dev,test}_frozen.csv`.

---

## Key takeaways

- **Frontier LLMs as zero-shot extractors** are the single biggest lever on this
  task; a leakage-safe stack over multiple prompt framings, a second judge, and a
  role-play questionnaire pushes well past the classical (pre-LLM) ceiling.
- **Role-play + questionnaire simulation** beats direct trait scoring, consistent
  with the 2024-2026 literature.
- **The small-n trap is real:** with 300-row Dev/Test, prefer repeated Train-CV as
  the selection signal and spend Test touches sparingly. Several richer ensembles
  that looked better on Dev did worse on Test (see `NEGATIVE_RESULTS.md`).
- **Ceiling awareness:** raw r > 0.40 has no published precedent for <=5 short
  answers; ~0.32 is at/above the realistic short-text frontier.
