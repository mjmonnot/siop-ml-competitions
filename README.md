# SIOP Machine Learning Competitions (2019–2026)

A year-over-year, reproducible collection of **reference implementations** and **teaching cases** for the SIOP Machine Learning Competitions. Each year directory is a self-contained case: the task, its scoring metric, runnable code, and notes on *why* particular approaches win or fail. The repo root holds shared infrastructure and cross-year teaching material.

## Why this repo exists

- Provide clean baselines and strong reference pipelines for each competition year
- Explain why methods work (or fail) as a compact, runnable teaching case
- Make it easy to rerun, compare, and extend across years
- Keep an honest line between what was **measured**, what was run on **synthetic/demo data**, and what is **projected**

## What's in this repo

| Year | Folder | Task | Metric | Status |
|------|--------|------|--------|--------|
| 2019 | `2019-personality-from-text/` | Big Five personality from open-ended SJI text | Mean Pearson *r* across five traits | Complete, runnable (results committed) |
| 2021 | `2021-fairness-pipeline-case/` | Fairness in a hiring/selection pipeline | Adverse impact ratio vs. accuracy trade-offs | Complete case study (drop in data to run) |
| 2023 | `2023-decision-making-from-text/` | Decision-making ratings from open-ended text | Correlation on a numeric rating (+ fairness audit) | Scaffold / in progress |
| 2024 | `2024-evaluate-LLMs-via-benchmark/` | Four LLM benchmarks: empathy, interview, clarity, fairness | 0.25-weighted composite | Runnable end-to-end on synthetic data; official inputs unavailable |
| 2026 | `2026-meta-analysis/` | Extracting bivariate Pearson *r* values from I-O psych PDFs | MSE vs. held-out *r* values | Complete; competed (dev 6/24, test 11/24) |

There is no 2020, 2022, or 2025 directory — years are added as cases get built, and the folders above are the ones that currently exist.

## The cases

**2019 — Personality from text.** Predicts Big Five traits from open-ended situational-judgment responses. Ships two reference pipelines: a TF-IDF + Ridge blend (the strong, competition-appropriate baseline) and an SBERT + Ridge contrast model. The intentional lesson: on a rank-order metric like Pearson *r*, a well-tuned linear text model can beat modern sentence embeddings.

**2021 — Fairness in the pipeline.** A case study built for lay readers (students, HR/compliance, executives) with two parallel implementations: a competition-style reference solution (one-hot + XGBoost + weighted ensemble + median cut, included as a *contrast* case) and a standards-aligned pipeline that generates scores without protected-group membership and uses that membership only for evaluation and governance. Includes notebooks for adverse-impact-ratio and accuracy-vs-AIR trade-off analysis. MIT licensed.

**2023 — Decision making from text.** A reproducible scaffold for the 2023 winners' task (predicting assessment-center decision-making ratings from open text). The planned pipeline is validate → preprocess → features → train → evaluate → fairness audit, with a TF-IDF + Ridge baseline and an SBERT-ready template. This year is still an early scaffold (README, requirements, and a synthetic demo dataset); the runnable `src/` is not committed yet.

**2024 — Evaluate LLMs via benchmark.** A post-hoc reconstruction of the four-benchmark 2024 competition, asking whether a single unified LLM pipeline with task-specific adapters can keep up with four separately hand-tuned winning submissions. The pipeline runs end-to-end against a fixed-seed synthetic data generator; the official EvalAI input files are not publicly archived, so synthetic-data scores are **not** comparable to the winners' published numbers. The headline table separates winner scores, a projected band, and measured synthetic scores. A retrospective deck (PDF) lives in `docs/`.

**2026 — Meta-analysis ("One Hot Key").** Automated extraction of zero-order bivariate *r* values from I-O psychology PDFs, built as a one-person-plus-agents entry. Uses a four-tier extraction cascade (pdfplumber geometric tables → Docling TableFormer → qwen2.5-VL on page images → regex candidates classified by phi4), all running locally. Placed 6/24 on the dev set and 11/24 on the test set. Includes the competition poster, presentation video, and deck.

## How the repo is organized

```
siop-ml-competitions/
├── README.md                      ← this file (repo overview)
├── environment.yml                ← shared conda environment (siop-ml)
├── .gitignore                     ← global data/output safety rules
│
├── docs/                          ← cross-year teaching material
│   ├── metrics.md                 ← how Pearson r works as a metric (general)
│   ├── teaching-notes.md          ← how to use these as applied ML cases
│   └── repo-conventions.md        ← data/output layout conventions
│
├── 2019-personality-from-text/    ← src + notebooks + results + README
├── 2021-fairness-pipeline-case/   ← two implementations + evaluation notebooks
├── 2023-decision-making-from-text/← scaffold (README + requirements)
├── 2024-evaluate-LLMs-via-benchmark/ ← pipeline, adapters, scoring, notebooks, docs
└── 2026-meta-analysis/            ← extraction pipelines, docs, media
```

Each year directory carries its own README, dependencies, and (where present) `data/`, `src/`, `results/`, and `docs/`.

## Conventions and design principles

- **Each year is independent and authoritative.** A year's own README, metric, and code override repo-level defaults. Framing, visual style, and methods are not carried across years by default — each case is built on its own terms.
- **Root = shared infrastructure; years = the actual cases.** The root holds the environment, ignore rules, and cross-year teaching docs; everything task-specific lives under the year.
- **Reproducibility is a standing requirement.** Fix random seeds, write key outputs to `results/`, and keep code runnable from within the year directory.
- **Honest reporting.** Keep a clear line between measured results, synthetic/demo-data results, and projected or estimated numbers — never blur them. (See 2024's `docs/STATUS.md` for the fullest example.)
- **Data hygiene.** Raw competition inputs are never committed (see `.gitignore`); only label files, synthetic/demo data, and small useful artifacts are tracked.

## Setup

A shared conda environment covers the common stack:

```bash
conda env create -f environment.yml
conda activate siop-ml
```

Individual years pin their own extras — see that year's `requirements.txt` or its README quick-start. Notably, the dependency stories differ a lot by year: 2019/2021/2023 are scikit-learn-centric, **2024** calls the OpenAI API (needs `OPENAI_API_KEY`), and **2026** runs local models via Ollama. Open the year you care about and follow its README.

## Cross-year docs

- [`docs/metrics.md`](docs/metrics.md) — how Pearson *r* behaves as a scoring metric and why it rewards rank order over calibration.
- [`docs/teaching-notes.md`](docs/teaching-notes.md) — the recommended per-year teaching pattern.
- [`docs/repo-conventions.md`](docs/repo-conventions.md) — data and output layout conventions.

## Original competitions

- 2019: https://github.com/izk8/2019_SIOP_Machine_Learning_Winners
- 2021: https://github.com/izk8/2021_SIOP_Machine_Learning_Winners 
- 2023: https://github.com/izk8/2023_SIOP_Machine_Learning_Winners
- 2024: https://github.com/izk8/2024_SIOP_Machine_Learning_Competition

## Credits

Pipelines and reconstructions by Matthew J. Monnot, PhD. Competition tasks, data, and benchmarks are credited to the respective SIOP ML Competition organizers and data sponsors within each year directory. Licensing is per-year (2021 is MIT; others TBD before public release).
