# SIOP Machine Learning Competitions (2019–Present)

This repo is a year-over-year, reproducible set of solutions and teaching cases for the SIOP Machine Learning Competitions.

## Why this repo exists
- Provide **clean baselines** and **strong reference pipelines** for each year
- Explain *why* methods work (or fail) as a compact teaching case
- Make it easy to rerun, compare, and extend across years

## Repo architecture (at a glance)

This repo is organized as a **shared framework + year-specific cases**.

siop-ml-competitions/
│
├── README.md                 ← What this repo is (overview)
├── .gitignore                ← Global safety rules (data, outputs)
├── requirements.txt          ← Shared baseline deps
├── environment.yml
│
├── docs/                     ← Cross-year concepts
│   ├── metrics.md            ← How competition metrics work (general)
│   ├── teaching-notes.md     ← How to use these cases pedagogically
│   └── repo-conventions.md
│
├── 2019-personality-from-text/
│   ├── README.md             ← 2019 rules + teaching case
│   ├── instructor-notes.md
│   ├── requirements.txt      ← (optional overrides)
│   │
│   ├── data/
│   │   ├── raw/              ← competition data (not committed)
│   │   └── processed/
│   │
│   ├── src/                  ← runnable models for 2019
│   ├── notebooks/            ← optional exploration / teaching
│   └── results/              ← CV, figures, submissions
│
├── 2020-<challenge-name>/
│   └── ...
└── 2021-<challenge-name>/
    └── ...

### Design principle
- **Root** = shared infrastructure + teaching context  
- **Each year** = authoritative rules, metric, and implementation for that competition  
- Year-level documentation always overrides repo-level defaults


## Quick start (2019)
1) Install dependencies (choose one route)
- **pip**:
  ```bash
  pip install -r requirements.txt
