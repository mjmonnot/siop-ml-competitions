# 2021 SIOP ML Competition — Fairness-in-the-Pipeline (Fully Annotated Repo)

This folder is a **complete, upload-ready case study repo** designed for **lay readers** (students, HR practitioners, compliance/legal partners, and executives).

It includes TWO parallel implementations:

1. **Competition-style reference solution** (what a leaderboard-optimized approach looks like)
2. **Standards-aligned pipeline** (what a more professionally defensible approach looks like)

> **Guiding principle taught here:**  
> Protected-group membership should generally be used to **evaluate outcomes** (auditing/governance), not to **generate individual prediction scores**.

---

## What’s included

### 01_competition_solution/
A readable, heavily-commented version of the “Team Procrustination–style” approach:
- one-hot encoding for categorical fields
- XGBoost models
- z-score standardization
- weighted ensemble
- median cut (hire top half)

This is included as a **contrast case**. It may score well in a competition, but it is *not* the recommended pattern for operational hiring.

### 02_standards_aligned_pipeline/
A teaching implementation of a standards-aligned workflow:
- **scores** are generated without using protected-group membership
- protected-group membership is used only for **evaluation and policy governance** (AIR, unfairness)
- hiring is determined by a **transparent cut score** chosen via documented trade-offs

### 03_evaluation/
Two notebooks you can run to visually explore:
- adverse impact ratio (AIR)
- accuracy vs AIR trade-offs across different cut scores

### figures/
- pipeline diagram
- illustrative trade-off curve

---

## Quickstart

### Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data placement
Put these files in `00_data/` (not tracked in git):
- `00_data/train.csv`
- `00_data/participant_test.csv`

### Run: competition-style reference (contrast)
```bash
python 01_competition_solution/procrustination_reference.py --train 00_data/train.csv --test 00_data/participant_test.csv --out final_submission.csv
```

### Run: standards-aligned pipeline (recommended)
```bash
python 02_standards_aligned_pipeline/job_success_model.py --train 00_data/train.csv --test 00_data/participant_test.csv --out scored_test.csv
python 02_standards_aligned_pipeline/cutoff_optimization.py --train 00_data/train.csv --scores scored_test.csv --out final_submission.csv
```

---

## Citation
Koenig, N., & Thompson, I. (2021). *The 2020–2021 SIOP Machine Learning Competition*. Presented at the 36th Annual Conference of the Society for Industrial and Organizational Psychology, New Orleans, LA.

---

## License
MIT (see `LICENSE`).
