# 2021 SIOP ML Competition — Fairness-in-the-Pipeline Teaching Case

This repository is a **teaching case** built around the 2020–2021 SIOP Machine Learning Competition (often referred to as “SIOP 2021”).
It uses the first-place approach (Team Procrustination) as a **didactic artifact** to show how *where* fairness is introduced in an ML hiring pipeline
changes technical behavior **and** professional defensibility.

The case provides:

1) A **reference implementation** of the winning-style solution (competition-optimized; not recommended for operational use).
2) A **Standards-aligned pipeline** that keeps protected-group variables out of model scoring and uses them **only for evaluation and decision governance**.
3) **Evaluation notebooks** (adverse impact + trade-off curves).
4) **Figures** (pipeline diagram + illustrative trade-off curve).

> **Important:** This repo is not legal advice. It is an educational resource showing how to align ML selection workflows with common professional expectations
(e.g., transparency, job relatedness, documentation) and classic adverse impact evaluation practices.

---

## Learning objectives

By the end of this case study, readers should be able to:

- Distinguish **predictive**, **decision**, and **outcome** fairness in hiring pipelines
- Identify fairness “insertion points” (feature design, training, scoring, thresholding, evaluation)
- Explain why protected-group variables should generally be used for **evaluation**, not **scoring**
- Implement a **cutoff optimization** approach that examines accuracy–fairness trade-offs
- Communicate a defensible operating point with clear documentation

---

## Competition scoring function (ground truth)

The competition ranked submissions using:

- **Final_score = Overall_accuracy − Unfairness**

Where:

- **Overall_accuracy**
  - % of true top performers hired × 25
  - % of true retained hired × 25
  - % of true retained *and* top performers hired × 50

- **Unfairness = |1 − Adverse_impact_ratio| × 100**

- **Adverse_impact_ratio (AIR)**  
  (Selection rate for protected group) ÷ (Selection rate for non-protected group)

This repo implements the metric in: `02_standards_aligned_pipeline/scoring_function.py`

---

## Repo structure

```
2021-fairness-pipeline-case/
├── README.md
├── requirements.txt
├── 00_data/                      # optional local data (not tracked)
├── 01_competition_solution/
│   ├── procrustination_reference.py
│   └── README.md
├── 02_standards_aligned_pipeline/
│   ├── scoring_function.py
│   ├── job_success_model.py
│   ├── cutoff_optimization.py
│   └── README.md
├── 03_evaluation/
│   ├── adverse_impact_analysis.ipynb
│   └── tradeoff_curves.ipynb
├── figures/
│   ├── pipeline_diagram.png
│   └── tradeoff_curve_example.png
└── teaching_notes.md
```

---

## Quickstart

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Place data

Put files here (or adjust paths in scripts):

- `00_data/train.csv`
- `00_data/participant_test.csv`

### 3) Run the competition-style baseline (reference only)

```bash
python 01_competition_solution/procrustination_reference.py --train 00_data/train.csv --test 00_data/participant_test.csv --out final_submission.csv
```

### 4) Run the Standards-aligned pipeline

```bash
python 02_standards_aligned_pipeline/job_success_model.py --train 00_data/train.csv --test 00_data/participant_test.csv --out scored_test.csv
python 02_standards_aligned_pipeline/cutoff_optimization.py --train 00_data/train.csv --scores scored_test.csv --out final_submission.csv
```

---

## What this case teaches (big picture)

### Competition-optimized fairness (illustrative)
The reference solution introduces fairness by **creating a proxy target** tied to protected-group membership and blending that signal into the final score.
That can improve leaderboard metrics, but it reduces transparency and can be hard to defend in practice.

### Standards-aligned fairness (recommended pattern)
The Standards-aligned pipeline:
- Builds prediction models **without protected-group variables**
- Uses protected-group membership **only** to compute AIR and fairness metrics during evaluation
- Chooses a decision threshold (“cut score”) based on documented trade-offs (utility vs AIR), consistent with common adverse-impact governance workflows

---

## Citation

Koenig, N., & Thompson, I. (2021). *The 2020–2021 SIOP Machine Learning Competition*. Presented at the 36th Annual Conference of the Society for Industrial and Organizational Psychology, New Orleans, LA.

---

## License

MIT (Just cite my repo...you know the deal).
