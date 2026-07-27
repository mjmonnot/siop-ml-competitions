# SIOP 2021 ML Competition — Fairness in the Pipeline

A post-hoc reconstruction of the **2020–2021 SIOP Machine Learning Competition**:
build hiring decisions that balance job-success outcomes against adverse impact.
This case asks one comparative question:

> **How much of Place 1’s leaderboard edge is a protected-group proxy inside the
> score — and what does a standards-aligned alternative look like on the same
> freeze protocol?**

Original competition / winners archive:
https://github.com/izk8/2021_SIOP_Machine_Learning_Winners

---

## Headline results

### Published private-test leaderboard (organizer; not remeasured)

| Place | Team | Final |
|------:|------|------:|
| 1 | Team Procrustination | **62.53** |
| 2 | Axiom Consulting Partners | 62.50 |
| 3 | RHDS | 61.09 |
| 4 | Go Ahead, Make My Data | 60.72 |

### Measured holdout-Train comparison (this repo, 2026-07-27)

Frozen protocol: labeled train rows only (n=7,890) → stratified 80/20 fit/holdout
(seed=42). Selection rate held near 50% except the unconstrained demo row.
**Not comparable** to the private-test column above — see [STATUS.md](docs/STATUS.md).

| Pipeline | Final | Accuracy | AIR | Unfairness | Sel. rate | Protected in score? |
|----------|------:|---------:|----:|-----------:|----------:|:-------------------:|
| Competition-style (+ Place-1 proxy) | **80.91** | 88.39 | 0.925 | 7.48 | 0.500 | yes |
| Competition-style, no proxy | 72.92 | 88.47 | 0.845 | 15.55 | 0.500 | no |
| Standards-aligned (OOF cut @ ~50%) | 72.93 | 88.53 | 0.844 | 15.60 | 0.494 | no |
| Unconstrained standards-aligned *(landmine demo)* | 99.03 | 99.26 | 0.998 | 0.23 | 0.958 | no |

**Takeaway:** the protected×retained proxy buys ~8 holdout final points almost
entirely by cutting unfairness. A blind-score + audited-cut pipeline matches the
no-proxy ensemble. Unconstrained cut search “wins” by hiring everyone — that is
a metric landmine, not a deployment strategy.

---

## The experiment

> This is a post-hoc teaching reconstruction. The 2021 winners optimized a
> composite of job-success hit rates minus an AIR-based unfairness penalty. Place 1
> put fairness *inside* the score via a protected×retained XGBoost head. This repo
> re-implements that recipe, ablates the proxy, and contrasts it with a
> standards-aligned path that keeps protected status out of scoring and uses it
> only for threshold governance / audit.
>
> The framing is pedagogical for students, HR/compliance partners, and executives:
> leaderboard literacy on one side, professionally defensible workflow on the other.

Metric:

```text
Final = Overall_accuracy − Unfairness
Unfairness = |1 − AIR| × 100
```

---

## Architecture summary

Two parallel paths (details in [ARCHITECTURE.md](docs/ARCHITECTURE.md)):

1. **Competition-style** — Procrustination ensemble with optional protected proxy; median cut.
2. **Standards-aligned** — Job Success Index without protected features; OOF threshold search; optional AIR floor.

Shared scorer: `02_standards_aligned_pipeline/scoring_function.py`.  
Freeze comparison entrypoint: `python -m src.run_compare`.

---

## Repo structure

```text
2021-fairness-pipeline-case/
├── README.md
├── SOLUTION.md                 Final write-up + reproduction
├── NEGATIVE_RESULTS.md         What not to claim
├── CHANGELOG.md
├── teaching_notes.md           Classroom flow (lay audience)
├── requirements.txt
├── LICENSE                     MIT
├── 00_data/                    train.csv etc. (gitignored; see README there)
├── 01_competition_solution/    Annotated Place-1–style reference
├── 02_standards_aligned_pipeline/
├── 03_evaluation/              AIR + tradeoff notebooks
├── src/
│   ├── data.py                 Load + freeze split
│   ├── pipelines.py            Both modeling paths
│   └── run_compare.py          Frozen comparison CLI
├── results/cv/                 compare_summary.csv, protocol.json, …
├── results/submissions/        Holdout Hire files
├── docs/
│   ├── STATUS.md
│   ├── ARCHITECTURE.md
│   ├── WINNERS_SYNTHESIS.md
│   └── KNOWN_LANDMINES.md
└── figures/
```

---

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Drop competition CSVs into 00_data/ (train.csv required for comparison)
python -m src.run_compare
```

Legacy single-pipeline CLIs (still supported):

```powershell
python 01_competition_solution/procrustination_reference.py `
  --train 00_data/train.csv --test 00_data/participant_test.csv `
  --out results/submissions/competition_style_test.csv

python 02_standards_aligned_pipeline/job_success_model.py `
  --train 00_data/train.csv --test 00_data/participant_test.csv `
  --out results/submissions/scored_test.csv
python 02_standards_aligned_pipeline/cutoff_optimization.py `
  --train 00_data/train.csv --scores results/submissions/scored_test.csv `
  --out results/submissions/standards_aligned_test.csv
```

---

## Docs map

| Doc | Purpose |
|-----|---------|
| [SOLUTION.md](SOLUTION.md) | Measured answer + how to reproduce |
| [docs/WINNERS_SYNTHESIS.md](docs/WINNERS_SYNTHESIS.md) | Place 1–4 methods side-by-side |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Score-path vs decision-path fairness |
| [docs/KNOWN_LANDMINES.md](docs/KNOWN_LANDMINES.md) | Hire-all gaming, leakage, missing labels |
| [docs/STATUS.md](docs/STATUS.md) | What was run; published vs measured |
| [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) | Claims we deliberately do *not* make |
| [teaching_notes.md](teaching_notes.md) | 60–90 min classroom outline |

---

## Citation

Koenig, N., & Thompson, I. (2021). *The 2020–2021 SIOP Machine Learning Competition*.
Presented at the 36th Annual Conference of the Society for Industrial and
Organizational Psychology, New Orleans, LA.

## License

MIT (see `LICENSE`).
