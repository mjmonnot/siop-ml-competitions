# Solution — 2021 post-hoc comparison

## Question

On a frozen, labeled holdout of the 2020–2021 SIOP ML Competition train data,
how does a **Team Procrustination–style** pipeline (protected×retained proxy
inside the score) compare to a **standards-aligned** pipeline (protected status
only at the decision/audit layer), holding hiring volume near 50%?

## Answer (measured, holdout-Train, seed=42)

| Pipeline | Final | Accuracy | AIR | Unfairness | Sel. rate | Protected in score? |
|----------|------:|---------:|----:|-----------:|----------:|:-------------------:|
| Competition-style (+ proxy) | **80.91** | 88.39 | 0.925 | 7.48 | 0.500 | yes |
| Competition-style, no proxy | 72.92 | 88.47 | 0.845 | 15.55 | 0.500 | no |
| Standards-aligned (OOF @ ~50%) | 72.93 | 88.53 | 0.844 | 15.60 | 0.494 | no |

Full table: `results/cv/compare_summary.csv`. Protocol: `docs/STATUS.md`.

### Interpretation

1. **The proxy is the Place-1 edge (qualitatively).** Removing it collapses the
   competition-style final onto the standards-aligned final (~72.9). Accuracy
   barely moves; unfairness doubles.
2. **Standards-aligned matches the no-proxy ensemble** under a 50% selection
   constraint. Blind scoring + OOF cut-score search does not free-fall relative
   to a fairness-unaware performance stack.
3. **Do not compare these finals to 62.53.** Private-test labels are unavailable;
   absolute levels differ. Published Place 1–4 scores stay in the README as
   historical context only.

## Reproduction

```powershell
cd 2021-fairness-pipeline-case
python -m pip install -r requirements.txt

# Place train.csv (and optional participant_*.csv) in 00_data/
# Local archive copy (gitignored): 2021 Winners and Data/train.csv

python -m src.run_compare
```

Outputs:

- `results/cv/compare_summary.csv` — measured holdout scorecard
- `results/cv/published_leaderboard.csv` — organizer Place 1–4 finals
- `results/cv/protocol.json` — freeze metadata
- `results/submissions/holdout_*.csv` — per-pipeline holdout decisions

## Recommended applied pattern

Use **standards-aligned** (`02_standards_aligned_pipeline/`):

1. Predict a Job Success Index without `Protected_Group`.
2. Choose a cut score on out-of-fold scores (document AIR / utility tradeoffs).
3. Audit subgroup outcomes after decisions; do not bake protected status into
   individual scores.

Keep the competition-style path as a **contrast case** for metric literacy.
