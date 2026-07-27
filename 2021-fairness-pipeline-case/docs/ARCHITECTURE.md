# Architecture — where fairness enters the pipeline

The 2021 competition rewarded a scalar:

```
Final = Overall_accuracy − Unfairness
Unfairness = |1 − AIR| × 100
AIR = selection_rate(protected) / selection_rate(non-protected)
```

`Overall_accuracy` is a weighted blend of three recall-like hit rates
(top performers hired, retained hired, both hired). That design creates two
very different engineering strategies, which this repo implements side by side.

## Path A — Competition-style (leaderboard-aware)

```
features (one-hot SJ / biodata / scenarios)
        │
        ├─ XGB: High_Performer          ─┐
        ├─ XGB: Overall_Rating          ─┼─ z-score → weighted ensemble (90%)
        ├─ XGB: High_Performer (tuned)  ─┘
        │
        └─ XGB: (Protected==1 ∧ Retained==1)  → z-score → 10% correction
                          │
                     median cut (hire top half)
                          │
                        Hire
```

**Fairness enters inside scoring.** Model 4 is trained on a label that *includes*
`Protected_Group`. The ensemble is nudged toward people who look like
protected+retained employees in training. That is metric-aware and historically
won the competition; it is **not** the recommended applied pattern.

Implemented in `01_competition_solution/procrustination_reference.py` and
`src/pipelines.competition_style_scores`.

## Path B — Standards-aligned (governance-aware)

```
features (same assessment items; Protected_Group excluded)
        │
        └─ XGB regressor → Job Success Index
              JSI = 0.25·Top + 0.25·Retained + 0.50·(Top∧Retained)
        │
        OOF scores on fit set → choose cut score
        (optionally require AIR ≥ floor)
        │
        apply threshold to holdout/test scores
        │
      Hire   ← Protected_Group used ONLY to evaluate AIR / document policy
```

**Fairness enters at the decision / governance layer**, not in the individual
score. Scores are blind to protected status; group membership is used to audit
outcomes and to select a documented operating point.

Implemented in `02_standards_aligned_pipeline/` and
`src/pipelines.standards_aligned_*`.

## Shared evaluation harness

`src/run_compare.py` freezes a stratified holdout of labeled train rows, runs
both paths (plus ablations), and writes `results/cv/compare_summary.csv`.
Scoring always goes through `02_standards_aligned_pipeline/scoring_function.py`
so the metric definition stays single-sourced.

## Why both paths exist

| Question | Path A answers | Path B answers |
|----------|----------------|----------------|
| How did Place 1 win? | Protected proxy + median cut | — |
| What would a defensible workflow look like? | — | Blind score + audited cut |
| What does the proxy buy on the metric? | Holdout +8 pts vs no-proxy ablation | — |
| Where do practitioners get sued / challenged? | Using protected status (or proxies) in scoring | Documented cut-score tradeoffs |

The pedagogical claim of this case is not “Path B beats Path A on the
leaderboard.” It is: **the same metric can be attacked from inside the score or
from the decision policy — and those choices have different professional
consequences.**
