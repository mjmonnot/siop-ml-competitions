# Known landmines — 2021 fairness pipeline

## 1. The metric wants you to hire everyone

`Overall_accuracy` is built from **recalls** (fraction of true top / retained /
both who were hired). Hiring every candidate drives those recalls to 1.0 while
AIR drifts toward 1.0, so `Final = accuracy − unfairness` explodes.

On our freeze holdout, unconstrained threshold search scores **99.0** at a
**96%** selection rate. That number is a bug in the *interpretation*, not a
breakthrough. Winners hired ~top half. This repo’s primary standards-aligned
path therefore constrains selection rate to **0.50 ± 0.02**.

**Classroom move:** show the unconstrained row in `compare_summary.csv`, then
ask what business constraint the metric forgot to encode.

## 2. Protected status inside the score vs at the cut

Place 1’s Model 4 trains on `(Protected_Group==1 ∧ Retained==1)`. That is
legal in a contest and devastatingly effective on this metric (+~8 holdout
final points in our ablation). In operational hiring it is usually the wrong
place to use protected-group membership.

**Classroom move:** Path A vs Path B in `ARCHITECTURE.md`. Same features, same
metric, different governance story.

## 3. Private-test labels are gone

You cannot reproduce 62.53. Anyone claiming a new private-test number without
organizer labels is inventing. Report **published** and **holdout-measured**
columns separately (see `STATUS.md`).

## 4. Label missingness is the real sample size

`train.csv` has 44,102 rows, but `High_Performer` / `Overall_Rating` are
missing on ~82% of them. Scorable rows: **7,890**. Pipelines that silently
`dropna()` across *all* columns shrink further. Always state the labeled-n
you evaluated on.

## 5. Complete-case training ≠ production missingness

Procrustination drops incomplete feature rows. RHDS instead codes categorical
`"MISSING"`. Those choices change who is eligible to be scored and can move AIR
even when the model family stays fixed.

## 6. Threshold chosen on in-sample scores is leaking

The original teaching `cutoff_optimization.py` placeholder built a governance
score from labels when CV scores were absent. That demonstrates the *API* of
cut-score search but is not an honest operating point. `src/run_compare.py`
uses **out-of-fold** JSI scores on the fit set only.

## 7. AIR floors can be infeasible at fixed selection rates

Requiring AIR ≥ 0.80 while forcing a 50% hire rate may admit no threshold if
score distributions differ sharply by group. Handle that as a documented
infeasibility, not a silent fallback to an unconstrained cut.

## 8. One-hot train/test column mismatch

Categorical levels that appear only in test (or only in fit) break naive
`get_dummies`. Always concatenate before dummy-coding (as both Path A and
Path B do), or you ship a pipeline that only works on the training schema.
