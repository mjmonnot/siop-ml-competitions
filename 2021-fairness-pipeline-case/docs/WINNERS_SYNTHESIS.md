# Winners synthesis — 2020–2021 SIOP ML Competition

Published private-test finals (organizer deck / [izk8 winners repo](https://github.com/izk8/2021_SIOP_Machine_Learning_Winners)):

| Place | Team | Final |
|------:|------|------:|
| 1 | Team Procrustination (Guo & McAbee) | **62.53** |
| 2 | Axiom Consulting Partners | 62.50 |
| 3 | RHDS | 61.09 |
| 4 | Go Ahead, Make My Data | 60.72 |

The top two are separated by **0.03** points. That is the headline: on this
metric, several strong stacks land in a very tight band, and small fairness
corrections decide the podium.

---

## Place 1 — Team Procrustination

**Stack:** one-hot categorical SJ / scenario / biodata items → four XGBoost
models → z-score → weighted ensemble → **median cut (hire top half)**.

| Model | Target | Role |
|-------|--------|------|
| 1 | `High_Performer` | Default XGBClassifier, `scale_pos_weight=1.5` |
| 2 | `Overall_Rating` | XGBRegressor (shallow, subsampled) |
| 3 | `High_Performer` | Bayesian-tuned XGBClassifier |
| 4 | `(Protected==1 ∧ Retained==1)` | Fairness correction, 10% ensemble weight |

Final score recipe (from their published script):

```text
score = (0.5·m1 + 0.2·m2 + 0.3·m3) · 0.9  +  0.1 · m4
Hire  = 1{score > median(score)}
```

**What worked:** explicit metric engineering. Model 4 injects protected-group
information into the *score*, which lifts AIR and shrinks the unfairness
penalty without giving up much of the performance ensemble. Our holdout
ablation reproduces the qualitative result: dropping Model 4 costs ~8 final
points (80.91 → 72.92) almost entirely via unfairness (7.5 → 15.6).

**Teaching read:** this is the cleanest illustration in the SIOP ML series of
“the metric asked for fairness, so the winning team put fairness inside the
model.” Professionally, that move is the contrast case — not the template.

Source in this repo: `01_competition_solution/procrustination_reference.py`
and the original script under `2021 Winners and Data/` (local, gitignored).

---

## Place 2 — Axiom Consulting Partners

**Stack (from their tidymodels pipeline):** heavy preprocessing / imputation /
scale construction (`scales.Rmd`, `impute_combine.Rmd`), then a broad
tidymodels search (`models.Rmd`) across discriminant, bagged, and rule-based
learners with feature selection (`recipeselectors`) and racing (`finetune`).
They also engineered combined targets such as `hp_ret`
(high-performer × retained).

**What worked:** exhaustive, disciplined model search on carefully prepared
features — finishing **0.03** behind Place 1 without (from the public materials)
the same blunt protected×retained proxy weight. The gap says the leaderboard
was saturating; small fairness adjustments mattered more than another base
learner.

**Teaching read:** when validity signals are strong and shared, the residual
contest is governance of subgroup outcomes.

---

## Place 3 — RHDS

**Stack:** tidyverse preprocessing (categorical SJ/biodata as characters,
`"MISSING"` level for categoricals), group-aware EDA on personality scales and
timing features, then (per their R script / deck) predictive models with
explicit attention to protected-group differences in predictors.

**What worked:** thoughtful missingness handling and subgroup diagnostics.
Their deck is a good classroom artifact for “look at the predictors by group
before you regularize the problem away.”

---

## Place 4 — Go Ahead, Make My Data

**Stack:** TPOT-driven AutoML (`TPOT Alt Weights 3 Clean.ipynb`) with alternate
weighting schemes for the multi-objective score.

**What worked:** letting genetic AutoML search pipelines under custom fitness
proxies for the competition metric. They were competitive (60.72) but the
search did not recover Place 1’s particular fairness correction.

**Teaching read:** AutoML optimizes what you measure. If the fitness function
under-weights AIR relative to the true EvalAI metric — or cannot express a
protected proxy cleanly — you leave points on the table.

---

## Synthesis

1. **Everyone modeled job success.** Top / retained / ratings signals dominate.
2. **Fairness was the differentiator.** Place 1’s protected×retained head is the
   clearest metric hack; Place 2 got almost the same final without that exact
   recipe.
3. **Decision volume was ~50%.** Median cut / top-half hiring appears across
   competitive solutions. Unconstrained threshold search on this metric
   collapses to “hire almost everyone” (see `KNOWN_LANDMINES.md`).
4. **Applied lesson ≠ leaderboard lesson.** For teaching HR/compliance audiences,
   Path B (blind scores + audited cut) is the recommended pattern even when
   Path A scores higher on the contest formula.

This repo’s post-hoc question is therefore:

> Holding hiring volume at ~50%, how much of Place 1’s edge is the protected
> proxy — and what does a standards-aligned alternative look like on the same
> freeze protocol?
