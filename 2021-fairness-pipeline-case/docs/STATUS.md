# Status — 2021 post-hoc comparison

Last frozen run: **2026-07-27** (`python -m src.run_compare`, seed=42).

## What has been measured

| Pipeline | Holdout final | Accuracy | AIR | Unfairness | Selection rate |
|----------|---------------|----------|-----|------------|----------------|
| Competition-style (+ protected×retained proxy) | **80.91** | 88.39 | 0.925 | 7.48 | 0.500 |
| Competition-style ablation (no proxy) | 72.92 | 88.47 | 0.844 | 15.55 | 0.500 |
| Standards-aligned (OOF cut @ ~50% hire) | 72.93 | 88.53 | 0.844 | 15.60 | 0.494 |
| Standards-aligned + AIR≥0.80 @ ~50% | 72.93 | 88.53 | 0.844 | 15.60 | 0.494 |
| Standards-aligned **unconstrained** (demo) | 99.03 | 99.26 | 0.998 | 0.23 | 0.958 |

Artifacts: `results/cv/compare_summary.csv`, `results/cv/protocol.json`,
`results/submissions/holdout_*.csv`.

## What has *not* been remeasured

Private-test labels were never released. Published Place 1–4 scores
(`results/cv/published_leaderboard.csv`) come from the organizer deck / winners
README and **cannot** be recomputed here:

| Place | Team | Published private-test final |
|------:|------|-----------------------------:|
| 1 | Team Procrustination | 62.53 |
| 2 | Axiom Consulting Partners | 62.50 |
| 3 | RHDS | 61.09 |
| 4 | Go Ahead, Make My Data | 60.72 |

Holdout-Train finals (~73–81) are **not comparable** to those private-test
numbers. Different population, different missingness, and a labeled-complete
subset (~7.9k of 44k train rows) all inflate absolute levels. Use holdout
numbers only for **within-repo** contrasts (proxy vs no-proxy; competition-style
vs standards-aligned).

## Protocol (frozen)

1. Keep rows with non-null `High_Performer`, `Retained`, `Protected_Group` (n=7,890).
2. Stratified 80/20 fit/holdout split on
   `Protected_Group × High_Performer × Retained` (seed=42) → fit 6,312 / holdout 1,578.
3. Fit pipelines on fit only; score holdout once.
4. Standards-aligned thresholds chosen on **fit OOF** scores only (5-fold),
   with selection rate constrained to 0.50 ± 0.02 unless explicitly
   marked `unconstrained`.

## Media package (pending)

Poster, presentation deck, and narrated video are intentionally **not** in this
upgrade. They will be produced separately (Claude Design + HeyGen) and dropped
into `docs/` + `media/` later. See `docs/MEDIA_TODO.md`.
