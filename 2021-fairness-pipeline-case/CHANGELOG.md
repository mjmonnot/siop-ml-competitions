# Changelog — 2021-fairness-pipeline-case

## v1.1 — 2026-07-27

Post-hoc comparative upgrade (code + docs).

- Add `src/run_compare.py` freeze protocol (stratified labeled holdout, seed=42)
- Measure competition-style (+/− protected proxy) vs standards-aligned paths
- Add `docs/STATUS.md`, `ARCHITECTURE.md`, `WINNERS_SYNTHESIS.md`, `KNOWN_LANDMINES.md`
- Add `SOLUTION.md`, `NEGATIVE_RESULTS.md`
- Commit measured artifacts under `results/cv/` and `results/submissions/`
- Rewrite README with headline scorecard (published vs holdout-measured)

## v1.0

Initial fully annotated teaching case: competition-style reference,
standards-aligned pipeline, evaluation notebooks, teaching notes.
