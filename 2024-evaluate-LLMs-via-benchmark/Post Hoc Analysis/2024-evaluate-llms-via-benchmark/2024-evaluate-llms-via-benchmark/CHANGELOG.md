# Changelog

All notable changes to this repo. Format borrowed from the 2026 meta-analysis repo.

## v1.2 — 2026-05-31 — Synthetic-data pipeline executed end-to-end

- Ran the full unified harness against synthetic inputs (`gpt-4o-2024-08-06`, default adapters, cache enabled).
- Measured test-split scores: empathy acc=1.000, interview cos=.309, clarity r=.959, fairness acc=1.000, composite=.817.
- Measured dev-split scores: empathy acc=1.000, interview cos=.292, clarity r=.962, fairness acc=1.000, composite=.814.
- Self-consistency (N=5) on empathy/fairness test: no change from base run (already at ceiling on synthetic data).
- Updated README headline table, `docs/STATUS.md`, and `notebooks/05_comparison.ipynb` with measured numbers.
- Submission CSVs written to `submissions/`.

## v1.1 — 2026-05 — Synthetic data fallback

- Added `scripts/make_synthetic_data.py` which generates plausible input CSVs that join correctly to the public label files. Lets the pipeline run end-to-end when the official EvalAI input files aren't available.
- Updated STATUS.md to be explicit about what synthetic-data scores tell you and don't tell you (good for engineering validation; not comparable to the 2024 winners' published numbers).
- README's "Input data" section now mentions the synthetic-data option as the default fallback.

## v1.0 — 2026-05 — Initial post-hoc submission

- All four adapters (empathy, interview, clarity, fairness) implemented against a shared harness in `src/`.
- Five notebooks: one per task plus a comparison scorecard.
- Five docs (ARCHITECTURE, KNOWN_LANDMINES, WINNERS_SYNTHESIS, STATUS, ADAPTING_TO_NEW_TASKS).
- Self-consistency wrapper baked into the harness (off by default; the original PAID Team didn't use it, but Akben did — see KNOWN_LANDMINES.md Landmine 3).
- Few-shot example selection supports random and similarity-based (default: similarity for empathy and fairness; random for clarity to avoid leakage; full-history for interview).
- Default model: `gpt-4o-2024-08-06`. CLI flag to switch to `gpt-4-0125-preview` to match what the winners actually ran.

## v0.x — pre-release sketches

Versioning here mirrors what I usually do in real submissions: rough numbered exploration before a clean v1 cut. None of the v0.x branches are preserved in this repo.

- v0.1 — sketch of the harness against a single empathy row, no caching, no retries
- v0.2 — added retries and structured-output JSON mode; switched to a registry pattern for adapters
- v0.3 — pulled scoring out of the adapters into a separate module; added the `--selftest` flags
- v0.4 — added the similarity-based example picker; immediately noticed clarity got worse with it (Landmine 6)
- v0.5 — added Akben-style self-consistency as an opt-in flag rather than a default
