# Teaching Notes (Lay-reader friendly)

## The main teaching idea
Fairness can be introduced at different points in the pipeline:
- during scoring (competition-style)
- during decision policy (standards-aligned)

This repo shows why many professional guidance frameworks prefer:
- **scores that are blind to protected group membership**
- **fairness auditing and governance at the evaluation/threshold stage**

## Recommended classroom flow (60–90 minutes)
1) Read the scoring function (what the competition rewards)
2) Skim [docs/WINNERS_SYNTHESIS.md](docs/WINNERS_SYNTHESIS.md) — how Place 1–4 differed
3) Inspect the competition-style baseline (how fairness enters scoring)
4) Inspect the standards-aligned pipeline (fairness enters decision governance)
5) Show `results/cv/compare_summary.csv` (proxy vs no-proxy vs standards-aligned)
6) Open the unconstrained landmine row / [KNOWN_LANDMINES.md](docs/KNOWN_LANDMINES.md)
7) Optional: run the trade-off notebook and discuss operating points

## Discussion questions
- Why does using protected status inside scoring create defensibility challenges?
- How does the chosen cut score affect both utility and AIR?
- Why is a 99-point “score” from hiring everyone not a success?
- What additional evidence would you need before operationalizing a model (job analysis, validation)?

## Deeper reading in this case
- [SOLUTION.md](SOLUTION.md) — measured comparative answer
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — score-path vs decision-path
- [docs/STATUS.md](docs/STATUS.md) — published vs holdout-measured honesty
