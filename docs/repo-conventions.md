# Repo conventions

## Data
- Raw competition data goes in: `YEAR/data/raw/` (not committed)
- Processed data goes in: `YEAR/data/processed/` (not committed)

## Outputs
- CV logs: `YEAR/results/cv/`
- Figures: `YEAR/results/figures/`
- Submissions: `YEAR/results/submissions/`

## Runs should be reproducible
- Fix random seeds
- Write key outputs to `results/`
- Keep code runnable from the year directory
