# Instructor Notes (2019 — Personality from Text)

## Learning objectives
- Understand the SIOP 2019 competition metric (mean Pearson r across 5 traits)
- Build a reproducible baseline using TF-IDF + Ridge
- Compare a classic linear text baseline vs modern embeddings
- Diagnose why “newer” methods may not improve performance

## What to run
- Example A (TF-IDF blend): `python -m src.run_tfidf_prompt_plus_all_blend --data_path "data/raw/2019_siop_ml_comp_data.txt"`
- Example B (SBERT + Ridge): `python -m src.run_example_b_sbert_ridge --data_path "data/raw/2019_siop_ml_comp_data.txt"`

## Discussion prompts
- Why does Openness lag other traits?
- Why do regularized linear models often perform strongly on small NLP datasets?
- What does “overfitting to the public leaderboard” look like?
