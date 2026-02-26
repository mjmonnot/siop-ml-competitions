# SIOP 2023 ML Competition (Post‑Hoc Replication) — Decision Making from Text

Drop‑in, reproducible scaffold for the **SIOP 2023 Machine Learning Competition** winners repo:
- https://github.com/izk8/2023_SIOP_Machine_Learning_Winners

Organizers describe the task as predicting **assessment‑center “decision making” ratings** from **open‑ended text**.

## What’s included
- End‑to‑end pipeline: **validate → preprocess → features → train → evaluate → fairness audit**
- Transparent baseline: **TF‑IDF + Ridge regression**
- Transformer‑ready template (SBERT) you can enable later
- Example **outputs + figures** generated from an **included synthetic demo dataset**
  - Replace `data/raw/train.csv` with real competition files and rerun.

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_all
```

## Data contract (defaults)
Required:
- `text` (string)
- `target` (numeric)

Optional group columns (for fairness audit):
- `group_gender`
- `group_race`

Edit `src/config.py` if your columns differ.

## Outputs
- `results/metrics/`
- `results/predictions/`
- `results/fairness/`
- `figures/`

