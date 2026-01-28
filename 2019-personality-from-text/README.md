# 2019 SIOP Machine Learning Competition  
## Predicting Personality Traits from Open-Ended Text

This directory contains **reproducible, fully runnable solutions** for the  
**2019 SIOP Machine Learning Competition**, which focused on predicting Big Five
personality traits from open-ended situational judgment item (SJI) responses.

The goal of the competition was to **maximize mean Pearson correlation (r)**
between predicted and observed trait scores across five traits:
- Agreeableness (A)
- Conscientiousness (C)
- Extraversion (E)
- Neuroticism (N)
- Openness (O)

---

## 📂 Repository Structure

019-personality-from-text/
│
├── data/
│ └── raw/
│ └── 2019_siop_ml_comp_data.txt
│
├── src/
│ ├── run_example_a_tfidf_blend.py
│ ├── run_example_b_sbert_ridge.py
│ ├── features.py
│ ├── models.py
│ ├── evaluation.py
│ └── init.py
│
├── results/
│ ├── figures/
│ │ ├── example_a/
│ │ └── example_b/
│ └── submissions/
│ ├── submission_dev.csv
│ └── submission_test.csv
│
├── requirements.txt
└── README.md

---

## 🧠 Modeling Approaches

### **Example A — TF-IDF + Ridge (Blended Baseline)**

**File:** `src/run_example_a_tfidf_blend.py`

A strong, competition-appropriate baseline that closely approximates top
submissions from 2019.

**Key characteristics:**
- Word + character n-gram TF-IDF features
- Prompt-specific models + all-text models
- Out-of-fold (OOF) predictions
- Z-scored Ridge blending per trait
- Explicit optimization for **correlation**, not MSE

**Why it works well:**
- Linear models preserve rank order
- High-dimensional sparse features capture stylistic signals
- Blending stabilizes variance across prompts

---

### **Example B — SBERT + Ridge (Modern Contrast Model)**

**File:** `src/run_example_b_sbert_ridge.py`

A modern embedding-based approach using sentence transformers.

**Key characteristics:**
- SBERT embeddings (`all-mpnet-base-v2`)
- Mean-pooled sentence representations
- Ridge regression per trait
- Identical CV and evaluation protocol

**Why it underperforms Example A:**
- Embeddings optimize semantic similarity, not stylistic variance
- Reduced dimensionality limits rank-order sensitivity
- Strong regularization compresses predictive spread

This contrast is intentional and instructional.

---

## 📊 Evaluation Metric

The competition metric is:

> **Mean Pearson correlation (r) across all five traits**

All scripts report:
- Per-trait out-of-fold r
- Mean r (competition metric)
- Reproducible CV splits

---

## ▶️ How to Run

### Install dependencies
```powershell
python -m pip install -r requirements.txt

Example A (TF-IDF Blend)
python -m src.run_example_a_tfidf_blend --data_path "data/raw/2019_siop_ml_comp_data.txt"

Example B (SBERT)
python -m src.run_example_b_sbert_ridge --data_path "data/raw/2019_siop_ml_comp_data.txt"

Outputs:
Submission files → results/submissions/
Diagnostic figures → results/figures/

🎯 Key Takeaways

Modern deep embeddings do not automatically outperform well-tuned linear text models
Competition metrics matter more than model sophistication
Reproducibility and interpretability remain critical in applied ML

Thompson, I., Koenig, N., & Lui, M. The 2019 SIOP Machine Learning Competition. Presented at the 34th annual Society for Industrial and Organizational Psychology conference in Austin, TX.
