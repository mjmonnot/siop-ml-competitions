"""
SIOP ML 2019 – Prompt-Specific + Concatenated TF-IDF (word ngrams) + Ridge Regression

What this script does (in plain English):
- Loads the full 2019 competition dataset (Train/Dev/Test rows in one file)
- Uses the 5 open-ended SJT responses as predictors (text)
- Predicts each Big Five trait score (A, C, E, N, O) using only text
- Builds TF-IDF features:
    • one vectorizer per prompt (open_ended_1 ... open_ended_5)
    • plus one additional vectorizer on all prompts concatenated (text_all)
- Fits a separate Ridge regression model per trait
- Evaluates with 5-fold CV using the competition metric: mean Pearson r across traits
- Writes:
    • results/figures/{trait}_pred_vs_true.png
    • results/submissions/submission_dev.csv
    • results/submissions/submission_test.csv

Run:
  python .\src\run_prompt_tfidf_ridge.py --data_path "data/raw/2019_siop_ml_comp_data.txt"
"""

import argparse
import os
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# seaborn is optional (nice plots). If missing, we fall back to matplotlib-only.
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


# -----------------------------
# Metrics / plotting utilities
# -----------------------------
def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r with a safeguard for constant predictions."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def mean_r_across_traits(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, trait_cols: list[str]) -> tuple[float, pd.Series]:
    """Competition metric: mean Pearson r across traits."""
    rs = {t: safe_pearsonr(y_true_df[t].values, y_pred_df[t].values) for t in trait_cols}
    s = pd.Series(rs).sort_index()
    return float(s.mean()), s


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, trait: str, out_dir: str) -> None:
    """Scatter plot of predictions vs true with r in the title."""
    r = safe_pearsonr(y_true, y_pred)
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(5, 5))
    if _HAS_SNS:
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    else:
        plt.scatter(y_true, y_pred, alpha=0.5)

    plt.xlabel("True score")
    plt.ylabel("Predicted score")
    plt.title(f"{trait} (OOF r = {r:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{trait}_pred_vs_true.png"))
    plt.close()


# -----------------------------
# Main workflow
# -----------------------------
def main(data_path: str, n_splits: int = 5, random_state: int = 42) -> None:
    # 1) Load
    df = pd.read_csv(data_path)

    text_cols = [f"open_ended_{i}" for i in range(1, 6)]
    trait_cols = ["A_Scale_score", "C_Scale_score", "E_Scale_score", "N_Scale_score", "O_Scale_score"]

    train_df = df[df["Dataset"] == "Train"].reset_index(drop=True)
    dev_df = df[df["Dataset"] == "Dev"].reset_index(drop=True)
    test_df = df[df["Dataset"] == "Test"].reset_index(drop=True)

    # 2) Ensure clean text (no NaNs, all strings)
    for col in text_cols:
        train_df[col] = train_df[col].fillna("").astype(str)
        dev_df[col] = dev_df[col].fillna("").astype(str)
        test_df[col] = test_df[col].fillna("").astype(str)

    # 3) Add concatenated column (often restores + improves baseline)
    train_df["text_all"] = train_df[text_cols].agg(" ".join, axis=1)
    dev_df["text_all"] = dev_df[text_cols].agg(" ".join, axis=1)
    test_df["text_all"] = test_df[text_cols].agg(" ".join, axis=1)

    text_cols_plus = text_cols + ["text_all"]

    y_true = train_df[trait_cols].copy()

    # 4) TF-IDF + Ridge pipeline
    # One TF-IDF per column. This avoids the FeatureUnion/DataFrame pitfalls.
    tfidf = ColumnTransformer(
        transformers=[
            (
                f"tfidf_{col}",
                TfidfVectorizer(
                    max_features=8000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    stop_words="english",
                ),
                col,
            )
            for col in text_cols_plus
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    model = Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )

    # 5) CV (OOF predictions)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof = pd.DataFrame(
        np.zeros((len(train_df), len(trait_cols))),
        columns=trait_cols,
    )

    print(f"\nRunning {n_splits}-fold CV...\n")

    for trait in trait_cols:
        print(f"Trait: {trait}")
        y_trait = train_df[trait].values

        for tr_idx, val_idx in kf.split(train_df):
            X_tr = train_df.loc[tr_idx, text_cols_plus]
            X_val = train_df.loc[val_idx, text_cols_plus]

            model.fit(X_tr, y_trait[tr_idx])
            preds = model.predict(X_val)
            oof.loc[val_idx, trait] = preds

    # 6) Evaluation (competition metric)
    mean_r, per_trait = mean_r_across_traits(y_true, oof, trait_cols)

    print("\nCV results:")
    for t in trait_cols:
        print(f"{t}: {per_trait[t]:.3f}")
        plot_pred_vs_true(
            y_true=y_true[t].values,
            y_pred=oof[t].values,
            trait=t,
            out_dir="results/figures",
        )

    print(f"\nCV mean r: {mean_r:.3f}")

    # 7) Fit full model per trait and predict Dev/Test
    dev_out = dev_df[["Respondent_ID"]].copy()
    test_out = test_df[["Respondent_ID"]].copy()

    for trait in trait_cols:
        model.fit(train_df[text_cols_plus], train_df[trait].values)
        dev_out[trait.replace("_Scale_score", "_Pred")] = model.predict(dev_df[text_cols_plus])
        test_out[trait.replace("_Scale_score", "_Pred")] = model.predict(test_df[text_cols_plus])

    os.makedirs("results/submissions", exist_ok=True)
    dev_out.to_csv("results/submissions/submission_dev.csv", index=False)
    test_out.to_csv("results/submissions/submission_test.csv", index=False)

    print("\nWrote:")
    print("  results/submissions/submission_dev.csv")
    print("  results/submissions/submission_test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to full competition dataset (contains Train/Dev/Test rows).",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    main(args.data_path, n_splits=args.n_splits, random_state=args.random_state)
