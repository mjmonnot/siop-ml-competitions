"""
SIOP ML 2019 – Concatenated TF-IDF (word + char n-grams) + Ridge

Why this baseline:
- Word n-grams capture content (what people say)
- Character n-grams capture style (how people say it): typos, punctuation, affixes, hedging, etc.
- For personality-from-text, char n-grams are often a big lift over word-only TF-IDF.

What this script does:
- Loads full dataset with Train/Dev/Test rows
- Builds one concatenated text field from open_ended_1..5
- Trains one Ridge regression per trait
- 5-fold CV with competition metric: mean Pearson r across traits
- Writes:
  - results/figures/{trait}_pred_vs_true.png
  - results/submissions/submission_dev.csv
  - results/submissions/submission_test.csv

Run:
  python .\src\run_tfidf_word_char_ridge.py --data_path "data/raw/2019_siop_ml_comp_data.txt"
"""

import argparse
import os
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# seaborn optional
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


# -----------------------------
# Utilities
# -----------------------------
def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def mean_r_across_traits(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, trait_cols: list[str]) -> tuple[float, pd.Series]:
    rs = {t: safe_pearsonr(y_true_df[t].values, y_pred_df[t].values) for t in trait_cols}
    s = pd.Series(rs).sort_index()
    return float(s.mean()), s


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, trait: str, out_dir: str) -> None:
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


def build_model(alpha: float, random_state: int) -> Pipeline:
    # Word TF-IDF (content)
    word_tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=60000,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )

    # Character TF-IDF (style)
    char_tfidf = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=80000,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )

    feats = FeatureUnion(
        transformer_list=[
            ("word", word_tfidf),
            ("char", char_tfidf),
        ],
        n_jobs=None,
    )

    model = Pipeline(
        steps=[
            ("feats", feats),
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", Ridge(alpha=alpha, random_state=random_state)),
        ]
    )
    return model


# -----------------------------
# Main
# -----------------------------
def main(data_path: str, n_splits: int = 5, random_state: int = 42, alpha: float = 10.0) -> None:
    df = pd.read_csv(data_path)

    text_cols = [f"open_ended_{i}" for i in range(1, 6)]
    trait_cols = ["A_Scale_score", "C_Scale_score", "E_Scale_score", "N_Scale_score", "O_Scale_score"]

    train_df = df[df["Dataset"] == "Train"].reset_index(drop=True)
    dev_df = df[df["Dataset"] == "Dev"].reset_index(drop=True)
    test_df = df[df["Dataset"] == "Test"].reset_index(drop=True)

    # Clean text
    for col in text_cols:
        train_df[col] = train_df[col].fillna("").astype(str)
        dev_df[col] = dev_df[col].fillna("").astype(str)
        test_df[col] = test_df[col].fillna("").astype(str)

    # Concatenate prompts into one document per person
    train_text = train_df[text_cols].agg(" ".join, axis=1).astype(str).values
    dev_text = dev_df[text_cols].agg(" ".join, axis=1).astype(str).values
    test_text = test_df[text_cols].agg(" ".join, axis=1).astype(str).values

    y_true = train_df[trait_cols].copy()

    model = build_model(alpha=alpha, random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof = pd.DataFrame(np.zeros((len(train_df), len(trait_cols))), columns=trait_cols)

    print(f"\nRunning {n_splits}-fold CV (alpha={alpha})...\n")

    for trait in trait_cols:
        print(f"Trait: {trait}")
        y = train_df[trait].values

        for tr_idx, val_idx in kf.split(train_text):
            X_tr = train_text[tr_idx]
            X_val = train_text[val_idx]

            model.fit(X_tr, y[tr_idx])
            preds = model.predict(X_val)
            oof.loc[val_idx, trait] = preds

    mean_r, per_trait = mean_r_across_traits(y_true, oof, trait_cols)

    print("\nCV results:")
    for t in trait_cols:
        print(f"{t}: {per_trait[t]:.3f}")
        plot_pred_vs_true(y_true[t].values, oof[t].values, t, out_dir="results/figures")

    print(f"\nCV mean r: {mean_r:.3f}")

    # Fit full model and predict Dev/Test
    dev_out = dev_df[["Respondent_ID"]].copy()
    test_out = test_df[["Respondent_ID"]].copy()

    for trait in trait_cols:
        y = train_df[trait].values
        model.fit(train_text, y)
        dev_out[trait.replace("_Scale_score", "_Pred")] = model.predict(dev_text)
        test_out[trait.replace("_Scale_score", "_Pred")] = model.predict(test_text)

    os.makedirs("results/submissions", exist_ok=True)
    dev_out.to_csv("results/submissions/submission_dev.csv", index=False)
    test_out.to_csv("results/submissions/submission_test.csv", index=False)

    print("\nWrote:")
    print("  results/submissions/submission_dev.csv")
    print("  results/submissions/submission_test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=10.0)
    args = parser.parse_args()

    main(args.data_path, n_splits=args.n_splits, random_state=args.random_state, alpha=args.alpha)
