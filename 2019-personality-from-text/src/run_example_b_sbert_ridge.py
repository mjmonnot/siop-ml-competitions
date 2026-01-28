# ============================================================
# Example B (2019 SIOP ML): SBERT Embeddings + Ridge Regression
# ============================================================
# PURPOSE (HIGH LEVEL)
# ------------------------------------------------------------
# This script demonstrates a "modern NLP" approach to the
# 2019 SIOP Machine Learning Competition:
#
#   - Use a pretrained Sentence-BERT (SBERT) model to convert
#     each participant’s text into dense semantic embeddings
#   - Train a simple Ridge regression model to predict each
#     Big Five trait score from those embeddings
#
# The goal is NOT to win the competition, but to show:
#   (a) how an SBERT-based pipeline can be implemented cleanly
#   (b) why it does NOT necessarily outperform classic TF-IDF
#       approaches on small, short-text datasets
#
# As in Example A, we evaluate using:
#   - 5-fold cross-validated out-of-fold (OOF) predictions
#   - Pearson correlation (r) per trait
#   - Mean r across traits (competition metric)
#
# OUTPUTS
# ------------------------------------------------------------
# - Prints:
#     per-trait OOF r and mean r
# - Writes:
#     results/submissions/submission_dev.csv
#     results/submissions/submission_test.csv
# - Writes figures:
#     results/figures/example_b/*
# ============================================================


# ============================================================
# IMPORTS
# ============================================================
import os
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer


# ============================================================
# FILE LOCATIONS & COLUMN DEFINITIONS
# ============================================================
DATA_PATH = "data/raw/2019_siop_ml_comp_data.txt"  # update if needed

TEXT_COLS = [f"open_ended_{i}" for i in range(1, 6)]
TARGETS = ["A_Scale_score", "C_Scale_score", "E_Scale_score", "N_Scale_score", "O_Scale_score"]

# SBERT model choice:
# - all-MiniLM-L6-v2 is fast, stable, and commonly used
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# METRIC HELPERS (COMPETITION SCORING)
# ============================================================
def safe_pearsonr(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def mean_r_across_traits(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, targets: list[str]) -> pd.Series:
    rs = {t: safe_pearsonr(y_true_df[t].values, y_pred_df[t].values) for t in targets}
    return pd.Series(rs)


# ============================================================
# DIAGNOSTICS & FIGURES
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_diagnostics_plots(train_df, targets, oof_pred_df, out_dir):
    ensure_dir(out_dir)

    # Trait intercorrelations
    corr = train_df[targets].corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(targets)), targets, rotation=45, ha="right")
    plt.yticks(range(len(targets)), targets)
    plt.colorbar()
    plt.title("Train: Trait Intercorrelations (Ground Truth)")
    save_fig(os.path.join(out_dir, "01_trait_intercorrelations.png"))

    for t in targets:
        y = train_df[t].values
        p = oof_pred_df[t].values
        r = safe_pearsonr(y, p)

        # Pred vs actual
        plt.figure(figsize=(5, 4))
        plt.scatter(y, p, s=10)
        plt.xlabel("Actual")
        plt.ylabel("OOF Predicted")
        plt.title(f"{t}: Pred vs Actual (OOF) | r={r:.3f}")
        save_fig(os.path.join(out_dir, f"02_{t}_pred_vs_actual.png"))

        # Residuals
        plt.figure(figsize=(5, 4))
        plt.scatter(p, y - p, s=10)
        plt.axhline(0, linewidth=1)
        plt.xlabel("OOF Predicted")
        plt.ylabel("Residual")
        plt.title(f"{t}: Residuals vs Pred (OOF)")
        save_fig(os.path.join(out_dir, f"03_{t}_residuals.png"))


# ============================================================
# MAIN WORKFLOW
# ============================================================
def main(data_path: str = DATA_PATH):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    df = pd.read_csv(data_path)

    # Clean text and concatenate all prompts
    for c in TEXT_COLS:
        df[c] = df[c].fillna("").astype(str)

    df["all_text"] = df[TEXT_COLS].agg(" ".join, axis=1)

    train = df[df["Dataset"] == "Train"].copy().reset_index(drop=True)
    dev = df[df["Dataset"] == "Dev"].copy().reset_index(drop=True)
    test = df[df["Dataset"] == "Test"].copy().reset_index(drop=True)

    print("Dataset sizes:")
    print(train["Dataset"].value_counts())

    # --------------------------------------------------------
    # SBERT embedding step
    # --------------------------------------------------------
    print("\nLoading SBERT model...")
    sbert = SentenceTransformer(SBERT_MODEL_NAME)

    print("Encoding text with SBERT...")
    X_train = sbert.encode(train["all_text"].tolist(), show_progress_bar=True)
    X_dev = sbert.encode(dev["all_text"].tolist(), show_progress_bar=False) if not dev.empty else None
    X_test = sbert.encode(test["all_text"].tolist(), show_progress_bar=False) if not test.empty else None

    # --------------------------------------------------------
    # Cross-validated OOF predictions
    # --------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = pd.DataFrame(index=train.index)

    print("\nRunning 5-fold CV (SBERT + Ridge)...\n")

    for t in TARGETS:
        y = train[t].values
        p_oof = np.zeros(len(train))

        for tr_idx, va_idx in kf.split(X_train):
            model = Ridge(alpha=10.0)
            model.fit(X_train[tr_idx], y[tr_idx])
            p_oof[va_idx] = model.predict(X_train[va_idx])

        oof_preds[t] = p_oof
        r = safe_pearsonr(y, p_oof)
        print(f"{t}: OOF r = {r:.3f}")

    # --------------------------------------------------------
    # Competition metric
    # --------------------------------------------------------
    rs = mean_r_across_traits(train[TARGETS], oof_preds[TARGETS], TARGETS)
    print("\nOOF CV results (Pearson r):")
    for t in TARGETS:
        print(f"{t}: {rs[t]:.3f}")
    print(f"\nCV mean r (competition metric): {rs.mean():.3f}\n")

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    fig_dir = os.path.join("results", "figures", "example_b")
    make_diagnostics_plots(train, TARGETS, oof_preds[TARGETS], fig_dir)
    print(f"Wrote figures to: {fig_dir}")

    # --------------------------------------------------------
    # Fit on full Train and predict Dev/Test
    # --------------------------------------------------------
    dev_out = pd.DataFrame({"Respondent_ID": dev["Respondent_ID"]}) if not dev.empty else None
    test_out = pd.DataFrame({"Respondent_ID": test["Respondent_ID"]}) if not test.empty else None

    for t in TARGETS:
        model = Ridge(alpha=10.0)
        model.fit(X_train, train[t].values)

        if dev_out is not None:
            dev_out[t.replace("_Scale_score", "_Pred")] = model.predict(X_dev)

        if test_out is not None:
            test_out[t.replace("_Scale_score", "_Pred")] = model.predict(X_test)

    ensure_dir(os.path.join("results", "submissions"))

    if dev_out is not None:
        dev_path = "results/submissions/submission_dev.csv"
        dev_out.to_csv(dev_path, index=False)
        print(f"Wrote: {dev_path}")

    if test_out is not None:
        test_path = "results/submissions/submission_test.csv"
        test_out.to_csv(test_path, index=False)
        print(f"Wrote: {test_path}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Example B: SBERT + Ridge (2019 SIOP ML)")
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    args = parser.parse_args()

    main(args.data_path)
