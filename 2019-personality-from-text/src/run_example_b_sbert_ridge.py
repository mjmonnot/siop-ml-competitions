# ============================================================
# Example B (TEACHING NOTE VERSION)
# 2019 SIOP ML: SBERT Embeddings + Ridge Regression
# ============================================================
# BIG IDEA (PLAIN ENGLISH)
# ------------------------------------------------------------
# In Example A we used "classic" text features (TF-IDF).
# Here we try a more "modern NLP" approach:
#
#   1) Use a pretrained SBERT model to turn each person’s text
#      into a dense numeric fingerprint called an "embedding".
#   2) Fit Ridge regression to predict each Big Five trait score
#      from those embeddings.
#
# What is an embedding?
# - Think of it as a compact summary of meaning:
#   each essay becomes a vector of (say) 384 numbers.
# - Similar texts tend to have embeddings close together.
#
# Why try this?
# - SBERT can capture meaning beyond simple word counts.
#
# Why might it NOT outperform TF-IDF here?
# - Competition data can be relatively small/short-text.
# - TF-IDF is often surprisingly strong when data is limited.
# - Pretrained embeddings may not align perfectly with the
#   specific psychological constructs being predicted.
#
# EVALUATION (HOW WE GRADE OURSELVES)
# ------------------------------------------------------------
# The competition metric is:
#   - Pearson correlation (r) per trait
#   - Mean r across all five traits
#
# We use 5-fold "out-of-fold" (OOF) predictions so the evaluation
# is honest (no training on the row you are scoring).
#
# OUTPUTS
# ------------------------------------------------------------
# - Prints: per-trait OOF r and mean r
# - Writes figures: results/figures/example_b/*
# - Writes submission-style CSVs:
#       results/submissions/submission_dev.csv
#       results/submissions/submission_test.csv
#
# NOTE ON DEPENDENCIES
# ------------------------------------------------------------
# This script requires:
#   pip install sentence-transformers
# Optionally:
#   pip install matplotlib
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd

# ----------------------------
# Correlation (competition metric)
# ----------------------------
# Correlation measures whether we get the ranking right:
# do higher-scoring people get higher predictions?
try:
    from scipy.stats import pearsonr
    def corr(x, y):
        return pearsonr(x, y)[0]
except Exception:
    def corr(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + 1e-12
        return float((x * y).sum() / denom)

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# SBERT
# SentenceTransformer downloads a pretrained model the first time you run it
# and caches it locally.
from sentence_transformers import SentenceTransformer


# ============================================================
# SECTION 1 — Configuration (edit here if needed)
# ============================================================

DEFAULT_DATA_PATH = "data/raw/2019_siop_ml_comp_data.txt"

TEXT_COLS = [f"open_ended_{i}" for i in range(1, 6)]
TARGETS = ["A_Scale_score", "C_Scale_score", "E_Scale_score", "N_Scale_score", "O_Scale_score"]

# A popular default SBERT model:
# - Fast
# - Often a good “general purpose” semantic encoder
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# SECTION 2 — Small helpers (kept in-file for teaching)
# ============================================================

def ensure_dir(path: str) -> None:
    """Create a folder if it does not exist."""
    os.makedirs(path, exist_ok=True)


def safe_pearsonr(y_true, y_pred) -> float:
    """
    Pearson correlation that won’t crash on weird edge cases.

    Plain English:
    - Correlation is undefined if all values are identical (std=0).
    - If that happens, we return 0.0 to avoid exploding the script.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(corr(y_true, y_pred))


def mean_r_across_traits(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, targets: list[str]) -> pd.Series:
    """
    Compute r for each trait and return a Series of r values.

    Plain English:
    - This is “the scoreboard” for the competition.
    - Each trait has its own correlation.
    """
    rs = {t: safe_pearsonr(y_true_df[t].values, y_pred_df[t].values) for t in targets}
    return pd.Series(rs)


def save_fig(path: str) -> None:
    """
    Save a plot safely (if matplotlib exists).
    """
    if plt is None:
        return
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# SECTION 3 — Diagnostics plots (optional but helpful for teaching)
# ============================================================

def make_diagnostics_plots(train_df, targets, oof_pred_df, out_dir):
    """
    Create quick diagnostic figures.

    Why bother?
    - Plots help students see what “good” vs “bad” predictions look like.
    - For example:
      * Predicted vs actual scatter should slope upward if working.
      * Residual plot should look like noise (no obvious patterns).
    """
    if plt is None:
        return

    ensure_dir(out_dir)

    # ---- 3.1 Trait intercorrelations (the ground truth structure)
    # Teaching note:
    # Big Five traits are not independent; they often correlate.
    corr_mat = train_df[targets].corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(corr_mat.values, aspect="auto")
    plt.xticks(range(len(targets)), targets, rotation=45, ha="right")
    plt.yticks(range(len(targets)), targets)
    plt.colorbar()
    plt.title("Train: Trait Intercorrelations (Ground Truth)")
    save_fig(os.path.join(out_dir, "01_trait_intercorrelations.png"))

    # ---- 3.2 Per-trait diagnostics
    for t in targets:
        y = train_df[t].values
        p = oof_pred_df[t].values
        r = safe_pearsonr(y, p)

        # Pred vs actual
        plt.figure(figsize=(5, 4))
        plt.scatter(y, p, s=10, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("OOF Predicted")
        plt.title(f"{t}: Pred vs Actual (OOF) | r={r:.3f}")
        save_fig(os.path.join(out_dir, f"02_{t}_pred_vs_actual.png"))

        # Residuals
        plt.figure(figsize=(5, 4))
        plt.scatter(p, y - p, s=10, alpha=0.6)
        plt.axhline(0, linewidth=1)
        plt.xlabel("OOF Predicted")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.title(f"{t}: Residuals vs Pred (OOF)")
        save_fig(os.path.join(out_dir, f"03_{t}_residuals.png"))


# ============================================================
# SECTION 4 — Core idea: SBERT embeddings
# ============================================================

def build_all_text(df: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    """
    Clean and concatenate the 5 prompts into one text field.

    Plain English:
    - Many NLP models work best with a single text input per person.
    - We create a single essay by joining the prompts.
    """
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)
    return df[text_cols].agg(" ".join, axis=1)


def encode_with_sbert(sbert: SentenceTransformer, texts: list[str], show_progress: bool) -> np.ndarray:
    """
    Encode a list of texts into embeddings.

    Plain English:
    - For each person, SBERT outputs a fixed-length vector.
    - These vectors become the “features” for our regression model.

    Practical note:
    - This can be slow the first time because the model downloads.
    - After that, it uses a local cache.
    """
    return np.asarray(
        sbert.encode(texts, show_progress_bar=show_progress),
        dtype=np.float32
    )


# ============================================================
# SECTION 5 — Cross-validation (honest evaluation)
# ============================================================

def oof_ridge_predictions(X: np.ndarray, y: np.ndarray, alpha: float, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """
    Out-of-fold predictions for Ridge regression.

    Plain English:
    - Split training data into K parts
    - Train on K-1 parts
    - Predict on the held-out part
    - Repeat so every row gets a prediction from a model that never saw its label

    Why it matters:
    - Prevents “training on the test”.
    - Produces a realistic estimate of performance.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=float)

    for tr_idx, va_idx in kf.split(X):
        model = Ridge(alpha=alpha)
        model.fit(X[tr_idx], y[tr_idx])
        preds[va_idx] = model.predict(X[va_idx])

    return preds


# ============================================================
# SECTION 6 — Main workflow
# ============================================================

def main(data_path: str, seed: int = 42, ridge_alpha: float = 10.0):
    # ---- 6.1 Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find data file at: {data_path}")

    df = pd.read_csv(data_path)

    # ---- 6.2 Build single text field
    df["all_text"] = build_all_text(df, TEXT_COLS)

    # ---- 6.3 Split into Train / Dev / Test
    train = df[df["Dataset"] == "Train"].copy().reset_index(drop=True)
    dev = df[df["Dataset"] == "Dev"].copy().reset_index(drop=True)
    test = df[df["Dataset"] == "Test"].copy().reset_index(drop=True)

    print("Dataset sizes:")
    print(f"  Train: {len(train)}")
    print(f"  Dev  : {len(dev)}")
    print(f"  Test : {len(test)}")

    # ---- 6.4 Load SBERT model
    print("\nLoading SBERT model...")
    sbert = SentenceTransformer(SBERT_MODEL_NAME)

    # ---- 6.5 Encode text -> embeddings
    # Teaching note:
    # After this step, text is no longer "words", it's numeric vectors.
    print("Encoding text with SBERT (this may take a bit the first time)...")
    X_train = encode_with_sbert(sbert, train["all_text"].tolist(), show_progress=True)

    # Dev/Test embeddings can be computed too (if they exist)
    X_dev = encode_with_sbert(sbert, dev["all_text"].tolist(), show_progress=False) if len(dev) else None
    X_test = encode_with_sbert(sbert, test["all_text"].tolist(), show_progress=False) if len(test) else None

    # ---- 6.6 Cross-validated OOF predictions
    print("\nRunning 5-fold CV (SBERT + Ridge)...\n")

    oof_preds = pd.DataFrame(index=train.index)

    for t in TARGETS:
        y = train[t].values.astype(float)

        # OOF predictions for this trait
        p_oof = oof_ridge_predictions(X_train, y, alpha=ridge_alpha, n_splits=5, seed=seed)

        oof_preds[t] = p_oof
        r = safe_pearsonr(y, p_oof)
        print(f"{t}: OOF r = {r:.3f}")

    # ---- 6.7 Competition metric summary
    rs = mean_r_across_traits(train[TARGETS], oof_preds[TARGETS], TARGETS)

    print("\nOOF CV results (Pearson r):")
    for t in TARGETS:
        print(f"  {t}: {rs[t]:.3f}")
    print(f"\nCV mean r (competition metric): {rs.mean():.3f}\n")

    # ---- 6.8 Diagnostics plots
    fig_dir = os.path.join("results", "figures", "example_b")
    make_diagnostics_plots(train, TARGETS, oof_preds[TARGETS], fig_dir)
    if plt is not None:
        print(f"Wrote figures to: {fig_dir}")
    else:
        print("matplotlib not installed; skipping figures.")

    # ---- 6.9 Fit on full Train and predict Dev/Test
    # Teaching note:
    # After CV evaluation, we train on ALL Train rows to maximize learning.
    ensure_dir(os.path.join("results", "submissions"))

    dev_out = pd.DataFrame({"Respondent_ID": dev["Respondent_ID"]}) if len(dev) else None
    test_out = pd.DataFrame({"Respondent_ID": test["Respondent_ID"]}) if len(test) else None

    for t in TARGETS:
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_train, train[t].values.astype(float))

        # Predict Dev/Test if available
        if dev_out is not None:
            dev_out[t.replace("_Scale_score", "_Pred")] = model.predict(X_dev)

        if test_out is not None:
            test_out[t.replace("_Scale_score", "_Pred")] = model.predict(X_test)

    if dev_out is not None:
        dev_path = "results/submissions/submission_dev.csv"
        dev_out.to_csv(dev_path, index=False)
        print(f"Wrote: {dev_path}")

    if test_out is not None:
        test_path = "results/submissions/submission_test.csv"
        test_out.to_csv(test_path, index=False)
        print(f"Wrote: {test_path}")

    # ---- 6.10 Teaching reflection (printed, optional)
    print("\nTeaching note:")
    print(" - If SBERT underperforms TF-IDF, that's common on small datasets.")
    print(" - TF-IDF can exploit surface cues (word choices) efficiently.")
    print(" - SBERT embeddings capture meaning, but may not align with trait signals.")
    print(" - Either way, transparent evaluation (OOF) is what makes this credible.\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example B (Teaching Note): SBERT + Ridge (2019 SIOP ML)")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge_alpha", type=float, default=10.0, help="Ridge regularization strength.")
    args = parser.parse_args()

    main(args.data_path, seed=args.seed, ridge_alpha=args.ridge_alpha)

