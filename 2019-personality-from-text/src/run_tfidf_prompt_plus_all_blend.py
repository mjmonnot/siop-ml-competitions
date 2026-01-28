# ============================================================
# Example A (2019 SIOP ML): TF-IDF (word+char) + Ridge + Blend
# ============================================================
# PURPOSE (HIGH LEVEL)
# ------------------------------------------------------------
# We want to predict 5 continuous Big Five personality trait scores:
#   - Agreeableness (A), Conscientiousness (C), Extraversion (E),
#     Neuroticism (N), Openness (O)
# using ONLY open-ended text responses to 5 situational judgment prompts.
#
# Competition metric: mean Pearson correlation (r) across the 5 traits.
# That means we mainly care about rank-order accuracy and preserving
# meaningful variance in predictions (not necessarily minimizing MSE).
#
# This script is designed as a *teaching case*:
# - clean, readable, fully annotated
# - single-file, end-to-end runnable in PyCharm/terminal
# - produces:
#     (1) 5-fold CV OOF correlations per trait + mean r
#     (2) diagnostic figures (OOF) to explain model behavior
#     (3) submission-style CSVs for Dev and Test
#
# MODELING IDEA (what we’re doing)
# ------------------------------------------------------------
# For each trait, we train TWO base text models:
#   Base Model 1 (Prompt-only): uses the *trait-aligned* prompt text
#   Base Model 2 (All-text): uses a concatenation of all five prompts
#
# Each base model is:
#   TF-IDF features (word n-grams + char n-grams) + Ridge regression
#
# Then we learn a simple blender:
#   Ridge regression on z-scored OOF predictions from the two base models.
#
# This blending is learned ONLY from leakage-safe OOF predictions
# (so the blender cannot “cheat” by seeing in-sample predictions).
#
# OUTPUTS
# ------------------------------------------------------------
# - Prints:
#     per-trait OOF r and CV mean r (competition metric)
# - Writes:
#     results/submissions/submission_dev.csv
#     results/submissions/submission_test.csv
# - Writes figures:
#     results/figures/example_a/*
# ============================================================


# ============================================================
# IMPORTS
# ============================================================
import os
import numpy as np
import pandas as pd

from scipy.stats import pearsonr

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt


# ============================================================
# FILE LOCATIONS & COLUMN DEFINITIONS
# ============================================================
# Update DATA_PATH to point at your combined 2019 competition dataset file.
# (This file includes Train/Dev/Test rows together, identified by the "Dataset" column.)
#
# Expected columns:
#   Respondent_ID
#   open_ended_1 ... open_ended_5
#   A_Scale_score, C_Scale_score, E_Scale_score, N_Scale_score, O_Scale_score (Train only)
#   Dataset in {"Train","Dev","Test"}
# ============================================================

DATA_PATH = "data/raw/2019_siop_ml_comp_data.txt"  # update if needed

TEXT_COLS = [f"open_ended_{i}" for i in range(1, 6)]
TARGETS = ["A_Scale_score", "C_Scale_score", "E_Scale_score", "N_Scale_score", "O_Scale_score"]

# Trait-aligned prompt mapping (how prompts were designed in the competition)
PROMPT_MAP = {
    "A_Scale_score": "open_ended_1",
    "C_Scale_score": "open_ended_2",
    "E_Scale_score": "open_ended_3",
    "N_Scale_score": "open_ended_4",
    "O_Scale_score": "open_ended_5",
}


# ============================================================
# METRIC HELPERS (COMPETITION SCORING)
# ============================================================
def safe_pearsonr(y_true, y_pred) -> float:
    """Pearson r that won't crash if predictions are constant (returns 0.0 in that case)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def mean_r_across_traits(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, targets: list[str]) -> pd.Series:
    """
    Returns a Series of per-trait r, plus mean r (as Series.mean()).
    y_true_df and y_pred_df must have columns matching `targets`.
    """
    rs = {t: safe_pearsonr(y_true_df[t].values, y_pred_df[t].values) for t in targets}
    return pd.Series(rs)


# ============================================================
# MODEL DEFINITIONS
# ============================================================
def make_text_model(alpha: float = 30.0, max_features: int = 30000) -> Pipeline:
    """
    Build a strong baseline text regression model:
      TF-IDF(word 1-2 grams + char 3-5 grams) -> Ridge regression

    Why TF-IDF word n-grams?
      Captures content and short phrases ("compromise", "plan ahead", "talk to my manager", etc.)

    Why TF-IDF char n-grams?
      Captures style and morphology (prefixes/suffixes, punctuation patterns, spelling habits)

    Why Ridge?
      Great for very high-dimensional sparse text. Stable, fast, and usually strong for correlation metrics.
    """
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=max_features,
    )
    char = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=max_features,
    )
    feats = FeatureUnion([("word", word), ("char", char)])
    reg = Ridge(alpha=alpha, solver="sag", random_state=0)
    return Pipeline([("tfidf", feats), ("reg", reg)])


# ============================================================
# OUT-OF-FOLD (OOF) PREDICTION HELPERS
# ============================================================
def oof_preds(model: Pipeline, X: pd.Series, y: pd.Series, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """
    Generates leakage-safe out-of-fold predictions.
    Each training row is predicted by a model that was NOT trained on that row.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = np.zeros(len(y), dtype=float)

    for tr_idx, va_idx in kf.split(X):
        m = model  # sklearn Pipeline is re-fit each fold
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p[va_idx] = m.predict(X.iloc[va_idx])

    return p


# ============================================================
# BLENDING & POST-PROCESSING HELPERS
# ============================================================
def zscore(a: np.ndarray) -> np.ndarray:
    """
    Z-score predictions to put multiple model outputs on comparable scales before blending.
    This helps preserve useful variance for correlation-based scoring.
    """
    a = np.asarray(a, dtype=float)
    return (a - a.mean()) / (a.std(ddof=0) + 1e-12)


# ============================================================
# DIAGNOSTICS & FIGURES (POST-MODEL EVALUATION)
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_diagnostics_plots(train_df: pd.DataFrame, targets: list[str], oof_pred_df: pd.DataFrame, out_dir: str) -> None:
    """
    Writes a small, high-value diagnostic set based on Train OOF predictions:
      1) Ground-truth trait intercorrelations
      2) Pred vs Actual scatter (OOF) per trait + r
      3) Actual vs Pred distribution overlay (OOF) per trait
      4) Residuals vs Pred (OOF) per trait
    """
    ensure_dir(out_dir)

    # 1) Trait intercorrelations (ground truth)
    corr = train_df[targets].corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(targets)), targets, rotation=45, ha="right")
    plt.yticks(range(len(targets)), targets)
    plt.colorbar()
    plt.title("Train: Trait Intercorrelations (Ground Truth)")
    save_fig(os.path.join(out_dir, "01_trait_intercorrelations.png"))

    # 2-4) Per-trait diagnostics
    for t in targets:
        y = train_df[t].values
        p = oof_pred_df[t].values
        r = safe_pearsonr(y, p)

        # 2) Pred vs Actual
        plt.figure(figsize=(5, 4))
        plt.scatter(y, p, s=10)
        plt.xlabel("Actual")
        plt.ylabel("OOF Predicted")
        plt.title(f"{t}: Pred vs Actual (OOF) | r={r:.3f}")
        save_fig(os.path.join(out_dir, f"02_{t}_pred_vs_actual.png"))

        # 3) Distribution overlay
        plt.figure(figsize=(5, 4))
        plt.hist(y, bins=30, alpha=0.6, label="Actual")
        plt.hist(p, bins=30, alpha=0.6, label="OOF Pred")
        plt.legend()
        plt.title(f"{t}: Distribution (Actual vs OOF Pred)")
        save_fig(os.path.join(out_dir, f"03_{t}_dist_overlay.png"))

        # 4) Residuals vs Pred
        resid = y - p
        plt.figure(figsize=(5, 4))
        plt.scatter(p, resid, s=10)
        plt.axhline(0, linewidth=1)
        plt.xlabel("OOF Predicted")
        plt.ylabel("Residual (Actual - Pred)")
        plt.title(f"{t}: Residuals vs Pred (OOF)")
        save_fig(os.path.join(out_dir, f"04_{t}_residuals.png"))


# ============================================================
# MAIN WORKFLOW
# ============================================================
def main(data_path: str = DATA_PATH) -> None:
    # --- Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find data file at: {data_path}\n"
            f"Update DATA_PATH or pass --data_path to this script."
        )

    df = pd.read_csv(data_path)

    # --- Ensure clean text: fill NA, cast to string
    for c in TEXT_COLS:
        df[c] = df[c].fillna("").astype(str)

    # --- Create all_text (concatenation of all prompts)
    df["all_text"] = df[TEXT_COLS].agg(" ".join, axis=1)

    # --- Split into Train/Dev/Test
    train = df[df["Dataset"] == "Train"].copy()
    dev = df[df["Dataset"] == "Dev"].copy()
    test = df[df["Dataset"] == "Test"].copy()

    print("Dataset sizes:")
    print(train["Dataset"].value_counts())
    if not dev.empty:
        print(dev["Dataset"].value_counts())
    if not test.empty:
        print(test["Dataset"].value_counts())

    # --- Storage for per-trait models and blenders
    blend_models: dict[str, Ridge] = {}
    base_models: dict[str, tuple[Pipeline, Pipeline]] = {}

    # --- Store OOF blended predictions for scoring + diagnostics
    oof_blend_store = pd.DataFrame(index=train.index)

    print("\nRunning 5-fold CV (OOF blending per trait)...\n")

    for t in TARGETS:
        y = train[t]

        # Base Model 1: trait prompt only
        X1 = train[PROMPT_MAP[t]]
        m1 = make_text_model(alpha=30.0, max_features=30000)
        p1_oof = oof_preds(m1, X1, y, n_splits=5, seed=42)

        # Base Model 2: all_text
        X2 = train["all_text"]
        m2 = make_text_model(alpha=30.0, max_features=20000)  # slightly smaller for speed
        p2_oof = oof_preds(m2, X2, y, n_splits=5, seed=42)

        # Blender: Ridge on z-scored base OOF predictions
        Z = np.vstack([zscore(p1_oof), zscore(p2_oof)]).T
        blender = Ridge(alpha=1.0)
        blender.fit(Z, y)

        p_blend = blender.predict(Z)
        oof_blend_store[t] = p_blend

        r = safe_pearsonr(y, p_blend)
        print(f"{t}: blended OOF r = {r:.3f}")

        blend_models[t] = blender
        base_models[t] = (m1, m2)

    # --- Print competition metric: mean r across traits
    oof_rs = mean_r_across_traits(train[TARGETS], oof_blend_store[TARGETS], TARGETS)
    print("\nOOF CV results (Pearson r):")
    for t in TARGETS:
        print(f"{t}: {oof_rs[t]:.3f}")
    print(f"\nCV mean r (competition metric): {oof_rs.mean():.3f}\n")

    # --- Diagnostic figures (OOF)
    fig_dir = os.path.join("results", "figures", "example_a")
    make_diagnostics_plots(train_df=train, targets=TARGETS, oof_pred_df=oof_blend_store[TARGETS], out_dir=fig_dir)
    print(f"Wrote figures to: {fig_dir}")

    # --- Fit base models on all Train and predict Dev/Test, then apply blender
    ensure_dir(os.path.join("results", "submissions"))

    dev_out = pd.DataFrame({"Respondent_ID": dev["Respondent_ID"]}) if not dev.empty else None
    test_out = pd.DataFrame({"Respondent_ID": test["Respondent_ID"]}) if not test.empty else None

    for t in TARGETS:
        m1, m2 = base_models[t]
        blender = blend_models[t]

        # Fit base models on full train
        m1.fit(train[PROMPT_MAP[t]], train[t])
        m2.fit(train["all_text"], train[t])

        # Predict + blend on Dev
        if dev_out is not None:
            p1_dev = m1.predict(dev[PROMPT_MAP[t]])
            p2_dev = m2.predict(dev["all_text"])
            Z_dev = np.vstack([zscore(p1_dev), zscore(p2_dev)]).T
            dev_out[t.replace("_Scale_score", "_Pred")] = blender.predict(Z_dev)

        # Predict + blend on Test
        if test_out is not None:
            p1_test = m1.predict(test[PROMPT_MAP[t]])
            p2_test = m2.predict(test["all_text"])
            Z_test = np.vstack([zscore(p1_test), zscore(p2_test)]).T
            test_out[t.replace("_Scale_score", "_Pred")] = blender.predict(Z_test)

    # --- Write submission-style CSVs
    if dev_out is not None:
        dev_path = os.path.join("results", "submissions", "submission_dev.csv")
        dev_out.to_csv(dev_path, index=False)
        print(f"Wrote: {dev_path}")

    if test_out is not None:
        test_path = os.path.join("results", "submissions", "submission_test.csv")
        test_out.to_csv(test_path, index=False)
        print(f"Wrote: {test_path}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Example A: TF-IDF + Ridge + Blend (2019 SIOP ML)")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to 2019 combined dataset CSV/TXT file")
    args = parser.parse_args()

    main(args.data_path)
