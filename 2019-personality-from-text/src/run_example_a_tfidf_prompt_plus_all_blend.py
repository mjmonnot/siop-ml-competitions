# ============================================================
# Example A (TEACHING NOTE VERSION)
# TF-IDF + Ridge, Prompt-only + All-text Blend
# ============================================================
# Big idea in plain English:
# - People wrote answers to 5 short prompts.
# - We want to predict 5 personality trait scores: A, C, E, N, O.
# - A common and surprisingly strong baseline for text prediction is:
#     Text -> TF-IDF features -> Ridge regression
#
# Why TF-IDF?
# - Computers cannot learn directly from raw words.
# - TF-IDF turns text into a huge table of numbers:
#     - "word n-grams" capture words and short phrases
#     - "character n-grams" capture style patterns (punctuation, spelling, tone)
#
# Why Ridge regression?
# - Text features are extremely high-dimensional.
# - Ridge is stable and resists overfitting by shrinking weights.
#
# Why blend prompt-only and all-text?
# - The “target” trait may be best expressed in its intended prompt,
#   but other prompts often contain helpful signal too.
# - Blending lets us combine both sources safely.
#
# What this script does:
#  1) Loads a combined Train/Dev/Test file
#  2) Creates an "all_text" column (all prompts concatenated)
#  3) For each trait, trains TWO base models:
#       - Base model 1: prompt-specific text only
#       - Base model 2: all_text (all prompts)
#  4) Uses out-of-fold (OOF) predictions to learn blend weights WITHOUT leakage
#  5) Fits final models on Train, predicts Dev/Test
#  6) Saves figures + submission-ready CSVs
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd

# ----------------------------
# Correlation (competition metric)
# ----------------------------
# The 2019 competition metric is the MEAN Pearson correlation (r) across traits.
# Correlation focuses on rank-order accuracy: do we correctly order people high vs low?
# It is less sensitive to absolute scale shifts (e.g., predictions slightly compressed).
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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.base import clone  # IMPORTANT for safe CV loops

# Optional plotting. If matplotlib isn't installed, we simply skip plotting.
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================================================
# SECTION 1 — Small helper functions (kept in-file for teaching)
# ============================================================

def zscore(a: np.ndarray) -> np.ndarray:
    """
    Standardize an array: mean = 0, std = 1.

    Plain English:
    - Different models can output predictions on slightly different scales.
    - If we blend them, a model with a larger numeric range can “dominate”
      even if it isn't actually better.
    - Z-scoring makes both inputs comparable before blending.

    Note:
    We add a tiny constant so we never divide by zero.
    """
    a = np.asarray(a, dtype=float)
    return (a - a.mean()) / (a.std(ddof=0) + 1e-12)


def mean_r_across_traits(r_by_trait: dict) -> float:
    """Competition score = mean Pearson r across all trait columns."""
    return float(np.mean(list(r_by_trait.values())))


def ensure_dir(path: str) -> None:
    """Create a folder if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ============================================================
# SECTION 2 — Model factory (the “engine”)
# ============================================================

def make_text_model(alpha=30.0, max_features=30000) -> Pipeline:
    """
    Build a text model: TF-IDF features -> Ridge regression

    In plain English:
    - We create two separate TF-IDF feature sets:
        (A) word n-grams (1–2): words + short phrases
        (B) char n-grams (3–5): writing style signals
    - Then we combine them (FeatureUnion).
    - Then Ridge regression learns weights for these features to predict a trait score.

    Key knobs:
    - max_features: caps vocabulary size (helps speed and prevents noise explosion)
    - alpha: Ridge penalty strength (higher = more shrinkage, less overfit)
    """
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,        # ignore tokens that appear in only 1 doc (often noise)
        max_df=0.9,      # ignore tokens that appear in almost all docs (not informative)
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

    # Ridge is a strong baseline for sparse, high-dimensional text features
    reg = Ridge(alpha=alpha, solver="auto", random_state=0)

    return Pipeline([("tfidf", feats), ("reg", reg)])


def oof_predictions(model: Pipeline, X: pd.Series, y: pd.Series, n_splits=5, seed=42) -> np.ndarray:
    """
    Out-of-fold predictions (OOF)

    Plain English:
    - We want predictions for every training row,
      but we must avoid "peeking" at that row's true label.
    - So we do K-fold cross-validation:
        * Split training into K folds
        * Train on K-1 folds
        * Predict on the held-out fold
    - Collect predictions for all folds => every row gets a prediction made
      by a model that did NOT train on it.

    Why this matters:
    - We use these OOF predictions to learn blend weights safely.
    - If we trained and predicted on the same rows, the blender would “cheat”
      and overestimate performance.

    Implementation note:
    - We CLONE the model each fold so each fold starts fresh.
      This is safer and avoids accidental state carryover.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=float)

    for tr_idx, te_idx in kf.split(X):
        m = clone(model)  # IMPORTANT: fresh model each fold
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds[te_idx] = m.predict(X.iloc[te_idx])

    return preds


# ============================================================
# SECTION 3 — Plotting helpers (optional but high teaching value)
# ============================================================

def plot_pred_vs_actual(y_true, y_pred, title, outpath):
    if plt is None:
        return
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.35)
    plt.xlabel("Actual score")
    plt.ylabel("Predicted score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_distribution_overlay(y_true, y_pred, title, outpath):
    if plt is None:
        return
    plt.figure()
    plt.hist(y_true, bins=30, alpha=0.5, label="Actual")
    plt.hist(y_pred, bins=30, alpha=0.5, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_residuals(y_true, y_pred, title, outpath):
    if plt is None:
        return
    resid = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure()
    plt.scatter(y_pred, resid, alpha=0.35)
    plt.axhline(0.0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_trait_intercorrelations(train_df, trait_cols, outpath):
    if plt is None:
        return
    corr_mat = train_df[trait_cols].corr()
    plt.figure()
    plt.imshow(corr_mat.values)
    plt.xticks(range(len(trait_cols)), trait_cols, rotation=45, ha="right")
    plt.yticks(range(len(trait_cols)), trait_cols)
    plt.title("Trait intercorrelations (Train)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ============================================================
# SECTION 4 — Main workflow
# ============================================================

def main(data_path: str, seed: int = 42):
    # ---- 4.1 Column definitions (edit here if your data differs)
    text_cols = [f"open_ended_{i}" for i in range(1, 6)]
    trait_cols = ["A_Scale_score", "C_Scale_score", "E_Scale_score", "N_Scale_score", "O_Scale_score"]

    # Which prompt was designed to elicit which trait
    prompt_map = {
        "A_Scale_score": "open_ended_1",
        "C_Scale_score": "open_ended_2",
        "E_Scale_score": "open_ended_3",
        "N_Scale_score": "open_ended_4",
        "O_Scale_score": "open_ended_5",
    }

    # ---- 4.2 Load data
    df = pd.read_csv(data_path)

    # ---- 4.3 Clean text and build "all_text"
    # Plain English:
    # - Missing text is common; we replace it with empty strings.
    # - Then we concatenate prompts into a single long essay per person.
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)

    df["all_text"] = df[text_cols].agg(" ".join, axis=1)

    # ---- 4.4 Split Train / Dev / Test
    train_df = df[df["Dataset"] == "Train"].copy()
    dev_df = df[df["Dataset"] == "Dev"].copy()
    test_df = df[df["Dataset"] == "Test"].copy()

    print("Dataset sizes:")
    print("  Train:", len(train_df))
    print("  Dev  :", len(dev_df))
    print("  Test :", len(test_df))

    # ---- 4.5 Output locations
    fig_dir = os.path.join("results", "figures", "example_a")
    sub_dir = os.path.join("results", "submissions")
    cv_dir = os.path.join("results", "cv")
    ensure_dir(fig_dir)
    ensure_dir(sub_dir)
    ensure_dir(cv_dir)

    # ---- 4.6 Optional: visualize trait correlations (teaching: traits are related)
    plot_trait_intercorrelations(train_df, trait_cols, os.path.join(fig_dir, "01_trait_intercorrelations.png"))

    # ---- 4.7 Train base models + learn blend weights per trait
    blend_models = {}   # store blender per trait
    base_models = {}    # store (m_prompt, m_alltext) per trait
    r_by_trait = {}

    print("\nRunning 5-fold CV (TF-IDF prompt-only + all-text blend)...\n")

    for t in trait_cols:
        # The target labels for this trait
        y = train_df[t]

        # -------- Base model 1: prompt-specific text
        X1 = train_df[prompt_map[t]]
        m1 = make_text_model(alpha=30.0, max_features=30000)
        p1_oof = oof_predictions(m1, X1, y, n_splits=5, seed=seed)

        # -------- Base model 2: all prompts concatenated
        X2 = train_df["all_text"]
        m2 = make_text_model(alpha=30.0, max_features=20000)
        p2_oof = oof_predictions(m2, X2, y, n_splits=5, seed=seed)

        # -------- Learn blending weights safely (using OOF predictions only)
        # Plain English:
        # - We now have TWO predictions for each training person:
        #     * one from prompt-only
        #     * one from all-text
        # - We z-score them so they are comparable
        # - Then we train a tiny model (Ridge) to combine them:
        #     blended = w1*prompt_pred + w2*alltext_pred + intercept
        Z = np.vstack([zscore(p1_oof), zscore(p2_oof)]).T
        blender = Ridge(alpha=1.0, random_state=seed)
        blender.fit(Z, y)

        # OOF blended prediction (still leakage-safe)
        p_blend_oof = blender.predict(Z)

        # Evaluate with Pearson r (what the competition cares about)
        r = corr(y.values, p_blend_oof)
        r_by_trait[t] = r

        print(f"{t}: blended OOF r = {r:.3f}")

        # Save models for later (full-train fit)
        blend_models[t] = blender
        base_models[t] = (m1, m2)

        # Figures: show behavior on training OOF predictions
        plot_pred_vs_actual(
            y.values, p_blend_oof,
            f"{t}: OOF Pred vs Actual (r={r:.3f})",
            os.path.join(fig_dir, f"02_{t}_pred_vs_actual.png")
        )
        plot_distribution_overlay(
            y.values, p_blend_oof,
            f"{t}: OOF Distribution Overlay",
            os.path.join(fig_dir, f"03_{t}_dist_overlay.png")
        )
        plot_residuals(
            y.values, p_blend_oof,
            f"{t}: OOF Residuals",
            os.path.join(fig_dir, f"04_{t}_residuals.png")
        )

    # Summarize CV
    mean_r = mean_r_across_traits(r_by_trait)

    print("\nOOF CV results (Pearson r):")
    for t in trait_cols:
        print(f"  {t}: {r_by_trait[t]:.3f}")
    print(f"\nCV mean r (competition metric): {mean_r:.3f}\n")

    # Save CV summary table
    cv_summary = pd.DataFrame({"trait": list(r_by_trait.keys()), "r": list(r_by_trait.values())})
    cv_summary.loc[len(cv_summary)] = ["MEAN", mean_r]
    cv_summary.to_csv(os.path.join(cv_dir, "cv_summary.csv"), index=False)

    # ---- 4.8 Fit on full Train + predict Dev/Test
    # Important: once blending weights are learned, we re-fit base models using ALL Train data.
    dev_out = pd.DataFrame({"Respondent_ID": dev_df["Respondent_ID"]})
    test_out = pd.DataFrame({"Respondent_ID": test_df["Respondent_ID"]})

    for t in trait_cols:
        m1, m2 = base_models[t]
        blender = blend_models[t]

        # Fit base models on ALL training rows
        m1.fit(train_df[prompt_map[t]], train_df[t])
        m2.fit(train_df["all_text"], train_df[t])

        # Predict base outputs for Dev/Test
        p1_dev = m1.predict(dev_df[prompt_map[t]])
        p2_dev = m2.predict(dev_df["all_text"])
        p1_test = m1.predict(test_df[prompt_map[t]])
        p2_test = m2.predict(test_df["all_text"])

        # Blend
        # Teaching note:
        # - We z-score within each split so the blender sees “normalized” inputs.
        # - This keeps the blend stable even if Dev/Test have slightly different scale.
        Z_dev = np.vstack([zscore(p1_dev), zscore(p2_dev)]).T
        Z_test = np.vstack([zscore(p1_test), zscore(p2_test)]).T

        dev_out[t.replace("_Scale_score", "_Pred")] = blender.predict(Z_dev)
        test_out[t.replace("_Scale_score", "_Pred")] = blender.predict(Z_test)

    # ---- 4.9 Write submission files
    dev_path = os.path.join(sub_dir, "submission_dev.csv")
    test_path = os.path.join(sub_dir, "submission_test.csv")
    dev_out.to_csv(dev_path, index=False)
    test_out.to_csv(test_path, index=False)

    print(f"Wrote figures to: {fig_dir}")
    print(f"Wrote: {dev_path}")
    print(f"Wrote: {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to 2019_siop_ml_comp_data.txt (CSV)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.data_path, seed=args.seed)

