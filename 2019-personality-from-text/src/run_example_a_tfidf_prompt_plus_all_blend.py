# ============================================================
# Example A (TF-IDF + Ridge, Prompt-only + All-text Blend)
# ============================================================
# In plain English:
# - Each person answered 5 open-ended prompts (text).
# - We want to predict 5 personality scores (A, C, E, N, O).
# - The competition metric is the MEAN Pearson correlation (r)
#   across the 5 traits, so we care about rank-order accuracy.
#
# This script:
#  1) Loads the combined Train/Dev/Test file
#  2) Builds an "all_text" column by concatenating all prompts
#  3) For each trait, trains TWO text models:
#       Model 1: prompt-only (the prompt designed for that trait)
#       Model 2: all_text (all prompts concatenated)
#     Both are TF-IDF (word ngrams + char ngrams) -> Ridge regression.
#  4) Creates out-of-fold (OOF) predictions for each base model
#     so we can learn blend weights without leakage.
#  5) Learns a small blender model per trait (Ridge on z-scored preds).
#  6) Fits final models on all Train and predicts Dev/Test.
#  7) Prints per-trait r and mean r; writes figures and submission CSVs.
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd

# Prefer scipy pearsonr if available; otherwise compute ourselves.
try:
    from scipy.stats import pearsonr
    def corr(x, y):
        return pearsonr(x, y)[0]
except Exception:
    def corr(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        x = x - x.mean()
        y = y - y.mean()
        denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + 1e-12
        return float((x * y).sum() / denom)

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

# Optional plotting (safe if matplotlib is installed)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ----------------------------
# Helpers: scoring + standardization
# ----------------------------

def zscore(a: np.ndarray) -> np.ndarray:
    """Standardize an array to mean=0, std=1 (to keep prediction variance healthy)."""
    a = np.asarray(a)
    return (a - a.mean()) / (a.std(ddof=0) + 1e-12)


def mean_r_across_traits(r_by_trait: dict) -> float:
    """Compute the competition metric: mean Pearson r across the 5 traits."""
    return float(np.mean(list(r_by_trait.values())))


# ----------------------------
# Text model factory
# ----------------------------

def make_text_model(alpha=30.0, max_features=30000) -> Pipeline:
    """
    In plain English:
    - Convert text into numeric features using TF-IDF:
        * word n-grams (1–2): words + short phrases
        * character n-grams (3–5): style/spelling/punctuation patterns
    - Fit Ridge regression (strong baseline for sparse text features)
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
    reg = Ridge(alpha=alpha, solver="auto", random_state=0)
    return Pipeline([("tfidf", feats), ("reg", reg)])


def oof_predictions(model: Pipeline, X: pd.Series, y: pd.Series, n_splits=5, seed=42) -> np.ndarray:
    """
    Out-of-fold predictions:
    - Split Train into K folds
    - Train on K-1 folds, predict held-out fold
    - Produces predictions for every training row WITHOUT training on that row's label
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=float)

    for tr_idx, te_idx in kf.split(X):
        m = model  # sklearn clones internals during fit; reusing is fine here
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds[te_idx] = m.predict(X.iloc[te_idx])

    return preds


# ----------------------------
# Figures (high-value, lightweight)
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_pred_vs_actual(y_true, y_pred, title, outpath):
    if plt is None:
        return
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.35)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
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


# ----------------------------
# Main
# ----------------------------

def main(data_path: str, seed: int = 42):
    # ---- Column definitions
    text_cols = [f"open_ended_{i}" for i in range(1, 6)]
    trait_cols = ["A_Scale_score", "C_Scale_score", "E_Scale_score", "N_Scale_score", "O_Scale_score"]

    # “Which prompt was designed for which trait”
    prompt_map = {
        "A_Scale_score": "open_ended_1",
        "C_Scale_score": "open_ended_2",
        "E_Scale_score": "open_ended_3",
        "N_Scale_score": "open_ended_4",
        "O_Scale_score": "open_ended_5",
    }

    # ---- Load
    df = pd.read_csv(data_path)

    # ---- Clean text and build "all_text"
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)
    df["all_text"] = df[text_cols].agg(" ".join, axis=1)

    # ---- Split Train / Dev / Test
    train_df = df[df["Dataset"] == "Train"].copy()
    dev_df = df[df["Dataset"] == "Dev"].copy()
    test_df = df[df["Dataset"] == "Test"].copy()

    print("Dataset sizes:")
    print(train_df["Dataset"].value_counts())
    print(dev_df["Dataset"].value_counts())
    print(test_df["Dataset"].value_counts())

    # ---- Output locations
    fig_dir = os.path.join("results", "figures", "example_a")
    sub_dir = os.path.join("results", "submissions")
    cv_dir = os.path.join("results", "cv")
    ensure_dir(fig_dir)
    ensure_dir(sub_dir)
    ensure_dir(cv_dir)

    # ---- Optional: trait intercorrelations
    plot_trait_intercorrelations(train_df, trait_cols, os.path.join(fig_dir, "01_trait_intercorrelations.png"))

    # ---- Train models + learn blend weights per trait
    blend_models = {}
    base_models = {}
    r_by_trait = {}

    print("\nRunning 5-fold CV (Example A: TF-IDF prompt-only + all-text blend)...\n")

    for t in trait_cols:
        y = train_df[t]

        # Base 1: prompt-specific
        X1 = train_df[prompt_map[t]]
        m1 = make_text_model(alpha=30.0, max_features=30000)
        p1_oof = oof_predictions(m1, X1, y, n_splits=5, seed=seed)

        # Base 2: all_text
        X2 = train_df["all_text"]
        m2 = make_text_model(alpha=30.0, max_features=20000)
        p2_oof = oof_predictions(m2, X2, y, n_splits=5, seed=seed)

        # Blend using z-scored base predictions (reduces scale mismatch)
        Z = np.vstack([zscore(p1_oof), zscore(p2_oof)]).T
        blender = Ridge(alpha=1.0, random_state=seed)
        blender.fit(Z, y)

        p_blend_oof = blender.predict(Z)
        r = corr(y.values, p_blend_oof)
        r_by_trait[t] = r

        print(f"{t}: blended OOF r = {r:.3f}")

        blend_models[t] = blender
        base_models[t] = (m1, m2)

        # Figures for OOF behavior (what the model is doing on Train)
        plot_pred_vs_actual(y.values, p_blend_oof, f"{t}: OOF Pred vs Actual (r={r:.3f})",
                            os.path.join(fig_dir, f"02_{t}_pred_vs_actual.png"))
        plot_distribution_overlay(y.values, p_blend_oof, f"{t}: OOF Distribution Overlay",
                                  os.path.join(fig_dir, f"03_{t}_dist_overlay.png"))
        plot_residuals(y.values, p_blend_oof, f"{t}: OOF Residuals",
                       os.path.join(fig_dir, f"04_{t}_residuals.png"))

    mean_r = mean_r_across_traits(r_by_trait)

    print("\nOOF CV results (Pearson r):")
    for t in trait_cols:
        print(f"{t}: {r_by_trait[t]:.3f}")
    print(f"\nCV mean r (competition metric): {mean_r:.3f}\n")

    # Save CV summary
    cv_summary = pd.DataFrame({"trait": list(r_by_trait.keys()), "r": list(r_by_trait.values())})
    cv_summary.loc[len(cv_summary)] = ["MEAN", mean_r]
    cv_summary.to_csv(os.path.join(cv_dir, "cv_summary.csv"), index=False)

    # ---- Fit on full Train + predict Dev/Test
    dev_out = pd.DataFrame({"Respondent_ID": dev_df["Respondent_ID"]})
    test_out = pd.DataFrame({"Respondent_ID": test_df["Respondent_ID"]})

    for t in trait_cols:
        m1, m2 = base_models[t]
        blender = blend_models[t]

        # Fit base models on full Train
        m1.fit(train_df[prompt_map[t]], train_df[t])
        m2.fit(train_df["all_text"], train_df[t])

        # Base predictions
        p1_dev = m1.predict(dev_df[prompt_map[t]])
        p2_dev = m2.predict(dev_df["all_text"])
        p1_test = m1.predict(test_df[prompt_map[t]])
        p2_test = m2.predict(test_df["all_text"])

        # Blend (z-score within split)
        Z_dev = np.vstack([zscore(p1_dev), zscore(p2_dev)]).T
        Z_test = np.vstack([zscore(p1_test), zscore(p2_test)]).T

        dev_out[t.replace("_Scale_score", "_Pred")] = blender.predict(Z_dev)
        test_out[t.replace("_Scale_score", "_Pred")] = blender.predict(Z_test)

    # ---- Write submissions
    dev_path = os.path.join(sub_dir, "submission_dev.csv")
    test_path = os.path.join(sub_dir, "submission_test.csv")
    dev_out.to_csv(dev_path, index=False)
    test_out.to_csv(test_path, index=False)

    print(f"Wrote figures to: {fig_dir}")
    print(f"Wrote: {dev_path}")
    print(f"Wrote: {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to 2019_siop_ml_comp_data.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.data_path, seed=args.seed)
