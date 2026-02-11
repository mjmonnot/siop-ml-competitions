"""CUTOFF OPTIMIZATION (FULLY ANNOTATED)

This script illustrates a governance-friendly approach:

- We do NOT adjust individual prediction scores using Protected_Group.
- We DO use Protected_Group to evaluate AIR and fairness outcomes **after** scoring.

The key idea:
1) Generate scores (from job_success_model.py).
2) Explore many possible cut scores (thresholds).
3) For each threshold, compute utility and AIR (using training data where labels exist).
4) Select an operating point that you can document and defend.
5) Apply that threshold to the test scores to create Hire/No-hire decisions.

Outputs:
- final_submission.csv with columns: UNIQUE_ID, Hire
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from scoring_function import summarize_policy

def choose_threshold(train_df: pd.DataFrame,
                     score_col: str,
                     protected_col: str = "Protected_Group",
                     top_col: str = "High_Performer",
                     retained_col: str = "Retained",
                     air_floor: float | None = None,
                     n_grid: int = 200) -> dict:
    """Choose the threshold that maximizes final score, optionally requiring AIR >= air_floor."""
    scores = train_df[score_col].to_numpy()

    # Candidate thresholds from quantiles (stable selection-rate exploration)
    thresholds = np.quantile(scores, np.linspace(0.05, 0.95, n_grid))

    best = None
    for t in thresholds:
        hire = (scores >= t).astype(int)
        s = summarize_policy(
            hire,
            train_df[top_col].to_numpy(),
            train_df[retained_col].to_numpy(),
            train_df[protected_col].to_numpy()
        )
        s["threshold"] = float(t)

        if air_floor is not None:
            if np.isnan(s["AIR"]) or s["AIR"] < air_floor:
                continue

        if best is None or s["final_score"] > best["final_score"]:
            best = s

    if best is None:
        raise RuntimeError("No threshold satisfied constraints. Relax air_floor or change search grid.")
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.csv (labels exist here)")
    ap.add_argument("--scores", required=True, help="Path to scored_test.csv (from job_success_model.py)")
    ap.add_argument("--out", default="final_submission.csv")
    ap.add_argument("--air_floor", type=float, default=None, help="Optional AIR floor (e.g., 0.80)")
    ap.add_argument("--n_grid", type=int, default=200)
    args = ap.parse_args()

    train = pd.read_csv(args.train, na_values=[' ', '.'])
    if "split" in train.columns:
        train.drop(columns=["split"], inplace=True)

    # IMPORTANT TEACHING NOTE:
    # In an ideal applied workflow, the threshold should be chosen using *out-of-sample* (CV) predicted scores on the training set.
    # Here, we provide a simple placeholder 'score_cv' if you don't have CV scores available yet.
    #
    # If you want the most principled version, add a CV step that creates out-of-fold predictions and store them in train['score_cv'].
    if "score_cv" not in train.columns:
        # A transparent placeholder proxy (NOT a deployment recommendation):
        # It uses only labels to create a "governance score" so the threshold selection process can be demonstrated.
        top = train["High_Performer"].astype(int)
        ret = train["Retained"].astype(int)
        both = ((top == 1) & (ret == 1)).astype(int)
        train["score_cv"] = 0.25 * train["Overall_Rating"].fillna(train["Overall_Rating"].median()) + 0.75 * both

    best = choose_threshold(train_df=train, score_col="score_cv", air_floor=args.air_floor, n_grid=args.n_grid)

    scored_test = pd.read_csv(args.scores)
    hire = (scored_test["score"].to_numpy() >= best["threshold"]).astype(int)

    sub = pd.DataFrame({"UNIQUE_ID": scored_test["UNIQUE_ID"], "Hire": hire})
    sub.to_csv(args.out, index=False)

    print("Chosen decision policy (from training governance scores):")
    for k, v in best.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"Wrote submission to: {args.out}")

if __name__ == "__main__":
    main()
