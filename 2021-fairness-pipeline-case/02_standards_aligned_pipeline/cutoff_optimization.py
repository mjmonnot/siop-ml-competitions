from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from scoring_function import adverse_impact_ratio, unfairness_from_air, summarize_policy

def choose_threshold(train_df: pd.DataFrame,
                     score_col: str = "score_cv",
                     protected_col: str = "Protected_Group",
                     top_col: str = "High_Performer",
                     retained_col: str = "Retained",
                     strategy: str = "max_final_score",
                     air_floor: float | None = None,
                     selection_rate: float | None = 0.50,
                     n_grid: int = 200) -> dict:
    """Choose a threshold on training scores using a documented strategy.

    Parameters
    ----------
    strategy:
        - 'max_final_score': maximize (overall accuracy - unfairness)
        - 'max_final_score_with_air_floor': maximize final score subject to AIR >= air_floor (requires air_floor)
    selection_rate:
        If not None, thresholds are derived from quantiles to keep selection rate stable.
        If None, a linear grid over score range is used.
    """
    scores = train_df[score_col].to_numpy()
    if selection_rate is not None:
        # derive candidate thresholds from quantiles (stable selection-rate exploration)
        qs = np.linspace(0.05, 0.95, n_grid)
        thresholds = np.quantile(scores, qs)
    else:
        thresholds = np.linspace(scores.min(), scores.max(), n_grid)

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

        if strategy == "max_final_score_with_air_floor":
            if air_floor is None:
                raise ValueError("air_floor must be set for max_final_score_with_air_floor")
            if np.isnan(s["AIR"]) or s["AIR"] < air_floor:
                continue

        if best is None or s["final_score"] > best["final_score"]:
            best = s

    if best is None:
        raise RuntimeError("No threshold satisfied constraints; relax air_floor or change strategy.")
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.csv (for threshold governance)")
    ap.add_argument("--scores", required=True, help="Path to scored_test.csv from job_success_model.py")
    ap.add_argument("--out", default="final_submission.csv", help="Output submission CSV")
    ap.add_argument("--air_floor", type=float, default=None, help="Optional AIR floor constraint (e.g., 0.80)")
    ap.add_argument("--strategy", default="max_final_score", choices=["max_final_score", "max_final_score_with_air_floor"])
    ap.add_argument("--n_grid", type=int, default=200)
    args = ap.parse_args()

    train = pd.read_csv(args.train, na_values=[' ', '.'])
    if "split" in train.columns:
        train = train.drop(columns=["split"])

    # NOTE: In a real applied workflow, the threshold should be chosen on validation predictions (CV or a holdout),
    # not on in-sample scores. For teaching simplicity, we assume the user will swap in CV scores here.
    # To encourage good practice, we create a naive score proxy:
    # score_cv = 0.25*Overall_Rating + 0.75*(High_Performer & Retained) (placeholder).
    # Replace this with proper out-of-fold predictions in a real deployment.
    # We include this explicit comment to teach the separation between *prediction* and *policy*.
    if "score_cv" not in train.columns:
        top = train["High_Performer"].astype(int)
        ret = train["Retained"].astype(int)
        both = ((top == 1) & (ret == 1)).astype(int)
        train["score_cv"] = 0.25 * train["Overall_Rating"].fillna(train["Overall_Rating"].median()) + 0.75 * both

    best = choose_threshold(
        train_df=train,
        score_col="score_cv",
        strategy=args.strategy,
        air_floor=args.air_floor,
        n_grid=args.n_grid
    )

    # Apply chosen threshold to test scores
    scored_test = pd.read_csv(args.scores)
    hire = (scored_test["score"].to_numpy() >= best["threshold"]).astype(int)
    sub = pd.DataFrame({"UNIQUE_ID": scored_test["UNIQUE_ID"], "Hire": hire})
    sub.to_csv(args.out, index=False)

    # Report decision policy summary
    print("Chosen policy (based on training governance scores):")
    for k, v in best.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"Wrote submission to: {args.out}")

if __name__ == "__main__":
    main()
