"""
Frozen post-hoc comparison for the 2021 SIOP ML Competition.

Protocol:
  1) Restrict to rows with High_Performer, Retained, Protected_Group present.
  2) Single stratified 80/20 fit/holdout split (seed=42). Touch holdout once.
  3) Score competition-style (with/without protected proxy) and standards-aligned.
  4) Write results under results/cv/ and example submissions under results/submissions/.

Private-test labels are not public, so measured numbers are holdout-Train only.
Published Place 1–4 private-test scores are recorded alongside for context.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    PUBLISHED_LEADERBOARD,
    default_train_path,
    freeze_split,
    labeled_subset,
    load_train,
)
from src.pipelines import (  # noqa: E402
    choose_threshold_from_oof,
    competition_style_hires,
    competition_style_scores,
    standards_aligned_oof_scores,
    standards_aligned_predict,
)

sys.path.insert(0, str(ROOT / "02_standards_aligned_pipeline"))
from scoring_function import summarize_policy  # noqa: E402


def _score_row(name: str, hire: np.ndarray, hold: pd.DataFrame, extra: dict | None = None) -> dict:
    s = summarize_policy(
        hire,
        hold["High_Performer"].to_numpy(),
        hold["Retained"].to_numpy(),
        hold["Protected_Group"].to_numpy(),
    )
    row = {"pipeline": name, **s}
    if extra:
        row.update(extra)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=str(default_train_path()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=float, default=0.20)
    ap.add_argument("--air-floor", type=float, default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cv_dir = out_dir / "cv"
    sub_dir = out_dir / "submissions"
    cv_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    train = load_train(args.train)
    labeled = labeled_subset(train)
    fit, hold = freeze_split(labeled, test_size=args.holdout, seed=args.seed)

    print(f"Labeled rows: {len(labeled):,} | fit: {len(fit):,} | holdout: {len(hold):,}")

    rows: list[dict] = []

    # --- Competition-style (Procrustination recipe) ---
    print("Fitting competition-style (+ protected×retained proxy)...")
    cs_scores = competition_style_scores(fit, hold, use_protected_proxy=True)
    cs_hire = competition_style_hires(cs_scores)
    rows.append(
        _score_row(
            "this_repo_competition_style",
            cs_hire,
            hold,
            {"uses_protected_in_scoring": True, "decision_rule": "median_cut"},
        )
    )
    pd.DataFrame({"UNIQUE_ID": hold["UNIQUE_ID"], "Hire": cs_hire, "score": cs_scores}).to_csv(
        sub_dir / "holdout_competition_style.csv", index=False
    )

    # --- Ablation: same ensemble without protected proxy ---
    print("Fitting competition-style ablation (no protected proxy)...")
    abl_scores = competition_style_scores(fit, hold, use_protected_proxy=False)
    abl_hire = competition_style_hires(abl_scores)
    rows.append(
        _score_row(
            "this_repo_competition_style_no_proxy",
            abl_hire,
            hold,
            {"uses_protected_in_scoring": False, "decision_rule": "median_cut"},
        )
    )

    # --- Standards-aligned: OOF threshold on fit, apply to holdout ---
    print("Fitting standards-aligned (OOF threshold governance)...")
    oof = standards_aligned_oof_scores(fit, n_splits=5, seed=args.seed)
    sa_scores = standards_aligned_predict(fit, hold, seed=args.seed)

    # Primary: ~50% selection rate (competition hiring volume) + metric-optimal cut.
    policy = choose_threshold_from_oof(
        fit, oof, air_floor=args.air_floor, selection_rate=0.50, n_grid=200
    )
    sa_hire = (sa_scores >= policy["threshold"]).astype(int)
    rows.append(
        _score_row(
            "this_repo_standards_aligned",
            sa_hire,
            hold,
            {
                "uses_protected_in_scoring": False,
                "decision_rule": "oof_metric_threshold_sel50",
                "threshold": policy["threshold"],
                "fit_oof_final_score": policy["final_score"],
                "fit_oof_AIR": policy["AIR"],
            },
        )
    )
    pd.DataFrame({"UNIQUE_ID": hold["UNIQUE_ID"], "Hire": sa_hire, "score": sa_scores}).to_csv(
        sub_dir / "holdout_standards_aligned.csv", index=False
    )

    # Teaching contrast: 4/5ths AIR floor at ~50% selection.
    print("Fitting standards-aligned with AIR floor 0.80 @ ~50% selection...")
    try:
        policy80 = choose_threshold_from_oof(
            fit, oof, air_floor=0.80, selection_rate=0.50, n_grid=200
        )
        sa80_hire = (sa_scores >= policy80["threshold"]).astype(int)
        rows.append(
            _score_row(
                "this_repo_standards_aligned_air80",
                sa80_hire,
                hold,
                {
                    "uses_protected_in_scoring": False,
                    "decision_rule": "oof_metric_threshold_sel50_air_floor_0.80",
                    "threshold": policy80["threshold"],
                    "fit_oof_final_score": policy80["final_score"],
                    "fit_oof_AIR": policy80["AIR"],
                },
            )
        )
    except RuntimeError as exc:
        print(f"  AIR-floor policy unavailable: {exc}")

    # Landmine demo: unconstrained threshold (hire-nearly-all) — documented, not primary.
    print("Fitting unconstrained standards-aligned (metric-gaming demo)...")
    policy_free = choose_threshold_from_oof(
        fit, oof, air_floor=None, selection_rate=None, n_grid=200
    )
    free_hire = (sa_scores >= policy_free["threshold"]).astype(int)
    rows.append(
        _score_row(
            "this_repo_standards_aligned_unconstrained",
            free_hire,
            hold,
            {
                "uses_protected_in_scoring": False,
                "decision_rule": "oof_metric_threshold_unconstrained",
                "threshold": policy_free["threshold"],
                "fit_oof_final_score": policy_free["final_score"],
                "fit_oof_AIR": policy_free["AIR"],
            },
        )
    )

    summary = pd.DataFrame(rows)
    summary_path = cv_dir / "compare_summary.csv"
    summary.to_csv(summary_path, index=False)

    published = pd.DataFrame(PUBLISHED_LEADERBOARD)
    published_path = cv_dir / "published_leaderboard.csv"
    published.to_csv(published_path, index=False)

    meta = {
        "protocol": "stratified_holdout_on_labeled_train",
        "seed": args.seed,
        "holdout_fraction": args.holdout,
        "n_labeled": int(len(labeled)),
        "n_fit": int(len(fit)),
        "n_holdout": int(len(hold)),
        "train_path": str(Path(args.train).resolve()),
        "note": (
            "Private-test labels are not public. Measured scores are holdout-Train only "
            "and are NOT directly comparable to published Place 1–4 private-test scores."
        ),
    }
    (cv_dir / "protocol.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== Measured holdout results ===")
    cols = [
        "pipeline",
        "final_score",
        "overall_accuracy",
        "AIR",
        "unfairness",
        "selection_rate_overall",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {published_path}")
    print("Published private-test (not remeasured):")
    print(published.to_string(index=False))


if __name__ == "__main__":
    main()
