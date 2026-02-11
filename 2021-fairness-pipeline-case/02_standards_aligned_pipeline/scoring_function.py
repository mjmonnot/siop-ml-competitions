"""SCORING FUNCTION (FULLY EXPLAINED)

This file implements the official competition metric in a transparent way.

Key terms:
- Selection rate: % of people hired in a group
- AIR: protected selection rate / non-protected selection rate
- Unfairness: |1 - AIR| * 100  (penalty)

Final score:
Final_score = Overall_accuracy - Unfairness

Overall_accuracy is built from three "hit rates" (all are recall-like):
- % of true top performers hired (weight 25)
- % of true retained hired (weight 25)
- % of true retained AND top performers hired (weight 50)

Even if you are not doing the competition, these functions are useful as teaching tools.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def adverse_impact_ratio(hire: np.ndarray, protected: np.ndarray) -> float:
    """Compute AIR from a vector of 0/1 hire decisions and 0/1 protected-group membership."""
    hire = np.asarray(hire).astype(int)
    protected = np.asarray(protected).astype(int)

    # selection rate within each group
    sel_prot = hire[protected == 1].mean() if (protected == 1).any() else np.nan
    sel_non  = hire[protected == 0].mean() if (protected == 0).any() else np.nan

    if sel_non == 0 or np.isnan(sel_non):
        return np.nan
    return float(sel_prot / sel_non)

def unfairness_from_air(air: float) -> float:
    """Unfairness penalty used in the competition."""
    if air is None or np.isnan(air):
        return np.nan
    return float(abs(1.0 - air) * 100.0)

def overall_accuracy_components(hire: np.ndarray, top: np.ndarray, retained: np.ndarray):
    """Compute the 3 recall-like components used in overall accuracy."""
    hire = np.asarray(hire).astype(int)
    top = np.asarray(top).astype(int)
    retained = np.asarray(retained).astype(int)

    both = (top == 1) & (retained == 1)

    pct_top = (hire[top == 1].sum() / max(1, (top == 1).sum()))
    pct_ret = (hire[retained == 1].sum() / max(1, (retained == 1).sum()))
    pct_both = (hire[both].sum() / max(1, both.sum()))
    return float(pct_top), float(pct_ret), float(pct_both)

def overall_accuracy_score(hire: np.ndarray, top: np.ndarray, retained: np.ndarray) -> float:
    pct_top, pct_ret, pct_both = overall_accuracy_components(hire, top, retained)
    return float(pct_top * 25.0 + pct_ret * 25.0 + pct_both * 50.0)

def final_score(hire: np.ndarray, top: np.ndarray, retained: np.ndarray, protected: np.ndarray) -> float:
    oa = overall_accuracy_score(hire, top, retained)
    air = adverse_impact_ratio(hire, protected)
    unf = unfairness_from_air(air)
    return float(oa - unf)

def summarize_policy(hire: np.ndarray, top: np.ndarray, retained: np.ndarray, protected: np.ndarray) -> dict:
    """Create a compact report for a given hire/no-hire policy."""
    hire = np.asarray(hire).astype(int)
    air = adverse_impact_ratio(hire, protected)
    pct_top, pct_ret, pct_both = overall_accuracy_components(hire, top, retained)
    oa = overall_accuracy_score(hire, top, retained)
    unf = unfairness_from_air(air)
    fs = oa - unf
    return {
        "selection_rate_overall": float(hire.mean()),
        "AIR": float(air) if air is not None and not np.isnan(air) else np.nan,
        "unfairness": float(unf) if unf is not None and not np.isnan(unf) else np.nan,
        "pct_true_top_hired": float(pct_top),
        "pct_true_retained_hired": float(pct_ret),
        "pct_true_retained_top_hired": float(pct_both),
        "overall_accuracy": float(oa),
        "final_score": float(fs),
    }

def policy_table(candidates: pd.DataFrame, score_col: str, protected_col: str, top_col: str, retained_col: str,
                 thresholds: np.ndarray) -> pd.DataFrame:
    """Evaluate a range of thresholds for one score column."""
    rows = []
    scores = candidates[score_col].to_numpy()
    prot = candidates[protected_col].to_numpy().astype(int)
    top = candidates[top_col].to_numpy().astype(int)
    ret = candidates[retained_col].to_numpy().astype(int)

    for t in thresholds:
        hire = (scores >= t).astype(int)
        s = summarize_policy(hire, top, ret, prot)
        s["threshold"] = float(t)
        rows.append(s)
    return pd.DataFrame(rows)
