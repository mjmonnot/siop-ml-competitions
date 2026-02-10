from __future__ import annotations
import numpy as np
import pandas as pd

def adverse_impact_ratio(hire: np.ndarray, protected: np.ndarray) -> float:
    """Compute AIR = (selection rate protected) / (selection rate non-protected).

    Parameters
    ----------
    hire : array of {0,1}
    protected : array of {0,1}

    Returns
    -------
    float
        AIR. If a denominator selection rate is 0, returns np.nan.
    """
    hire = np.asarray(hire).astype(int)
    protected = np.asarray(protected).astype(int)

    sel_prot = hire[protected == 1].mean() if (protected == 1).any() else np.nan
    sel_non  = hire[protected == 0].mean() if (protected == 0).any() else np.nan
    if sel_non == 0 or np.isnan(sel_non):
        return np.nan
    return sel_prot / sel_non

def unfairness_from_air(air: float) -> float:
    """Unfairness = |1 - AIR| * 100"""
    if air is None or np.isnan(air):
        return np.nan
    return abs(1.0 - air) * 100.0

def overall_accuracy_components(hire: np.ndarray, top: np.ndarray, retained: np.ndarray):
    """Compute the three competition hit-rates (as proportions)."""
    hire = np.asarray(hire).astype(int)
    top = np.asarray(top).astype(int)
    retained = np.asarray(retained).astype(int)

    both = (top == 1) & (retained == 1)

    # Selected true / total true (recall-like)
    pct_top = (hire[top == 1].sum() / max(1, (top == 1).sum()))
    pct_ret = (hire[retained == 1].sum() / max(1, (retained == 1).sum()))
    pct_both = (hire[both].sum() / max(1, both.sum()))
    return pct_top, pct_ret, pct_both

def overall_accuracy_score(hire: np.ndarray, top: np.ndarray, retained: np.ndarray) -> float:
    pct_top, pct_ret, pct_both = overall_accuracy_components(hire, top, retained)
    return pct_top * 25.0 + pct_ret * 25.0 + pct_both * 50.0

def final_score(hire: np.ndarray, top: np.ndarray, retained: np.ndarray, protected: np.ndarray) -> float:
    """Final_score = Overall_accuracy - Unfairness."""
    oa = overall_accuracy_score(hire, top, retained)
    air = adverse_impact_ratio(hire, protected)
    unf = unfairness_from_air(air)
    return oa - unf

def summarize_policy(hire: np.ndarray, top: np.ndarray, retained: np.ndarray, protected: np.ndarray) -> dict:
    """Return a compact summary dict for reporting."""
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
    """Evaluate many thresholds for a single scoring column.

    candidates must include:
    - score_col: higher is better
    - protected_col, top_col, retained_col: true labels (for training/validation only)

    Returns DataFrame with one row per threshold.
    """
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
