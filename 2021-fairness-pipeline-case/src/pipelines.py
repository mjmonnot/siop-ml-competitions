"""Competition-style and standards-aligned pipelines for freeze evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor

from .data import feature_matrix


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def competition_style_scores(
    fit: pd.DataFrame,
    predict_on: pd.DataFrame,
    *,
    use_protected_proxy: bool = True,
) -> np.ndarray:
    """
    Team Procrustination–style ensemble scores.

    When use_protected_proxy=True (default contest recipe), Model 4 trains on
    (Protected_Group==1 AND Retained==1). Set False for an ablation that keeps
    the same performance ensemble without the fairness hack.
    """
    # Match the winner's complete-case habit on the fit set for the proxy labels,
    # but keep rows that have the required outcomes.
    fit_cc = fit.dropna(subset=["High_Performer", "Retained", "Protected_Group"]).copy()
    if "Overall_Rating" in fit_cc.columns:
        fit_cc = fit_cc.dropna(subset=["Overall_Rating"])

    X_fit, X_pred = feature_matrix(fit_cc, predict_on, drop_protected=True)

    # Model 1: High_Performer (simple)
    m1 = XGBClassifier(
        scale_pos_weight=1.5,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )
    m1.fit(X_fit, fit_cc["High_Performer"].astype(int))
    p1 = _zscore(m1.predict_proba(X_pred)[:, 1])

    # Model 2: Overall_Rating
    m2 = XGBRegressor(
        subsample=0.6,
        n_estimators=200,
        learning_rate=0.05,
        colsample_bytree=0.6,
        max_depth=2,
        min_child_weight=4,
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )
    m2.fit(X_fit, fit_cc["Overall_Rating"].astype(float))
    p2 = _zscore(m2.predict(X_pred))

    # Model 3: High_Performer (winner hyperparameters; slightly fewer trees for CV speed)
    m3 = XGBClassifier(
        learning_rate=0.005411872947900535,
        n_estimators=800,
        max_depth=3,
        min_child_weight=5.818676232053935,
        gamma=0.05591980172280099,
        subsample=0.5744551033482959,
        colsample_bytree=0.5226781217635239,
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )
    m3.fit(X_fit, fit_cc["High_Performer"].astype(int))
    p3 = _zscore(m3.predict_proba(X_pred)[:, 1])

    performance = (p1 * 0.5 + p2 * 0.2 + p3 * 0.3)

    if not use_protected_proxy:
        return performance

    y4 = ((fit_cc["Protected_Group"] == 1) & (fit_cc["Retained"] == 1)).astype(int)
    m4 = XGBClassifier(
        scale_pos_weight=3,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )
    m4.fit(X_fit, y4)
    p4 = _zscore(m4.predict_proba(X_pred)[:, 1])
    return performance * 0.9 + p4 * 0.1


def competition_style_hires(scores: np.ndarray) -> np.ndarray:
    """Median cut: hire the top half (winner decision rule)."""
    cutoff = np.median(scores)
    return (scores > cutoff).astype(int)


def job_success_index(df: pd.DataFrame) -> pd.Series:
    top = df["High_Performer"].astype(int)
    ret = df["Retained"].astype(int)
    both = ((top == 1) & (ret == 1)).astype(int)
    return 0.25 * top + 0.25 * ret + 0.50 * both


def standards_aligned_oof_scores(
    fit: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Out-of-fold Job Success Index scores for threshold governance."""
    fit = fit.reset_index(drop=True)
    y = job_success_index(fit).to_numpy()
    oof = np.zeros(len(fit), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr_idx, va_idx in kf.split(fit):
        tr, va = fit.iloc[tr_idx], fit.iloc[va_idx]
        X_tr, X_va = feature_matrix(tr, va, drop_protected=True)
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=4,
            verbosity=0,
        )
        model.fit(X_tr, y[tr_idx])
        oof[va_idx] = model.predict(X_va)
    return oof


def standards_aligned_predict(
    fit: pd.DataFrame,
    predict_on: pd.DataFrame,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Fit JSI model on fit set; score predict_on. Never uses Protected_Group."""
    y = job_success_index(fit).to_numpy()
    X_fit, X_pred = feature_matrix(fit, predict_on, drop_protected=True)
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        verbosity=0,
    )
    model.fit(X_fit, y)
    return model.predict(X_pred)


def choose_threshold_from_oof(
    fit: pd.DataFrame,
    oof_scores: np.ndarray,
    *,
    air_floor: float | None = None,
    selection_rate: float | None = 0.50,
    selection_tol: float = 0.02,
    n_grid: int = 200,
) -> dict:
    """
    Pick a cut score on OOF predictions using the competition metric.

    Default ``selection_rate=0.50`` mirrors the winners' top-half hiring volume.
    Without a selection-rate constraint the metric is gamed by hiring nearly
    everyone (recall components saturate; AIR ≈ 1) — see KNOWN_LANDMINES.md.
    """
    import sys
    from pathlib import Path

    scoring_dir = Path(__file__).resolve().parents[1] / "02_standards_aligned_pipeline"
    if str(scoring_dir) not in sys.path:
        sys.path.insert(0, str(scoring_dir))
    from scoring_function import summarize_policy

    if selection_rate is None:
        thresholds = np.quantile(oof_scores, np.linspace(0.05, 0.95, n_grid))
    else:
        # Search a tight band around the target hire rate.
        q_lo = max(0.01, 1.0 - selection_rate - selection_tol)
        q_hi = min(0.99, 1.0 - selection_rate + selection_tol)
        thresholds = np.quantile(oof_scores, np.linspace(q_lo, q_hi, n_grid))

    best = None
    for t in thresholds:
        hire = (oof_scores >= t).astype(int)
        s = summarize_policy(
            hire,
            fit["High_Performer"].to_numpy(),
            fit["Retained"].to_numpy(),
            fit["Protected_Group"].to_numpy(),
        )
        s["threshold"] = float(t)
        if selection_rate is not None:
            if abs(s["selection_rate_overall"] - selection_rate) > selection_tol:
                continue
        if air_floor is not None and (np.isnan(s["AIR"]) or s["AIR"] < air_floor):
            continue
        if best is None or s["final_score"] > best["final_score"]:
            best = s
    if best is None:
        raise RuntimeError("No threshold satisfied constraints.")
    return best
