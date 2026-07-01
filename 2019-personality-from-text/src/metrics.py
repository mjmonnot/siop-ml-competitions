"""Scoring helpers for the competition metric: mean Pearson r across 5 traits.

Because the metric is correlation, only the rank/linear ordering of predictions
within each trait matters; absolute calibration is irrelevant. Pearson r is
invariant to affine transforms, so per-trait standardization never changes the
score (it only matters when *combining* models on a common scale).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from .data import TRAITS


def safe_pearsonr(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def per_trait_r(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred are (n, 5) arrays in TRAITS order."""
    return {t: safe_pearsonr(y_true[:, i], y_pred[:, i]) for i, t in enumerate(TRAITS)}


def mean_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rs = per_trait_r(y_true, y_pred)
    return float(np.mean(list(rs.values())))


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Print a one-line per-trait + mean report and return the dict."""
    rs = per_trait_r(y_true, y_pred)
    m = float(np.mean(list(rs.values())))
    cells = "  ".join(f"{t}={rs[t]:.3f}" for t in TRAITS)
    print(f"[{name:>16}] mean_r={m:.4f}   {cells}")
    rs["MEAN"] = m
    return rs
