import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold


def cross_val_mean_r(X, y, model, n_splits, random_state):
    """
    Compute out-of-fold Pearson r for a single target.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof = np.zeros(len(y))

    for train_idx, val_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict(X[val_idx])

    r, _ = pearsonr(y, oof)
    return r, oof


def summarize_cv(results_dict):
    """
    results_dict: {target: r}
    """
    s = pd.Series(results_dict)
    return {
        "per_trait_r": s,
        "mean_r": s.mean()
    }
