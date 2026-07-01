"""Leakage-safe out-of-fold (OOF) stacking engine.

Protocol:
- Base learners produce OOF predictions on Train via shared K-fold splits
  (every base uses identical folds so meta-features are aligned).
- Each base is refit on full Train to predict Dev/Test.
- A per-trait meta-learner is fit on the Train OOF meta-features only.

Nothing here ever fits on Dev/Test. Vectorizers/scalers live inside the
estimator pipelines, so they are refit fold-internally during OOF.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import SEED, TRAITS, TRAIT_TO_PROMPT


def make_splits(n: int, n_splits: int = 5, seed: int = SEED):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n)))


# ---------------------------------------------------------------------------
# Base learners
# ---------------------------------------------------------------------------
class StackBase:
    """A single-feature-matrix, multi-output base learner.

    extract_fn(df) -> 1-D array of text (for sparse pipelines) or 2-D dense X.
    est_factory()  -> a fresh unfitted sklearn estimator (Pipeline).
    """

    def __init__(self, name: str, extract_fn, est_factory):
        self.name = name
        self.extract_fn = extract_fn
        self.est_factory = est_factory
        self._full = None

    def oof(self, train_df, Y, splits) -> np.ndarray:
        X = self.extract_fn(train_df)
        oof = np.zeros((len(train_df), len(TRAITS)))
        for tr, va in splits:
            est = self.est_factory()
            est.fit(_take(X, tr), Y[tr])
            oof[va] = est.predict(_take(X, va))
        return oof

    def fit_full(self, train_df, Y):
        X = self.extract_fn(train_df)
        self._full = self.est_factory()
        self._full.fit(X, Y)
        return self

    def predict(self, df) -> np.ndarray:
        return self._full.predict(self.extract_fn(df))


class LLMBase:
    """Zero-shot LLM trait scores as a base learner (no label fitting).

    The LLM never sees ground-truth labels, so the same cached scores serve as
    both OOF and full-fit predictions.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        provider: str = "anthropic",
        variant: str = "general",
    ):
        self.name = name
        self.model_name = model_name
        self.provider = provider
        self.variant = variant

    def _scores(self, df) -> np.ndarray:
        from .llm_extract import score_rows
        return score_rows(
            df,
            model_name=self.model_name,
            provider=self.provider,
            variant=self.variant,
        )

    def oof(self, train_df, Y, splits) -> np.ndarray:
        return self._scores(train_df)

    def fit_full(self, train_df, Y):
        return self

    def predict(self, df) -> np.ndarray:
        return self._scores(df)


class LLMFeatureBase(StackBase):
    """LLM behavioral subfeatures with a supervised head fit on Train only."""

    def __init__(self, name: str, model_name: str, est_factory, provider: str = "anthropic"):
        from .llm_extract import score_subfeatures

        self.model_name = model_name
        self.provider = provider

        def extract(df):
            return score_subfeatures(df, model_name=model_name, provider=provider)

        super().__init__(name, extract, est_factory)


class LLMQBase:
    """Role-play + questionnaire (item-level BFI-2) with a reverse-scored direct
    aggregate. No label fitting -> cached scores serve OOF and full-fit alike.

    use_summary=True runs the two-stage persona-summary variant (Liu et al. 2025).
    """

    def __init__(self, name: str, model_name: str, provider: str = "anthropic",
                 use_summary: bool = False):
        self.name = name
        self.model_name = model_name
        self.provider = provider
        self.use_summary = use_summary

    def _scores(self, df) -> np.ndarray:
        from .llm_extract import (aggregate_questionnaire, score_questionnaire,
                                  score_questionnaire_summary)
        fn = score_questionnaire_summary if self.use_summary else score_questionnaire
        items = fn(df, model_name=self.model_name, provider=self.provider)
        return aggregate_questionnaire(items)

    def oof(self, train_df, Y, splits) -> np.ndarray:
        return self._scores(train_df)

    def fit_full(self, train_df, Y):
        return self

    def predict(self, df) -> np.ndarray:
        return self._scores(df)


class LLMQFeatureBase(StackBase):
    """Role-play + questionnaire item responses fed to a supervised head (learned
    aggregation, fit on Train only).

    use_summary=True runs the two-stage persona-summary variant (Liu et al. 2025).
    """

    def __init__(self, name: str, model_name: str, est_factory, provider: str = "anthropic",
                 use_summary: bool = False):
        from .llm_extract import score_questionnaire, score_questionnaire_summary

        self.model_name = model_name
        self.provider = provider
        self.use_summary = use_summary
        fn = score_questionnaire_summary if use_summary else score_questionnaire

        def extract(df):
            return fn(df, model_name=model_name, provider=provider)

        super().__init__(name, extract, est_factory)


class MatchedPromptBase:
    """Per-trait single-target learners, each on that trait's eliciting prompt.

    Captures prompt-specific signal that an all-text model dilutes.
    """

    def __init__(self, name: str, est_factory):
        self.name = name
        self.est_factory = est_factory
        self._full = {}

    def oof(self, train_df, Y, splits) -> np.ndarray:
        oof = np.zeros((len(train_df), len(TRAITS)))
        for i, t in enumerate(TRAITS):
            col = TRAIT_TO_PROMPT[t]
            X = train_df[col].to_numpy()
            for tr, va in splits:
                est = self.est_factory()
                est.fit(X[tr], Y[tr, i])
                oof[va, i] = est.predict(X[va])
        return oof

    def fit_full(self, train_df, Y):
        for i, t in enumerate(TRAITS):
            col = TRAIT_TO_PROMPT[t]
            est = self.est_factory()
            est.fit(train_df[col].to_numpy(), Y[:, i])
            self._full[t] = est
        return self

    def predict(self, df) -> np.ndarray:
        out = np.zeros((len(df), len(TRAITS)))
        for i, t in enumerate(TRAITS):
            col = TRAIT_TO_PROMPT[t]
            out[:, i] = self._full[t].predict(df[col].to_numpy())
        return out


def _take(X, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


# ---------------------------------------------------------------------------
# Meta layer
# ---------------------------------------------------------------------------
def build_meta_features(base_preds: dict, cross_trait: bool):
    """Turn {base_name: (n,5)} into per-trait meta-feature matrices.

    Returns a function trait_index -> (n, k) selecting columns.
    If cross_trait, every trait sees all bases x all traits (target-conditioning);
    otherwise each trait sees only its own column from each base.
    """
    names = list(base_preds.keys())
    stacked = np.concatenate([base_preds[n] for n in names], axis=1)  # (n, 5*B)

    def select(i):
        if cross_trait:
            return stacked
        cols = [b * len(TRAITS) + i for b in range(len(names))]
        return stacked[:, cols]

    return select


def run_stack(bases, train_df, eval_frames: dict, Y_train,
              n_splits: int = 5, seed: int = SEED,
              meta_alpha: float = 1.0, cross_trait: bool = False):
    """Run the full stack.

    eval_frames: {tag: dataframe} to predict (e.g. {'dev': dev_df}).
    Returns dict with 'oof' (n,5), and per-tag predictions (n,5), plus
    the raw base OOF/eval predictions for inspection.
    """
    splits = make_splits(len(train_df), n_splits=n_splits, seed=seed)

    base_oof = {}
    base_eval = {tag: {} for tag in eval_frames}
    for b in bases:
        base_oof[b.name] = b.oof(train_df, Y_train, splits)
        b.fit_full(train_df, Y_train)
        for tag, frame in eval_frames.items():
            base_eval[tag][b.name] = b.predict(frame)

    sel_oof = build_meta_features(base_oof, cross_trait)
    sel_eval = {tag: build_meta_features(base_eval[tag], cross_trait) for tag in eval_frames}

    oof_pred = np.zeros((len(train_df), len(TRAITS)))
    eval_pred = {tag: np.zeros((len(eval_frames[tag]), len(TRAITS))) for tag in eval_frames}
    meta_models = {}
    for i, t in enumerate(TRAITS):
        meta = Pipeline([("sc", StandardScaler()), ("rdg", Ridge(alpha=meta_alpha))])
        meta.fit(sel_oof(i), Y_train[:, i])
        meta_models[t] = meta
        oof_pred[:, i] = meta.predict(sel_oof(i))
        for tag in eval_frames:
            eval_pred[tag][:, i] = meta.predict(sel_eval[tag](i))

    return {
        "oof": oof_pred,
        "eval": eval_pred,
        "base_oof": base_oof,
        "base_eval": base_eval,
        "meta_models": meta_models,
    }


def average_blend(base_preds: dict, weights: dict | None = None) -> np.ndarray:
    """Simple weighted z-score average of base predictions (no meta-learner)."""
    names = list(base_preds.keys())
    w = {n: (weights.get(n, 1.0) if weights else 1.0) for n in names}
    n_rows = base_preds[names[0]].shape[0]
    out = np.zeros((n_rows, len(TRAITS)))
    for i in range(len(TRAITS)):
        acc = np.zeros(n_rows)
        wsum = 0.0
        for n in names:
            col = base_preds[n][:, i]
            sd = col.std()
            z = (col - col.mean()) / sd if sd > 0 else col * 0
            acc += w[n] * z
            wsum += w[n]
        out[:, i] = acc / wsum if wsum else acc
    return out
