"""Run bleeding-edge LLM experiments on Train-CV + Dev only.

Stages:
1. Haiku prompt self-consistency (multiple prompt variants).
2. Trait-focused prompt variant.
3. Second frontier judge (Sonnet).
4. LLM behavioral subfeatures.
5. Rank-oriented meta layer over the best stage.

No Test evaluation is performed here.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import SEED, TARGETS, TRAITS, load_data, split_data
from .experiment import build_bases, evaluate
from .metrics import report
from .stack import build_meta_features, make_splits

E5 = "intfloat/e5-large-v2"
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"


def core_specs():
    return [
        ("llm:" + HAIKU, {"variant": "general"}),
        ("embpsvr:" + E5, {"C": 4.0, "gamma": "scale"}),
        ("embsvr:" + E5, {"C": 4.0, "gamma": "scale"}),
        ("tfidf_char_all", {"alpha": 8.0}),
        ("tfidf_word1_all", {"alpha": 8.0}),
        ("engineered", {"alpha": 5.0}),
        ("engineered_gbm", {}),
    ]


def rank_normalize_matrix(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for j in range(x.shape[1]):
        r = rankdata(x[:, j], method="average")
        out[:, j] = (r - 0.5) / len(r)
    return out


def evaluate_rank_meta(specs, label: str):
    """Evaluate a rank-oriented meta learner on Train-CV + Dev."""
    df = load_data()
    train, dev, _test = split_data(df)
    y = train[TARGETS].to_numpy(dtype=float)
    ydev = dev[TARGETS].to_numpy(dtype=float)
    splits = make_splits(len(train), seed=SEED)

    base_oof = {}
    base_dev = {}
    for base in build_bases(specs):
        base_oof[base.name] = rank_normalize_matrix(base.oof(train, y, splits))
        base.fit_full(train, y)
        base_dev[base.name] = rank_normalize_matrix(base.predict(dev))

    sel_oof = build_meta_features(base_oof, cross_trait=False)
    sel_dev = build_meta_features(base_dev, cross_trait=False)
    pred_oof = np.zeros_like(y)
    pred_dev = np.zeros_like(ydev)

    for i in range(len(TRAITS)):
        target_rank = rank_normalize_matrix(y[:, [i]]).ravel()
        meta = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=4.0))])
        meta.fit(sel_oof(i), target_rank)
        pred_oof[:, i] = meta.predict(sel_oof(i))
        pred_dev[:, i] = meta.predict(sel_dev(i))

    print(f"\n=== {label} rank-meta ===")
    report("RANK oof", y, pred_oof)
    report("RANK dev", ydev, pred_dev)


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY must be set.")

    specs0 = core_specs()
    evaluate(specs0, meta_alpha=4.0, cross_trait=False, label="current winning core", show_bases=False)

    specs1 = specs0 + [
        ("llm:" + HAIKU, {"variant": "evidence"}),
        ("llm:" + HAIKU, {"variant": "ranked"}),
    ]
    evaluate(specs1, meta_alpha=4.0, cross_trait=False, label="stage 1: haiku self-consistency")

    specs2 = specs1 + [
        ("llm:" + HAIKU, {"variant": "trait_focus"}),
    ]
    evaluate(specs2, meta_alpha=4.0, cross_trait=False, label="stage 2: trait-focused haiku")

    specs3 = specs2 + [
        ("llm:" + SONNET, {"variant": "general"}),
    ]
    evaluate(specs3, meta_alpha=4.0, cross_trait=False, label="stage 3: add sonnet judge")

    specs4 = specs3 + [
        ("llmfeat:" + HAIKU, {"alpha": 5.0}),
    ]
    evaluate(specs4, meta_alpha=4.0, cross_trait=False, label="stage 4: add llm subfeatures")

    evaluate_rank_meta(specs4, label="stage 5: rank-oriented decision layer")


if __name__ == "__main__":
    main()
